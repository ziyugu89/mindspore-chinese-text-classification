"""训练 / 评估循环（MindSpore 2.10 CPU 友好 · numpy 手工 batch）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from tqdm import tqdm


def setup_mindspore(device: str = "CPU", seed: int = 42) -> None:
    ms.set_seed(seed)
    ms.set_device(device)
    # GRAPH_MODE 在 CPU 上通常明显快于 PyNative（首次构图稍慢）
    ms.set_context(mode=ms.GRAPH_MODE)


def _to_numpy(
    xs: Sequence[Sequence[int]],
    ys: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
    )


def _iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        sl = indices[start : start + batch_size]
        yield x[sl], y[sl]


def accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    pred = logits.argmax(axis=1)
    correct = (pred == labels).asnumpy().sum()
    total = int(labels.shape[0])
    return float(correct) / float(total)


def train_one_epoch(
    model: nn.Cell,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: nn.Cell,
    optimizer: nn.Optimizer,
    batch_size: int,
    epoch: int,
) -> Tuple[float, float]:
    model.set_train(True)

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    losses: List[float] = []
    accs: List[float] = []
    batches = list(_iter_batches(x, y, batch_size, shuffle=True, seed=epoch))
    for bx, by in tqdm(batches, desc="train", leave=False):
        data = Tensor(bx)
        label = Tensor(by)
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        losses.append(float(loss.asnumpy()))
        accs.append(accuracy_from_logits(logits, label))
    return float(np.mean(losses)), float(np.mean(accs))


def evaluate(
    model: nn.Cell,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: nn.Cell,
    batch_size: int = 64,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """兼容旧签名：也可传入 GeneratorDataset，但推荐传 numpy。"""
    # 兼容 notebook / CLI 可能仍传 dataset 的情况
    if not isinstance(x, np.ndarray):
        raise TypeError("evaluate 期望 numpy 特征矩阵；请升级调用方")

    model.set_train(False)
    losses: List[float] = []
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    for bx, by in tqdm(
        list(_iter_batches(x, y, batch_size, shuffle=False)),
        desc="eval",
        leave=False,
    ):
        data = Tensor(bx)
        label = Tensor(by)
        logits = model(data)
        loss = loss_fn(logits, label)
        losses.append(float(loss.asnumpy()))
        all_pred.append(logits.argmax(axis=1).asnumpy())
        all_true.append(by)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_true = np.concatenate(all_true) if all_true else np.array([])
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc, y_true, y_pred


def make_dataset(xs, ys, batch_size, shuffle):
    """保留符号以兼容旧 notebook 单元格；内部改为 numpy 打包提示。"""
    raise RuntimeError(
        "已改为 numpy batch：请直接把 train_xs/train_ys 传给 fit/evaluate，"
        "不要再调用 make_dataset。"
    )


def fit(
    model: nn.Cell,
    train_xs: List[List[int]],
    train_ys: List[int],
    test_xs: List[List[int]],
    test_ys: List[int],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    ckpt_path: Path | None = None,
) -> Dict[str, List[float]]:
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    x_train, y_train = _to_numpy(train_xs, train_ys)
    x_test, y_test = _to_numpy(test_xs, test_ys)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, x_train, y_train, loss_fn, optimizer, batch_size, epoch
        )
        te_loss, te_acc, _, _ = evaluate(
            model, x_test, y_test, loss_fn, batch_size=batch_size
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}",
            flush=True,
        )

        if ckpt_path is not None and te_acc >= best_acc:
            best_acc = te_acc
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, str(ckpt_path))
            print(f"  → 保存最优权重 test_acc={best_acc:.4f} → {ckpt_path}", flush=True)

    return history


def load_weights(model: nn.Cell, ckpt_path: Path) -> None:
    param_dict = load_checkpoint(str(ckpt_path))
    load_param_into_net(model, param_dict)
