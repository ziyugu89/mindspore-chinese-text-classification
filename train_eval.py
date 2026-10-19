#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行入口（可选）：与 notebooks/train_eval.ipynb 同一流水线。

用法（在 conda ms_docs_try 中）:
  python train_eval.py --epochs 5 --model textcnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    build_vocab,
    ensure_chnsenticorp,
    load_reviews,
    prepare_arrays,
    save_vocab,
    train_test_split,
)
from src.model import build_model  # noqa: E402
from src.train import evaluate, fit, setup_mindspore  # noqa: E402
from src.viz import plot_confusion, plot_history, plot_sample_predictions  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChnSentiCorp 中文情感分类（MindSpore）")
    p.add_argument("--model", default="textcnn", choices=["textcnn", "bilstm"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--max-vocab", type=int, default=8000)
    p.add_argument("--device", default="CPU")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_mindspore(device=args.device, seed=args.seed)

    csv_path = ensure_chnsenticorp(ROOT / "data")
    samples = load_reviews(csv_path)
    print(f"样本总数: {len(samples)}")

    train_samples, test_samples = train_test_split(samples, test_ratio=0.2, seed=args.seed)
    vocab = build_vocab([t for t, _ in train_samples], max_vocab_size=args.max_vocab)
    save_vocab(vocab, ROOT / "checkpoints" / "vocab.txt")
    print(f"词表大小: {len(vocab)}  训练:{len(train_samples)}  测试:{len(test_samples)}")

    train_xs, train_ys = prepare_arrays(train_samples, vocab, max_len=args.max_len)
    test_xs, test_ys = prepare_arrays(test_samples, vocab, max_len=args.max_len)

    model, model_name = build_model(args.model, len(vocab), embed_dim=args.embed_dim)
    print(f"模型: {model_name}")

    ckpt = ROOT / "checkpoints" / f"{args.model}_best.ckpt"
    history = fit(
        model,
        train_xs,
        train_ys,
        test_xs,
        test_ys,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=ckpt,
    )

    from mindspore import nn

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    x_test = np.asarray(test_xs, dtype=np.int32)
    y_test = np.asarray(test_ys, dtype=np.int32)
    _, acc, y_true, y_pred = evaluate(
        model, x_test, y_test, loss_fn, batch_size=args.batch_size
    )
    print(f"最终测试准确率: {acc:.4f}")

    demo = ROOT / "assets" / "demo"
    plot_history(history, demo / "accuracy_loss_curves.png")
    plot_confusion(y_true, y_pred, demo / "confusion_matrix.png")

    # 抽若干样例画图
    show = []
    for i, ((text, y_t), y_p) in enumerate(zip(test_samples, y_pred)):
        if len(show) >= 12:
            break
        show.append((text, int(y_t), int(y_p)))
    plot_sample_predictions(show, demo / "sample_predictions.png")

    if acc >= 0.80:
        print("✅ 已达到 ≥80% 准确率门槛")
    else:
        print("⚠️ 未达 80%。可增大 --epochs / --embed-dim，或检查数据是否下载完整。")


if __name__ == "__main__":
    main()
