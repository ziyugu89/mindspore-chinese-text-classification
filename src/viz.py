"""训练曲线、混淆矩阵与样例预测可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# 中文字体：Windows 常见字体回退；没有则用默认（仍可出图）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_history(history: Dict[str, List[float]], save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["test_acc"], label="test")
    axes[1].axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% 门槛")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    labels: Sequence[str] = ("负向", "正向"),
) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (ChnSentiCorp)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sample_predictions(
    samples: List[Tuple[str, int, int]],
    save_path: Path,
    max_chars: int = 40,
) -> Path:
    """samples: (text, true_label, pred_label)"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    label_name = {0: "负", 1: "正"}
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(samples) + 1)))
    ax.axis("off")
    lines = ["样例预测（真值 / 预测）", ""]
    for i, (text, y_t, y_p) in enumerate(samples, 1):
        short = text.replace("\n", " ")[:max_chars]
        mark = "✓" if y_t == y_p else "✗"
        lines.append(
            f"{i}. [{mark}] 真={label_name[y_t]} 预={label_name[y_p]}  {short}"
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="sans-serif", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
