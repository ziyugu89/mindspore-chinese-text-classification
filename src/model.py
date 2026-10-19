"""TextCNN / BiLSTM 模型定义（MindSpore 2.10 · CPU 友好）。"""

from __future__ import annotations

from typing import Sequence, Tuple

import mindspore as ms
from mindspore import nn, ops


class TextCNN(nn.Cell):
    """多尺度一维卷积文本分类（Kim 2014 思路，Conv1d 实现更利 CPU）。

    输入: (batch, seq_len) int32 token ids
    输出: (batch, num_classes) logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 2,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Conv1d: 输入通道=embed_dim，在序列维上卷积
        self.convs = nn.CellList(
            [
                nn.Conv1d(
                    embed_dim,
                    num_filters,
                    kernel_size=k,
                    pad_mode="valid",
                    has_bias=True,
                )
                for k in kernel_sizes
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Dense(num_filters * len(kernel_sizes), num_classes)
        self.num_filters = num_filters
        self.kernel_sizes = tuple(kernel_sizes)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # x: (N, L) → emb: (N, L, D) → (N, D, L) 供 Conv1d
        emb = self.embedding(x)
        emb = ops.transpose(emb, (0, 2, 1))
        pooled = []
        for conv in self.convs:
            h = self.relu(conv(emb))  # (N, F, L-k+1)
            h = ops.ReduceMax(keep_dims=False)(h, 2)  # (N, F)
            pooled.append(h)
        out = ops.concat(pooled, axis=1)
        out = self.dropout(out)
        return self.fc(out)


class BiLSTMClassifier(nn.Cell):
    """可选对比模型：双向 LSTM + 最后时刻拼接后接全连接。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            has_bias=True,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Dense(hidden_size * 2, num_classes)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        emb = self.embedding(x)
        _outputs, (h, _c) = self.lstm(emb)
        forward_last = h[-2]
        backward_last = h[-1]
        feat = ops.concat((forward_last, backward_last), axis=1)
        feat = self.dropout(feat)
        return self.fc(feat)


def build_model(
    name: str,
    vocab_size: int,
    embed_dim: int = 128,
    num_classes: int = 2,
    num_filters: int = 100,
) -> Tuple[nn.Cell, str]:
    name = name.lower()
    if name == "textcnn":
        return (
            TextCNN(
                vocab_size,
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_filters=num_filters,
            ),
            "TextCNN",
        )
    if name in ("bilstm", "lstm"):
        return (
            BiLSTMClassifier(vocab_size, embed_dim=embed_dim, num_classes=num_classes),
            "BiLSTM",
        )
    raise ValueError(f"未知模型: {name}，可选 textcnn / bilstm")
