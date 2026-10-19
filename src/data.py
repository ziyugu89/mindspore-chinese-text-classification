"""数据下载、字级分词与 Dataset 构建。"""

from __future__ import annotations

import csv
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ChnSentiCorp 酒店评论全集（label, review）
# 来源：SophonPlus/ChineseNlpCorpus
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/"
    "master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_chnsenticorp(
    data_dir: Path | None = None,
    url: str = DEFAULT_DATA_URL,
    timeout: int = 120,
) -> Path:
    """下载 ChnSentiCorp CSV（若不存在）。返回本地路径。"""
    data_dir = data_dir or (project_root() / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ChnSentiCorp_htl_all.csv"
    if csv_path.exists() and csv_path.stat().st_size > 1000:
        return csv_path

    print(f"[data] 正在下载 ChnSentiCorp → {csv_path}")
    print(f"[data] URL: {url}")
    try:
        urllib.request.urlretrieve(url, csv_path)
    except Exception as exc:  # noqa: BLE001 — 给用户明确本地补救提示
        raise RuntimeError(
            "自动下载失败。请手动将 ChnSentiCorp_htl_all.csv 放到 data/ 目录。\n"
            f"原始地址: {url}\n"
            f"错误: {exc}"
        ) from exc

    if not csv_path.exists() or csv_path.stat().st_size < 1000:
        raise RuntimeError(f"下载文件异常或过小: {csv_path}")
    print(f"[data] 下载完成，大小 {csv_path.stat().st_size / 1024:.1f} KB")
    return csv_path


def load_reviews(csv_path: Path) -> List[Tuple[str, int]]:
    """读取 (text, label)。label: 1=正向, 0=负向。"""
    rows: List[Tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        # 常见列名：label, review
        for row in reader:
            if not row:
                continue
            label_raw = row.get("label") or row.get("Label") or row.get("sentiment")
            text = row.get("review") or row.get("Review") or row.get("text") or ""
            text = (text or "").strip()
            if label_raw is None or not text:
                continue
            try:
                label = int(float(label_raw))
            except ValueError:
                continue
            if label not in (0, 1):
                continue
            rows.append((text, label))
    if len(rows) < 100:
        raise RuntimeError(f"有效样本过少（{len(rows)}），请检查 CSV 格式。")
    return rows


def tokenize_chars(text: str) -> List[str]:
    """字级分词：中文情感分类在 CPU 上足够稳，无需 jieba。"""
    return [ch for ch in text.strip() if not ch.isspace()]


def build_vocab(
    texts: Sequence[str],
    max_vocab_size: int = 8000,
    min_freq: int = 2,
) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize_chars(t))
    # 保留 pad / unk
    vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab_size:
            break
    return vocab


def encode(
    text: str,
    vocab: Dict[str, int],
    max_len: int = 128,
) -> List[int]:
    ids = [vocab.get(ch, UNK_ID) for ch in tokenize_chars(text)]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def train_test_split(
    samples: List[Tuple[str, int]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    data = list(samples)
    rng.shuffle(data)
    n_test = max(1, int(len(data) * test_ratio))
    return data[n_test:], data[:n_test]


def prepare_arrays(
    samples: List[Tuple[str, int]],
    vocab: Dict[str, int],
    max_len: int = 128,
) -> Tuple[List[List[int]], List[int]]:
    xs, ys = [], []
    for text, label in samples:
        xs.append(encode(text, vocab, max_len=max_len))
        ys.append(label)
    return xs, ys


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{idx}\n")


def load_vocab(path: Path) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            vocab[parts[0]] = int(parts[1])
    return vocab
