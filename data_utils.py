from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bpe_tokenizer import ByteBPETokenizerWrapper


def load_poetry_texts(
    dataset_name: str,
    split: str,
    sample_fraction: float = 1.0,
    seed: int = 3407,
) -> list[str]:
    """Loads poem bodies from Hugging Face and optionally subsamples them.

    Args:
        dataset_name: Hub dataset id.
        split: Split name such as `train`.
        sample_fraction: Fraction of rows kept when below `1.0`.
        seed: Shuffle seed used before subsampling.

    Returns:
        List of non-empty poem strings.
    """

    ds = load_dataset(dataset_name, split=split)
    texts = [str(row["text"]).strip() for row in ds if str(row["text"]).strip()]
    if sample_fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(len(texts) * sample_fraction))
        idx = rng.choice(len(texts), size=n, replace=False)
        texts = [texts[int(i)] for i in idx]
    return texts


def iter_poetry_texts(
    dataset_name: str,
    split: str,
    sample_fraction: float = 1.0,
    seed: int = 3407,
) -> Iterator[str]:
    """Streams poem strings for tokenizer training without materialising the full corpus.

    Args:
        dataset_name: Hub dataset id.
        split: Split name such as `train`.
        sample_fraction: Fraction of rows kept when below `1.0`.
        seed: Shuffle seed used before subsampling.

    Returns:
        Iterator over poem strings.
    """

    ds = load_dataset(dataset_name, split=split, streaming=True)
    if sample_fraction >= 1.0:
        for row in ds:
            text = str(row["text"]).strip()
            if text:
                yield text
        return

    rng = np.random.default_rng(seed)
    for row in ds:
        if rng.random() <= sample_fraction:
            text = str(row["text"]).strip()
            if text:
                yield text


class PoetryTokenDataset(Dataset):
    """On-the-fly tokenisation of poem strings."""

    def __init__(self, texts: list[str], tokenizer: ByteBPETokenizerWrapper, max_seq_len: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.texts[idx], add_eos=True)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long)


def collate_poetry_batch(
    batch: list[torch.Tensor],
    pad_id: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads variable-length token rows into a rectangular batch.

    Args:
        batch: List of 1D token tensors.
        pad_id: Padding token id.
        max_seq_len: Hard upper bound on sequence length inside a batch.

    Returns:
        Tuple `(input_ids, attention_mask)` with shapes `[batch, sequence]`.
    """

    max_len = min(max_seq_len, max(int(x.shape[0]) for x in batch))
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, seq in enumerate(batch):
        cur = seq[:max_len]
        n = int(cur.shape[0])
        input_ids[i, :n] = cur
        attention_mask[i, :n] = True

    return input_ids, attention_mask


def build_dataloader(
    dataset: Dataset,
    pad_id: int,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Constructs a DataLoader with the poetry collate function.

    Args:
        dataset: Tokenised poetry dataset.
        pad_id: Padding token id.
        max_seq_len: Context window.
        batch_size: Micro batch size.
        shuffle: Whether to shuffle rows each epoch.
        num_workers: Worker processes for loading.

    Returns:
        DataLoader yielding `(input_ids, attention_mask)` batches.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda rows: collate_poetry_batch(rows, pad_id=pad_id, max_seq_len=max_seq_len),
    )


def token_length_histogram(
    texts: list[str],
    tokenizer: ByteBPETokenizerWrapper,
    max_seq_len: int,
    output_path: Path,
) -> dict[str, float]:
    """Computes token length statistics and writes a histogram image.

    Args:
        texts: Poem strings.
        tokenizer: Trained BPE wrapper.
        max_seq_len: Upper cap used for histogram range.
        output_path: PNG destination.

    Returns:
        Dictionary with count, mean, std, min, max, and quartiles.
    """

    import matplotlib.pyplot as plt

    lengths = np.array([len(tokenizer.encode(t, add_eos=True)) for t in tqdm(texts, desc="Token lengths")], dtype=np.int64)
    stats = {
        "count": float(lengths.size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": float(lengths.min()),
        "p25": float(np.percentile(lengths, 25)),
        "p50": float(np.percentile(lengths, 50)),
        "p75": float(np.percentile(lengths, 75)),
        "max": float(lengths.max()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=128, range=(0, max_seq_len), color="steelblue")
    plt.title("Token length distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return stats


def save_json(path: Path, payload: dict) -> None:
    """Writes a JSON dictionary with UTF-8 encoding.

    Args:
        path: Destination file.
        payload: Serializable mapping.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
