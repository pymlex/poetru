from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class WatermarkConfig:
    """Parameters for the soft green-list watermark of Kirchenbauer et al., 2023."""

    gamma: float = 0.25
    delta: float = 2.0
    seed_scheme: str = "kirchenbauer_soft"


@dataclass(frozen=True)
class WatermarkDetectionResult:
    """Outcome of a one-sided green-list z-test on token ids."""

    green_count: int
    total_count: int
    z_score: float
    p_value: float
    is_watermarked: bool


def _rng_for_prev_token(prev_token_id: int, vocab_size: int, cfg: WatermarkConfig) -> np.random.Generator:
    """Builds a deterministic NumPy generator keyed by the previous token id.

    Args:
        prev_token_id: Integer id of token at position `t-1`.
        vocab_size: Vocabulary size used to hash the seed string.
        cfg: Watermark hyperparameters including the seed scheme label.

    Returns:
        NumPy Generator initialised from a SHA-256 digest.
    """

    seed_material = f"{cfg.seed_scheme}:{prev_token_id}:{vocab_size}".encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def greenlist_for_prev_token(prev_token_id: int, vocab_size: int, cfg: WatermarkConfig) -> np.ndarray:
    """Samples the green token set for position `t` given token `t-1`.

    Args:
        prev_token_id: Previous token id.
        vocab_size: Vocabulary size.
        cfg: Watermark configuration with `gamma` controlling list mass.

    Returns:
        Sorted integer array of green token ids with length `floor(gamma * vocab_size)`.
    """

    rng = _rng_for_prev_token(prev_token_id, vocab_size, cfg)
    green_size = max(1, int(cfg.gamma * vocab_size))
    perm = rng.permutation(vocab_size)
    return np.sort(perm[:green_size])


def apply_watermark_bias(logits: Tensor, prev_token_id: int, cfg: WatermarkConfig) -> Tensor:
    """Adds `delta` to logits of green-list tokens before sampling.

    Args:
        logits: Last-step logits shaped `[vocab_size]`.
        prev_token_id: Token id at position `t-1`.
        cfg: Watermark configuration.

    Returns:
        Logits with green-list bias applied.
    """

    vocab_size = int(logits.shape[-1])
    green = greenlist_for_prev_token(prev_token_id, vocab_size, cfg)
    out = logits.clone()
    out[green] = out[green] + cfg.delta
    return out


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_p: float,
    prev_token_id: int,
    cfg: WatermarkConfig,
    apply_watermark: bool,
) -> int:
    """Samples one token with temperature, nucleus filtering, and optional watermark bias.

    Args:
        logits: Last-step logits shaped `[vocab_size]`.
        temperature: Softmax temperature.
        top_p: Nucleus mass threshold in `(0, 1]`.
        prev_token_id: Previous token id used for green-list hashing.
        cfg: Watermark configuration.
        apply_watermark: When `True`, applies green-list logit bias.

    Returns:
        Sampled token id as Python integer.
    """

    scaled = logits / max(temperature, 1e-8)
    if apply_watermark:
        scaled = apply_watermark_bias(scaled, prev_token_id, cfg)

    probs = F.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    filtered_probs = sorted_probs * keep.to(dtype=sorted_probs.dtype)
    filtered_probs = filtered_probs / filtered_probs.sum()
    choice = torch.multinomial(filtered_probs, num_samples=1)
    return int(sorted_idx[choice].item())


def detect_watermark(
    token_ids: list[int],
    vocab_size: int,
    cfg: WatermarkConfig,
    z_threshold: float = 4.0,
) -> WatermarkDetectionResult:
    """Runs the one-sided green-list z-test from Kirchenbauer et al., 2023.

    Args:
        token_ids: Generated token sequence including optional prompt tokens.
        vocab_size: Vocabulary size.
        cfg: Watermark configuration with `gamma`.
        z_threshold: Detection threshold on the z statistic.

    Returns:
        WatermarkDetectionResult with counts, z-score, p-value, and boolean flag.
    """

    if len(token_ids) < 2:
        return WatermarkDetectionResult(0, 0, 0.0, 1.0, False)

    green_hits = 0
    total = 0
    for prev_id, cur_id in zip(token_ids[:-1], token_ids[1:]):
        green = greenlist_for_prev_token(int(prev_id), vocab_size, cfg)
        total += 1
        if int(cur_id) in set(green.tolist()):
            green_hits += 1

    if total == 0:
        return WatermarkDetectionResult(0, 0, 0.0, 1.0, False)

    expected = cfg.gamma * total
    variance = cfg.gamma * (1.0 - cfg.gamma) * total
    z = (green_hits - expected) / np.sqrt(max(variance, 1e-8))
    p_value = float(1.0 - stats.norm.cdf(z))
    return WatermarkDetectionResult(
        green_count=green_hits,
        total_count=total,
        z_score=float(z),
        p_value=p_value,
        is_watermarked=bool(z >= z_threshold),
    )
