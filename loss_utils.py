from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_language_modeling_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> Tensor:
    """Masked next-token cross entropy averaged over valid positions.

    Args:
        logits: Model outputs `[batch, sequence, vocab_size]`.
        labels: Targets `[batch, sequence]` aligned so position `t` predicts `labels[..., t]` from logits at `t-1`.
        mask: `{0,1}` weights `[batch, sequence]` marking supervised positions.

    Returns:
        Scalar tensor with mean negative log likelihood over supervised tokens.
    """

    logits_shift = logits[:, :-1, :].contiguous()
    labels_shift = labels[:, 1:].contiguous()
    mask_shift = mask[:, 1:].contiguous().to(dtype=torch.float32)

    vocab = logits_shift.shape[-1]
    flat_logits = logits_shift.view(-1, vocab)
    flat_labels = labels_shift.view(-1)
    flat_mask = mask_shift.view(-1)

    token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    denom = flat_mask.sum().clamp_min(1.0)
    return (token_loss * flat_mask).sum() / denom
