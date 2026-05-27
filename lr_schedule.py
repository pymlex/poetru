from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Linear warmup from zero followed by cosine decay to a small eta min fraction."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("warmup_steps must lie in [0, total_steps].")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes multiplicative factors per param group."""

        step = self.last_epoch + 1
        out: list[float] = []

        for base_lr in self.base_lrs:
            eta_min = base_lr * self.eta_min_ratio
            if step < self.warmup_steps:
                scale = float(step) / float(max(1, self.warmup_steps))
                out.append(base_lr * scale)
                continue

            t = step - self.warmup_steps
            T = max(1, self.total_steps - self.warmup_steps)
            cos_part = 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))
            out.append(eta_min + (base_lr - eta_min) * cos_part)

        return out
