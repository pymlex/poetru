from __future__ import annotations

import json
from pathlib import Path

import torch

from configs import TransformerConfig
from model import PoetruCausalLM


def count_parameters(model: torch.nn.Module) -> int:
    """Counts trainable parameters.

    Args:
        model: PyTorch module.

    Returns:
        Integer parameter count.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: PoetruCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: float,
) -> None:
    """Persists model, optimiser, scheduler, and counters.

    Args:
        path: Destination `.pt` file.
        model: Language model.
        optimizer: AdamW instance.
        scheduler: LR scheduler.
        step: Global optimisation step.
        epoch: Fractional epoch counter.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": model.cfg.__dict__,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[PoetruCausalLM, dict]:
    """Restores a checkpoint into a freshly constructed model.

    Args:
        path: Checkpoint file.
        device: Target device.

    Returns:
        Tuple `(model, metadata)` where metadata holds optimiser and counters when present.
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = TransformerConfig(**payload["config"])
    model = PoetruCausalLM(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Appends rows to a JSONL file.

    Args:
        path: Destination JSONL path.
        rows: Serializable dictionaries.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
