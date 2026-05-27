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


def latest_step_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Returns the newest `step_*.pt` file by step index.

    Args:
        checkpoint_dir: Directory where the trainer writes periodic checkpoints.

    Returns:
        Path to the highest-step checkpoint, or `None` when no step files exist.
    """

    step_paths = list(checkpoint_dir.glob("step_*.pt"))
    if not step_paths:
        return None
    return max(step_paths, key=lambda path: int(path.stem.split("_", maxsplit=1)[1]))


def resolve_publish_checkpoint(checkpoint_dir: Path) -> Path:
    """Selects `final.pt` when present, otherwise the latest `step_*.pt`.

    Args:
        checkpoint_dir: Checkpoint root used by training.

    Returns:
        Path to the checkpoint file that should be published.

    Raises:
        FileNotFoundError: When no publishable checkpoint exists.
    """

    final_path = checkpoint_dir / "final.pt"
    if final_path.exists():
        return final_path
    latest = latest_step_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return latest


def read_checkpoint_meta(path: Path) -> dict:
    """Loads non-weight metadata from a training checkpoint on CPU.

    Args:
        path: Checkpoint `.pt` file.

    Returns:
        Dictionary with keys such as `step`, `epoch`, and `config`.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "step": int(payload["step"]),
        "epoch": float(payload["epoch"]),
        "config": dict(payload["config"]),
    }


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
