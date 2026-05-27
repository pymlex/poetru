from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history_csv: Path, output_dir: Path) -> None:
    """Draws loss, validation loss, and learning rate curves from trainer logs.

    Args:
        history_csv: CSV written during training with columns `step`, `loss`, `val_loss`, `lr`.
        output_dir: Directory for PNG exports.

    Returns:
        None.
    """

    df = pd.read_csv(history_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["loss"], label="train")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        plt.plot(df["step"], df["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["lr"])
    plt.title("Learning rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate.png", dpi=160)
    plt.close()
