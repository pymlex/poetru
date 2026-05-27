from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_loss_linear_and_loglog(
    history_csv: Path,
    out_linear: Path,
    out_loglog: Path,
    title: str,
) -> None:
    """Writes linear CE curves and a Chinchilla-style log(step) vs log(log L) figure.

    Args:
        history_csv: Trainer log with `step`, `loss`, and `val_loss`.
        out_linear: Destination PNG for the linear plot.
        out_loglog: Destination PNG for the log-log scaling plot.
        title: Figure title shared by both exports.

    Returns:
        None.
    """

    df = pd.read_csv(history_csv)
    step = df["step"].to_numpy(dtype=np.float64)
    train = df["loss"].to_numpy(dtype=np.float64)
    val = df["val_loss"].to_numpy(dtype=np.float64)

    out_linear.parent.mkdir(parents=True, exist_ok=True)
    out_loglog.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(step, train, label="train CE")
    ax.plot(step, val, label="val CE")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_linear, dpi=160)
    plt.close(fig)

    log_step = np.log(step)
    loglog_train = np.log(np.log(train))
    loglog_val = np.log(np.log(val))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log_step, loglog_train, label="train CE")
    ax.plot(log_step, loglog_val, label="val CE")
    ax.set_xlabel(r"$\log(\mathrm{step})$")
    ax.set_ylabel(r"$\log\log L$")
    ax.set_title(title)
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_loglog, dpi=160)
    plt.close(fig)
