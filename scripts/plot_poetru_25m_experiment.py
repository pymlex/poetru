from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_experiment(csv_path: Path, out_dir: Path) -> None:
    """Writes linear-scale and log-log loss curves from a training history CSV."""

    df = pd.read_csv(csv_path)
    step = df["step"].to_numpy(dtype=np.float64)
    train = df["loss"].to_numpy(dtype=np.float64)
    val = df["val_loss"].to_numpy(dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(step, train, label="train CE")
    ax.plot(step, val, label="val CE")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Poetru-25M pilot run")
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "poetru_25m_loss_linear.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(step, train, label="train CE")
    ax.loglog(step, val, label="val CE")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Poetru-25M pilot run")
    ax.grid(alpha=0.5, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "poetru_25m_loss_loglog.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "docs" / "experiments"
    plot_experiment(data_dir / "poetru_25m_train_to_53k.csv", data_dir)
