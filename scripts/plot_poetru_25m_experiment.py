from __future__ import annotations

from pathlib import Path

from plot_utils import plot_loss_linear_and_loglog


def plot_experiment(csv_path: Path, out_dir: Path) -> None:
    """Writes linear-scale and log-log loss curves from a training history CSV."""

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_linear_and_loglog(
        csv_path,
        out_dir / "poetru_25m_loss_linear.png",
        out_dir / "poetru_25m_loss_loglog.png",
        title="Poetru-25M pilot run",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "docs" / "experiments"
    plot_experiment(data_dir / "poetru_25m_train_to_53k.csv", data_dir)
