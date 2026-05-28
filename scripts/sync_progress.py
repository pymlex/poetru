from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from checkpoint_utils import read_checkpoint_meta, resolve_publish_checkpoint
from hub_utils import load_env, publish_model_bundle
from plot_utils import plot_loss_linear_and_loglog
from pydantic_models import ProjectPaths


def copy_file_snapshot(source: Path, destination: Path) -> None:
    """Copies a file without removing the source used by a live trainer.

    Args:
        source: Existing file path.
        destination: Target path to overwrite.

    Returns:
        None.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_hub_checkpoint(source_ckpt: Path, staging_dir: Path) -> Path:
    """Copies a checkpoint into the Hub staging directory.

    Args:
        source_ckpt: Trainer checkpoint file.
        staging_dir: `artifacts/hub_staging` root.

    Returns:
        Path to `staging_dir / model.pt`.
    """

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / "model.pt"
    copy_file_snapshot(source_ckpt, staged)
    return staged


def export_github_progress(
    paths: ProjectPaths,
    history_csv: Path,
    meta: dict,
) -> list[Path]:
    """Mirrors logs and plots into `docs/experiments` for version control.

    Args:
        paths: Project path bundle.
        history_csv: Live trainer CSV under `artifacts/logs`.
        meta: Serialisable sync metadata.

    Returns:
        Paths that should be passed to `git add`.
    """

    docs_dir = paths.root / "docs" / "experiments"
    docs_dir.mkdir(parents=True, exist_ok=True)

    tracked = [
        docs_dir / "poetru_75m_train_history.csv",
        docs_dir / "poetru_75m_loss_linear.png",
        docs_dir / "poetru_75m_loss_loglog.png",
        docs_dir / "poetru_75m_sync_meta.json",
    ]

    copy_file_snapshot(history_csv, tracked[0])
    plot_loss_linear_and_loglog(
        tracked[0],
        tracked[1],
        tracked[2],
        title="Poetru-75M training progress",
    )

    hist_src = paths.metrics_dir / "token_length_hist.png"
    if hist_src.exists():
        hist_dst = docs_dir / "token_length_hist.png"
        copy_file_snapshot(hist_src, hist_dst)
        tracked.append(hist_dst)

    with tracked[3].open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return tracked


def push_github(paths: list[Path], message: str) -> None:
    """Commits only the supplied paths and pushes to `origin`.

    Args:
        paths: Files to stage.
        message: Commit subject line.

    Returns:
        None.
    """

    path_args = [str(path) for path in paths]
    subprocess.run(["git", "add", *path_args], check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        print("GitHub: nothing new to commit.")
        return
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], check=True)
    print("GitHub: push complete.")


def sync_progress(
    root: Path,
    skip_github: bool,
    skip_hf: bool,
    git_message: str | None,
) -> None:
    """Publishes a read-only snapshot while training continues in another process.

    Args:
        root: Repository root.
        skip_github: When `True`, skip CSV and plot upload to Git.
        skip_hf: When `True`, skip Hugging Face uploads.
        git_message: Optional override for the Git commit subject.

    Returns:
        None.
    """

    paths = ProjectPaths(root)
    history_csv = paths.logs_dir / "train_history.csv"
    if not history_csv.exists():
        raise FileNotFoundError(f"Missing training log: {history_csv}")

    source_ckpt = resolve_publish_checkpoint(paths.checkpoint_dir)
    meta = read_checkpoint_meta(source_ckpt)
    meta["source_checkpoint"] = source_ckpt.name
    meta["synced_at_utc"] = datetime.now(timezone.utc).isoformat()

    if not skip_github:
        tracked = export_github_progress(paths, history_csv, meta)
        subject = git_message or f"Poetru-75M training snapshot at step {meta['step']}."
        push_github(tracked, subject)

    if not skip_hf:
        env = load_env()
        staging_dir = paths.artifacts_dir / "hub_staging"
        staged_ckpt = stage_hub_checkpoint(source_ckpt, staging_dir)
        publish_model_bundle(
            staged_ckpt,
            meta["config"],
            paths.tokenizer_dir,
            paths.metrics_dir,
            paths.generated_poems_path,
            env.hf_model_repo,
            env.hf_token,
            commit_message=f"Poetru-75M checkpoint step {meta['step']}",
        )
        print(f"Hugging Face: uploaded step {meta['step']} from {source_ckpt.name}.")


def build_parser() -> argparse.ArgumentParser:
    """CLI for mid-training Git and Hub sync."""

    parser = argparse.ArgumentParser(
        description="Snapshot training progress to GitHub and Hugging Face without stopping train_model.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-github", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--message", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sync_progress(args.root.resolve(), args.skip_github, args.skip_hf, args.message)
