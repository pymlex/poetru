from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder

from pydantic_models import EnvSettings


def publish_tokenizer(local_dir: Path, repo_id: str, token: str | None) -> None:
    """Uploads `tokenizer.json` to a Hub model repository.

    Args:
        local_dir: Directory containing `tokenizer.json`.
        repo_id: Target Hub repository id.
        token: Hugging Face access token.

    Returns:
        None.
    """

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="model",
        token=token,
        commit_message="Upload ByteLevel BPE tokenizer",
    )


def publish_model_bundle(
    checkpoint_path: Path,
    config_dict: dict,
    tokenizer_dir: Path,
    metrics_dir: Path,
    generated_poems_path: Path,
    repo_id: str,
    token: str | None,
    commit_message: str = "Upload poetru checkpoint and evaluation artefacts",
) -> None:
    """Uploads weights, config, metrics, generated poems, and model card to the Hub.

    Args:
        checkpoint_path: Final `.pt` checkpoint.
        config_dict: Serialised TransformerConfig mapping.
        tokenizer_dir: Local tokenizer directory.
        metrics_dir: Directory with JSON metrics and plots.
        generated_poems_path: JSONL with generated poems.
        repo_id: Hub model repository id.
        token: Hugging Face access token.
        commit_message: Hub commit description.

    Returns:
        None.
    """

    publish_root = checkpoint_path.parent / "hub_publish"
    if publish_root.exists():
        shutil.rmtree(publish_root)
    publish_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, publish_root / "model.pt")
    with (publish_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    shutil.copytree(tokenizer_dir, publish_root / "tokenizer")

    metrics_out = publish_root / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)
    for item in metrics_dir.glob("*"):
        if item.is_file():
            shutil.copy2(item, metrics_out / item.name)

    if generated_poems_path.exists():
        shutil.copy2(generated_poems_path, publish_root / "generated_poems.jsonl")

    card_src = Path(__file__).resolve().parent / "MODEL_CARD.md"
    if card_src.exists():
        shutil.copy2(card_src, publish_root / "README.md")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(publish_root),
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )


def load_env() -> EnvSettings:
    """Loads environment-backed Hub settings.

    Args:
        None.

    Returns:
        EnvSettings instance.
    """

    return EnvSettings()
