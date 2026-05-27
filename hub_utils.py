from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, upload_folder

from pydantic_models import EnvSettings, ProjectPaths


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


def download_inference_artifacts(
    root: Path,
    model_repo_id: str | None = None,
    tokenizer_repo_id: str | None = None,
    token: str | None = None,
) -> tuple[Path, Path]:
    """Fetches `model.pt` and `tokenizer.json` into the local `artifacts/` tree.

    Args:
        root: Repository root.
        model_repo_id: Hub model repository with the checkpoint bundle.
        tokenizer_repo_id: Optional separate tokenizer repository.
        token: Hugging Face token for private repositories.

    Returns:
        Tuple `(checkpoint_path, tokenizer_json_path)`.
    """

    env = load_env()
    model_repo = model_repo_id if model_repo_id is not None else env.hf_model_repo
    tokenizer_repo = tokenizer_repo_id if tokenizer_repo_id is not None else env.hf_tokenizer_repo
    hub_token = token if token is not None else env.hf_token

    paths = ProjectPaths(root)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = paths.checkpoint_dir / "final.pt"
    tokenizer_path = paths.tokenizer_dir / "tokenizer.json"

    if not checkpoint_path.exists():
        cached_ckpt = hf_hub_download(
            repo_id=model_repo,
            filename="model.pt",
            token=hub_token,
        )
        shutil.copy2(cached_ckpt, checkpoint_path)

    if not tokenizer_path.exists():
        api = HfApi(token=hub_token)
        model_files = api.list_repo_files(model_repo)
        if "tokenizer/tokenizer.json" in model_files:
            cached_tok = hf_hub_download(
                repo_id=model_repo,
                filename="tokenizer/tokenizer.json",
                token=hub_token,
            )
            shutil.copy2(cached_tok, tokenizer_path)
        else:
            cached_tok = hf_hub_download(
                repo_id=tokenizer_repo,
                filename="tokenizer.json",
                token=hub_token,
            )
            shutil.copy2(cached_tok, tokenizer_path)

    return checkpoint_path, tokenizer_path


def load_env() -> EnvSettings:
    """Loads environment-backed Hub settings.

    Args:
        None.

    Returns:
        EnvSettings instance.
    """

    return EnvSettings()
