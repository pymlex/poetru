from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Runtime credentials and Hub identifiers loaded from environment variables.

    Args:
        None; values are read from the process environment and optional `.env`.

    Returns:
        EnvSettings: Validated configuration object.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_model_repo: str = Field(default="pymlex/poetru-75m", alias="HF_MODEL_REPO")
    hf_tokenizer_repo: str = Field(default="pymlex/poetru-75m-tokenizer", alias="HF_TOKENIZER_REPO")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations for artefacts used by training and evaluation scripts.

    Args:
        root: Repository root directory.

    Returns:
        ProjectPaths: Concrete paths under `root`.
    """

    root: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def tokenizer_dir(self) -> Path:
        return self.artifacts_dir / "tokenizer"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def generated_poems_path(self) -> Path:
        return self.artifacts_dir / "generated_poems.jsonl"
