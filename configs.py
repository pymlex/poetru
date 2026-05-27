from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for the causal Transformer language model."""

    vocab_size: int = 24000
    max_seq_len: int = 512
    hidden_dim: int = 384
    intermediate_dim: int = 1024
    n_layer: int = 5
    n_head: int = 8
    n_kv_head: int = 4
    latent_dim: int = 384
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainConfig:
    """Training loop controls aligned with cosine decay after linear warmup."""

    dataset_name: str = "IlyaGusev/stihi_ru"
    dataset_split: str = "train"
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    num_epochs: float = 2.5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 3407
    num_workers: int = 4
    log_every_steps: int = 50
    checkpoint_every_steps: int = 2000
    eval_batches: int = 200
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class GenerationConfig:
    """Constants shared by poem generation, watermark evaluation, and PCA overlay."""

    target_poem_count: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.85
    top_p: float = 0.92
    watermark_gamma: float = 0.25
    watermark_delta: float = 2.0
    watermark_seed_scheme: str = "kirchenbauer_soft"


@dataclass(frozen=True)
class TokenizerTrainConfig:
    """Training-time controls for Hugging Face `tokenizers` ByteLevel BPE."""

    vocab_size: int = 24000
    min_frequency: int = 2
    dataset_sample_fraction: float = 1.0


@dataclass(frozen=True)
class AuthorPCConfig:
    """Author centroid embedding experiment."""

    min_poems_per_author: int = 120
    max_poems_per_author: int = 48
    max_authors: int = 200
    embedding_chunk_tokens: int = 512
