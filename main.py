from __future__ import annotations

import argparse
from pathlib import Path

from configs import AuthorPCConfig, GenerationConfig, TokenizerTrainConfig, TrainConfig, TransformerConfig
from data_utils import (
    build_dataloader,
    iter_poetry_texts,
    load_poetry_texts,
    save_json,
    token_length_histogram,
)
from gpu_telemetry import seed_everything
from pydantic_models import ProjectPaths


def train_tokenizer(root: Path) -> None:
    """Trains ByteLevel BPE on the poetry corpus and writes length statistics."""

    paths = ProjectPaths(root)
    tok_cfg = TokenizerTrainConfig()
    train_cfg = TrainConfig()
    seed_everything(train_cfg.seed)

    texts_for_stats = load_poetry_texts(
        train_cfg.dataset_name,
        train_cfg.dataset_split,
        sample_fraction=0.05,
        seed=train_cfg.seed,
    )

    from bpe_tokenizer import ByteBPETokenizerWrapper, train_byte_level_bpe

    train_byte_level_bpe(
        iter_poetry_texts(
            train_cfg.dataset_name,
            train_cfg.dataset_split,
            sample_fraction=tok_cfg.dataset_sample_fraction,
            seed=train_cfg.seed,
        ),
        vocab_size=tok_cfg.vocab_size,
        output_dir=paths.tokenizer_dir,
        min_frequency=tok_cfg.min_frequency,
    )

    wrapper = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    stats = token_length_histogram(
        texts_for_stats,
        wrapper,
        max_seq_len=TransformerConfig().max_seq_len,
        output_path=paths.metrics_dir / "token_length_hist.png",
    )
    save_json(paths.metrics_dir / "token_length_stats.json", stats)


def train_model(root: Path) -> None:
    """Runs the full language modelling training loop."""

    from bpe_tokenizer import ByteBPETokenizerWrapper
    from checkpoint_utils import count_parameters
    from data_utils import PoetryTokenDataset
    from model import PoetruCausalLM
    from trainer import Trainer

    paths = ProjectPaths(root)
    tcfg = TransformerConfig()
    train_cfg = TrainConfig()
    seed_everything(train_cfg.seed)

    tokenizer = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    texts = load_poetry_texts(train_cfg.dataset_name, train_cfg.dataset_split, seed=train_cfg.seed)

    rng = __import__("numpy").random.default_rng(train_cfg.seed)
    perm = rng.permutation(len(texts))
    split = int(len(texts) * 0.995)
    train_texts = [texts[int(i)] for i in perm[:split]]
    val_texts = [texts[int(i)] for i in perm[split:]]

    train_ds = PoetryTokenDataset(train_texts, tokenizer, tcfg.max_seq_len)
    val_ds = PoetryTokenDataset(val_texts, tokenizer, tcfg.max_seq_len)

    train_loader = build_dataloader(
        train_ds,
        pad_id=tokenizer.pad_id,
        max_seq_len=tcfg.max_seq_len,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = build_dataloader(
        val_ds,
        pad_id=tokenizer.pad_id,
        max_seq_len=tcfg.max_seq_len,
        batch_size=train_cfg.micro_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    model = PoetruCausalLM(tcfg)
    save_json(
        paths.metrics_dir / "model_param_count.json",
        {"trainable_parameters": count_parameters(model), "config": tcfg.__dict__},
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        tcfg=tcfg,
        checkpoint_dir=paths.checkpoint_dir,
        logs_dir=paths.logs_dir,
    )
    trainer.run()


def generate_poems(root: Path, count: int | None = None) -> None:
    """Generates watermarked poems and stores them as JSONL."""

    import json
    import torch

    from bpe_tokenizer import ByteBPETokenizerWrapper
    from checkpoint_utils import load_checkpoint
    from trainer import generate_poem

    paths = ProjectPaths(root)
    gen_cfg = GenerationConfig()
    target = count if count is not None else gen_cfg.target_poem_count
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    model, _ = load_checkpoint(paths.checkpoint_dir / "final.pt", device)
    model.eval()

    prompts = [
        "В тишине ночной",
        "Я помню чудное мгновенье",
        "Люблю грозу в начале мая",
        "Белеет парус одинокий",
        "Мой дух омрачен",
        "Осень. Холодные ветры",
        "Звезда падала",
        "Ты помнишь",
    ]

    rows = []
    for idx in range(target):
        prompt = prompts[idx % len(prompts)]
        prompt_ids = tokenizer.encode(prompt, add_eos=False)
        token_ids, _ = generate_poem(
            model,
            prompt_ids,
            eos_id=tokenizer.eos_id,
            gen_cfg=gen_cfg,
            device=device,
            apply_watermark=True,
        )
        text = tokenizer.decode(token_ids)
        rows.append({"id": idx, "prompt": prompt, "text": text, "token_ids": token_ids})

    paths.generated_poems_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.generated_poems_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_perplexity(root: Path) -> None:
    """Computes validation perplexity on a capped batch count."""

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    from bpe_tokenizer import ByteBPETokenizerWrapper
    from checkpoint_utils import load_checkpoint
    from data_utils import PoetryTokenDataset, build_dataloader
    from loss_utils import causal_language_modeling_loss

    paths = ProjectPaths(root)
    train_cfg = TrainConfig()
    tcfg = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    texts = load_poetry_texts(train_cfg.dataset_name, train_cfg.dataset_split, sample_fraction=0.01, seed=train_cfg.seed)
    val_ds = PoetryTokenDataset(texts[-5000:], tokenizer, tcfg.max_seq_len)
    val_loader = build_dataloader(
        val_ds,
        pad_id=tokenizer.pad_id,
        max_seq_len=tcfg.max_seq_len,
        batch_size=train_cfg.micro_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model, _ = load_checkpoint(paths.checkpoint_dir / "final.pt", device)
    model.eval()

    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(val_loader, desc="Perplexity")):
            if batch_idx >= train_cfg.eval_batches:
                break
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            loss = causal_language_modeling_loss(logits, input_ids, attention_mask)
            total += float(loss.item())
            seen += 1

    mean_loss = total / max(seen, 1)
    ppl = float(np.exp(mean_loss))
    save_json(paths.metrics_dir / "perplexity.json", {"val_loss": mean_loss, "perplexity": ppl})


def evaluate_watermark(root: Path) -> None:
    """Runs watermark detection ROC metrics on generated and real poems."""

    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

    from bpe_tokenizer import ByteBPETokenizerWrapper
    from configs import GenerationConfig, TrainConfig
    from data_utils import load_poetry_texts
    from watermark import WatermarkConfig, detect_watermark

    paths = ProjectPaths(root)
    gen_cfg = GenerationConfig()
    train_cfg = TrainConfig()
    wm_cfg = WatermarkConfig(gamma=gen_cfg.watermark_gamma, delta=gen_cfg.watermark_delta)

    tokenizer = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    vocab_size = TokenizerTrainConfig().vocab_size

    generated_rows = []
    with paths.generated_poems_path.open("r", encoding="utf-8") as f:
        for line in f:
            generated_rows.append(json.loads(line))
    generated_rows = generated_rows[: gen_cfg.target_poem_count]

    real_texts = load_poetry_texts(train_cfg.dataset_name, train_cfg.dataset_split, sample_fraction=0.001, seed=train_cfg.seed)
    real_texts = real_texts[: gen_cfg.target_poem_count]

    scores = []
    labels = []

    for row in generated_rows:
        det = detect_watermark(row["token_ids"], vocab_size, wm_cfg)
        scores.append(det.z_score)
        labels.append(1)

    for text in real_texts:
        ids = tokenizer.encode(text, add_eos=True)
        det = detect_watermark(ids, vocab_size, wm_cfg)
        scores.append(det.z_score)
        labels.append(0)

    y = np.array(labels, dtype=np.int64)
    s = np.array(scores, dtype=np.float64)
    y_pred = (s >= 4.0).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }

    fpr, tpr, _ = roc_curve(y, s)
    metrics["roc_auc"] = float(auc(fpr, tpr))
    save_json(paths.metrics_dir / "watermark_metrics.json", metrics)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.title("Watermark detection ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(paths.metrics_dir / "watermark_roc.png", dpi=160)
    plt.close()

    pd.DataFrame({"label": y, "z_score": s}).to_csv(paths.metrics_dir / "watermark_scores.csv", index=False)


def author_pca(root: Path) -> None:
    """Builds author centroid embeddings and overlays generated poems in 2D PCA space."""

    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from tqdm.auto import tqdm

    from bpe_tokenizer import ByteBPETokenizerWrapper
    from checkpoint_utils import load_checkpoint
    from configs import AuthorPCConfig, GenerationConfig, TrainConfig
    from trainer import mean_pool_hidden

    paths = ProjectPaths(root)
    ap_cfg = AuthorPCConfig()
    gen_cfg = GenerationConfig()
    train_cfg = TrainConfig()
    tcfg = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from collections import defaultdict

    from datasets import load_dataset

    ds = load_dataset(train_cfg.dataset_name, split=train_cfg.dataset_split, streaming=True)
    author_to_texts: dict[str, list[str]] = defaultdict(list)
    author_counts: dict[str, int] = defaultdict(int)

    for row in tqdm(ds, desc="Scanning authors"):
        author = row.get("author")
        text = str(row.get("text", "")).strip()
        if author is None or not text:
            continue
        name = str(author)
        author_counts[name] += 1
        if author_counts[name] <= ap_cfg.max_poems_per_author:
            author_to_texts[name].append(text)

    eligible = [
        author
        for author, count in author_counts.items()
        if count >= ap_cfg.min_poems_per_author
    ]
    eligible = sorted(eligible, key=lambda a: author_counts[a], reverse=True)[: ap_cfg.max_authors]

    tokenizer = ByteBPETokenizerWrapper.from_file(paths.tokenizer_dir / "tokenizer.json")
    model, _ = load_checkpoint(paths.checkpoint_dir / "final.pt", device)
    model.eval()

    author_vectors = []
    author_names = []

    for author in tqdm(eligible, desc="Author embeddings"):
        poems = author_to_texts[author]
        chunk_embs = []
        for poem in poems:
            ids = tokenizer.encode(poem, add_eos=True)
            if len(ids) > ap_cfg.embedding_chunk_tokens:
                ids = ids[: ap_cfg.embedding_chunk_tokens]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            emb = mean_pool_hidden(model, input_ids, mask)
            chunk_embs.append(emb.detach().cpu().numpy()[0])
        centroid = np.mean(np.stack(chunk_embs, axis=0), axis=0)
        author_vectors.append(centroid)
        author_names.append(author)

    author_matrix = np.stack(author_vectors, axis=0)
    np.savez(
        paths.metrics_dir / "author_embeddings.npz",
        authors=np.array(author_names),
        embeddings=author_matrix,
    )

    generated_rows = []
    with paths.generated_poems_path.open("r", encoding="utf-8") as f:
        for line in f:
            generated_rows.append(json.loads(line))
    generated_rows = generated_rows[: gen_cfg.target_poem_count]

    gen_vectors = []
    for row in tqdm(generated_rows, desc="Generated embeddings"):
        ids = row["token_ids"]
        if len(ids) > ap_cfg.embedding_chunk_tokens:
            ids = ids[: ap_cfg.embedding_chunk_tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        emb = mean_pool_hidden(model, input_ids, mask)
        gen_vectors.append(emb.detach().cpu().numpy()[0])

    gen_matrix = np.stack(gen_vectors, axis=0)
    np.savez(paths.metrics_dir / "generated_embeddings.npz", embeddings=gen_matrix)

    all_matrix = np.concatenate([author_matrix, gen_matrix], axis=0)
    pca = PCA(n_components=2, random_state=train_cfg.seed)
    coords = pca.fit_transform(all_matrix)

    author_coords = coords[: author_matrix.shape[0]]
    gen_coords = coords[author_matrix.shape[0] :]

    plt.figure(figsize=(10, 8))
    plt.scatter(author_coords[:, 0], author_coords[:, 1], s=12, alpha=0.7, label="authors")
    plt.scatter(gen_coords[:, 0], gen_coords[:, 1], s=18, alpha=0.9, label="generated")
    plt.title("Author centroids and generated poems in PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.metrics_dir / "author_pca.png", dpi=160)
    plt.close()


def publish_hub(root: Path) -> None:
    """Publishes tokenizer and trained model bundle to Hugging Face."""

    from hub_utils import load_env, publish_model_bundle, publish_tokenizer

    paths = ProjectPaths(root)
    env = load_env()
    tcfg = TransformerConfig()

    publish_tokenizer(paths.tokenizer_dir, env.hf_tokenizer_repo, env.hf_token)
    publish_model_bundle(
        paths.checkpoint_dir / "final.pt",
        tcfg.__dict__,
        paths.tokenizer_dir,
        paths.metrics_dir,
        paths.generated_poems_path,
        env.hf_model_repo,
        env.hf_token,
    )


def build_parser() -> argparse.ArgumentParser:
    """Creates the CLI parser for pipeline stages."""

    parser = argparse.ArgumentParser(description="Poetru Russian poetry LM pipeline")
    parser.add_argument(
        "stage",
        choices=[
            "train_tokenizer",
            "train_model",
            "generate",
            "perplexity",
            "watermark_eval",
            "author_pca",
            "publish",
            "all",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--count", type=int, default=None, help="Override generated poem count")
    return parser


def main() -> None:
    """Entry point for scripted pipeline execution."""

    args = build_parser().parse_args()
    root = args.root.resolve()

    if args.stage in {"train_tokenizer", "all"}:
        train_tokenizer(root)
    if args.stage in {"train_model", "all"}:
        train_model(root)
    if args.stage in {"generate", "all"}:
        generate_poems(root, count=args.count)
    if args.stage in {"perplexity", "all"}:
        evaluate_perplexity(root)
    if args.stage in {"watermark_eval", "all"}:
        evaluate_watermark(root)
    if args.stage in {"author_pca", "all"}:
        author_pca(root)
    if args.stage in {"publish", "all"}:
        publish_hub(root)


if __name__ == "__main__":
    main()
