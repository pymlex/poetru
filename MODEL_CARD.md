---
language: ru
license: gpl-3.0
tags:
  - poetry
  - russian
  - causal-lm
  - watermark
datasets:
  - IlyaGusev/stihi_ru
---

# poetru-25m

Compact Russian poetry language model trained on [IlyaGusev/stihi_ru](https://huggingface.co/datasets/IlyaGusev/stihi_ru) with ByteLevel BPE, RoPE, GQA, MLA-style latent KV compression, SwiGLU blocks, and tied embeddings.

## Architecture

| Field | Value |
| --- | --- |
| Parameters | ~24.5M trainable |
| Layers | 5 |
| Hidden size | 384 |
| Heads | 8 query, 4 KV (GQA) |
| MLA latent dim | 384 |
| FFN | SwiGLU, intermediate 1024 |
| Context | 512 tokens |
| Vocabulary | 24000 ByteLevel BPE |
| Positional encoding | RoPE, $\theta = 10000$ |

Chinchilla scaling targets ~20 tokens per parameter. With ~455M BPE tokens in the corpus and 2.5 epochs the model sees about $1.14 \times 10^9$ token updates, which matches a ~25M parameter budget.

## Watermarking

Generation applies the soft green-list bias from Kirchenbauer et al., 2023 with $\gamma = 0.25$ and $\delta = 2.0$. Detection uses the one-sided z-test shipped in `watermark.py`.

## Metrics

Perplexity, watermark ROC-AUC, and author PCA artefacts are stored under `metrics/` in this repository snapshot.

## Usage

```python
import torch
from pathlib import Path
from bpe_tokenizer import ByteBPETokenizerWrapper
from checkpoint_utils import load_checkpoint
from configs import GenerationConfig
from trainer import generate_poem

device = torch.device("cuda")
tokenizer = ByteBPETokenizerWrapper.from_file(Path("artifacts/tokenizer/tokenizer.json"))
model, _ = load_checkpoint(Path("artifacts/checkpoints/final.pt"), device)
model.eval()

prompt_ids = tokenizer.encode("В тишине ночной", add_eos=False)
token_ids, _ = generate_poem(
    model,
    prompt_ids,
    eos_id=tokenizer.eos_id,
    gen_cfg=GenerationConfig(),
    device=device,
    apply_watermark=True,
)
print(tokenizer.decode(token_ids))
```

## Files

- `model.pt` — final checkpoint
- `config.json` — architecture hyperparameters
- `tokenizer/tokenizer.json` — ByteLevel BPE model
- `generated_poems.jsonl` — watermarked samples
- `metrics/` — evaluation outputs
