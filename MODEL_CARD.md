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

# poetru-75m

Compact Russian poetry causal language model trained on [IlyaGusev/stihi_ru](https://huggingface.co/datasets/IlyaGusev/stihi_ru) with ByteLevel BPE, RoPE, GQA, MLA-style latent KV compression, SwiGLU blocks, and tied embeddings.

## Architecture

| Field | Value |
| --- | --- |
| Parameters | ~74.9M trainable |
| Layers | 12 |
| Hidden size | 640 |
| Heads | 8 query, 4 KV (GQA) |
| MLA latent dim | 640 |
| FFN | SwiGLU, intermediate 1728 |
| Context | 512 tokens |
| Vocabulary | 24000 ByteLevel BPE |
| Positional encoding | RoPE base 10000 |

Chinchilla-style scaling uses on the rough order of 20 training tokens per parameter. The stihi_ru ByteLevel-BPE token count is on the rough order of 455 million. Training for 2.5 epochs yields a cumulative token budget on the rough order of

$$
T_{\mathrm{train}} \approx 1.14 \times 10^{9}
$$

which implies a compute-optimal parameter count near $5.7 \times 10^{7}$. This checkpoint uses about $7.5 \times 10^{7}$ trainable parameters, roughly 30 percent above that estimate.

## Watermarking

Generation applies the soft green-list bias from Kirchenbauer et al., 2023 ([arXiv:2301.10226](https://arxiv.org/abs/2301.10226)). Default detection parameters are packaged as gamma and delta in the published code:

$$
\gamma = 0.25
$$

$$
\delta = 2.0
$$

Detection uses the one-sided green-list proportion test outlined in `watermark.py`.

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
