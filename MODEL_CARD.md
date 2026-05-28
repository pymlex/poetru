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

Poetru-75M is a Russian poetry SLM trained on [IlyaGusev/stihi_ru](https://huggingface.co/datasets/IlyaGusev/stihi_ru).

GitHub source: [github.com/pymlex/poetru](https://github.com/pymlex/poetru)

## Architecture

| Field | Value |
| --- | --- |
| Parameters | 74,899,072 trainable |
| Layers | 12 |
| Hidden size | 640 |
| Heads | 8 query, 4 KV (GQA) |
| MLA latent dim | 640 |
| FFN | SwiGLU, intermediate 1728 |
| Context | 512 tokens |
| Vocabulary | 24000 ByteLevel BPE |
| Positional encoding | RoPE base 10000 |

Chinchilla budget:

$$
T_{\mathrm{train}} \approx 1.37 \times 10^{9}
$$

$$
N_* \approx \frac{T_{\mathrm{train}}}{20} \approx 6.8 \times 10^7
$$

## Watermarking

Generation applies the soft green-list bias from Kirchenbauer et al., 2023 ([arXiv:2301.10226](https://arxiv.org/abs/2301.10226)).

$$
\gamma = 0.25
$$

$$
\delta = 2.0
$$

$$
z=\frac{K-\gamma T}{\sqrt{T\gamma(1-\gamma)}}
$$

## Training Setup

- CPU: Ryzen 9 9900X
- GPU: RTX 5090 32GB
- epochs: 3
- steps: 240,246
- wall-clock: 18h 31m
- effective batch: 64 with `grad_accum_steps = 1`

## Metrics

From uploaded `metrics/`:

| Metric | Value |
| --- | ---: |
| val loss | 3.2713 |
| perplexity | 26.3448 |
| watermark accuracy | 0.963 |
| watermark precision | 1.000 |
| watermark recall | 0.926 |
| watermark F1 | 0.9616 |
| watermark ROC-AUC | 0.9992 |

![Token length histogram](metrics/token_length_hist.png)
![Training loss](metrics/loss_curve.png)
![Training loss log-step log-log-loss](metrics/loss_curve_loglog.png)
![Learning rate](metrics/learning_rate.png)
![Watermark ROC](metrics/watermark_roc.png)
![Author PCA](metrics/author_pca.png)

## 25M Pilot

Poetru-25M was trained as a pilot before the current checkpoint. The 75M configuration is the active model line for stronger generalisation.

## Inference

Install and run from GitHub:

```bash
git clone https://github.com/pymlex/poetru.git
cd poetru
pip install -r requirements.txt
```

```python
from pathlib import Path
import torch
from hub_utils import download_inference_artifacts
from bpe_tokenizer import ByteBPETokenizerWrapper
from checkpoint_utils import load_checkpoint
from configs import GenerationConfig
from trainer import generate_poem

root = Path(".")
download_inference_artifacts(root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ByteBPETokenizerWrapper.from_file(root / "artifacts/tokenizer/tokenizer.json")
model, _ = load_checkpoint(root / "artifacts/checkpoints/final.pt", device)
model.eval()

prompt_ids = tokenizer.encode("Раз, два, три", add_eos=False)
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
