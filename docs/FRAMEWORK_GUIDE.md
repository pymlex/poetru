# Poetru Framework Guide

## Project Identity

Poetru-75M is a Russian poetry SLM trained with a decoder-only Transformer and watermark-aware generation.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` fields:

| Variable | Meaning |
| --- | --- |
| `HF_TOKEN` | Hugging Face access token |
| `HF_MODEL_REPO` | model repository id |
| `HF_TOKENIZER_REPO` | tokenizer repository id |
| `GITHUB_TOKEN` | optional token for scripted GitHub APIs |

## Core Pipeline Commands

```bash
python main.py train_tokenizer --root .
python main.py train_model --root .
python scripts/generate_poems.py --root . --count 1000
python main.py perplexity --root .
python scripts/evaluate_watermark.py
python main.py author_pca --root .
python main.py publish --root .
```

Single pipeline call:

```bash
python main.py all --root .
```

## Resume Training

Explicit checkpoint:

```bash
python main.py train_model --root . --resume artifacts/checkpoints/step_53000.pt
```

Newest step checkpoint:

```bash
python main.py train_model --root . --resume latest
```

## Results Push Helpers

Push all tracked changes:

```bash
bash scripts/push_github.sh "Update poetru run"
```

Push only run outputs:

```bash
bash scripts/push_results.sh "Update run artifacts"
```

Mid-training snapshot to GitHub and HF:

```bash
bash scripts/sync_progress.sh
```

## Inference Setup

Download checkpoint and tokenizer from HF into local `artifacts/`:

```python
from pathlib import Path
from hub_utils import download_inference_artifacts

root = Path(".").resolve()
download_inference_artifacts(root)
```

## Watermark Configuration

Configuration object:

```python
from watermark import WatermarkConfig

wm_cfg = WatermarkConfig(
    gamma=0.25,
    delta=2.0,
    seed_scheme="kirchenbauer_soft",
)
```

Generation call with watermark enabled:

```python
token_ids, _ = generate_poem(
    model,
    prompt_ids,
    eos_id=tokenizer.eos_id,
    gen_cfg=GenerationConfig(),
    device=device,
    apply_watermark=True,
)
```

Detection call:

```python
from watermark import detect_watermark

det = detect_watermark(token_ids, tokenizer.vocab_size, wm_cfg)
print(det.z_score)
```
