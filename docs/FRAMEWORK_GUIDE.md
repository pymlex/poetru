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
| `HF_TOKENIZER_REPO` | keep equal to `HF_MODEL_REPO` for single-repo layout |
| `GITHUB_TOKEN` | optional token for scripted GitHub APIs |

## Core Pipeline Commands

`python main.py train_tokenizer --root .`  
Trains ByteLevel BPE and writes `artifacts/tokenizer/tokenizer.json` plus token-length plots and summary stats.

`python main.py train_model --root .`  
Runs full optimisation from step 0 with config defaults. Writes periodic `step_*.pt`, `final.pt`, and `artifacts/logs/train_history.csv`.

`python main.py train_model --root . --resume latest`  
Restores optimizer, scheduler, epoch, and step from newest checkpoint in `artifacts/checkpoints`.

`python scripts/generate_poems.py --root . --count 1000`  
Generates watermarked continuations and writes `artifacts/generated_poems.jsonl`. `--count` overrides `GenerationConfig.target_poem_count`.

`python main.py perplexity --root .`  
Computes validation loss and perplexity on a capped number of batches and writes `artifacts/metrics/perplexity.json`.

`python scripts/evaluate_watermark.py`  
Builds watermark metrics, ROC curve, confusion matrix, and `watermark_scores.csv`.

`python main.py author_pca --root .`  
Builds author and generated embedding sets, PCA projection figure, and permutation-test significance statistics.

`python main.py publish --root .`  
Publishes checkpoint, tokenizer, metrics, generated poems, and model card to Hugging Face model repo.

`python main.py all --root .`  
Runs tokenizer, training, generation, perplexity, watermark evaluation, author PCA, and publish in sequence.

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
