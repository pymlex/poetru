# poetru

Russian poetry causal language model on [IlyaGusev/stihi_ru](https://huggingface.co/datasets/IlyaGusev/stihi_ru). The repository trains a ~25M parameter Transformer from scratch with ByteLevel BPE, RoPE, grouped-query attention, MLA-style KV compression, SwiGLU feed-forward blocks, cosine warmup scheduling, digital watermarking at inference, and post-training evaluation for perplexity, watermark detection, and author-space PCA.

Hub model: [pymlex/poetru-25m](https://huggingface.co/pymlex/poetru-25m)

## Architecture

The model follows the Chinchilla token-to-parameter budget for a ~455M BPE-token corpus seen for 2.5 epochs, which yields about $1.14 \times 10^9$ training tokens and motivates a parameter count near $2.5 \times 10^7$.

| Component | Setting |
| --- | --- |
| Vocabulary | 24000 ByteLevel BPE tokens |
| Context length | 512 |
| Layers | 5 |
| Hidden dimension $d$ | 384 |
| Query heads $H$ | 8 |
| KV heads $H_{kv}$ | 4 |
| Head dimension $d_h = d/H$ | 48 |
| MLA latent dimension | 384 |
| SwiGLU intermediate | 1024 |
| Positional encoding | RoPE with $\theta = 10000$ |
| Normalisation | RMSNorm pre-layer |
| Output projection | tied with token embeddings |

### RoPE

Rotary embeddings act on query and key head vectors. For head dimension $d_h$, dimension pairs $(2k, 2k+1)$ are rotated by angle $m\theta_k$ at position $m$:

$$
\theta_k = \theta_0^{-2k/d_h}, \quad k \in \{0, \ldots, d_h/2 - 1\}
$$

$$
\begin{pmatrix} q'_{2k} \\ q'_{2k+1} \end{pmatrix}
=
\begin{pmatrix} \cos m\theta_k & -\sin m\theta_k \\ \sin m\theta_k & \cos m\theta_k \end{pmatrix}
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
$$

The same transform applies to keys before scaled dot-product attention.

### MLA with GQA

Keys and values are compressed through a latent projection $c_t = W_d h_t \in \mathbb{R}^{d_l}$ with $d_l = 384$, then expanded to KV heads:

$$
k_t = W_k c_t, \quad v_t = W_v c_t
$$

Queries use $H=8$ heads while KV tensors use $H_{kv}=4$ heads. Each KV head is repeated with `repeat_interleave` so every query head attends to a KV head.

### SwiGLU

The feed-forward block uses gated linear units:

$$
\mathrm{SwiGLU}(x) = W_2 \big(\mathrm{SiLU}(W_1 x) \odot W_3 x\big)
$$

with intermediate width 1024.

### Watermarking

Generation follows Kirchenbauer et al., 2023. Given previous token $s_{t-1}$, a pseudo-random green list $G_t \subset \mathcal{V}$ of size $\gamma|\mathcal{V}|$ is formed. Logits receive additive bias $\delta$ on green tokens before nucleus sampling. Detection counts green tokens among generated positions and applies a one-sided z-test against expected fraction $\gamma$.

## Dataset

Source: `IlyaGusev/stihi_ru`, about 5.15M Russian poems. Raw character lengths have median near 464 and 75th percentile near 673 characters. After ByteLevel BPE training the token histogram is written to `artifacts/metrics/token_length_hist.png`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `HF_TOKEN`, `HF_MODEL_REPO`, and `HF_TOKENIZER_REPO` in `.env`.

## Pipeline

```bash
python main.py train_tokenizer
python main.py train_model
python scripts/generate_poems.py
python main.py perplexity
python scripts/evaluate_watermark.py
python main.py author_pca
python main.py publish
```

Single command:

```bash
python main.py all
```

Override generated poem count:

```bash
python scripts/generate_poems.py --count 400
```

GitHub push helper without system-wide git configuration:

```bash
bash scripts/push_github.sh "Train poetru tokenizer"
```

Hub upload helper:

```bash
bash scripts/push_hub.sh publish
```

Interactive workflow lives in `notebooks/poetru.ipynb`.

## Outputs

| Path | Content |
| --- | --- |
| `artifacts/tokenizer/tokenizer.json` | trained BPE |
| `artifacts/checkpoints/final.pt` | model weights |
| `artifacts/logs/train_history.csv` | loss, val loss, LR, GPU telemetry |
| `artifacts/generated_poems.jsonl` | watermarked generations |
| `artifacts/metrics/perplexity.json` | validation perplexity |
| `artifacts/metrics/watermark_metrics.json` | accuracy, precision, recall, F1, ROC-AUC |
| `artifacts/metrics/author_pca.png` | author centroids and generated poems |
| `artifacts/metrics/author_embeddings.npz` | saved author centroid vectors |

## License

GPL-3.0. See `LICENSE`.
