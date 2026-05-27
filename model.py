from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from configs import TransformerConfig


class RMSNorm(nn.Module):
    """Root mean square layer normalisation used inside Transformer blocks."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Hidden states with shape `[batch, sequence, dim]`.

        Returns:
            Tensor with the same shape as `x` after RMS normalisation and scaling.
        """

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.scale * x_norm


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Applies rotary embeddings to query and key tensors.

    Args:
        q: Queries shaped `[batch, heads, sequence, head_dim]`.
        k: Keys shaped `[batch, kv_heads, sequence, head_dim]` before repeat if needed.
        cos: Cosine factors shaped `[1, 1, sequence, head_dim // 2]`.
        sin: Sine factors shaped `[1, 1, sequence, head_dim // 2]`.

    Returns:
        Tuple `(q_rot, k_rot)` with RoPE applied.
    """

    d = q.shape[-1]
    half = d // 2
    q1 = q[..., :half]
    q2 = q[..., half:]
    k1 = k[..., :half]
    k2 = k[..., half:]
    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_out, k_out


class MultiHeadLatentAttention(nn.Module):
    """Multi-head causal attention with GQA, RoPE, and MLA-style latent KV compression."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.n_head != 0:
            raise ValueError("hidden_dim must divide n_head.")
        if cfg.n_head % cfg.n_kv_head != 0:
            raise ValueError("n_head must divide n_kv_head.")

        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.n_head
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.q_heads = cfg.n_head
        self.kv_heads = cfg.n_kv_head
        self.group = cfg.n_head // cfg.n_kv_head

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.n_head * self.head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.hidden_dim, cfg.latent_dim, bias=False)
        self.k_up = nn.Linear(cfg.latent_dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_up = nn.Linear(cfg.latent_dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)

        rope_half = self.head_dim // 2
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, rope_half, dtype=torch.float32) / float(rope_half))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, cfg.max_seq_len, cfg.max_seq_len), persistent=False)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Projections, latent KV expansion, RoPE, softmax attention, and output projection.

        Args:
            x: Hidden states `[batch, sequence, hidden_dim]`.
            attention_mask: Boolean or `{0,1}` mask `[batch, sequence]` where `True` marks valid tokens.

        Returns:
            Tensor `[batch, sequence, hidden_dim]` after attention and residual pathway outside this module.
        """

        bsz, seqlen, _ = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.q_heads, self.head_dim).transpose(1, 2)

        latent = self.kv_down(x)
        k = self.k_up(latent).view(bsz, seqlen, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(bsz, seqlen, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        t = torch.arange(seqlen, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.rope_inv_freq.to(device=x.device))
        cos = freqs.cos().to(dtype=q.dtype).view(1, 1, seqlen, -1)
        sin = freqs.sin().to(dtype=q.dtype).view(1, 1, seqlen, -1)
        q, k = apply_rotary_emb(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:, :, :seqlen, :seqlen], float("-inf"))

        if attention_mask is not None:
            m = attention_mask.to(dtype=torch.bool)
            if m.dim() != 2:
                raise ValueError("attention_mask must have shape [batch, sequence].")
            key_mask = m.view(bsz, 1, 1, seqlen)
            attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))

        probs = F.softmax(attn_scores, dim=-1)
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v)

        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seqlen, self.q_heads * self.head_dim)
        return self.o_proj(ctx)


class SwiGLUMLP(nn.Module):
    """Feed-forward block with SwiGLU activation."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        hidden = cfg.intermediate_dim
        self.gate_up = nn.Linear(cfg.hidden_dim, hidden * 2, bias=True)
        self.down = nn.Linear(hidden, cfg.hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU transformation.

        Args:
            x: Hidden states `[batch, sequence, hidden_dim]`.

        Returns:
            Tensor `[batch, sequence, hidden_dim]` projected back to model width.
        """

        a, b = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class TransformerBlock(nn.Module):
    """Pre-norm residual block with MLA attention and SwiGLU MLP."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim)
        self.attn = MultiHeadLatentAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_dim)
        self.mlp = SwiGLUMLP(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Residual updates around attention and MLP sublayers.

        Args:
            x: Hidden states `[batch, sequence, hidden_dim]`.
            attention_mask: Padding mask broadcast into attention scores.

        Returns:
            Tensor with the same layout as `x`.
        """

        x = x + self.dropout(self.attn(self.attn_norm(x), attention_mask=attention_mask))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class PoetruCausalLM(nn.Module):
    """Causal Transformer language model with tied input and output embeddings."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.out_norm = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Token embeddings, stack of blocks, and logits.

        Args:
            input_ids: Token indices `[batch, sequence]`.
            attention_mask: Padding mask `[batch, sequence]`.

        Returns:
            Logits `[batch, sequence, vocab_size]`.
        """

        x = self.dropout(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.out_norm(x)
        return self.lm_head(x)
