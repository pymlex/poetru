from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm

from configs import GenerationConfig, TrainConfig, TransformerConfig
from gpu_telemetry import GpuTelemetry
from loss_utils import causal_language_modeling_loss
from lr_schedule import CosineWarmupScheduler
from model import PoetruCausalLM
from watermark import WatermarkConfig, sample_next_token


class Trainer:
    """AdamW trainer with cosine warmup, gradient accumulation, and GPU telemetry."""

    def __init__(
        self,
        model: PoetruCausalLM,
        train_loader,
        val_loader,
        cfg: TrainConfig,
        tcfg: TransformerConfig,
        checkpoint_dir: Path,
        logs_dir: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.tcfg = tcfg
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
        self.total_steps = int(steps_per_epoch * cfg.num_epochs)
        warmup_steps = int(self.total_steps * cfg.warmup_ratio)

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.total_steps,
        )

        self.autocast_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and torch.cuda.is_available() else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.autocast_dtype == torch.float16)

        self.history_csv = logs_dir / "train_history.csv"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.history_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss", "val_loss", "lr", "gpu_util", "gpu_mem_used_mb", "gpu_mem_total_mb"])

        self.gpu = GpuTelemetry() if torch.cuda.is_available() else None

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        """Computes mean validation loss over a capped number of batches.

        Args:
            max_batches: Optional upper bound on validation batches.

        Returns:
            Scalar validation loss.
        """

        self.model.eval()
        total = 0.0
        seen = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = causal_language_modeling_loss(logits, input_ids, attention_mask)
            total += float(loss.item())
            seen += 1
        self.model.train()
        return total / max(seen, 1)

    def _append_log(self, row: list) -> None:
        with self.history_csv.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    def run(self) -> Path:
        """Executes the full optimisation schedule and writes checkpoints.

        Args:
            None.

        Returns:
            Path to the final checkpoint file.
        """

        from checkpoint_utils import save_checkpoint

        self.model.train()
        global_step = 0
        epoch = 0.0
        running_loss = 0.0
        accum = 0

        data_iter = iter(self.train_loader)
        pbar = tqdm(total=self.total_steps, desc="Training")

        while global_step < self.total_steps:
            self.optimizer.zero_grad(set_to_none=True)
            micro_loss = 0.0

            for _ in range(self.cfg.grad_accum_steps):
                batch = next(data_iter, None)
                if batch is None:
                    epoch += 1.0
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    loss = causal_language_modeling_loss(logits, input_ids, attention_mask)
                    loss = loss / self.cfg.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_loss += float(loss.item())
                accum += 1

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            global_step += 1
            running_loss += micro_loss
            pbar.update(1)

            gpu_util = 0.0
            mem_used = 0
            mem_total = 0
            if self.gpu is not None:
                snap = self.gpu.read()
                gpu_util = snap.utilisation_percent
                mem_used = snap.mem_used_bytes // (1024 * 1024)
                mem_total = snap.mem_total_bytes // (1024 * 1024)

            if global_step % self.cfg.log_every_steps == 0:
                avg_loss = running_loss / max(self.cfg.log_every_steps, 1)
                val_loss = self.evaluate(max_batches=self.cfg.eval_batches)
                lr = self.scheduler.get_last_lr()[0]
                self._append_log(
                    [global_step, epoch, avg_loss, val_loss, lr, gpu_util, mem_used, mem_total]
                )
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val": f"{val_loss:.4f}", "gpu": f"{gpu_util:.0f}%"})
                running_loss = 0.0

            if global_step % self.cfg.checkpoint_every_steps == 0:
                ckpt = self.checkpoint_dir / f"step_{global_step}.pt"
                save_checkpoint(ckpt, self.model, self.optimizer, self.scheduler, global_step, epoch)

        final_ckpt = self.checkpoint_dir / "final.pt"
        save_checkpoint(final_ckpt, self.model, self.optimizer, self.scheduler, global_step, epoch)
        pbar.close()

        if self.gpu is not None:
            self.gpu.close()

        return final_ckpt


@torch.inference_mode()
def generate_poem(
    model: PoetruCausalLM,
    prompt_ids: list[int],
    eos_id: int,
    gen_cfg: GenerationConfig,
    device: torch.device,
    apply_watermark: bool = True,
) -> tuple[list[int], str]:
    """Autoregressively continues a prompt with optional watermarking.

    Args:
        model: Trained causal LM in eval mode.
        prompt_ids: Prefix token ids without trailing EOS unless desired.
        eos_id: End-of-sequence id.
        gen_cfg: Sampling and watermark hyperparameters.
        device: Torch device.
        apply_watermark: Enables green-list logit bias during sampling.

    Returns:
        Tuple `(token_ids, decoded_text)`.
    """

    wm_cfg = WatermarkConfig(gamma=gen_cfg.watermark_gamma, delta=gen_cfg.watermark_delta)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = prompt_ids.copy()

    for _ in range(gen_cfg.max_new_tokens):
        ctx = ids if ids.shape[1] <= model.cfg.max_seq_len else ids[:, -model.cfg.max_seq_len :]
        logits = model(ctx)[0, -1, :]
        prev_id = int(generated[-1])
        next_id = sample_next_token(
            logits,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            prev_token_id=prev_id,
            cfg=wm_cfg,
            apply_watermark=apply_watermark,
        )
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
        if next_id == eos_id:
            break

    return generated, ""


def mean_pool_hidden(model: PoetruCausalLM, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pools final hidden states over valid tokens.

    Args:
        model: Language model used as an encoder.
        token_ids: `[batch, sequence]` indices.
        attention_mask: `[batch, sequence]` mask.

    Returns:
        Tensor `[batch, hidden_dim]`.
    """

    x = model.dropout(model.embed(token_ids))
    for layer in model.layers:
        x = layer(x, attention_mask=attention_mask)
    x = model.out_norm(x)
    mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom
