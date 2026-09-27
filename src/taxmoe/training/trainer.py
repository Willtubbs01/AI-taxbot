from __future__ import annotations

import math
import torch

from .objectives import DenseCausalLMAdapter
from .progress import commit_step


class DenseCPTTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        optimization_config,
        *,
        unique_train_tokens: int,
        device="cuda",
        adapter=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = optimization_config
        self.unique_train_tokens = unique_train_tokens
        self.device = device
        self.adapter = adapter or DenseCausalLMAdapter()

        use_fp16_scaler = self.cfg.precision == "fp16" and str(device).startswith("cuda")
        if use_fp16_scaler:
            fp16_trainable = any(
                p.requires_grad and p.dtype == torch.float16 for p in self.model.parameters()
            )
            if fp16_trainable:
                raise ValueError(
                    "TRAIN-FP16-PARAMETERS-WITH-GRADSCALER: FP16 AMP requires FP32 master parameters; "
                    "use BF16 parameters on a supported GPU or load FP32 parameters for FP16 AMP"
                )
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    def _to_device(self, batch):
        return {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in batch.items()
            if k != "record_ids"
        }

    def train_optimizer_step(self, microbatches, progress):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        examples = tokens = supervised = 0
        loss_total = 0.0
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.cfg.precision)
        lm_total = 0.0
        aux_totals = {}

        for batch in microbatches:
            batch = self._to_device(batch)
            examples += batch["input_ids"].shape[0]
            tokens += int(batch["attention_mask"].sum())
            supervised += int((batch["labels"] != -100).sum())

            if amp_dtype is not None and str(self.device).startswith("cuda"):
                ctx = torch.autocast("cuda", dtype=amp_dtype)
            else:
                ctx = torch.autocast("cpu", enabled=False)

            with ctx:
                loss_out = self.adapter.compute_loss(self.model, batch)
                loss = loss_out.total_loss / self.cfg.gradient_accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("TRAIN-NONFINITE-LOSS")

            self.scaler.scale(loss).backward()
            loss_total += float(loss.detach().float().cpu())
            lm_total += (
                float(loss_out.lm_loss.detach().float().cpu())
                / self.cfg.gradient_accumulation_steps
            )
            for name, value in loss_out.auxiliary_losses.items():
                if hasattr(value, "detach") and getattr(value, "ndim", 1) == 0:
                    aux_totals[name] = aux_totals.get(name, 0.0) + (
                        float(value.detach().float().cpu())
                        / self.cfg.gradient_accumulation_steps
                    )

        self.scaler.unscale_(self.optimizer)
        grad = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip_norm
            )
        )
        if not math.isfinite(grad):
            raise FloatingPointError("TRAIN-NONFINITE-GRADIENT")

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        progress = commit_step(
            progress,
            examples=examples,
            tokens=tokens,
            supervised_tokens=supervised,
            unique_train_tokens=self.unique_train_tokens,
        )
        metrics = {
            "loss": loss_total,
            "lm_loss": lm_total,
            "grad_norm": grad,
            "tokens": tokens,
            "examples": examples,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        metrics.update({f"aux/{k}": v for k, v in aux_totals.items()})
        return progress, metrics
