from __future__ import annotations

import torch

from taxmoe.moe.balance import aggregate_balance_losses, load_balance_loss
from taxmoe.moe.config import BalanceLossConfig
from taxmoe.moe.upcycle import iter_sparse_moe

from .objectives import TrainingLossOutput


class MoECausalLMAdapter:
    """Stage 6 LM + load-balancing objective without changing Qwen's public forward contract."""

    def __init__(self, balance_config: BalanceLossConfig | None = None):
        self.balance_config = balance_config or BalanceLossConfig()

    def compute_loss(self, model, batch):
        if "ngram_feature_ids" in batch:
            raise ValueError("MOE-HIDDEN-ROUTER-UNEXPECTED-NGRAM-FEATURES")
        modules = sorted(iter_sparse_moe(model), key=lambda m: m.layer_index)
        if not modules:
            raise ValueError("MOE-INTEGRATION-NO-MOE-LAYERS")
        for module in modules:
            module.clear_aux()
            module.set_collect_aux(self.balance_config.enabled)
        try:
            out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels"))
            lm_loss = out.loss
            if lm_loss is None:
                raise ValueError("MOE-TRAIN-LM-LOSS-MISSING")
            if not self.balance_config.enabled:
                return TrainingLossOutput(total_loss=lm_loss, lm_loss=lm_loss)
            layer_losses = []
            layer_stats = {}
            scalar_metrics = {}
            attention_mask = batch.get("attention_mask")
            for module in modules:
                aux = module.pop_router_aux()
                if aux is None:
                    raise ValueError(f"MOE-INTEGRATION-MISSING-LAYER-AUX:{module.layer_index}")
                result = load_balance_loss(
                    aux.probabilities,
                    aux.expert_indices,
                    attention_mask,
                    num_experts=module.num_experts,
                    top_k=module.top_k,
                )
                layer_losses.append(result.loss)

                if attention_mask is None:
                    valid = torch.ones(aux.probabilities.shape[:-1], dtype=torch.bool, device=aux.probabilities.device)
                else:
                    valid = attention_mask.to(device=aux.probabilities.device, dtype=torch.bool)
                probs = aux.probabilities[valid].detach().float()
                top1 = torch.bincount(probs.argmax(-1), minlength=module.num_experts).float() / float(probs.shape[0])
                entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1).mean()

                layer_key = str(module.layer_index)
                layer_stats[layer_key] = {
                    "balance_loss": result.loss.detach(),
                    "assignment_fraction": result.assignment_fraction.detach(),
                    "probability_mass": result.probability_mass.detach(),
                    "token_inclusion_rate": result.token_inclusion_rate.detach(),
                    "top1_share": top1.detach(),
                    "mean_entropy": entropy.detach(),
                }
                scalar_metrics[f"router/{layer_key}/entropy"] = entropy.detach()
                scalar_metrics[f"router/{layer_key}/balance"] = result.loss.detach()
                for expert_id in range(module.num_experts):
                    scalar_metrics[f"router/{layer_key}/assignment_e{expert_id}"] = result.assignment_fraction[expert_id].detach()
                    scalar_metrics[f"router/{layer_key}/prob_mass_e{expert_id}"] = result.probability_mass[expert_id].detach()
                    scalar_metrics[f"router/{layer_key}/top1_e{expert_id}"] = top1[expert_id].detach()
            balance = aggregate_balance_losses(layer_losses)
            if torch.is_grad_enabled() and any(p.requires_grad for m in modules for p in m.router.parameters()) and not balance.requires_grad:
                raise ValueError("MOE-BALANCE-DISCONNECTED; use non-reentrant gradient checkpointing")
            total = lm_loss.float() + self.balance_config.coefficient * balance
            return TrainingLossOutput(
                total_loss=total,
                lm_loss=lm_loss,
                auxiliary_losses={"moe_balance": balance, "moe_layers": layer_stats, **scalar_metrics},
            )
        finally:
            for module in modules:
                module.set_collect_aux(False)


def enable_moe_gradient_checkpointing(model) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise ValueError("MOE-GRADIENT-CHECKPOINTING-UNSUPPORTED")
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError as exc:
        raise ValueError("MOE-GRADIENT-CHECKPOINTING-NONREENTRANT-UNSUPPORTED") from exc
    if hasattr(model, "config"):
        model.config.use_cache = False
