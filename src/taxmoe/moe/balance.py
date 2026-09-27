from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoadBalanceResult:
    loss: torch.Tensor
    probability_mass: torch.Tensor
    assignment_fraction: torch.Tensor
    assignment_count: torch.Tensor
    token_inclusion_rate: torch.Tensor
    valid_token_count: int


def load_balance_loss(
    router_probabilities: torch.Tensor,
    expert_indices: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    num_experts: int,
    top_k: int,
) -> LoadBalanceResult:
    if router_probabilities.shape[:-1] != expert_indices.shape[:-1]:
        raise ValueError("MOE-BALANCE-SHAPE-MISMATCH")
    if router_probabilities.shape[-1] != num_experts or expert_indices.shape[-1] != top_k:
        raise ValueError("MOE-BALANCE-SHAPE-MISMATCH")
    if attention_mask is None:
        valid = torch.ones(router_probabilities.shape[:-1], dtype=torch.bool, device=router_probabilities.device)
    else:
        if tuple(attention_mask.shape) != tuple(router_probabilities.shape[:-1]):
            raise ValueError("MOE-BALANCE-SHAPE-MISMATCH")
        valid = attention_mask.to(device=router_probabilities.device, dtype=torch.bool)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        raise ValueError("MOE-BALANCE-NO-VALID-TOKENS")

    probs = router_probabilities[valid].float()
    if not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("MOE-BALANCE-INVALID-PROBABILITY")
    sums = probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("MOE-BALANCE-PROBABILITY-NOT-NORMALIZED")

    indices = expert_indices[valid].reshape(-1)
    if indices.numel():
        if int(indices.min()) < 0 or int(indices.max()) >= num_experts:
            raise ValueError("MOE-BALANCE-EXPERT-ID-OOB")
    counts = torch.bincount(indices, minlength=num_experts)
    assignment_fraction = counts.float() / float(n_valid * top_k)
    probability_mass = probs.mean(dim=0)
    token_inclusion_rate = counts.float() / float(n_valid)
    loss = float(num_experts) * torch.sum(assignment_fraction.to(probability_mass.device) * probability_mass)
    if not torch.isfinite(loss):
        raise ValueError("MOE-BALANCE-NONFINITE-LOSS")
    return LoadBalanceResult(
        loss=loss,
        probability_mass=probability_mass,
        assignment_fraction=assignment_fraction.to(probability_mass.device),
        assignment_count=counts,
        token_inclusion_rate=token_inclusion_rate.to(probability_mass.device),
        valid_token_count=n_valid,
    )


def aggregate_balance_losses(losses: list[torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ValueError("MOE-BALANCE-MISSING-LAYER-AUX")
    return torch.stack([x.float() for x in losses]).mean()
