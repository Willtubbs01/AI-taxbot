from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterStats:
    valid_token_count: int
    probability_mass: torch.Tensor
    assignment_fraction: torch.Tensor
    token_inclusion_rate: torch.Tensor
    top1_share: torch.Tensor
    mean_entropy: float
    pair_counts: dict[str, int]


def compute_router_stats(probabilities: torch.Tensor, expert_indices: torch.Tensor, attention_mask: torch.Tensor | None, *, num_experts: int, top_k: int) -> RouterStats:
    if attention_mask is None:
        valid = torch.ones(probabilities.shape[:-1], dtype=torch.bool, device=probabilities.device)
    else:
        valid = attention_mask.to(device=probabilities.device, dtype=torch.bool)
    probs = probabilities[valid].detach().float()
    idx = expert_indices[valid].detach().long()
    n = int(probs.shape[0])
    if n == 0:
        raise ValueError("MOE-STATS-NO-VALID-TOKENS")
    counts = torch.bincount(idx.reshape(-1), minlength=num_experts).float()
    assignment = counts / float(n * top_k)
    inclusion = counts / float(n)
    p_mass = probs.mean(0)
    top1 = torch.bincount(probs.argmax(-1), minlength=num_experts).float() / float(n)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1).mean()
    pairs: dict[str, int] = {}
    if top_k == 2:
        for row in idx.cpu().tolist():
            a, b = sorted((int(row[0]), int(row[1])))
            key = f"{a}-{b}"
            pairs[key] = pairs.get(key, 0) + 1
    return RouterStats(
        valid_token_count=n,
        probability_mass=p_mass.cpu(),
        assignment_fraction=assignment.cpu(),
        token_inclusion_rate=inclusion.cpu(),
        top1_share=top1.cpu(),
        mean_entropy=float(entropy.cpu()),
        pair_counts=pairs,
    )
