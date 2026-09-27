from __future__ import annotations

from dataclasses import dataclass

import torch

from .experts import ExpertBank
from .router import RoutingDecision


@dataclass
class DispatchStats:
    assignment_count: torch.Tensor
    weighted_mass: torch.Tensor


def _validate(hidden_states: torch.Tensor, experts: ExpertBank, decision: RoutingDecision) -> tuple[int, int, int]:
    if hidden_states.ndim < 2:
        raise ValueError("MOE-DISPATCH-HIDDEN-SHAPE-INVALID")
    if decision.expert_indices.shape != decision.expert_weights.shape:
        raise ValueError("MOE-DISPATCH-SHAPE-MISMATCH")
    if tuple(decision.expert_indices.shape[:-1]) != tuple(hidden_states.shape[:-1]):
        raise ValueError("MOE-DISPATCH-SHAPE-MISMATCH")
    if not torch.isfinite(decision.expert_weights).all():
        raise ValueError("MOE-DISPATCH-NONFINITE-WEIGHTS")
    if (decision.expert_weights < 0).any():
        raise ValueError("MOE-DISPATCH-INVALID-WEIGHT")
    sums = decision.expert_weights.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("MOE-DISPATCH-WEIGHTS-NOT-NORMALIZED")
    if decision.expert_indices.numel():
        lo = int(decision.expert_indices.min())
        hi = int(decision.expert_indices.max())
        if lo < 0 or hi >= experts.num_experts:
            raise ValueError("MOE-DISPATCH-EXPERT-ID-OOB")
    n = hidden_states.numel() // hidden_states.shape[-1]
    return n, hidden_states.shape[-1], decision.expert_indices.shape[-1]


def dispatch_to_experts(
    hidden_states: torch.Tensor,
    experts: ExpertBank,
    decision: RoutingDecision,
    *,
    return_stats: bool = False,
):
    n, hidden_size, top_k = _validate(hidden_states, experts, decision)
    flat_hidden = hidden_states.reshape(n, hidden_size)
    flat_indices = decision.expert_indices.reshape(n, top_k)
    flat_weights = decision.expert_weights.reshape(n, top_k)

    token_ids = torch.arange(n, device=hidden_states.device, dtype=torch.long).unsqueeze(1).expand(n, top_k).reshape(-1)
    expert_ids = flat_indices.reshape(-1)
    weights = flat_weights.reshape(-1)
    if expert_ids.numel() != n * top_k:
        raise ValueError("MOE-DISPATCH-ASSIGNMENT-COUNT-MISMATCH")

    output = torch.zeros_like(flat_hidden)
    counts = torch.bincount(expert_ids, minlength=experts.num_experts)
    mass = torch.zeros(experts.num_experts, device=weights.device, dtype=torch.float32)
    mass = mass.scatter_add(0, expert_ids, weights.float())

    for expert_id, expert in enumerate(experts.experts):
        mask = expert_ids == expert_id
        ids = token_ids[mask]
        if ids.numel() == 0:
            continue
        expert_input = flat_hidden.index_select(0, ids)
        expert_output = expert(expert_input)
        selected_weights = weights[mask].to(dtype=expert_output.dtype).unsqueeze(-1)
        weighted = expert_output * selected_weights
        output = output.index_add(0, ids, weighted.to(dtype=output.dtype))

    output = output.reshape_as(hidden_states)
    if not torch.isfinite(output).all():
        raise ValueError("MOE-DISPATCH-NONFINITE-OUTPUT")
    if return_stats:
        return output, DispatchStats(assignment_count=counts.detach(), weighted_mass=mass.detach())
    return output


def reference_dispatch(hidden_states: torch.Tensor, experts: ExpertBank, decision: RoutingDecision) -> torch.Tensor:
    """Slow correctness oracle for tiny test tensors only."""
    _validate(hidden_states, experts, decision)
    shape = hidden_states.shape
    h = hidden_states.reshape(-1, shape[-1])
    idx = decision.expert_indices.reshape(h.shape[0], -1)
    weights = decision.expert_weights.reshape(h.shape[0], -1)
    rows = []
    for token in range(h.shape[0]):
        value = torch.zeros_like(h[token])
        for slot in range(idx.shape[1]):
            eid = int(idx[token, slot])
            y = experts.experts[eid](h[token : token + 1])[0]
            value = value + y * weights[token, slot].to(y.dtype)
        rows.append(value)
    return torch.stack(rows, dim=0).reshape(shape)
