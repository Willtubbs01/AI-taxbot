from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from taxmoe.ingestion.hashing import stable_int


@dataclass
class RoutingDecision:
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor


@dataclass
class RouterAux:
    logits: torch.Tensor
    probabilities: torch.Tensor
    expert_indices: torch.Tensor


@dataclass
class RouterForwardOutput:
    decision: RoutingDecision
    aux: RouterAux | None = None


class TopKRoutingPolicy(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("MOE-ROUTER-TOPK-INVALID")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

    def forward(self, logits: torch.Tensor) -> RoutingDecision:
        if logits.shape[-1] != self.num_experts:
            raise ValueError("MOE-ROUTER-EXPERT-DIM-MISMATCH")
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("MOE-ROUTER-NONFINITE-LOGITS")
        selected_logits, indices = torch.topk(logits, k=self.top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)
        return RoutingDecision(expert_indices=indices.long(), expert_weights=weights)


class HiddenStateRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        bias: bool = True,
        init_std: float = 0.01,
        seed: int = 1701,
        layer_index: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("MOE-ROUTER-HIDDEN-DIM-INVALID")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.layer_index = int(layer_index)
        self.global_seed = int(seed)
        self.layer_seed = stable_int("taxmoe-router", seed, layer_index, bits=32)
        self.init_std = float(init_std)
        self.hidden_proj = nn.Linear(hidden_size, num_experts, bias=bias, device=device, dtype=dtype)
        self.policy = TopKRoutingPolicy(num_experts, top_k)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.layer_seed)
        values = torch.randn(tuple(self.hidden_proj.weight.shape), generator=gen, dtype=torch.float32) * self.init_std
        with torch.no_grad():
            self.hidden_proj.weight.copy_(values.to(device=self.hidden_proj.weight.device, dtype=self.hidden_proj.weight.dtype))
            if self.hidden_proj.bias is not None:
                self.hidden_proj.bias.zero_()

    def forward(self, hidden_states: torch.Tensor, *, return_aux: bool = False) -> RouterForwardOutput:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("MOE-ROUTER-HIDDEN-DIM-MISMATCH")
        logits = self.hidden_proj(hidden_states).float()
        decision = self.policy(logits)
        aux = None
        if return_aux:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux = RouterAux(logits=logits, probabilities=probs, expert_indices=decision.expert_indices)
        return RouterForwardOutput(decision=decision, aux=aux)
