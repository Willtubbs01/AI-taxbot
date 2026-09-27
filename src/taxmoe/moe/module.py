from __future__ import annotations

import torch
from torch import nn

from .dispatch import dispatch_to_experts
from .experts import ExpertBank
from .router import HiddenStateRouter, RouterAux


class SparseMoE(nn.Module):
    """Drop-in replacement for a Qwen MLP: Tensor [...,H] -> Tensor [...,H]."""

    def __init__(self, *, layer_index: int, experts: ExpertBank, router: HiddenStateRouter):
        super().__init__()
        if experts.num_experts != router.num_experts:
            raise ValueError("MOE-INTEGRATION-EXPERT-ROUTER-COUNT-MISMATCH")
        self.layer_index = int(layer_index)
        self.experts = experts
        self.router = router
        self._collect_aux = False
        self._last_router_aux: RouterAux | None = None

    @property
    def num_experts(self) -> int:
        return self.experts.num_experts

    @property
    def top_k(self) -> int:
        return self.router.top_k

    def set_collect_aux(self, enabled: bool) -> None:
        self._collect_aux = bool(enabled)
        if not enabled:
            self._last_router_aux = None

    def clear_aux(self) -> None:
        self._last_router_aux = None

    def pop_router_aux(self) -> RouterAux | None:
        value = self._last_router_aux
        self._last_router_aux = None
        return value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Non-reentrant activation checkpointing re-enters this forward during
        # backward.  The adapter intentionally clears _collect_aux after the
        # original forward, so using _collect_aux to decide which tensor ops the
        # router executes would make recomputation differ from the original
        # graph.  Keep the router tensor graph invariant while training; only
        # the Python side-channel storage is conditional.
        routed = self.router(hidden_states, return_aux=(self.training or self._collect_aux))
        output = dispatch_to_experts(hidden_states, self.experts, routed.decision)
        if self._collect_aux:
            if routed.aux is None:
                raise ValueError("MOE-INTEGRATION-MISSING-LAYER-AUX")
            self._last_router_aux = routed.aux
        return output
