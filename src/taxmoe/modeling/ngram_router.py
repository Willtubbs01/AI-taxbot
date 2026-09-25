from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .router_config import MoERouterConfig


@dataclass
class RouterDiagnostics:
    hidden_logits: torch.Tensor
    scaled_ngram_logits: torch.Tensor | None
    combined_logits: torch.Tensor


class TaxMoERouter(nn.Module):
    """Hidden-state router with optional additive n-gram expert-logit bias."""

    def __init__(self, config: MoERouterConfig):
        super().__init__()
        self.config = config
        if config.top_k > config.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.hidden_router = nn.Linear(config.hidden_dim, config.num_experts, bias=True)
        if config.ngram.enabled:
            if config.ngram.projection_bias:
                raise ValueError("Stage 4 v1 requires n-gram projection bias=False")
            self.ngram_router = nn.Linear(config.ngram.input_dim, config.num_experts, bias=False)
            self.ngram_scale = nn.Parameter(torch.tensor(float(config.ngram.initial_scale)))
        else:
            self.ngram_router = None
            self.register_parameter("ngram_scale", None)

    def forward(self, hidden_states: torch.Tensor, ngram_features: torch.Tensor | None = None,
                *, return_diagnostics: bool = False):
        hidden_logits = self.hidden_router(hidden_states)
        if not self.config.ngram.enabled:
            if return_diagnostics:
                return RouterDiagnostics(hidden_logits, None, hidden_logits)
            return hidden_logits
        if ngram_features is None:
            raise ValueError("NGRAM-ROUTER-FEATURES-MISSING")
        if hidden_states.shape[:2] != ngram_features.shape[:2]:
            raise ValueError("NGRAM-ROUTER-SEQUENCE-SHAPE-MISMATCH")
        if ngram_features.shape[-1] != self.config.ngram.input_dim:
            raise ValueError("NGRAM-ROUTER-DIMENSION-MISMATCH")
        ngram_logits = self.ngram_router(ngram_features)
        scaled = self.ngram_scale * ngram_logits
        combined = hidden_logits + scaled
        if return_diagnostics:
            return RouterDiagnostics(hidden_logits, scaled, combined)
        return combined
