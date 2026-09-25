from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Stage4Model


class NGramRouterConfig(Stage4Model):
    enabled: bool = False
    fusion: Literal["additive_logits"] = "additive_logits"
    input_dim: int = 32
    initial_scale: float = 0.01
    projection_bias: bool = False


class MoERouterConfig(Stage4Model):
    hidden_dim: int = Field(default=1024, gt=0)
    num_experts: int = Field(default=4, gt=1)
    top_k: int = Field(default=2, gt=0)
    ngram: NGramRouterConfig = NGramRouterConfig()
