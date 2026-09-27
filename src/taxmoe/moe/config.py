from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from taxmoe.schemas.common import TaxMoEModel


class RouterInitConfig(TaxMoEModel):
    bias: bool = True
    init_std: float = Field(default=0.01, gt=0)
    seed: int = 1701
    logits_dtype: Literal["fp32"] = "fp32"


class DispatchConfig(TaxMoEModel):
    backend: Literal["native_v1"] = "native_v1"
    token_drop: bool = False
    capacity_limit: int | None = None
    execute_padding: bool = True
    expert_execution_order: Literal["ascending_id"] = "ascending_id"

    @model_validator(mode="after")
    def stage6_semantics(self):
        if self.token_drop:
            raise ValueError("MOE-CONFIG-TOKEN-DROP-FORBIDDEN")
        if self.capacity_limit is not None:
            raise ValueError("MOE-CONFIG-CAPACITY-LIMIT-FORBIDDEN")
        return self


class BalanceLossConfig(TaxMoEModel):
    enabled: bool = True
    version: Literal["switch_style_v1"] = "switch_style_v1"
    coefficient: float = Field(default=0.01, ge=0)
    layer_aggregation: Literal["mean"] = "mean"
    valid_tokens: Literal["attention_mask"] = "attention_mask"


class MoEUpcycleConfig(TaxMoEModel):
    version: str = "1"
    target_layers: list[int] = Field(default_factory=lambda: [3, 7, 11, 15, 19, 23, 27])
    num_experts: int = Field(default=4, gt=1)
    top_k: int = Field(default=2, gt=0)
    expert_init: Literal["exact_copy"] = "exact_copy"
    router: RouterInitConfig = Field(default_factory=RouterInitConfig)
    dispatch: DispatchConfig = Field(default_factory=DispatchConfig)
    balance_loss: BalanceLossConfig = Field(default_factory=BalanceLossConfig)

    @model_validator(mode="after")
    def valid(self):
        if self.top_k > self.num_experts:
            raise ValueError("MOE-CONFIG-TOPK-EXCEEDS-EXPERTS")
        if not self.target_layers:
            raise ValueError("MOE-CONFIG-NO-TARGET-LAYERS")
        if len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("MOE-CONFIG-DUPLICATE-TARGET-LAYER")
        if any(i < 0 for i in self.target_layers):
            raise ValueError("MOE-CONFIG-NEGATIVE-TARGET-LAYER")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw.get("moe", raw))
