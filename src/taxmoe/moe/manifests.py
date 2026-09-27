from __future__ import annotations

from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel


class LayerUpcycleReport(TaxMoEModel):
    layer_index: int
    source_mlp_class: str
    source_mlp_hash: str
    source_parameter_count: int
    expert_hashes: list[str] = Field(default_factory=list)
    expert_parameter_count: int
    router_parameter_count: int
    router_seed: int
    router_hash: str
    copy_verified: bool = True
    storage_independence_verified: bool = True


class MoEUpcycleReport(TaxMoEModel):
    schema_version: str = "1"
    target_layers: list[int]
    num_experts: int
    top_k: int
    parent_parameter_count: int
    converted_parameter_count: int
    expected_parameter_count: int
    preserved_tensor_count: int
    unexpected_preserved_changes: list[str] = Field(default_factory=list)
    layers: list[LayerUpcycleReport] = Field(default_factory=list)
    architecture_hash: str
    passed: bool = True
