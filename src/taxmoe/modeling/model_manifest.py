from __future__ import annotations
from typing import Any
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel

class BaseModelManifest(TaxMoEModel):
    schema_version: str = "0.1"
    model_id: str
    revision: str
    architecture: str
    model_class: str
    config_class: str
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    max_position_embeddings: int | None = None
    hidden_act: str | None = None
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool | None = None
    rope_theta: float | None = None
    parameter_count: int
    trainable_parameter_count: int
    config_sha256: str
    tokenizer_manifest_hash: str | None = None
    weight_files: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    base_model_fingerprint: str

class LayerInspection(TaxMoEModel):
    layer_index: int
    layer_class: str
    mlp_class: str
    mlp_path: str
    parameter_count: int
    projection_shapes: dict[str, list[int]] = Field(default_factory=dict)
    moe_candidate: bool = False

class ArchitectureReport(TaxMoEModel):
    model_class: str
    parameter_count: int
    trainable_parameter_count: int
    layer_count: int
    layers: list[LayerInspection] = Field(default_factory=list)
    candidate_layers: list[int] = Field(default_factory=list)
    candidate_layers_valid: bool = True
    tied_word_embeddings: bool | None = None
