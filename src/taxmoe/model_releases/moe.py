from __future__ import annotations

from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel

from .models import ModelReleaseStatus, ReleaseFile


class TaxMoEReleaseManifest(TaxMoEModel):
    schema_version: str = "0.1"
    release_id: str
    name: str = "TaxMoE-4E-Top2"
    version: str = "0.1"
    status: ModelReleaseStatus = ModelReleaseStatus.CANDIDATE
    architecture_type: str = "taxmoe_qwen3"
    taxdense_parent_id: str
    taxdense_parent_hash: str
    moe_init_id: str
    moe_init_hash: str
    input_release_id: str
    input_release_hash: str
    source_run_id: str
    source_checkpoint_id: str
    source_checkpoint_hash: str
    num_experts: int = 4
    moe_top_k: int = 2
    target_layers: list[int] = Field(default_factory=lambda: [3, 7, 11, 15, 19, 23, 27])
    router_type: str = "hidden_state"
    dispatch_semantics_version: str = "native_v1"
    balance_loss_version: str = "switch_style_v1"
    training_config_hash: str
    optimization_config_hash: str
    hardware_envelope_hash: str
    evaluation_hash: str
    routing_health_hash: str
    reproducibility_hash: str
    parameter_count: int
    architecture_hash: str
    weight_manifest_hash: str
    model_files: list[ReleaseFile] = Field(default_factory=list)
    release_content_hash: str = ""
