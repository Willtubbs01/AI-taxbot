from __future__ import annotations
from enum import Enum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class CheckpointType(str,Enum): FULL='full';WEIGHTS_ONLY='weights_only'
class ArtifactFile(TaxMoEModel): path:str; bytes:int; sha256:str
class TrainingCheckpointManifest(TaxMoEModel):
    checkpoint_version:str='1'; checkpoint_id:str; checkpoint_type:CheckpointType; run_id:str; run_spec_hash:str; global_step:int; consumed_training_tokens:int; effective_passes:float; parent_model_fingerprint:str; input_release_hash:str; training_data_manifest_hash:str; model_architecture_hash:str; training_config_hash:str; model_files:list[ArtifactFile]=Field(default_factory=list); optimizer_file:ArtifactFile|None=None; scheduler_file:ArtifactFile|None=None; scaler_file:ArtifactFile|None=None; rng_file:ArtifactFile|None=None; sampler_state_file:ArtifactFile|None=None; progress_file:ArtifactFile|None=None; checkpoint_content_hash:str=''; complete:bool=False
