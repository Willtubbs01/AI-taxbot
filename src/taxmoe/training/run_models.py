from __future__ import annotations
from enum import Enum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class RunStatus(str,Enum):
    CREATED='created';RUNNING='running';INTERRUPTED='interrupted';COMPLETED='completed';FAILED='failed';CANCELLED='cancelled'
class TrainingProgress(TaxMoEModel):
    global_step:int=0; optimizer_step:int=0; micro_step:int=0; gradient_accumulation_step:int=0; consumed_examples:int=0; consumed_tokens:int=0; supervised_tokens:int=0; effective_passes:float=0.0; current_pass_index:int=0
class TrainingRunManifest(TaxMoEModel):
    run_id:str; run_spec_hash:str; parent_model_fingerprint:str; input_release_id:str; input_release_hash:str; data_manifest_hash:str; objective_hash:str; optimizer_config_hash:str; scheduler_config_hash:str; precision:str; code_revision:str|None=None; status:RunStatus=RunStatus.CREATED; metadata:dict=Field(default_factory=dict)
