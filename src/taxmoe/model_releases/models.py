from __future__ import annotations
from enum import Enum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class ModelReleaseStatus(str,Enum): CANDIDATE='candidate';FROZEN='frozen';SUPERSEDED='superseded';WITHDRAWN='withdrawn'
class ReleaseFile(TaxMoEModel): path:str; bytes:int; sha256:str
class TaxDenseReleaseManifest(TaxMoEModel):
    schema_version:str='0.1'; release_id:str; name:str='TaxDense-0.6B'; version:str='0.1'; status:ModelReleaseStatus=ModelReleaseStatus.CANDIDATE; architecture:str; parent_model_id:str; parent_model_revision:str; parent_model_fingerprint:str; tokenizer_manifest_hash:str; input_release_id:str; input_release_hash:str; source_run_id:str; source_checkpoint_id:str; source_checkpoint_hash:str; training_data_manifest_hash:str; objective_config_hash:str; optimization_config_hash:str; evaluation_hash:str; reproducibility_hash:str; model_files:list[ReleaseFile]=Field(default_factory=list); parameter_count:int; architecture_hash:str; weight_manifest_hash:str; release_content_hash:str=''
