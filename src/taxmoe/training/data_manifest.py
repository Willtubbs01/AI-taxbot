from __future__ import annotations
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class SourceTrainingStats(TaxMoEModel):
    source_id:str; records:int=0; unique_tokens:int=0; sampled_tokens:int=0
class DenseTrainingDataManifest(TaxMoEModel):
    version:str='1'; input_release_id:str; input_release_hash:str; train_cache_hash:str; validation_cache_hash:str; sampling_strategy:str='token_uniform'; sampling_config_hash:str=''; unique_train_records:int=0; unique_train_tokens:int=0; source_statistics:list[SourceTrainingStats]=Field(default_factory=list); replay_enabled:bool=False
