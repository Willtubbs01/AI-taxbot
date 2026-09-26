from __future__ import annotations
from enum import Enum
from typing import Any,Literal
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class EvalCaseError(str,Enum):
    GENERATION_FAILED='generation_failed';MAX_NEW_TOKENS='max_new_tokens';JSON_PARSE_FAILED='json_parse_failed';SCHEMA_VALIDATION_FAILED='schema_validation_failed';METRIC_FAILED='metric_failed'
class EvaluationCaseResult(TaxMoEModel):
    record_id:str; suite:str; task:str; generated_text:str|None=None; prompt_token_count:int=0; generated_token_count:int=0; generation_truncated:bool=False; metrics:dict[str,Any]=Field(default_factory=dict); error_code:str|None=None
class TeacherForcedResult(TaxMoEModel):
    record_id:str; supervised_token_count:int; loss_sum:float; mean_loss:float; finite:bool=True
class EvaluationManifest(TaxMoEModel):
    eval_id:str; model_fingerprint:str; input_release_hash:str; suite_manifest_hash:str; evaluator_version:str='1'; generation_config_hash:str; metric_config_hash:str; environment:dict[str,Any]=Field(default_factory=dict)
class MetricDefinition(TaxMoEModel):
    name:str; direction:Literal['higher','lower']; unit:str='fraction'
class TaxDenseCandidateScorecard(TaxMoEModel):
    candidate_id:str; checkpoint_hash:str; tax_validation_loss:float; tax_metrics:dict[str,float]=Field(default_factory=dict); general_metrics:dict[str,float]=Field(default_factory=dict); temporal_metrics:dict[str,float]=Field(default_factory=dict); adversarial_metrics:dict[str,float]=Field(default_factory=dict); tax_deltas:dict[str,float]=Field(default_factory=dict); general_deltas:dict[str,float]=Field(default_factory=dict); critical_regressions:list[str]=Field(default_factory=list); material_regressions:list[str]=Field(default_factory=list); eligible_for_freeze:bool=False
