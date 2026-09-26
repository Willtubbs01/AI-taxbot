from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class ExperimentConclusion(str,Enum): SUPPORTS_BASELINE='supports_baseline';SUPPORTS_VARIANT='supports_variant';INCONCLUSIVE='inconclusive';UNSTABLE='unstable'
class AblationExperimentManifest(TaxMoEModel):
    experiment_id:str; parent_spec_hash:str; changed_variable:str; baseline_value:Any; experimental_value:Any; training_run_id:str; evaluation_id:str; result_hash:str=''
class ReproducibilityStudy(TaxMoEModel):
    study_id:str; run_spec_hash:str; seeds:list[int]; run_ids:list[str]; metric_summary:dict=Field(default_factory=dict); stability_result:str=''
