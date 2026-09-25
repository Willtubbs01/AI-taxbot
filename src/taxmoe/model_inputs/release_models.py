from __future__ import annotations
from enum import Enum
from pydantic import Field
from taxmoe.modeling.base import Stage4Model

class InputReleaseStatus(str,Enum):
    CANDIDATE='candidate';FROZEN='frozen';WITHDRAWN='withdrawn'

class InputReproducibilityResult(Stage4Model):
    passed:bool
    candidate_build_id:str
    reproduction_build_id:str
    tokenization_match:bool
    ngram_hash_match:bool
    alignment_match:bool
    cache_semantic_match:bool
    cache_file_hash_match:bool
    mismatches:list[str]=Field(default_factory=list)

class ModelInputReleaseManifest(Stage4Model):
    schema_version:str='0.1'
    release_id:str
    name:str
    version:str
    status:InputReleaseStatus
    dataset_release_id:str
    dataset_release_hash:str
    tokenizer_manifest_hash:str
    tokenization_manifest_hash:str
    ngram_normalization_manifest_hash:str
    ngram_hash_manifest_hash:str
    token_alignment_manifest_hash:str
    ngram_alignment_manifest_hash:str
    token_cache_hash:str
    ngram_cache_hash:str
    encoder_architecture_hash:str
    router_interface_hash:str
    validation_report_hash:str
    benchmark_report_hash:str
    release_content_hash:str=''
