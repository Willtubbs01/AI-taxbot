from __future__ import annotations
from typing import Literal
from pydantic import Field, model_validator
from taxmoe.modeling.base import Stage4Model

class CachedTrainingExample(Stage4Model):
    schema_version:str='0.1'
    record_id:str
    split:str
    source_kind:str
    input_ids:list[int]
    labels:list[int]
    ngram_feature_ids:list[list[int]]|None=None
    token_count:int=Field(ge=0)
    supervised_token_count:int=Field(ge=0)
    content_hash:str

    @model_validator(mode='after')
    def validate_all(self):
        n=len(self.input_ids)
        if len(self.labels)!=n or self.token_count!=n: raise ValueError('CACHE-TOKEN-LENGTH-MISMATCH')
        if self.ngram_feature_ids is not None:
            if len(self.ngram_feature_ids)!=n: raise ValueError('CACHE-NGRAM-LENGTH-MISMATCH')
            if any(len(row)!=24 for row in self.ngram_feature_ids): raise ValueError('CACHE-NGRAM-WIDTH-MISMATCH')
        return self

class CachedEvaluationExample(Stage4Model):
    schema_version:str='0.1'
    record_id:str
    split:str
    prompt_input_ids:list[int]
    prompt_ngram_feature_ids:list[list[int]]|None=None
    reference_input_ids:list[int]
    reference_labels:list[int]
    reference_ngram_feature_ids:list[list[int]]|None=None

class CacheShard(Stage4Model):
    split:str
    path:str
    records:int
    tokens:int
    bytes:int
    sha256:str
    first_record_id:str|None=None
    last_record_id:str|None=None

class TrainingCacheManifest(Stage4Model):
    version:str='1'
    cache_id:str
    cache_kind:Literal['token_only','token_ngram']
    dataset_release_id:str
    dataset_release_hash:str
    tokenizer_manifest_hash:str
    tokenization_manifest_hash:str
    ngram_normalization_hash:str|None=None
    ngram_hash_manifest_hash:str|None=None
    token_alignment_manifest_hash:str|None=None
    ngram_alignment_manifest_hash:str|None=None
    files:list[CacheShard]=Field(default_factory=list)
    total_records:int=0
    total_tokens:int=0
    content_hash:str=''
