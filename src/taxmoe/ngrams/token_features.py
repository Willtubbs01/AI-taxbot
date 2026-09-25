from __future__ import annotations
from pydantic import Field
from taxmoe.modeling.base import Stage4Model


class TokenNGramFeatures(Stage4Model):
    token_index: int = Field(ge=0)
    feature_ids: list[int]
    candidate_count: int = Field(ge=0)
    dropped_feature_count: int = Field(ge=0)


class NGramTokenAlignmentStatistics(Stage4Model):
    total_tokens: int = 0
    tokens_with_features: int = 0
    total_selected_features: int = 0
    total_dropped_features: int = 0
    tokens_hitting_cap: int = 0


class NGramAlignmentResult(Stage4Model):
    record_id: str
    strategy: str
    alignment_version: str
    max_features_per_token: int
    tokens: list[TokenNGramFeatures]
    statistics: NGramTokenAlignmentStatistics
