from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .base import Stage4Model


class LossMode(str, Enum):
    FULL_CAUSAL = "full_causal"
    ASSISTANT_ONLY = "assistant_only"


class ChatMessage(Stage4Model):
    role: Literal["system", "user", "assistant"]
    content: str


class TokenSpan(Stage4Model):
    token_index: int = Field(ge=0)
    token_id: int = Field(ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    special: bool = False

    @model_validator(mode="after")
    def validate_span(self):
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("start_char and end_char must both be set or both be None")
        if self.start_char is not None and self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class TokenizedText(Stage4Model):
    input_ids: list[int]
    attention_mask: list[int]
    spans: list[TokenSpan] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    tokenizer_id: str
    tokenizer_revision: str | None = None
    text_sha256: str
    rendered_text_sha256: str | None = None

    @model_validator(mode="after")
    def validate_lengths(self):
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask lengths differ")
        if self.spans and len(self.spans) != len(self.input_ids):
            raise ValueError("spans length differs from input_ids")
        if self.token_count != len(self.input_ids):
            raise ValueError("token_count differs from input_ids length")
        return self


class TokenizationLineage(Stage4Model):
    dataset_release_id: str
    dataset_release_hash: str
    input_record_id: str
    input_record_hash: str
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_manifest_hash: str
    tokenization_config_hash: str
    chat_template_hash: str | None = None


class TokenizedExample(Stage4Model):
    schema_version: str = "0.1"
    record_id: str
    source_kind: Literal["cpt", "sft", "eval"]
    split: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_count: int = Field(ge=0)
    supervised_token_count: int = Field(ge=0)
    lineage: TokenizationLineage | None = None

    @model_validator(mode="after")
    def validate_lengths(self):
        n = len(self.input_ids)
        if len(self.attention_mask) != n or len(self.labels) != n:
            raise ValueError("input_ids, attention_mask and labels must have equal lengths")
        if self.token_count != n:
            raise ValueError("token_count differs from input length")
        if any(x != -100 and x < 0 for x in self.labels):
            raise ValueError("labels may only use -100 as a negative sentinel")
        actual = sum(x != -100 for x in self.labels)
        if self.supervised_token_count != actual:
            raise ValueError("supervised_token_count is inconsistent with labels")
        return self
