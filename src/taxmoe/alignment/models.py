from __future__ import annotations
from pydantic import Field, model_validator
from taxmoe.modeling.base import Stage4Model

class CharacterSpan(Stage4Model):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    @model_validator(mode="after")
    def valid(self):
        if self.end < self.start:
            raise ValueError("span end must be >= start")
        return self

class RenderedSegment(Stage4Model):
    kind: str
    segment_id: str | None = None
    rendered_span: CharacterSpan
    role: str | None = None
    source_span: CharacterSpan | None = None

class TokenCharacterAlignment(Stage4Model):
    token_index: int = Field(ge=0)
    token_id: int = Field(ge=0)
    rendered_span: CharacterSpan | None = None
    segment_id: str | None = None
    segment_span: CharacterSpan | None = None
    role: str | None = None
    is_special: bool = False
    is_template_control: bool = False

class AlignmentStatistics(Stage4Model):
    total_tokens: int = 0
    content_tokens: int = 0
    template_control_tokens: int = 0
    special_tokens: int = 0
    unmapped_tokens: int = 0
    cross_segment_tokens: int = 0

class AlignmentResult(Stage4Model):
    record_id: str
    alignment_version: str = "1"
    rendered_text_hash: str
    token_alignments: list[TokenCharacterAlignment]
    segments: list[RenderedSegment]
    statistics: AlignmentStatistics
