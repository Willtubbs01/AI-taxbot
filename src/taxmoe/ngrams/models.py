from __future__ import annotations

from enum import Enum
from pydantic import Field, model_validator
from taxmoe.modeling.base import Stage4Model


class NGramKind(str, Enum):
    WORD = "word"
    CHAR = "char"


class LexicalTokenClass(str, Enum):
    WORD = "word"
    FORM_LIKE = "form_like"
    TAX_YEAR = "tax_year"
    MONEY = "money"
    PERCENT = "percent"
    NUMBER = "number"
    DATE = "date"
    SYMBOL = "symbol"


class TextSegment(Stage4Model):
    segment_id: str
    text: str
    role: str | None = None


class LexicalToken(Stage4Model):
    surface: str
    normalized: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_class: LexicalTokenClass
    segment_id: str
    derived_alias: bool = False

    @model_validator(mode="after")
    def valid(self):
        if self.end_char <= self.start_char:
            raise ValueError("lexical token span must be non-empty")
        return self


class NGramOccurrence(Stage4Model):
    kind: NGramKind
    order: int = Field(gt=0)
    normalized_key: str
    surface_text: str
    segment_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    derived_alias: bool = False

    @model_validator(mode="after")
    def valid(self):
        if self.end_char <= self.start_char:
            raise ValueError("n-gram span must be non-empty")
        if not self.normalized_key:
            raise ValueError("normalized_key cannot be empty")
        return self


class NGramExtractionStatistics(Stage4Model):
    lexical_tokens: int = 0
    word_occurrences: int = 0
    char_occurrences: int = 0


class NGramExtractionResult(Stage4Model):
    segment_id: str
    text_hash: str
    normalization_version: str
    lexical_tokens: list[LexicalToken]
    occurrences: list[NGramOccurrence]
    statistics: NGramExtractionStatistics
