from __future__ import annotations

from pydantic import Field
from taxmoe.modeling.base import Stage4Model


class WordNGramConfig(Stage4Model):
    enabled: bool = True
    min_order: int = 1
    max_order: int = 4


class CharNGramConfig(Stage4Model):
    enabled: bool = True
    orders: list[int] = Field(default_factory=lambda: [3, 4, 5])
    min_token_length: int = 7
    eligible_if_contains_digit: bool = True
    eligible_if_contains_hyphen: bool = True
    eligible_if_contains_slash: bool = True
    max_atom_chars: int = 64


class NGramNormalizationConfig(Stage4Model):
    version: str = "1"
    unicode_form: str = "NFKC"
    lowercase: bool = True
    preserve_tax_years: bool = True
    normalize_money: bool = True
    normalize_percent: bool = True
    normalize_general_numbers: bool = True
    max_aliases_per_atom: int = 4
    word: WordNGramConfig = WordNGramConfig()
    char: CharNGramConfig = CharNGramConfig()
