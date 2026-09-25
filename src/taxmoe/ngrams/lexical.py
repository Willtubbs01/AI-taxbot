from __future__ import annotations

import re

from .config import NGramNormalizationConfig
from .models import LexicalToken
from .normalization import normalize_atom

# Purposefully small deterministic lexer. Punctuation creates boundaries.
TOKEN_RE = re.compile(
    r"(?:[+-]?\$\s*\d[\d,]*(?:\.\d+)?)"       # money
    r"|(?:\d+(?:\.\d+)?%)"                   # percent
    r"|(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"    # date-like
    r"|(?:[A-Za-z0-9]+(?:[-_/'][A-Za-z0-9]+)*)" # word/form/structured key
    r"|(?:\d+(?:[,.]\d+)*)"                   # number
)

SENTENCE_BREAK_RE = re.compile(r"[.!?;:\n\r]+")


def lexical_tokens(text: str, segment_id: str, config: NGramNormalizationConfig) -> list[LexicalToken]:
    out: list[LexicalToken] = []
    for match in TOKEN_RE.finditer(text):
        surface = match.group(0)
        norm = normalize_atom(surface, config)
        out.append(LexicalToken(
            surface=surface,
            normalized=norm.key,
            start_char=match.start(),
            end_char=match.end(),
            token_class=norm.token_class,
            segment_id=segment_id,
        ))
        for alias in norm.aliases:
            out.append(LexicalToken(
                surface=surface,
                normalized=alias,
                start_char=match.start(),
                end_char=match.end(),
                token_class=norm.token_class,
                segment_id=segment_id,
                derived_alias=True,
            ))
    return out


def sentence_break_between(text: str, left_end: int, right_start: int) -> bool:
    return bool(SENTENCE_BREAK_RE.search(text[left_end:right_start]))
