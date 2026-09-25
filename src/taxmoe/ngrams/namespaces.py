from __future__ import annotations

from enum import Enum
from .models import NGramKind


class NGramNamespace(str, Enum):
    WORD_1 = "word_1"
    WORD_2 = "word_2"
    WORD_3 = "word_3"
    WORD_4 = "word_4"
    CHAR_3 = "char_3"
    CHAR_4 = "char_4"
    CHAR_5 = "char_5"


NGRAM_NAMESPACE_ORDER_V1 = (
    NGramNamespace.WORD_1,
    NGramNamespace.WORD_2,
    NGramNamespace.WORD_3,
    NGramNamespace.WORD_4,
    NGramNamespace.CHAR_3,
    NGramNamespace.CHAR_4,
    NGramNamespace.CHAR_5,
)


def namespace_for(kind: NGramKind, order: int) -> NGramNamespace:
    key = f"{kind.value}_{order}"
    try:
        return NGramNamespace(key)
    except ValueError as exc:
        raise ValueError(f"unsupported n-gram namespace: {key}") from exc
