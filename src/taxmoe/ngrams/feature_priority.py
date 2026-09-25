from __future__ import annotations
from .hashing import HashedNGramOccurrence

_WORD_PRIORITY = {4: 0, 3: 1, 2: 2, 1: 3}
_CHAR_PRIORITY = {5: 4, 4: 5, 3: 6}


def priority_key(x: HashedNGramOccurrence):
    if x.namespace.value.startswith("word_"):
        group = _WORD_PRIORITY[x.order]
    else:
        group = _CHAR_PRIORITY[x.order]
    return (group, 1 if x.derived_alias else 0, x.feature_id)
