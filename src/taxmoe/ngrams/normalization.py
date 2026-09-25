from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .config import NGramNormalizationConfig
from .models import LexicalTokenClass

DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-", "‒": "-"})
MONEY_RE = re.compile(r"^[+-]?\$\s*\d[\d,]*(?:\.\d+)?$")
PERCENT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?%$")
DATE_RE = re.compile(r"^(?:\d{1,2}[/-]){2}\d{2,4}$")
NUMBER_RE = re.compile(r"^[+-]?\d+(?:[,.]\d+)*$")
FORMISH_RE = re.compile(r"^(?:w-?\d+[a-z]?|10\d\d(?:-[a-z]+)?|109[89](?:-[a-z]+)?|\d{4}-[a-z]+)$", re.I)
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True)
class NormalizedAtom:
    key: str
    token_class: LexicalTokenClass
    aliases: tuple[str, ...] = ()


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    return unicodedata.normalize(form, text).translate(DASHES)


def normalize_atom(surface: str, config: NGramNormalizationConfig) -> NormalizedAtom:
    s = normalize_unicode(surface, config.unicode_form)
    if config.lowercase:
        s = s.lower()
    if MONEY_RE.match(s) and config.normalize_money:
        return NormalizedAtom("<money>", LexicalTokenClass.MONEY)
    if PERCENT_RE.match(s) and config.normalize_percent:
        return NormalizedAtom("<percent>", LexicalTokenClass.PERCENT)
    if DATE_RE.match(s):
        return NormalizedAtom("<date>", LexicalTokenClass.DATE)
    if config.preserve_tax_years and YEAR_RE.match(s):
        return NormalizedAtom(s, LexicalTokenClass.TAX_YEAR)
    if FORMISH_RE.match(s):
        aliases = []
        compact = re.sub(r"[-_\s]", "", s)
        if compact != s:
            aliases.append(compact)
        return NormalizedAtom(s, LexicalTokenClass.FORM_LIKE, tuple(aliases[:config.max_aliases_per_atom]))
    if NUMBER_RE.match(s) and config.normalize_general_numbers:
        return NormalizedAtom("<number>", LexicalTokenClass.NUMBER)

    aliases: list[str] = []
    if "_" in s:
        aliases.extend(x for x in s.split("_") if x)
    elif "/" in s:
        aliases.extend(x for x in s.split("/") if x)
    elif "-" in s and any(c.isalpha() for c in s):
        aliases.extend(x for x in s.split("-") if x)
    return NormalizedAtom(s, LexicalTokenClass.WORD, tuple(dict.fromkeys(aliases))[:config.max_aliases_per_atom])
