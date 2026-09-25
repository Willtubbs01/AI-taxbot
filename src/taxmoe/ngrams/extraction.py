from __future__ import annotations

import hashlib

from .config import NGramNormalizationConfig
from .lexical import lexical_tokens, sentence_break_between
from .models import NGramExtractionResult, NGramExtractionStatistics, NGramKind, NGramOccurrence, TextSegment


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NGramExtractor:
    def __init__(self, config: NGramNormalizationConfig = NGramNormalizationConfig()):
        self.config = config

    def _base_tokens(self, text: str, segment_id: str):
        # Word phrase generation uses canonical (non-alias) tokens only.
        all_tokens = lexical_tokens(text, segment_id, self.config)
        base = [t for t in all_tokens if not t.derived_alias]
        aliases = [t for t in all_tokens if t.derived_alias]
        return base, aliases

    def extract_segment(self, segment: TextSegment) -> NGramExtractionResult:
        text = segment.text
        base, aliases = self._base_tokens(text, segment.segment_id)
        occurrences: list[NGramOccurrence] = []

        # Canonical unigrams plus lexical aliases.
        for tok in base + aliases:
            occurrences.append(NGramOccurrence(
                kind=NGramKind.WORD,
                order=1,
                normalized_key=tok.normalized,
                surface_text=tok.surface,
                segment_id=segment.segment_id,
                start_char=tok.start_char,
                end_char=tok.end_char,
                derived_alias=tok.derived_alias,
            ))

        if self.config.word.enabled:
            max_order = self.config.word.max_order
            for i in range(len(base)):
                for order in range(max(2, self.config.word.min_order), max_order + 1):
                    j = i + order
                    if j > len(base):
                        break
                    group = base[i:j]
                    if any(sentence_break_between(text, group[k].end_char, group[k+1].start_char) for k in range(len(group)-1)):
                        break
                    key = " ".join(t.normalized for t in group)
                    occurrences.append(NGramOccurrence(
                        kind=NGramKind.WORD,
                        order=order,
                        normalized_key=key,
                        surface_text=text[group[0].start_char:group[-1].end_char],
                        segment_id=segment.segment_id,
                        start_char=group[0].start_char,
                        end_char=group[-1].end_char,
                    ))

        if self.config.char.enabled:
            for tok in base:
                key = tok.normalized
                eligible = (
                    len(key) >= self.config.char.min_token_length
                    or (self.config.char.eligible_if_contains_digit and any(c.isdigit() for c in key))
                    or (self.config.char.eligible_if_contains_hyphen and "-" in key)
                    or (self.config.char.eligible_if_contains_slash and "/" in key)
                )
                if not eligible or len(key) > self.config.char.max_atom_chars or key.startswith("<"):
                    continue
                # Only emit exact-span char grams when normalization is length-preserving over the surface.
                # Normalized/alias lexical identity still exists as WORD_1 if normalization changed length.
                surface_norm = tok.surface.lower().translate(str.maketrans({"–":"-","—":"-","−":"-","‑":"-","‒":"-"}))
                if len(surface_norm) != len(key):
                    continue
                for order in self.config.char.orders:
                    if len(key) < order:
                        continue
                    for pos in range(0, len(key) - order + 1):
                        gram = key[pos:pos+order]
                        occurrences.append(NGramOccurrence(
                            kind=NGramKind.CHAR,
                            order=order,
                            normalized_key=gram,
                            surface_text=tok.surface[pos:pos+order],
                            segment_id=segment.segment_id,
                            start_char=tok.start_char + pos,
                            end_char=tok.start_char + pos + order,
                        ))

        occurrences.sort(key=lambda x: (x.start_char, x.end_char, x.kind.value, x.order, x.normalized_key, x.derived_alias))
        stats = NGramExtractionStatistics(
            lexical_tokens=len(base) + len(aliases),
            word_occurrences=sum(o.kind == NGramKind.WORD for o in occurrences),
            char_occurrences=sum(o.kind == NGramKind.CHAR for o in occurrences),
        )
        return NGramExtractionResult(
            segment_id=segment.segment_id,
            text_hash=_hash_text(text),
            normalization_version=self.config.version,
            lexical_tokens=base + aliases,
            occurrences=occurrences,
            statistics=stats,
        )

    def extract_text(self, text: str, segment_id: str = "text:0") -> NGramExtractionResult:
        return self.extract_segment(TextSegment(segment_id=segment_id, text=text))
