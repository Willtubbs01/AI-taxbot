from __future__ import annotations

from collections import defaultdict
from enum import Enum

from taxmoe.alignment.models import AlignmentResult
from .feature_priority import priority_key
from .hashing import HashedNGramOccurrence
from .token_features import NGramAlignmentResult, NGramTokenAlignmentStatistics, TokenNGramFeatures


class NGramAlignmentStrategy(str, Enum):
    RIGHT_EDGE_CAUSAL = "right_edge_causal"


class TokenNGramAligner:
    def __init__(self, max_features_per_token: int = 24):
        self.max_features_per_token = max_features_per_token

    def align(self, token_alignment: AlignmentResult, ngrams: list[HashedNGramOccurrence]) -> NGramAlignmentResult:
        by_segment: dict[str, list] = defaultdict(list)
        for t in token_alignment.token_alignments:
            if t.segment_id is not None and t.segment_span is not None:
                by_segment[t.segment_id].append(t)
        for values in by_segment.values():
            values.sort(key=lambda t: t.token_index)

        candidates: dict[int, list[HashedNGramOccurrence]] = defaultdict(list)
        for g in ngrams:
            if g.end_char <= g.start_char:
                raise ValueError("NGRAM-ALIGN-EMPTY-SPAN")
            tokens = by_segment.get(g.segment_id)
            if not tokens:
                raise ValueError(f"NGRAM-ALIGN-UNKNOWN-SEGMENT:{g.segment_id}")
            right_char = g.end_char - 1
            target = None
            for t in tokens:
                s, e = t.segment_span.start, t.segment_span.end
                if s <= right_char < e:
                    target = t
                    break
            if target is None:
                raise ValueError(f"NGRAM-ALIGN-RIGHT-EDGE-GAP:{g.segment_id}:{right_char}")
            if target.is_template_control:
                raise ValueError("NGRAM-ALIGN-TEMPLATE-LEAK")
            candidates[target.token_index].append(g)

        outputs = []
        total_selected = total_dropped = tokens_with = hits = 0
        for t in token_alignment.token_alignments:
            raw = candidates.get(t.token_index, [])
            # Deduplicate IDs while preserving the best-priority candidate for that ID.
            best = {}
            for c in raw:
                prev = best.get(c.feature_id)
                if prev is None or priority_key(c) < priority_key(prev):
                    best[c.feature_id] = c
            ranked = sorted(best.values(), key=priority_key)
            selected = ranked[:self.max_features_per_token]
            dropped = max(0, len(ranked) - len(selected))
            ids = [x.feature_id for x in selected]
            if ids:
                tokens_with += 1
            if dropped:
                hits += 1
            total_selected += len(ids)
            total_dropped += dropped
            outputs.append(TokenNGramFeatures(
                token_index=t.token_index,
                feature_ids=ids,
                candidate_count=len(ranked),
                dropped_feature_count=dropped,
            ))
        return NGramAlignmentResult(
            record_id=token_alignment.record_id,
            strategy=NGramAlignmentStrategy.RIGHT_EDGE_CAUSAL.value,
            alignment_version="1",
            max_features_per_token=self.max_features_per_token,
            tokens=outputs,
            statistics=NGramTokenAlignmentStatistics(
                total_tokens=len(outputs),
                tokens_with_features=tokens_with,
                total_selected_features=total_selected,
                total_dropped_features=total_dropped,
                tokens_hitting_cap=hits,
            ),
        )


def fixed_width_features(result: NGramAlignmentResult) -> list[list[int]]:
    width = result.max_features_per_token
    rows = []
    for t in result.tokens:
        row = t.feature_ids[:width] + [0] * (width - len(t.feature_ids))
        rows.append(row)
    return rows
