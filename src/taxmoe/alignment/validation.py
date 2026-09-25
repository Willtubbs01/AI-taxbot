from __future__ import annotations
from .models import AlignmentResult


def validate_alignment(result: AlignmentResult):
    expected=list(range(len(result.token_alignments)))
    actual=[x.token_index for x in result.token_alignments]
    if expected != actual:
        raise ValueError('ALIGN-TOKEN-INDEX-SEQUENCE-INVALID')
    for t in result.token_alignments:
        if t.is_template_control and t.segment_id is not None:
            raise ValueError('ALIGN-CONTROL-HAS-SEGMENT')
        if t.segment_span is not None and t.segment_span.end < t.segment_span.start:
            raise ValueError('ALIGN-INVALID-SEGMENT-SPAN')
    return True
