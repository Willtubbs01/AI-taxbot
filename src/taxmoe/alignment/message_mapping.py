from __future__ import annotations
from .exceptions import AlignmentError
from .models import CharacterSpan, RenderedSegment


def map_rendered_span_to_segment(span: CharacterSpan, segments: list[RenderedSegment]):
    hits=[]
    for seg in segments:
        s=seg.rendered_span
        if span.start < s.end and s.start < span.end:
            hits.append(seg)
    if len(hits) > 1:
        raise AlignmentError("ALIGN-CROSS-SEGMENT-TOKEN")
    if not hits:
        return None, None
    seg=hits[0]
    # Token must be fully inside the semantic segment; partial control/content token is rejected.
    if span.start < seg.rendered_span.start or span.end > seg.rendered_span.end:
        raise AlignmentError("ALIGN-CROSS-SEGMENT-TOKEN")
    local=CharacterSpan(start=span.start-seg.rendered_span.start, end=span.end-seg.rendered_span.start)
    return seg, local
