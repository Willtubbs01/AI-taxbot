from __future__ import annotations
import hashlib
from taxmoe.modeling.inputs import ChatMessage
from .exceptions import AlignmentError
from .models import CharacterSpan, RenderedSegment

_START = "\ue000TAXMOE_MSG_{:04d}_START\ue001"
_END = "\ue000TAXMOE_MSG_{:04d}_END\ue001"


def instrument_messages(messages: list[ChatMessage]) -> list[dict[str,str]]:
    out=[]
    for i,m in enumerate(messages):
        start, end = _START.format(i), _END.format(i)
        if start in m.content or end in m.content:
            raise AlignmentError("ALIGN-SENTINEL-COLLISION")
        out.append({"role":m.role,"content":start + m.content + end})
    return out


def derive_segments(messages: list[ChatMessage], canonical_render: str, instrumented_render: str):
    cleaned = instrumented_render
    segments=[]
    # Find each instrumented content and simultaneously compute cleaned offsets.
    for i,m in enumerate(messages):
        start_tag, end_tag = _START.format(i), _END.format(i)
        a=instrumented_render.find(start_tag)
        b=instrumented_render.find(end_tag)
        if a < 0 or b < 0 or b < a:
            raise AlignmentError(f"ALIGN-MESSAGE-NOT-FOUND:{i}")
    # Remove sentinels to ensure the template is compositional.
    for i in range(len(messages)):
        cleaned = cleaned.replace(_START.format(i), "").replace(_END.format(i), "")
    if cleaned != canonical_render:
        raise AlignmentError("ALIGN-TEMPLATE-NONCOMPOSITIONAL")
    # Exact mapping can now use deterministic occurrence positions in canonical render by walking in message order.
    cursor=0
    for i,m in enumerate(messages):
        pos=canonical_render.find(m.content, cursor)
        if pos < 0:
            raise AlignmentError(f"ALIGN-MESSAGE-NOT-FOUND:{i}")
        end=pos+len(m.content)
        segments.append(RenderedSegment(
            kind="message_content", segment_id=f"msg:{i}",
            rendered_span=CharacterSpan(start=pos,end=end), role=m.role,
            source_span=CharacterSpan(start=0,end=len(m.content))))
        cursor=end
    return segments
