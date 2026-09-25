from __future__ import annotations
from .exceptions import AlignmentError
from .models import CharacterSpan


def validate_offsets(offsets, text_length: int):
    prev=0
    for i,(a,b) in enumerate(offsets):
        a,b=int(a),int(b)
        if a < 0 or b < a or b > text_length:
            raise AlignmentError(f"ALIGN-OFFSET-OUT-OF-RANGE:{i}:{a}:{b}")
        if b > a and a < prev:
            raise AlignmentError(f"ALIGN-OFFSET-BACKTRACK:{i}:{a}<{prev}")
        if b > a:
            prev=a


def span_from_offset(offset, *, special: bool):
    a,b=map(int,offset)
    if special or b <= a:
        return None
    return CharacterSpan(start=a,end=b)
