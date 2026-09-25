import pytest
from taxmoe.alignment.models import CharacterSpan

def test_half_open_span():
    s=CharacterSpan(start=0,end=5)
    assert 'basis'[s.start:s.end]=='basis'

def test_bad_span_rejected():
    with pytest.raises(Exception):CharacterSpan(start=5,end=4)
