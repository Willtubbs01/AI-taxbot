from __future__ import annotations
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel


class ExtractedDocument(TaxMoEModel):
    source_id: str
    text: str
    pages: list[str] = Field(default_factory=list)
    extractor: str = "plain"


class SourceChunk(TaxMoEModel):
    chunk_id: str
    source_id: str
    text: str
    ordinal: int
    metadata: dict[str, object] = Field(default_factory=dict)
