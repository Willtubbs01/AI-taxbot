from datetime import datetime, timezone
from pydantic import Field
from .common import TaxMoEModel

class SourceReference(TaxMoEModel):
    source_id: str
    chunk_id: str | None = None
    page: int | None = None
    locator: str | None = None

class ArtifactLineage(TaxMoEModel):
    parent_ids: list[str] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "taxmoe"
