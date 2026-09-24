from datetime import datetime
from pydantic import Field
from .common import TaxMoEModel, Jurisdiction
from .enums import QualityLevel, ReviewLevel

class SourceRecord(TaxMoEModel):
    source_id: str
    title: str
    url: str | None = None
    raw_path: str
    sha256: str
    document_type: str
    authority_class: str
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    tax_year: int | None = None
    retrieved_at: datetime | None = None
    quality: QualityLevel = QualityLevel.Q1
    review: ReviewLevel = ReviewLevel.R0

class SourceManifest(TaxMoEModel):
    manifest_id: str
    version: str
    sources: list[SourceRecord]
