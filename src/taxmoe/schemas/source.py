from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import TaxMoEModel
from .identifiers import SourceId


class SourceLocator(TaxMoEModel):
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_id: str | None = None


class SourceRecord(TaxMoEModel):
    source_id: SourceId
    title: str
    jurisdiction: str
    authority_class: str
    tax_year: int | None = None
    source_type: str
    url: str | None = None
    raw_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime
    revision: str | None = None
    status: Literal["active", "superseded", "quarantined"] = "active"
