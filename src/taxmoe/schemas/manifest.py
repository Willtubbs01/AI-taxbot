from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import TaxMoEModel


class ArtifactFile(TaxMoEModel):
    path: str
    sha256: str
    bytes: int
    records: int | None = None


class BuildManifest(TaxMoEModel):
    build_id: str
    created_at: datetime
    spec_hash: str
    versions: dict[str, str]
    files: list[ArtifactFile] = Field(default_factory=list)
