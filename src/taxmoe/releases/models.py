from enum import StrEnum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel


class DatasetReleaseStatus(StrEnum):
    CANDIDATE = "candidate"
    FROZEN = "frozen"
    WITHDRAWN = "withdrawn"


class ReleaseFile(TaxMoEModel):
    path: str
    bytes: int
    sha256: str
    records: int | None = None


class DatasetReleaseManifest(TaxMoEModel):
    release_id: str
    name: str
    version: str
    status: DatasetReleaseStatus
    source_build_id: str
    source_build_spec_hash: str
    files: list[ReleaseFile] = Field(default_factory=list)
    release_content_hash: str
