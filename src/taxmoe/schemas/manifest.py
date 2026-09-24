from pydantic import Field
from .common import TaxMoEModel

class FileRecord(TaxMoEModel):
    path: str
    bytes: int
    sha256: str
    records: int | None = None

class BuildManifest(TaxMoEModel):
    build_id: str
    spec_hash: str
    versions: dict[str, str] = Field(default_factory=dict)
    files: list[FileRecord] = Field(default_factory=list)

class SplitAssignment(TaxMoEModel):
    cluster_id: str
    split: str
    split_version: str
    split_seed: int
    reservation_reason: str | None = None

class SplitManifest(TaxMoEModel):
    split_version: str
    clustering_version: str
    dedup_version: str
    seed: int
    scenario_to_cluster: dict[str, str] = Field(default_factory=dict)
    assignments: list[SplitAssignment] = Field(default_factory=list)
