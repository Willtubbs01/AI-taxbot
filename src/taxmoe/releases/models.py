from enum import Enum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
from taxmoe.schemas.manifest import FileRecord

class DatasetReleaseStatus(str, Enum):
    CANDIDATE = "candidate"
    FROZEN = "frozen"
    WITHDRAWN = "withdrawn"

class DatasetReleaseManifest(TaxMoEModel):
    release_id: str
    dataset_name: str
    dataset_version: str
    status: DatasetReleaseStatus
    source_build_id: str
    source_build_spec_hash: str
    release_content_hash: str = ""
    files: list[FileRecord] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
