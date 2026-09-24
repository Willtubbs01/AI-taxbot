from enum import Enum
from datetime import datetime, timezone
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel

class BuildStage(str, Enum):
    PREPARE = "prepare"
    GENERATE = "generate"
    MUTATE = "mutate"
    SPLIT = "split"
    VALIDATE = "validate"
    EXPORT = "export"
    REPORT = "report"

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"

class DatasetBuildRecord(TaxMoEModel):
    build_id: str
    build_name: str
    spec_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    current_stage: str | None = None
    passed_validation: bool = False
    export_id: str | None = None
