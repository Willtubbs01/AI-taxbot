from enum import StrEnum
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel


class BuildStage(StrEnum):
    PREPARE = "prepare"
    GENERATE = "generate"
    MUTATE = "mutate"
    SPLIT = "split"
    VALIDATE = "validate"
    EXPORT = "export"
    REPORT = "report"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class DatasetBuildRecord(TaxMoEModel):
    build_id: str
    name: str
    spec_hash: str
    stages: dict[BuildStage, StageStatus] = Field(default_factory=dict)
