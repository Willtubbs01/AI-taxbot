from __future__ import annotations

from pydantic import Field

from .common import TaxMoEModel
from .enums import DatasetSplit, QualityLevel


class TrainingExample(TaxMoEModel):
    example_id: str
    scenario_id: str
    family_id: str
    cluster_id: str
    task_id: str
    dataset_family: str
    split: DatasetSplit
    messages: list[dict[str, str]]
    metadata: dict[str, object] = Field(default_factory=dict)
    quality: QualityLevel = QualityLevel.Q2


class EvaluationExample(TaxMoEModel):
    example_id: str
    scenario_id: str
    cluster_id: str
    split: DatasetSplit
    messages: list[dict[str, str]]
    gold: dict[str, object]
    metadata: dict[str, object] = Field(default_factory=dict)
