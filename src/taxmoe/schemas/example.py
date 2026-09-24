from pydantic import Field
from .common import TaxMoEModel
from .enums import DatasetSplit, QualityLevel

class ExampleLineage(TaxMoEModel):
    scenario_id: str
    scenario_content_hash: str
    task_id: str
    renderer_id: str
    renderer_version: str
    exporter_version: str
    protocol_version: str = "0.1"
    split_version: str = "v1"

class TrainingExample(TaxMoEModel):
    example_id: str
    scenario_id: str
    family_id: str
    cluster_id: str
    task_id: str
    dataset_family: str
    split: DatasetSplit
    messages: list[dict[str, str]]
    metadata: dict = Field(default_factory=dict)
    lineage: ExampleLineage
    quality: QualityLevel = QualityLevel.Q2
