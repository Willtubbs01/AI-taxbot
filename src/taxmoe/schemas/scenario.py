from __future__ import annotations

from pydantic import Field

from .common import TaxMoEModel
from .enums import AnswerabilityStatus, QualityLevel, ReviewLevel, ScenarioOrigin
from .....taxmoe.schemas.fact import TaxFact
from .identifiers import EventId, FamilyId, ScenarioId


class TaxEvent(TaxMoEModel):
    event_id: EventId
    event_type: str
    fact_ids: list[str]
    concept_ids: list[str] = Field(default_factory=list)


class ScenarioDocument(TaxMoEModel):
    document_instance_id: str
    form_id: str
    form_version_id: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None


class ScenarioContextItem(TaxMoEModel):
    context_id: str
    role: str
    content_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    truth: str = "unknown"
    trusted: bool = False


class ScenarioInput(TaxMoEModel):
    scenario_id: ScenarioId
    jurisdiction: str
    tax_year: int
    facts: list[TaxFact]
    events: list[TaxEvent] = Field(default_factory=list)
    documents: list[ScenarioDocument] = Field(default_factory=list)
    context_items: list[ScenarioContextItem] = Field(default_factory=list)


class TaskAnalysis(TaxMoEModel):
    task_id: str
    status: AnswerabilityStatus
    active_concept_ids: list[str] = Field(default_factory=list)
    missing_fact_ids: list[str] = Field(default_factory=list)
    conflicting_fact_ids: list[str] = Field(default_factory=list)
    required_rule_lookups: list[str] = Field(default_factory=list)
    required_calculations: list[str] = Field(default_factory=list)
    candidate_forms: list[str] = Field(default_factory=list)
    required_forms: list[str] = Field(default_factory=list)
    rule_trace: list[str] = Field(default_factory=list)


class ScenarioAnalysis(TaxMoEModel):
    overall_status: AnswerabilityStatus
    task_analyses: list[TaskAnalysis]


class ScenarioGenerationLineage(TaxMoEModel):
    template_id: str
    template_revision: int
    seed: int
    generator_version: str
    generation_index: int | None = None
    rule_store_version: str
    taxonomy_version: str
    form_registry_version: str
    value_store_version: str


class MutationLineage(TaxMoEModel):
    parent_scenario_id: ScenarioId
    mutation_id: str
    mutation_type: str
    mutation_version: str
    seed: int
    target_fact_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, object] = Field(default_factory=dict)


class TaxScenario(TaxMoEModel):
    scenario_id: ScenarioId
    family_id: FamilyId
    origin: ScenarioOrigin
    input: ScenarioInput
    analysis: ScenarioAnalysis
    semantic_fingerprint: str
    structural_fingerprint: str
    content_hash_value: str
    generation: ScenarioGenerationLineage | None = None
    mutation: MutationLineage | None = None
    quality: QualityLevel = QualityLevel.Q2
    review: ReviewLevel = ReviewLevel.R0
