from __future__ import annotations
from pydantic import Field
from .common import TaxMoEModel, Jurisdiction
from .enums import AnswerabilityStatus, QualityLevel, ReviewLevel, ScenarioOrigin
from .fact import TaxFact

class TaxEvent(TaxMoEModel):
    event_id: str
    event_type: str
    fact_ids: list[str] = Field(default_factory=list)
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
    payload: dict = Field(default_factory=dict)
    truth: str = "unknown"
    trusted: bool = False

class ScenarioInput(TaxMoEModel):
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    tax_year: int
    facts: list[TaxFact] = Field(default_factory=list)
    events: list[TaxEvent] = Field(default_factory=list)
    documents: list[ScenarioDocument] = Field(default_factory=list)
    context_items: list[ScenarioContextItem] = Field(default_factory=list)

class TaskAnalysis(TaxMoEModel):
    task_id: str
    status: AnswerabilityStatus
    topic_ids: list[str] = Field(default_factory=list)
    missing_concept_ids: list[str] = Field(default_factory=list)
    conflicting_concept_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    candidate_forms: list[str] = Field(default_factory=list)
    required_forms: list[str] = Field(default_factory=list)
    required_rule_lookups: list[str] = Field(default_factory=list)
    required_calculations: list[str] = Field(default_factory=list)
    activated_rule_ids: list[str] = Field(default_factory=list)

class ScenarioAnalysis(TaxMoEModel):
    overall_status: AnswerabilityStatus
    task_analyses: list[TaskAnalysis] = Field(default_factory=list)

class ScenarioGenerationLineage(TaxMoEModel):
    template_id: str
    template_revision: int = 1
    seed: int
    generator_version: str = "0.1"
    generation_index: int = 0
    accepted_attempt: int = 1
    taxonomy_version: str = "0.1"
    form_registry_version: str = "0.1"
    rule_store_version: str = "0.1"
    value_store_version: str = "0.1"
    calculation_registry_version: str = "0.1"
    build_id: str = "BUILD-DEV"

class MutationLineage(TaxMoEModel):
    parent_scenario_id: str
    mutation_id: str
    mutation_type: str
    mutation_version: str = "0.1"
    seed: int
    target_fact_ids: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)

class TaxScenario(TaxMoEModel):
    scenario_id: str
    family_id: str
    origin: ScenarioOrigin
    input: ScenarioInput
    analysis: ScenarioAnalysis
    semantic_fingerprint: str
    structural_fingerprint: str
    content_hash: str
    generation: ScenarioGenerationLineage | None = None
    mutation: MutationLineage | None = None
    quality: QualityLevel = QualityLevel.Q2
    review: ReviewLevel = ReviewLevel.R0
