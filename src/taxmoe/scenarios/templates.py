from __future__ import annotations
from typing import Any, Literal
from pydantic import Field, model_validator
from taxmoe.schemas.common import TaxMoEModel, Jurisdiction

class ValueGeneratorSpec(TaxMoEModel):
    type: str
    values: list[Any] | None = None
    weights: list[float] | None = None
    value: Any | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | None = None
    reference: str | None = None
    min_days_after: int | None = None
    max_days_after: int | None = None
    value_ref: str | None = None
    boundary_position: Literal["below", "at", "above"] | None = None
    delta: int | float | None = None

class TemplateVariable(TaxMoEModel):
    variable_id: str
    concept_id: str
    value_type: str
    generator: ValueGeneratorSpec
    mutable: bool = True
    mutation_tags: list[str] = Field(default_factory=list)
    family_semantic: bool = False

class TemplateConstraint(TaxMoEModel):
    type: str
    left: str
    right: Any

class TemplateEvent(TaxMoEModel):
    event_id: str
    event_type: str
    variable_refs: list[str]

class TemplateDocument(TaxMoEModel):
    document_id: str
    form_id: str
    variable_refs: list[str] = Field(default_factory=list)

class GenerationPolicy(TaxMoEModel):
    max_attempts: int = 100
    require_complete_base_scenario: bool = True

class ScenarioTemplate(TaxMoEModel):
    template_id: str
    revision: int = 1
    name: str
    template_type: str = "base"
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    supported_tax_years: list[int] = Field(default_factory=list)
    domain_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    variables: list[TemplateVariable]
    constraints: list[TemplateConstraint] = Field(default_factory=list)
    events: list[TemplateEvent] = Field(default_factory=list)
    documents: list[TemplateDocument] = Field(default_factory=list)
    required_rule_ids: list[str] = Field(default_factory=list)
    generation: GenerationPolicy = Field(default_factory=GenerationPolicy)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_variables(self):
        ids = [v.variable_id for v in self.variables]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate template variable IDs")
        return self
