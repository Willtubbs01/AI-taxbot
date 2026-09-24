from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel


class TemplateValueType(StrEnum):
    MONEY = "money"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    STRING = "string"


class ValueGeneratorSpec(TaxMoEModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class TemplateVariable(TaxMoEModel):
    variable_id: str
    concept_id: str
    value_type: TemplateValueType
    generator: ValueGeneratorSpec
    mutable: bool = True
    mutation_tags: list[str] = Field(default_factory=list)
    family_semantic: bool = False
    field_path: str | None = None


class TemplateConstraint(TaxMoEModel):
    constraint_type: str
    left: str
    right: str | int | float | bool


class TemplateEvent(TaxMoEModel):
    event_id: str
    event_type: str
    variable_refs: list[str]
    concept_ids: list[str] = Field(default_factory=list)


class TemplateDocument(TaxMoEModel):
    document_id: str
    form_id: str
    field_bindings: dict[str, str] = Field(default_factory=dict)


class ScenarioTemplate(TaxMoEModel):
    schema_version: str = "0.1"
    template_id: str
    revision: int = 1
    name: str
    description: str = ""
    jurisdiction: str
    supported_tax_years: list[int] | None = None
    domain_ids: list[str]
    task_ids: list[str]
    variables: list[TemplateVariable]
    constraints: list[TemplateConstraint] = Field(default_factory=list)
    events: list[TemplateEvent] = Field(default_factory=list)
    documents: list[TemplateDocument] = Field(default_factory=list)
    required_rule_ids: list[str] = Field(default_factory=list)
    require_complete_base_scenario: bool = True
