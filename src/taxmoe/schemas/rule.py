from __future__ import annotations
from typing import Any
from pydantic import Field
from .common import TaxMoEModel, Jurisdiction
from .enums import QualityLevel, ReviewLevel
from .provenance import SourceReference

class RuleCondition(TaxMoEModel):
    concept_id: str
    operator: str
    value: Any | None = None
    value_ref: str | None = None

class RuleEffect(TaxMoEModel):
    effect_type: str
    target: str
    value: Any | None = None

class TaxRule(TaxMoEModel):
    rule_id: str
    revision: int = 1
    name: str
    category: str
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    tax_years: list[int] = Field(default_factory=list)
    conditions: list[RuleCondition] = Field(default_factory=list)
    effects: list[RuleEffect] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
    status: str = "verified"
    completeness: str = "complete"
    scenario_generation_allowed: bool = True
    quality: QualityLevel = QualityLevel.Q3
    review: ReviewLevel = ReviewLevel.R1
