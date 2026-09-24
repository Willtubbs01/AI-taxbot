from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import TaxMoEModel
from .enums import QualityLevel, ReviewLevel, RuleCompleteness, RuleStatus
from .identifiers import RuleId, SourceId


class RulePredicate(TaxMoEModel):
    op: Literal["exists", "eq", "ne", "gt", "gte", "lt", "lte", "and", "or", "not"]
    concept_id: str | None = None
    value: Any | None = None
    children: list["RulePredicate"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.op in {"and", "or"} and not self.children:
            raise ValueError(f"{self.op} requires children")
        if self.op == "not" and len(self.children) != 1:
            raise ValueError("not requires exactly one child")
        if self.op in {"exists", "eq", "ne", "gt", "gte", "lt", "lte"} and not self.concept_id:
            raise ValueError(f"{self.op} requires concept_id")
        return self


class RuleEffect(TaxMoEModel):
    effect_type: Literal[
        "activate_concept",
        "require_fact",
        "require_form",
        "candidate_form",
        "require_calculation",
        "needs_rule_lookup",
        "outside_scope",
    ]
    target: str
    task_id: str | None = None


class RuleSourceRef(TaxMoEModel):
    source_id: SourceId
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None


class TaxRule(TaxMoEModel):
    rule_id: RuleId
    revision: int = Field(ge=1)
    name: str
    jurisdiction: str
    tax_years: list[int] = Field(default_factory=list)
    status: RuleStatus = RuleStatus.DRAFT
    completeness: RuleCompleteness = RuleCompleteness.PARTIAL
    quality: QualityLevel = QualityLevel.Q0
    review: ReviewLevel = ReviewLevel.R0
    predicate: RulePredicate
    effects: list[RuleEffect]
    source_refs: list[RuleSourceRef] = Field(default_factory=list)
    scenario_generation_allowed: bool = False
