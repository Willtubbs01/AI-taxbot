from __future__ import annotations
from decimal import Decimal
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from .enums import Severity

class TaxMoEModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class Money(TaxMoEModel):
    amount_cents: int
    currency: str = "USD"

    @property
    def amount(self) -> Decimal:
        return Decimal(self.amount_cents) / Decimal(100)

class Jurisdiction(TaxMoEModel):
    country: str = "US"
    level: str = "federal"
    state: str | None = None

class ValidationIssue(TaxMoEModel):
    code: str
    severity: Severity
    message: str
    artifact_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

class ValidationResult(TaxMoEModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
