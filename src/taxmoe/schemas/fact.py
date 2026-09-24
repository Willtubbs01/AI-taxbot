from __future__ import annotations
from typing import Any
from pydantic import Field
from .common import TaxMoEModel
from .enums import FactOrigin, InformationState

class FactAssertion(TaxMoEModel):
    assertion_id: str
    value: Any
    source_type: str
    source_ref: str | None = None

class TaxFact(TaxMoEModel):
    fact_id: str
    concept_id: str
    state: InformationState = InformationState.PRESENT
    value: Any | None = None
    origin: FactOrigin = FactOrigin.SYNTHETIC
    event_id: str | None = None
    document_instance_id: str | None = None
    assertions: list[FactAssertion] = Field(default_factory=list)
    generation_variable_id: str | None = None
