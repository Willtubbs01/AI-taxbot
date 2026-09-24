from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Union

from pydantic import Field

from ...src.taxmoe.taxmoe.schemas.common import TaxMoEModel
from ...src.taxmoe.taxmoe.schemas.enums import FactOrigin, InformationState, TruthPolarity
from ...src.taxmoe.taxmoe.schemas.identifiers import EventId, FactId


class Money(TaxMoEModel):
    amount_cents: int
    currency: str = "USD"


class ScalarValue(TaxMoEModel):
    type: Literal["scalar"] = "scalar"
    value: str | int | bool | Decimal


class MoneyValue(TaxMoEModel):
    type: Literal["money"] = "money"
    value: Money


TaxFactValue = Annotated[Union[ScalarValue, MoneyValue], Field(discriminator="type")]


class FactAssertion(TaxMoEModel):
    assertion_id: str
    value: TaxFactValue
    source_type: FactOrigin
    document_instance_id: str | None = None


class TaxFact(TaxMoEModel):
    fact_id: FactId
    concept_id: str
    state: InformationState = InformationState.PRESENT
    value: TaxFactValue | None = None
    assertions: list[FactAssertion] = Field(default_factory=list)
    origin: FactOrigin = FactOrigin.SYNTHETIC
    truth: TruthPolarity = TruthPolarity.SYNTHETIC
    event_id: EventId | None = None
    document_instance_id: str | None = None
    field_path: str | None = None
    generation_variable_id: str | None = None
