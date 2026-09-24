from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel


class TaxValue(TaxMoEModel):
    value_id: str
    concept_id: str
    tax_year: int
    jurisdiction: str
    value: int | Decimal | str
    dimensions: dict[str, str] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


class TaxValueStore:
    def __init__(self, values: list[TaxValue], version: str = "0.1"):
        self.version = version
        self.values = values

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaxValueStore":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            [TaxValue.model_validate(x) for x in raw.get("values", [])],
            version=str(raw.get("version", "0.1")),
        )

    def resolve(
        self,
        value_id: str,
        tax_year: int,
        jurisdiction: str,
        dimensions: dict[str, str] | None = None,
    ) -> TaxValue:
        dimensions = dimensions or {}
        matches = [
            v for v in self.values
            if v.value_id == value_id
            and v.tax_year == tax_year
            and v.jurisdiction == jurisdiction
            and v.dimensions == dimensions
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one value for {value_id=} {tax_year=} {jurisdiction=} {dimensions=}; "
                f"found {len(matches)}"
            )
        return matches[0]
