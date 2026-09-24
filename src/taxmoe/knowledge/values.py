from pathlib import Path
from typing import Any
import yaml
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel

class TaxValue(TaxMoEModel):
    value_id: str
    tax_year: int
    value: Any
    dimensions: dict[str, str] = Field(default_factory=dict)
    source_refs: list[dict] = Field(default_factory=list)

class TaxValueStore:
    def __init__(self, values: list[TaxValue], version: str = "0.1"):
        self.version = version
        self.values = values

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaxValueStore":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            [TaxValue.model_validate(x) for x in data.get("values", [])],
            version=str(data.get("version", "0.1")),
        )

    def resolve(self, value_id: str, tax_year: int, dimensions: dict[str, str] | None = None):
        dimensions = dimensions or {}
        matches = [
            x for x in self.values
            if x.value_id == value_id and x.tax_year == tax_year and x.dimensions == dimensions
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Ambiguous tax value {value_id} for {tax_year}/{dimensions}")
        return matches[0].value
