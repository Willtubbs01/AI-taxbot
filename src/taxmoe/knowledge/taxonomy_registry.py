from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel


class TaxonomyConcept(TaxMoEModel):
    concept_id: str
    name: str
    domain_id: str
    aliases: list[str] = Field(default_factory=list)
    deprecated_by: str | None = None


class TaxonomyRegistry:
    def __init__(self, concepts: Iterable[TaxonomyConcept], version: str = "0.1"):
        self.version = version
        concepts = list(concepts)
        self._concepts = {c.concept_id: c for c in concepts}
        if len(self._concepts) != len(concepts):
            raise ValueError("Duplicate taxonomy concept IDs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaxonomyRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            [TaxonomyConcept.model_validate(x) for x in raw.get("concepts", [])],
            version=str(raw.get("version", "0.1")),
        )

    def get(self, concept_id: str) -> TaxonomyConcept | None:
        return self._concepts.get(concept_id)

    def require(self, concept_id: str) -> TaxonomyConcept:
        value = self.get(concept_id)
        if value is None:
            raise KeyError(f"Unknown concept: {concept_id}")
        return value

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts
