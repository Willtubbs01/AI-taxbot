from __future__ import annotations

from pydantic import Field
from .common import TaxMoEModel


class ArtifactLineage(TaxMoEModel):
    parent_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    version_refs: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)
