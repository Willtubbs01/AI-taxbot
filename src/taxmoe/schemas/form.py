from __future__ import annotations

from pydantic import Field

from .common import TaxMoEModel
from .identifiers import FormId, SourceId


class FormField(TaxMoEModel):
    field_id: str
    concept_id: str
    label: str
    box_or_line: str | None = None


class FormVersion(TaxMoEModel):
    form_version_id: str
    form_id: FormId
    tax_year: int
    source_ids: list[SourceId] = Field(default_factory=list)
    fields: list[FormField] = Field(default_factory=list)


class FormDefinition(TaxMoEModel):
    form_id: FormId
    name: str
    role: str
    aliases: list[str] = Field(default_factory=list)
    versions: list[FormVersion] = Field(default_factory=list)
