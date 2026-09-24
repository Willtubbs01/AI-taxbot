from pydantic import Field
from .common import TaxMoEModel, Jurisdiction
from .provenance import SourceReference

class FormField(TaxMoEModel):
    field_id: str
    label: str
    concept_id: str | None = None

class FormDefinition(TaxMoEModel):
    form_id: str
    name: str
    role: str  # source_document | return_form | schedule
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)

class FormVersion(TaxMoEModel):
    form_version_id: str
    form_id: str
    tax_year: int
    fields: list[FormField] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
