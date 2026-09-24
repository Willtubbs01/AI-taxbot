from pathlib import Path
import yaml
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
from taxmoe.schemas.enums import ValidationMode

class TemplateBuildSpec(TaxMoEModel):
    id: str
    count: int

class DatasetBuildConfig(TaxMoEModel):
    name: str
    build_version: str = "0.1"
    seed: int = 42017
    tax_year: int = 2025

    source_manifest: str
    taxonomy_domains: str
    taxonomy_tasks: str
    forms: str
    form_versions: str
    rule_files: list[str]
    value_store: str
    template_files: list[str]

    templates: list[TemplateBuildSpec]
    mutation_config: str
    split_config: str
    validation_policy: str
    export_config: str

    validation_mode: ValidationMode = ValidationMode.FULL

    @classmethod
    def from_yaml(cls, path: str | Path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw["build"] if "build" in raw else raw)
