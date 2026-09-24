from __future__ import annotations

from pathlib import Path

import yaml

from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.taxonomy_registry import TaxonomyRegistry
from ...src.taxmoe.taxmoe.scenarios.templates import ScenarioTemplate


class TemplateRegistry:
    def __init__(self, templates: list[ScenarioTemplate], version: str = "0.1"):
        self.version = version
        self._templates = {(t.template_id, t.revision): t for t in templates}

    @classmethod
    def from_directory(cls, root: str | Path, version: str = "0.1") -> "TemplateRegistry":
        templates: list[ScenarioTemplate] = []
        for path in sorted(Path(root).rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload = raw.get("template", raw)
            templates.append(ScenarioTemplate.model_validate(payload))
        return cls(templates, version)

    def get(self, template_id: str, revision: int | None = None) -> ScenarioTemplate | None:
        candidates = [t for (tid, _), t in self._templates.items() if tid == template_id]
        if not candidates:
            return None
        if revision is None:
            return max(candidates, key=lambda t: t.revision)
        return self._templates.get((template_id, revision))

    def validate_references(
        self,
        taxonomy: TaxonomyRegistry,
        forms: FormRegistry,
        rules: RuleRegistry,
    ) -> None:
        for template in self._templates.values():
            variable_ids = {v.variable_id for v in template.variables}
            if len(variable_ids) != len(template.variables):
                raise ValueError(f"Duplicate variable IDs in {template.template_id}")
            for variable in template.variables:
                taxonomy.require(variable.concept_id)
            for document in template.documents:
                forms.require(document.form_id)
                for variable_ref in document.field_bindings.values():
                    if variable_ref not in variable_ids:
                        raise ValueError(f"Unknown document variable ref: {variable_ref}")
            for rid in template.required_rule_ids:
                if rules.get(rid) is None:
                    raise ValueError(f"Unknown rule {rid} in {template.template_id}")
