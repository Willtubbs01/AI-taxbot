from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from taxmoe.schemas.form import FormDefinition, FormVersion


class FormRegistry:
    def __init__(self, forms: Iterable[FormDefinition], version: str = "0.1"):
        self.version = version
        self._forms = {str(f.form_id): f for f in forms}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FormRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            [FormDefinition.model_validate(x) for x in raw.get("forms", [])],
            version=str(raw.get("version", "0.1")),
        )

    def get(self, form_id: str) -> FormDefinition | None:
        return self._forms.get(form_id)

    def require(self, form_id: str) -> FormDefinition:
        form = self.get(form_id)
        if form is None:
            raise KeyError(f"Unknown form: {form_id}")
        return form

    def resolve_version(self, form_id: str, tax_year: int) -> FormVersion:
        form = self.require(form_id)
        matches = [v for v in form.versions if v.tax_year == tax_year]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one {form_id} version for tax year {tax_year}")
        return matches[0]
