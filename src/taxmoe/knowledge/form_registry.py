from pathlib import Path
import yaml
from taxmoe.schemas.form import FormDefinition, FormVersion

class FormRegistry:
    def __init__(self, forms: list[FormDefinition], versions: list[FormVersion], version: str = "0.1"):
        self.version = version
        self.forms = {x.form_id: x for x in forms}
        self.versions = {x.form_version_id: x for x in versions}

    @classmethod
    def from_yaml(cls, forms_path: str | Path, versions_path: str | Path) -> "FormRegistry":
        forms_data = yaml.safe_load(Path(forms_path).read_text(encoding="utf-8")) or {}
        ver_data = yaml.safe_load(Path(versions_path).read_text(encoding="utf-8")) or {}
        return cls(
            [FormDefinition.model_validate(x) for x in forms_data.get("forms", [])],
            [FormVersion.model_validate(x) for x in ver_data.get("versions", [])],
            version=str(forms_data.get("version", "0.1")),
        )

    def get_form(self, form_id: str):
        return self.forms.get(form_id)

    def get_version_for_year(self, form_id: str, tax_year: int):
        matches = [v for v in self.versions.values() if v.form_id == form_id and v.tax_year == tax_year]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous form version for {form_id}/{tax_year}")
        return matches[0] if matches else None
