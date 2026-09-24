from pathlib import Path
import yaml
from .templates import ScenarioTemplate

class TemplateRegistry:
    def __init__(self, templates: list[ScenarioTemplate], version: str = "0.1"):
        self.version = version
        self.templates = {(t.template_id, t.revision): t for t in templates}

    @classmethod
    def from_paths(cls, paths: list[str | Path], version: str = "0.1"):
        templates = []
        for path in paths:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if "template" in data:
                data = data["template"]
            templates.append(ScenarioTemplate.model_validate(data))
        return cls(templates, version)

    def get(self, template_id: str, revision: int | None = None) -> ScenarioTemplate | None:
        if revision is not None:
            return self.templates.get((template_id, revision))
        matches = [t for (tid, _), t in self.templates.items() if tid == template_id]
        return max(matches, key=lambda x: x.revision) if matches else None
