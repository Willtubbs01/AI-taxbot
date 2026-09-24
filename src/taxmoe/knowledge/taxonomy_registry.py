from pathlib import Path
import yaml

class TaxonomyRegistry:
    def __init__(self, concepts: dict[str, dict], tasks: dict[str, dict], version: str = "0.1"):
        self.concepts = concepts
        self.tasks = tasks
        self.version = version

    @classmethod
    def from_yaml(cls, domains_path: str | Path, tasks_path: str | Path) -> "TaxonomyRegistry":
        domains = yaml.safe_load(Path(domains_path).read_text(encoding="utf-8")) or {}
        tasks = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8")) or {}
        return cls(
            concepts=domains.get("concepts", {}),
            tasks=tasks.get("tasks", {}),
            version=str(domains.get("version", "0.1")),
        )

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self.concepts

    def has_task(self, task_id: str) -> bool:
        return task_id in self.tasks

    def require_concept(self, concept_id: str) -> dict:
        if not self.has_concept(concept_id):
            raise KeyError(f"Unknown concept: {concept_id}")
        return self.concepts[concept_id]
