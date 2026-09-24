from pathlib import Path
import yaml
from taxmoe.schemas.source import SourceManifest

class SourceRegistry:
    def __init__(self, manifest: SourceManifest):
        self.manifest = manifest
        self._by_id = {s.source_id: s for s in manifest.sources}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SourceRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(SourceManifest.model_validate(data))

    def get(self, source_id: str):
        return self._by_id.get(source_id)

    def require(self, source_id: str):
        value = self.get(source_id)
        if value is None:
            raise KeyError(f"Unknown source_id: {source_id}")
        return value

    def all(self):
        return list(self._by_id.values())
