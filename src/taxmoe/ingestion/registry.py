from __future__ import annotations
import json
from pathlib import Path
from taxmoe.schemas.source import SourceRecord


class SourceRegistry:
    def __init__(self, records: list[SourceRecord] | None = None):
        self._records = {str(r.source_id): r for r in (records or [])}

    def add(self, record: SourceRecord) -> None:
        key = str(record.source_id)
        if key in self._records:
            raise ValueError(f"Duplicate source_id: {key}")
        self._records[key] = record

    def get(self, source_id: str) -> SourceRecord | None:
        return self._records.get(source_id)

    def require(self, source_id: str) -> SourceRecord:
        record = self.get(source_id)
        if record is None:
            raise KeyError(source_id)
        return record

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for key in sorted(self._records):
                f.write(json.dumps(self._records[key].model_dump(mode="json"), sort_keys=True) + "\n")
