from __future__ import annotations
import json
from pathlib import Path


def write_jsonl(path: str | Path, records) -> int:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count
