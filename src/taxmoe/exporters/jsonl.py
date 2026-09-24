from __future__ import annotations
import json
from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file

def write_jsonl(path: str | Path, rows: list[dict]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
            f.write("\n")
    tmp.replace(path)
    return {
        "path": str(path),
        "records": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
