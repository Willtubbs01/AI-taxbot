from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file
from taxmoe.schemas.source import SourceRecord


def validate_source(record: SourceRecord, root: str | Path = ".") -> list[str]:
    path = Path(root) / record.raw_path
    issues = []
    if not path.exists():
        return ["SRC-MISSING-RAW"]
    if sha256_file(path) != record.sha256:
        issues.append("SRC-HASH-MISMATCH")
    return issues
