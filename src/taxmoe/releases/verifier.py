from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file
from .models import DatasetReleaseManifest

def verify_release(path: str | Path) -> dict:
    root = Path(path)
    manifest = DatasetReleaseManifest.model_validate_json(
        (root / "release_manifest.json").read_text(encoding="utf-8")
    )
    failures = []
    for rec in manifest.files:
        p = root / rec.path
        if not p.exists():
            failures.append({"path": rec.path, "reason": "missing"})
        elif sha256_file(p) != rec.sha256:
            failures.append({"path": rec.path, "reason": "hash_mismatch"})
    return {"passed": not failures, "failures": failures, "release_id": manifest.release_id}
