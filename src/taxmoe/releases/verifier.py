from pathlib import Path
import hashlib
import json


def verify_release(root: str | Path) -> list[str]:
    root = Path(root)
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    issues = []
    for entry in manifest.get("files", []):
        path = root / entry["path"]
        if not path.exists():
            issues.append(f"MISSING:{entry['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            issues.append(f"HASH-MISMATCH:{entry['path']}")
    return issues
