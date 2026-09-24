from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from taxmoe.schemas.common import stable_hash


def freeze_build(build_dir: str | Path, releases_root: str | Path, version: str) -> Path:
    build_dir = Path(build_dir)
    manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((build_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise RuntimeError("Build is not freeze-eligible")
    destination = Path(releases_root) / f"taxmoe-v{version}"
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(releases_root) / ".staging" / f"taxmoe-v{version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(build_dir / "exports", staging / "exports")
    files = []
    for path in sorted((staging / "exports").rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            files.append({
                "path": str(path.relative_to(staging)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
    release = {
        "release_id": f"RELEASE-{stable_hash(version, manifest['build_id'], manifest['spec_hash'])[:20]}",
        "name": "TaxMoE-Dataset",
        "version": version,
        "status": "frozen",
        "source_build_id": manifest["build_id"],
        "source_build_spec_hash": manifest["spec_hash"],
        "files": files,
    }
    release["release_content_hash"] = stable_hash(release)
    (staging / "release_manifest.json").write_text(json.dumps(release, sort_keys=True, indent=2), encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return destination
