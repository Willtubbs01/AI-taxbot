from __future__ import annotations
import json
import shutil
from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file, stable_hash
from taxmoe.schemas.manifest import FileRecord
from .models import DatasetReleaseManifest, DatasetReleaseStatus

class DatasetFreezer:
    def __init__(self, project_root: str | Path = "."):
        self.root = Path(project_root).resolve()

    def freeze(self, build_id: str, version: str, name: str = "TaxMoE-Dataset") -> DatasetReleaseManifest:
        build = self.root / "artifacts" / "builds" / build_id
        record = json.loads((build / "build.json").read_text(encoding="utf-8"))
        if not record.get("passed_validation"):
            raise RuntimeError("Build is not validation-approved")

        destination = self.root / "datasets" / "releases" / f"taxmoe-v{version}"
        if destination.exists():
            raise FileExistsError(f"Release already exists: {destination}")

        staging = self.root / "datasets" / "releases" / ".staging" / f"taxmoe-v{version}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(build, staging / "build_artifacts")

        file_records = []
        for path in sorted((staging / "build_artifacts").rglob("*")):
            if path.is_file():
                rel = path.relative_to(staging)
                file_records.append(FileRecord(
                    path=str(rel).replace("\\", "/"),
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                ))

        release_id = f"RELEASE-{stable_hash(name, version, build_id, [x.sha256 for x in file_records])[:20]}"
        manifest = DatasetReleaseManifest(
            release_id=release_id,
            dataset_name=name,
            dataset_version=version,
            status=DatasetReleaseStatus.FROZEN,
            source_build_id=build_id,
            source_build_spec_hash=record["spec_hash"],
            files=file_records,
            metadata={"scope": "US federal individual income tax; starter Stage 3 release"},
        )
        content_hash = stable_hash(manifest.model_dump(exclude={"release_content_hash"}, mode="json"))
        manifest.release_content_hash = content_hash
        (staging / "release_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        return manifest
