from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Mapping

from taxmoe.ingestion.hashing import sha256_file

from .models import ModelReleaseStatus, ReleaseFile
from .moe import TaxMoEReleaseManifest
from .verifier import manifest_hash


class TaxMoEFreezer:
    def freeze(
        self,
        candidate_model_dir,
        releases_root,
        manifest: TaxMoEReleaseManifest,
        *,
        evidence_files: Mapping[str, str | Path] | None = None,
    ):
        blockers = {
            "optimization_config_hash": "TAXMOE-FREEZE-OPTIMIZATION-SPEC-INCOMPLETE",
            "hardware_envelope_hash": "TAXMOE-FREEZE-HARDWARE-ENVELOPE-MISSING",
            "source_checkpoint_hash": "TAXMOE-FREEZE-TRAINED-CANDIDATE-MISSING",
            "evaluation_hash": "TAXMOE-FREEZE-CANDIDATE-EVALUATION-MISSING",
            "routing_health_hash": "TAXMOE-FREEZE-ROUTING-HEALTH-EVIDENCE-MISSING",
            "reproducibility_hash": "TAXMOE-FREEZE-REPRODUCIBILITY-EVIDENCE-MISSING",
        }
        for field, code in blockers.items():
            if not getattr(manifest, field):
                raise ValueError(code)

        src = Path(candidate_model_dir)
        if not src.exists():
            raise FileNotFoundError(src)
        root = Path(releases_root)
        dest = root / f"{manifest.name}-v{manifest.version}"
        if dest.exists():
            raise FileExistsError(dest)
        stage = root / ".staging" / f"{manifest.release_id}-{uuid.uuid4().hex[:8]}"
        stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, stage)

        for rel, source in sorted((evidence_files or {}).items()):
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"TAXMOE-FREEZE-EVIDENCE-PATH-INVALID:{rel}")
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            target = stage / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

        files = []
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "model_manifest.json":
                files.append(
                    ReleaseFile(
                        path=str(p.relative_to(stage)).replace("\\", "/"),
                        bytes=p.stat().st_size,
                        sha256=sha256_file(p),
                    )
                )
        frozen = manifest.model_copy(update={"status": ModelReleaseStatus.FROZEN, "model_files": files})
        frozen = frozen.model_copy(update={"release_content_hash": manifest_hash(frozen)})
        (stage / "model_manifest.json").write_text(
            frozen.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(dest)
        return dest, frozen
