from __future__ import annotations

from pathlib import Path

from taxmoe.ingestion.hashing import sha256_file, stable_hash


def model_artifact_hash(root: str | Path) -> str:
    """Stable sparse/dense HF model identity excluding mutable report sidecars."""
    root = Path(root)
    files: list[tuple[str, int, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        name = p.name
        is_weight = name.startswith("model") and (
            name.endswith(".safetensors") or name.endswith(".bin") or name.endswith(".json")
        )
        if rel in {"config.json", "generation_config.json", "moe_config.json"} or is_weight:
            files.append((rel, p.stat().st_size, sha256_file(p)))
    if not files:
        raise ValueError(f"MODEL-ARTIFACT-HASH-EMPTY:{root}")
    return stable_hash(files)
