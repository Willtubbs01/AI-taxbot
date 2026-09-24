from __future__ import annotations
import json
from pathlib import Path


def write_build_summary(manifest: dict, path: str | Path) -> None:
    lines = [
        f"# TaxMoE Build {manifest['build_id']}",
        "",
        f"- Name: `{manifest.get('name', '')}`",
        f"- Spec hash: `{manifest.get('spec_hash', '')}`",
        f"- Scenarios: {manifest.get('scenario_count', 0)}",
        f"- Clusters: {manifest.get('cluster_count', 0)}",
        "",
        "## Files",
    ]
    for name, info in sorted(manifest.get("files", {}).items()):
        lines.append(f"- {name}: `{info['path']}` — {info['bytes']} bytes — `{info['sha256']}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_build_summary_json(manifest: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
