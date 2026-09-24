import json
from pathlib import Path

def write_build_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    md = ["# TaxMoE Build Summary", ""]
    for key, value in summary.items():
        md.append(f"## {key.replace('_', ' ').title()}")
        if isinstance(value, (dict, list)):
            md.append("```json")
            md.append(json.dumps(value, indent=2, sort_keys=True, default=str))
            md.append("```")
        else:
            md.append(str(value))
        md.append("")
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
