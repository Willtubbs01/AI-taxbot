from __future__ import annotations

import json
from pathlib import Path

import typer

from taxmoe.data.cache_models import TrainingCacheManifest
from taxmoe.data.cache_verifier import verify_cache
from taxmoe.model_inputs.freezer import InputReleaseFreezer
from taxmoe.model_inputs.release_models import ModelInputReleaseManifest
from taxmoe.model_inputs.verifier import verify_release_manifest

app = typer.Typer(help="Stage 4 model-input cache and release commands.")


def _load_cache_manifest(path: str | Path) -> TrainingCacheManifest:
    return TrainingCacheManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _load_input_release_manifest(path: str | Path) -> ModelInputReleaseManifest:
    path = Path(path)
    if path.is_dir():
        path = path / "manifests" / "input_release_manifest.json"
    return ModelInputReleaseManifest.model_validate_json(path.read_text(encoding="utf-8"))


@app.command("build")
def build_command(
    source_manifest: str = typer.Option("sources/registry/source_manifest_2025.yaml", "--source-manifest", help="IRS/source registry YAML for CPT ingestion."),
    tokenization_config: str = typer.Option("configs/modeling/tokenization_v1.yaml", "--tokenization-config"),
    output: str = typer.Option("artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1", "--output"),
    project_root: str = typer.Option(".", "--project-root"),
    chunk_max_chars: int = typer.Option(6000, "--chunk-max-chars"),
    validation_percent: int = typer.Option(10, "--validation-percent", min=0, max=50),
):
    """Build the Stage 4 token-only CPT cache from verified authoritative PDF sources."""
    from taxmoe.data.stage4_builder import build_cpt_token_cache

    manifest_path, manifest = build_cpt_token_cache(
        source_manifest=source_manifest,
        tokenization_config=tokenization_config,
        output_dir=output,
        project_root=project_root,
        chunk_max_chars=chunk_max_chars,
        validation_percent=validation_percent,
    )
    typer.echo(json.dumps({
        "passed": True,
        "output": str(Path(output)),
        "manifest": str(manifest_path),
        "cache_id": manifest.cache_id,
        "records": manifest.total_records,
        "tokens": manifest.total_tokens,
        "content_hash": manifest.content_hash,
    }, indent=2))


@app.command("verify-cache")
def verify_cache_command(
    manifest: str = typer.Argument(..., help="Path to a Stage 4 cache manifest JSON."),
    root: str | None = typer.Option(None, "--root", help="Cache shard root. Defaults to the manifest directory."),
):
    manifest_path = Path(manifest)
    model = _load_cache_manifest(manifest_path)
    cache_root = Path(root) if root else manifest_path.parent
    verify_cache(model, cache_root)
    typer.echo(json.dumps({"passed": True, "cache_id": model.cache_id, "content_hash": model.content_hash}, indent=2))


@app.command("freeze")
def freeze_command(
    candidate_dir: str = typer.Argument(..., help="Completed Stage 4 input candidate directory."),
    manifest: str = typer.Option(..., "--manifest", help="Candidate ModelInputReleaseManifest JSON."),
    releases_root: str = typer.Option("artifacts/model_inputs/releases", "--releases-root"),
):
    candidate_manifest = _load_input_release_manifest(manifest)
    destination, frozen = InputReleaseFreezer().freeze(candidate_dir, releases_root, candidate_manifest)
    typer.echo(json.dumps({"release_path": str(destination), "manifest": frozen.model_dump(mode="json")}, indent=2))


@app.command("verify-release")
def verify_release_command(
    release: str = typer.Argument(..., help="Input release directory or input_release_manifest.json path."),
):
    model = _load_input_release_manifest(release)
    verify_release_manifest(model)
    typer.echo(json.dumps({"passed": True, "release_id": model.release_id, "content_hash": model.release_content_hash}, indent=2))
