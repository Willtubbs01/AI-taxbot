from pathlib import Path
import json
import typer
from taxmoe.builds.runner import DatasetBuildRunner
from taxmoe.releases.freezer import DatasetFreezer
from taxmoe.releases.verifier import verify_release

app = typer.Typer(help="Dataset build and release commands.")

@app.command("build")
def build(
    config: str = typer.Option(..., "--config"),
    project_root: str = typer.Option(".", "--project-root"),
):
    record = DatasetBuildRunner(project_root).run(config)
    typer.echo(record.model_dump_json(indent=2))

@app.command("status")
def status(
    build_id: str,
    project_root: str = typer.Option(".", "--project-root"),
):
    path = Path(project_root) / "artifacts" / "builds" / build_id / "build.json"
    typer.echo(path.read_text(encoding="utf-8"))

@app.command("freeze")
def freeze(
    build_id: str,
    version: str = typer.Option(..., "--version"),
    project_root: str = typer.Option(".", "--project-root"),
):
    manifest = DatasetFreezer(project_root).freeze(build_id, version)
    typer.echo(manifest.model_dump_json(indent=2))

@app.command("verify-release")
def verify_release_command(
    release_path: str,
):
    typer.echo(json.dumps(verify_release(release_path), indent=2))
