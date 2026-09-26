from __future__ import annotations
import json
from pathlib import Path
import typer

app=typer.Typer(help='Stage 5 model inspection and release commands.')

@app.command('inspect')
def inspect(config:str, output:str='artifacts/models/base/QwenDense-0.6B'):
    # Lazy import keeps the general CLI usable before optional/heavy model deps
    # are installed in a fresh environment.
    from taxmoe.modeling.model_loader import ModelLoadConfig,load_model,build_manifest
    cfg=ModelLoadConfig.from_yaml(config)
    model,c=load_model(cfg)
    manifest,report=build_manifest(cfg,model,c)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'model_manifest.json').write_text(manifest.model_dump_json(indent=2)+'\n')
    (out/'architecture_report.json').write_text(report.model_dump_json(indent=2)+'\n')
    typer.echo(json.dumps({'passed':True,'base_model_fingerprint':manifest.base_model_fingerprint,'output':str(out)},indent=2))

@app.command('verify-release')
def verify_release(path:str):
    from taxmoe.model_releases.verifier import verify_taxdense_release
    m=verify_taxdense_release(path)
    typer.echo(json.dumps({'passed':True,'release_id':m.release_id,'content_hash':m.release_content_hash},indent=2))
