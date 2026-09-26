from __future__ import annotations
import json
from pathlib import Path
import typer
from taxmoe.benchmarking.environment import collect_environment
app=typer.Typer(help='Stage 5 hardware profiling commands.')
@app.command('environment')
def environment(output:str|None=None):
    data=collect_environment(); text=json.dumps(data,indent=2)
    if output: Path(output).write_text(text+'\n')
    typer.echo(text)
