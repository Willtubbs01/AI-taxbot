from __future__ import annotations
import json
import typer
from taxmoe.training.optimization import OptimizationConfig
app=typer.Typer(help='Stage 5 training configuration commands.')
@app.command('check-optimization')
def check_optimization(config:str):
    c=OptimizationConfig.from_yaml(config); typer.echo(json.dumps(c.model_dump(mode='json'),indent=2))
