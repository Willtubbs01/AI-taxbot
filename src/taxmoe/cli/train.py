from __future__ import annotations

import json
import typer

from taxmoe.training.optimization import OptimizationConfig

app = typer.Typer(help="Stage 5/6 training configuration commands.")


@app.command("check-optimization")
def check_optimization(config: str):
    c = OptimizationConfig.from_yaml(config)
    typer.echo(json.dumps(c.model_dump(mode="json"), indent=2))


@app.command("check-moe")
def check_moe(config: str):
    from taxmoe.moe.config import MoEUpcycleConfig
    c = MoEUpcycleConfig.from_yaml(config)
    typer.echo(json.dumps(c.model_dump(mode="json"), indent=2))


@app.command("dense")
def train_dense(
    cache_manifest: str = typer.Argument(..., help="Stage 4 token-only CPT cache manifest."),
    training_config: str = typer.Option("configs/training/taxdense_v1.yaml", "--config"),
    output: str = typer.Option("artifacts/training/RUN-DENSE-CPT-v0.1", "--output"),
    device: str = typer.Option("cuda", "--device"),
    microbatch_size: int = typer.Option(1, "--microbatch-size", min=1),
    max_steps: int | None = typer.Option(None, "--max-steps", min=1),
    gradient_checkpointing: bool = typer.Option(True, "--gradient-checkpointing/--no-gradient-checkpointing"),
    seed: int = typer.Option(1701, "--seed"),
):
    """Run Stage 5 full-parameter dense CPT and write an HF-compatible TaxDense candidate."""
    from taxmoe.training.dense_run import run_dense_cpt
    result = run_dense_cpt(
        training_config=training_config,
        cache_manifest=cache_manifest,
        output_dir=output,
        device=device,
        microbatch_size=microbatch_size,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("moe")
def train_moe(
    model_path: str = typer.Argument(..., help="Initialized Stage 6 TaxMoE artifact containing moe_config.json."),
    cache_manifest: str = typer.Argument(..., help="Stage 4 token-only CPT cache manifest."),
    training_config: str = typer.Option("configs/training/taxmoe_v1.yaml", "--config"),
    output: str = typer.Option("artifacts/training/RUN-MOE-CPT-v0.1", "--output"),
    device: str = typer.Option("cuda", "--device"),
    microbatch_size: int = typer.Option(1, "--microbatch-size", min=1),
    sequence_length: int = typer.Option(1024, "--sequence-length", min=3),
    max_steps: int | None = typer.Option(None, "--max-steps", min=1),
    gradient_checkpointing: bool = typer.Option(True, "--gradient-checkpointing/--no-gradient-checkpointing"),
    seed: int = typer.Option(1701, "--seed"),
):
    """Run Stage 6 sparse-MoE CPT with LM + load-balancing loss."""
    from taxmoe.training.moe_run import run_moe_cpt

    result = run_moe_cpt(
        model_path=model_path,
        training_config=training_config,
        cache_manifest=cache_manifest,
        output_dir=output,
        device=device,
        microbatch_size=microbatch_size,
        sequence_length=sequence_length,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
    )
    typer.echo(json.dumps(result, indent=2))
