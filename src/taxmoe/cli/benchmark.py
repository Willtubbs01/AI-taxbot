from __future__ import annotations

import json
from pathlib import Path
import typer

from taxmoe.benchmarking.environment import collect_environment

app = typer.Typer(help="Stage 5/6 hardware profiling commands.")


@app.command("environment")
def environment(output: str | None = None):
    data = collect_environment(); text = json.dumps(data, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


@app.command("moe-case")
def moe_case(model_path: str, sequence_length: int = 512, microbatch_size: int = 1, precision: str = "fp16", gradient_checkpointing: bool = False, optimizer_config: str | None = None, output: str | None = typer.Option(None, "--output", help="Optional JSON file for release/hardware evidence.")):
    """Run one Stage 6 GPU training-profile case. Use separate processes for production sweeps."""
    import torch
    from taxmoe.benchmarking.moe_runner import run_moe_case
    from taxmoe.modeling.taxmoe_loader import load_taxmoe_model
    from taxmoe.training.optimization import OptimizationConfig, build_optimizer

    if not torch.cuda.is_available():
        typer.echo(json.dumps({"status": "unsupported", "error": "CUDA unavailable"}, indent=2)); raise typer.Exit(2)
    if precision not in {"fp16", "bf16"}:
        raise typer.BadParameter("--precision must be fp16 or bf16")
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            typer.echo(json.dumps({"status": "unsupported", "error": "CUDA BF16 unsupported"}, indent=2)); raise typer.Exit(2)
        parameter_dtype = torch.bfloat16
    else:
        # FP16 AMP must keep FP32 trainable parameters for GradScaler.
        parameter_dtype = torch.float32
    model, _ = load_taxmoe_model(model_path, device="cuda", torch_dtype=parameter_dtype)
    optimizer = None
    if optimizer_config:
        ocfg = OptimizationConfig.from_yaml(optimizer_config)
        optimizer, _ = build_optimizer(model, ocfg)
    result = run_moe_case(model, int(model.config.vocab_size), precision, sequence_length, microbatch_size, gradient_checkpointing=gradient_checkpointing, optimizer=optimizer)
    text = result.model_dump_json(indent=2)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
