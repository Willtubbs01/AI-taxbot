from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from taxmoe.data.cache_models import TrainingCacheManifest
from taxmoe.data.cache_verifier import verify_cache
from taxmoe.data.cached_dataset import CachedTaxMoEDataset
from taxmoe.data.collator import TaxMoETrainingCollator
from taxmoe.data.token_block_dataset import TokenBlockDataset
from taxmoe.ingestion.hashing import sha256_file, stable_hash
from taxmoe.modeling.taxmoe_loader import load_taxmoe_model, save_taxmoe_model
from taxmoe.model_releases.artifact_hashing import model_artifact_hash
from taxmoe.modeling.tokenization_config import TokenizationConfig
from taxmoe.modeling.tokenizer import TaxMoETokenizer
from taxmoe.training.moe_adapter import MoECausalLMAdapter, enable_moe_gradient_checkpointing
from taxmoe.training.optimization import OptimizationConfig, build_optimizer, build_scheduler
from taxmoe.training.run_models import RunStatus, TrainingProgress, TrainingRunManifest
from taxmoe.training.trainer import DenseCPTTrainer


def _yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _file_hash(path: str | Path) -> str:
    return stable_hash(_yaml(path))



def _load_cache_manifest(path: str | Path) -> tuple[Path, TrainingCacheManifest]:
    path = Path(path)
    manifest = TrainingCacheManifest.model_validate_json(path.read_text(encoding="utf-8"))
    verify_cache(manifest, path.parent)
    return path, manifest


def _resolve_shards(manifest_path: Path, manifest: TrainingCacheManifest, split: str) -> list[Path]:
    paths = [manifest_path.parent / f.path for f in manifest.files if f.split == split]
    if not paths:
        raise ValueError(f"TRAIN-CACHE-SPLIT-MISSING:{split}")
    return paths


def _objective_balance_coefficient(objective_path: Path, default: float) -> float:
    raw = _yaml(objective_path).get("objective", {})
    for item in raw.get("auxiliary_losses", []) or []:
        if item.get("name") == "moe_balance":
            return float(item.get("coefficient", default))
    return float(default)


def _routing_health(metrics: dict[str, Any], target_layers: list[int], num_experts: int) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    passed = True
    for layer in target_layers:
        prefix = f"aux/router/{layer}/"
        assignments = [float(metrics.get(prefix + f"assignment_e{e}", 0.0)) for e in range(num_experts)]
        probability_mass = [float(metrics.get(prefix + f"prob_mass_e{e}", 0.0)) for e in range(num_experts)]
        top1 = [float(metrics.get(prefix + f"top1_e{e}", 0.0)) for e in range(num_experts)]
        entropy = float(metrics.get(prefix + "entropy", float("nan")))
        balance = float(metrics.get(prefix + "balance", float("nan")))
        finite = all(math.isfinite(x) for x in assignments + probability_mass + top1 + [entropy, balance])
        all_experts_used = all(x > 0.0 for x in assignments)
        no_severe_top1_collapse = max(top1, default=1.0) < 0.95
        layer_passed = finite and all_experts_used and no_severe_top1_collapse
        passed = passed and layer_passed
        layers[str(layer)] = {
            "assignment_fraction": assignments,
            "probability_mass": probability_mass,
            "top1_share": top1,
            "mean_entropy": entropy,
            "balance_loss": balance,
            "all_experts_used": all_experts_used,
            "no_severe_top1_collapse": no_severe_top1_collapse,
            "passed": layer_passed,
        }
    return {"passed": passed, "layers": layers}


def run_moe_cpt(
    *,
    model_path: str | Path,
    training_config: str | Path,
    cache_manifest: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    microbatch_size: int = 1,
    sequence_length: int = 1024,
    max_steps: int | None = None,
    gradient_checkpointing: bool = True,
    seed: int = 1701,
) -> dict[str, Any]:
    """Run Stage 6 full-parameter sparse-MoE CPT from an initialized TaxMoE artifact."""
    if microbatch_size < 1:
        raise ValueError("TRAIN-MICROBATCH-INVALID")
    if sequence_length < 3:
        raise ValueError("TRAIN-SEQUENCE-LENGTH-INVALID")
    model_path = Path(model_path)
    if not model_path.exists() or not (model_path / "moe_config.json").exists():
        raise FileNotFoundError(f"MOE-TRAIN-INIT-ARTIFACT-INVALID:{model_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_path = Path(training_config)
    project_root = Path.cwd()
    run_cfg = _yaml(train_cfg_path)
    opt_cfg_path = _resolve(project_root, run_cfg["optimization"]["config"])
    objective_cfg_path = _resolve(project_root, run_cfg["objective"]["config"])

    cache_path, cache = _load_cache_manifest(cache_manifest)
    if cache.cache_kind != "token_only":
        raise ValueError("MOE-CPT-REQUIRES-TOKEN-ONLY-CACHE")
    train_shards = _resolve_shards(cache_path, cache, "train")
    train_token_count = sum(f.tokens for f in cache.files if f.split == "train")
    if train_token_count <= 0:
        raise ValueError("TRAIN-CACHE-NO-TRAIN-TOKENS")

    optimization = OptimizationConfig.from_yaml(opt_cfg_path)
    requested_precision = optimization.precision
    parameter_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float32,
        "fp32": torch.float32,
    }[requested_precision]
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("TRAIN-CUDA-NOT-AVAILABLE")
        if requested_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("TRAIN-BF16-UNSUPPORTED")

    model, moe_config = load_taxmoe_model(model_path, device=device, torch_dtype=parameter_dtype)
    original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if gradient_checkpointing:
        enable_moe_gradient_checkpointing(model)

    balance_coefficient = _objective_balance_coefficient(objective_cfg_path, moe_config.balance_loss.coefficient)
    balance_config = moe_config.balance_loss.model_copy(update={"coefficient": balance_coefficient})

    token_raw = _yaml(_resolve(project_root, "configs/modeling/tokenization_v1.yaml"))
    token_cfg = TokenizationConfig.model_validate(token_raw)
    tokenizer = TaxMoETokenizer(token_cfg)
    pad_id = tokenizer.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("TRAIN-TOKENIZER-NO-PAD-OR-EOS")

    base_dataset = CachedTaxMoEDataset(train_shards)
    dataset = TokenBlockDataset(base_dataset, sequence_length=sequence_length)
    if len(dataset) == 0:
        raise ValueError("TRAIN-DATALOADER-EMPTY")
    if dataset.total_tokens != train_token_count:
        raise ValueError(f"TRAIN-TOKEN-BLOCK-CONSERVATION:{dataset.total_tokens}!={train_token_count}")
    collator = TaxMoETrainingCollator(pad_token_id=int(pad_id), include_ngram_features=False)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )

    budget = run_cfg.get("budget", {})
    max_passes = float(budget.get("max_effective_passes", 2.0))
    configured_max_tokens = budget.get("max_training_tokens")
    average_tokens = train_token_count / max(len(dataset), 1)
    approximate_tokens_per_step = max(
        average_tokens * microbatch_size * optimization.gradient_accumulation_steps, 1.0
    )
    estimated_steps = max(1, math.ceil((train_token_count * max_passes) / approximate_tokens_per_step))
    if configured_max_tokens:
        estimated_steps = min(
            estimated_steps,
            max(1, math.ceil(float(configured_max_tokens) / approximate_tokens_per_step)),
        )
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("TRAIN-MAX-STEPS-INVALID")
        estimated_steps = min(estimated_steps, max_steps)

    optimizer, parameter_groups = build_optimizer(model, optimization)
    scheduler = build_scheduler(optimizer, optimization, estimated_steps)

    init_hash = model_artifact_hash(model_path)
    objective_hash = _file_hash(objective_cfg_path)
    optimization_hash = stable_hash(optimization.model_dump(mode="json"))
    scheduler_hash = stable_hash(optimization.scheduler, estimated_steps, optimization.warmup_ratio)
    architecture_hash = stable_hash(
        moe_config.model_dump(mode="json"),
        sum(p.numel() for p in model.parameters()),
    )
    run_spec_hash = stable_hash(
        init_hash,
        cache.content_hash,
        objective_hash,
        optimization_hash,
        sequence_length,
        microbatch_size,
        gradient_checkpointing,
        seed,
        estimated_steps,
    )
    run_id = "RUN-MOE-" + run_spec_hash[:20]
    run_manifest = TrainingRunManifest(
        run_id=run_id,
        run_spec_hash=run_spec_hash,
        parent_model_fingerprint=init_hash,
        input_release_id=cache.cache_id,
        input_release_hash=cache.content_hash,
        data_manifest_hash=cache.content_hash,
        objective_hash=objective_hash,
        optimizer_config_hash=optimization_hash,
        scheduler_config_hash=scheduler_hash,
        precision=optimization.precision,
        status=RunStatus.RUNNING,
        metadata={
            "training_config": str(train_cfg_path),
            "init_artifact": str(model_path),
            "init_artifact_hash": init_hash,
            "cache_manifest": str(cache_path),
            "device": device,
            "microbatch_size": microbatch_size,
            "sequence_length": sequence_length,
            "gradient_checkpointing": gradient_checkpointing,
            "seed": seed,
            "planned_optimizer_steps": estimated_steps,
            "balance_coefficient": balance_coefficient,
            "parameter_groups": {"decay": len(parameter_groups[0]), "no_decay": len(parameter_groups[1])},
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
        },
    )
    (output_dir / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "moe_config.resolved.json").write_text(moe_config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "optimization.resolved.json").write_text(optimization.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "objective.resolved.json").write_text(json.dumps(_yaml(objective_cfg_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    trainer = DenseCPTTrainer(
        model,
        optimizer,
        scheduler,
        optimization,
        unique_train_tokens=train_token_count,
        device=device,
        adapter=MoECausalLMAdapter(balance_config),
    )
    progress = TrainingProgress()
    metrics_path = output_dir / "metrics.jsonl"
    iterator = iter(loader)
    last_metrics: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_file:
        for _ in range(estimated_steps):
            microbatches = []
            for _micro in range(optimization.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                microbatches.append(batch)
            progress, metrics = trainer.train_optimizer_step(microbatches, progress)
            metrics["optimizer_step"] = progress.optimizer_step
            metrics["effective_passes"] = progress.effective_passes
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            last_metrics = metrics
            if configured_max_tokens and progress.consumed_tokens >= int(configured_max_tokens):
                break
            if progress.effective_passes >= max_passes:
                break

    routing_health = _routing_health(last_metrics, sorted(moe_config.target_layers), moe_config.num_experts)
    routing_health_hash = stable_hash(routing_health)
    (output_dir / "routing_health.json").write_text(
        json.dumps(routing_health, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    final_model = output_dir / "final_model"
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    save_taxmoe_model(model, final_model, moe_config)
    tokenizer.tokenizer.save_pretrained(final_model)
    (output_dir / "progress.json").write_text(progress.model_dump_json(indent=2) + "\n", encoding="utf-8")

    completed = run_manifest.model_copy(update={"status": RunStatus.COMPLETED})
    (output_dir / "run_manifest.json").write_text(completed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    candidate_hash = model_artifact_hash(final_model)
    result = {
        "passed": True,
        "run_id": run_id,
        "output": str(output_dir),
        "final_model": str(final_model),
        "optimizer_steps": progress.optimizer_step,
        "consumed_tokens": progress.consumed_tokens,
        "effective_passes": progress.effective_passes,
        "sequence_length": sequence_length,
        "microbatch_size": microbatch_size,
        "candidate_weight_hash": candidate_hash,
        "init_artifact_hash": init_hash,
        "cache_id": cache.cache_id,
        "cache_hash": cache.content_hash,
        "optimization_config_hash": optimization_hash,
        "architecture_hash": architecture_hash,
        "routing_health_hash": routing_health_hash,
        "routing_health_passed": bool(routing_health["passed"]),
        "balance_coefficient": balance_coefficient,
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
