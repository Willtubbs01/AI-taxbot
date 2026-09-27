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
from taxmoe.ingestion.hashing import sha256_file, stable_hash
from taxmoe.modeling.model_inspection import inspect_architecture
from taxmoe.modeling.model_loader import ModelLoadConfig, build_manifest, load_model
from taxmoe.modeling.tokenization_config import TokenizationConfig
from taxmoe.modeling.tokenizer import TaxMoETokenizer
from taxmoe.training.optimization import OptimizationConfig, build_optimizer, build_scheduler
from taxmoe.training.run_models import RunStatus, TrainingProgress, TrainingRunManifest
from taxmoe.training.trainer import DenseCPTTrainer


def _yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _file_config_hash(path: str | Path) -> str:
    path = Path(path)
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


def _candidate_weight_hash(model_dir: Path) -> str:
    files = []
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            files.append((str(p.relative_to(model_dir)).replace("\\", "/"), p.stat().st_size, sha256_file(p)))
    return stable_hash(files)


def run_dense_cpt(
    *,
    training_config: str | Path,
    cache_manifest: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    microbatch_size: int = 1,
    max_steps: int | None = None,
    gradient_checkpointing: bool = True,
    seed: int = 1701,
) -> dict[str, Any]:
    if microbatch_size < 1:
        raise ValueError("TRAIN-MICROBATCH-INVALID")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_path = Path(training_config)
    project_root = Path.cwd()
    run_cfg = _yaml(train_cfg_path)
    model_cfg_path = _resolve(project_root, run_cfg["model"]["config"])
    opt_cfg_path = _resolve(project_root, run_cfg["optimization"]["config"])
    objective_cfg_path = _resolve(project_root, run_cfg["objective"]["config"])

    cache_path, cache = _load_cache_manifest(cache_manifest)
    if cache.cache_kind != "token_only":
        raise ValueError("DENSE-CPT-REQUIRES-TOKEN-ONLY-CACHE")
    train_shards = _resolve_shards(cache_path, cache, "train")
    train_token_count = sum(f.tokens for f in cache.files if f.split == "train")
    if train_token_count <= 0:
        raise ValueError("TRAIN-CACHE-NO-TRAIN-TOKENS")

    optimization = OptimizationConfig.from_yaml(opt_cfg_path)

    # Full-parameter training precision must be resolved before loading weights.
    # BF16 can safely keep trainable parameters in BF16 without a GradScaler.
    # FP16 AMP, by contrast, requires FP32 master parameters for GradScaler.
    requested_precision = optimization.precision
    model_load = ModelLoadConfig.from_yaml(model_cfg_path)
    training_parameter_dtype = {
        "bf16": "bf16",
        "fp16": "fp32",
        "fp32": "fp32",
    }[requested_precision]
    model_load = model_load.model_copy(update={"dtype": training_parameter_dtype})

    if str(device).startswith("cuda") and requested_precision == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("TRAIN-CUDA-NOT-AVAILABLE")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "TRAIN-BF16-UNSUPPORTED: this CUDA device/runtime does not report BF16 support; "
                "use fp16 only with sufficient memory for FP32 master parameters, or use a BF16-capable CUDA runtime/device"
            )

    model, hf_config = load_model(model_load, device=device)
    original_use_cache = getattr(model.config, "use_cache", None)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    token_raw = _yaml(_resolve(project_root, "configs/modeling/tokenization_v1.yaml"))
    token_cfg = TokenizationConfig.model_validate(token_raw)
    tokenizer = TaxMoETokenizer(token_cfg)
    pad_id = tokenizer.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("TRAIN-TOKENIZER-NO-PAD-OR-EOS")

    dataset = CachedTaxMoEDataset(train_shards)
    collator = TaxMoETrainingCollator(pad_token_id=int(pad_id), include_ngram_features=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )
    if len(loader) == 0:
        raise ValueError("TRAIN-DATALOADER-EMPTY")

    budget = run_cfg.get("budget", {})
    max_passes = float(budget.get("max_effective_passes", 2.0))
    configured_max_tokens = budget.get("max_training_tokens")

    average_tokens = train_token_count / max(len(dataset), 1)
    approximate_tokens_per_step = max(average_tokens * microbatch_size * optimization.gradient_accumulation_steps, 1.0)
    estimated_steps = max(1, math.ceil((train_token_count * max_passes) / approximate_tokens_per_step))
    if configured_max_tokens:
        estimated_steps = min(estimated_steps, max(1, math.ceil(float(configured_max_tokens) / approximate_tokens_per_step)))
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("TRAIN-MAX-STEPS-INVALID")
        estimated_steps = min(estimated_steps, max_steps)

    optimizer, parameter_groups = build_optimizer(model, optimization)
    scheduler = build_scheduler(optimizer, optimization, estimated_steps)

    base_manifest, architecture_report = build_manifest(model_load, model, hf_config)
    objective_hash = _file_config_hash(objective_cfg_path)
    optimization_hash = stable_hash(optimization.model_dump(mode="json"))
    scheduler_hash = stable_hash(optimization.scheduler, estimated_steps, optimization.warmup_ratio)
    run_spec_hash = stable_hash(
        base_manifest.base_model_fingerprint,
        cache.content_hash,
        objective_hash,
        optimization_hash,
        microbatch_size,
        gradient_checkpointing,
        seed,
        estimated_steps,
    )
    run_id = "RUN-DENSE-" + run_spec_hash[:20]
    run_manifest = TrainingRunManifest(
        run_id=run_id,
        run_spec_hash=run_spec_hash,
        parent_model_fingerprint=base_manifest.base_model_fingerprint,
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
            "cache_manifest": str(cache_path),
            "device": device,
            "microbatch_size": microbatch_size,
            "gradient_checkpointing": gradient_checkpointing,
            "seed": seed,
            "planned_optimizer_steps": estimated_steps,
            "requested_precision": requested_precision,
            "model_parameter_dtype": training_parameter_dtype,
            "parameter_groups": {"decay": len(parameter_groups[0]), "no_decay": len(parameter_groups[1])},
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
        },
    )
    (output_dir / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "base_model_manifest.json").write_text(base_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "architecture_report.json").write_text(architecture_report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "optimization.resolved.json").write_text(optimization.model_dump_json(indent=2) + "\n", encoding="utf-8")

    trainer = DenseCPTTrainer(
        model,
        optimizer,
        scheduler,
        optimization,
        unique_train_tokens=train_token_count,
        device=device,
    )
    progress = TrainingProgress()
    metrics_path = output_dir / "metrics.jsonl"
    iterator = iter(loader)

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
            if configured_max_tokens and progress.consumed_tokens >= int(configured_max_tokens):
                break
            if progress.effective_passes >= max_passes:
                break

    final_model = output_dir / "final_model"
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    model.save_pretrained(final_model, safe_serialization=True)
    tokenizer.tokenizer.save_pretrained(final_model)
    progress_path = output_dir / "progress.json"
    progress_path.write_text(progress.model_dump_json(indent=2) + "\n", encoding="utf-8")

    completed = run_manifest.model_copy(update={"status": RunStatus.COMPLETED})
    (output_dir / "run_manifest.json").write_text(completed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    result = {
        "passed": True,
        "run_id": run_id,
        "output": str(output_dir),
        "final_model": str(final_model),
        "optimizer_steps": progress.optimizer_step,
        "consumed_tokens": progress.consumed_tokens,
        "effective_passes": progress.effective_passes,
        "candidate_weight_hash": _candidate_weight_hash(final_model),
        "cache_id": cache.cache_id,
        "cache_hash": cache.content_hash,
        "parent_model_fingerprint": base_manifest.base_model_fingerprint,
        "optimization_config_hash": optimization_hash,
        "architecture_hash": stable_hash(architecture_report.model_dump(mode="json")),
    }
    (output_dir / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
