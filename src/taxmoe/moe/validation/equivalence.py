from __future__ import annotations

import copy
import gc

import torch

from taxmoe.moe.config import MoEUpcycleConfig
from taxmoe.moe.hashing import module_sha256
from taxmoe.moe.module import SparseMoE
from taxmoe.moe.upcycle import find_transformer_layers

from .models import LayerEquivalenceResult, NumericalComparison, UpcycleValidationReport


def compare_tensors(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> NumericalComparison:
    da = a.detach().float().cpu()
    db = b.detach().float().cpu()
    diff = (da - db).abs()
    denom = db.abs().clamp_min(1e-12)
    return NumericalComparison(
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
        max_rel=float((diff / denom).max().item()) if diff.numel() else 0.0,
        passed=bool(torch.allclose(da, db, atol=atol, rtol=rtol)),
    )


def validate_structure_and_copies(dense_model, moe_model, config: MoEUpcycleConfig) -> list[str]:
    failures = []
    dense_layers = find_transformer_layers(dense_model).layers
    moe_layers = find_transformer_layers(moe_model).layers
    for i in range(len(dense_layers)):
        if i in config.target_layers:
            if not isinstance(moe_layers[i].mlp, SparseMoE):
                failures.append(f"target-not-moe:{i}")
                continue
            if moe_layers[i].mlp.num_experts != config.num_experts:
                failures.append(f"expert-count:{i}")
            src_hash = module_sha256(dense_layers[i].mlp)
            for eid, expert in enumerate(moe_layers[i].mlp.experts.experts):
                if module_sha256(expert) != src_hash:
                    failures.append(f"expert-copy:{i}:{eid}")
        else:
            if isinstance(moe_layers[i].mlp, SparseMoE):
                failures.append(f"unexpected-moe:{i}")
            elif module_sha256(moe_layers[i].mlp) != module_sha256(dense_layers[i].mlp):
                failures.append(f"dense-layer-changed:{i}")
    return failures


def _fp32_cpu_copy(module):
    """Create an isolated FP32/CPU evaluation copy for mathematical equivalence checks.

    Dense -> MoE upcycling is intended to preserve the MLP function before
    training because every expert is an exact copy of the dense source MLP and
    the selected router weights sum to one. In BF16/FP16 runtime arithmetic,
    however, multiplying identical expert outputs by non-uniform routing weights
    and accumulating them introduces normal low-precision rounding. Comparing
    that path with FP32-level tolerances produces false failures.

    The structural/hash checks still verify the actual saved low-precision
    tensors. This isolated copy only tests the mathematical transformation.
    """
    return copy.deepcopy(module).to(device="cpu", dtype=torch.float32).eval()


def validate_local_equivalence(dense_model, moe_model, config: MoEUpcycleConfig, *, atol=1e-5, rtol=1e-5, tokens: int = 7):
    dense_layers = find_transformer_layers(dense_model).layers
    moe_layers = find_transformer_layers(moe_model).layers
    results = []
    hidden = getattr(getattr(dense_model, "config", None), "hidden_size", None)
    for i in config.target_layers:
        source = dense_layers[i].mlp
        target = moe_layers[i].mlp
        p = next(source.parameters())
        layer_hidden = int(hidden or p.shape[-1])

        # Validate the upcycling identity in FP32 so the result measures the
        # transformation itself rather than BF16/FP16 routing-rounding noise.
        source_ref = _fp32_cpu_copy(source)
        target_ref = _fp32_cpu_copy(target)
        x = torch.randn(2, tokens, layer_hidden, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            a = source_ref(x)
            b = target_ref(x)
        results.append(LayerEquivalenceResult(layer_index=i, comparison=compare_tensors(b, a, atol=atol, rtol=rtol)))

        # Release the 4-expert FP32 copy before validating the next layer.
        del source_ref, target_ref, x, a, b
        gc.collect()
    return results


def validate_full_model(
    dense_model,
    moe_model,
    batch: dict[str, torch.Tensor],
    *,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    loss_atol: float = 1e-5,
):
    """Compare the complete dense and initialized-MoE causal LMs on one batch.

    Callers should load both models in FP32 for this gate. Low-precision runtime
    drift is profiled separately; this check is intended to verify the actual
    architectural transformation end-to-end.
    """
    dense_model.eval()
    moe_model.eval()
    with torch.inference_mode():
        a = dense_model(**batch)
        b = moe_model(**batch)
    comp = compare_tensors(b.logits, a.logits, atol=atol, rtol=rtol)
    delta = None
    loss_passed = None
    if getattr(a, "loss", None) is not None and getattr(b, "loss", None) is not None:
        delta = abs(float(a.loss.detach().float().cpu()) - float(b.loss.detach().float().cpu()))
        loss_passed = bool(delta <= loss_atol)
    return comp, delta, loss_passed


def build_cache_validation_batch(cache_manifest: str, *, sequence_length: int = 128) -> dict[str, torch.Tensor]:
    """Build a deterministic real-CPT batch for full-model equivalence."""
    if sequence_length < 2:
        raise ValueError("MOE-EQUIVALENCE-SEQUENCE-LENGTH-INVALID")
    from pathlib import Path

    from taxmoe.data.cache_models import TrainingCacheManifest
    from taxmoe.data.cache_verifier import verify_cache
    from taxmoe.data.cached_dataset import CachedTaxMoEDataset

    manifest_path = Path(cache_manifest)
    manifest = TrainingCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    verify_cache(manifest, manifest_path.parent)
    preferred = [f for f in manifest.files if f.split == "validation"]
    shards = preferred or [f for f in manifest.files if f.split == "train"]
    if not shards:
        raise ValueError("MOE-EQUIVALENCE-CACHE-SPLIT-MISSING")
    dataset = CachedTaxMoEDataset([manifest_path.parent / f.path for f in shards])
    if len(dataset) == 0:
        raise ValueError("MOE-EQUIVALENCE-CACHE-EMPTY")
    example = dataset[0]
    n = min(len(example["input_ids"]), int(sequence_length))
    if n < 2:
        raise ValueError("MOE-EQUIVALENCE-EXAMPLE-TOO-SHORT")
    ids = torch.tensor(example["input_ids"][:n], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(example["labels"][:n], dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


def validate_upcycle(
    dense_model,
    moe_model,
    config: MoEUpcycleConfig,
    *,
    batch=None,
    atol=1e-5,
    rtol=1e-5,
    full_atol=1e-3,
    full_rtol=1e-4,
    loss_atol=1e-5,
) -> UpcycleValidationReport:
    failures = validate_structure_and_copies(dense_model, moe_model, config)
    local = validate_local_equivalence(dense_model, moe_model, config, atol=atol, rtol=rtol)
    failures.extend(f"local:{r.layer_index}" for r in local if not r.comparison.passed)
    full = None
    delta = None
    loss_passed = None
    if batch is not None:
        full, delta, loss_passed = validate_full_model(
            dense_model, moe_model, batch, atol=full_atol, rtol=full_rtol, loss_atol=loss_atol
        )
        if not full.passed:
            failures.append("full-logits")
        if loss_passed is None:
            failures.append("full-loss-missing")
        elif not loss_passed:
            failures.append("full-loss")
    return UpcycleValidationReport(
        target_layers=sorted(config.target_layers),
        structure_passed=not any(x.startswith(("target-", "unexpected-", "dense-layer")) for x in failures),
        expert_copy_passed=not any(x.startswith("expert-copy") for x in failures),
        local_equivalence=local,
        full_model_logits=full,
        lm_loss_delta=delta,
        lm_loss_tolerance=loss_atol if batch is not None else None,
        lm_loss_passed=loss_passed,
        passed=not failures,
        failures=failures,
    )
