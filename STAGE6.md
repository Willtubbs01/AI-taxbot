# Stage 6 — Qwen → MoE Upcycling

This project bundle includes the Stage 6 implementation for the hidden-state-routed `TaxMoE-4E-Top2` baseline.

## Added runtime components

- Exact-copy `ExpertBank` construction with storage-alias checks.
- Deterministic per-layer hidden-state routers.
- FP32 router logits and normalized Top-k selected weights.
- Native vectorized sparse expert dispatch with no token dropping/capacity limit.
- Switch-style load-balancing loss over `attention_mask`-valid tokens.
- Dense → MoE model upcycling and conversion manifests.
- MoE save/load helpers that rebuild the sparse skeleton before strict state loading.
- MoE training adapter that adds `lambda * balance_loss` while keeping Qwen's own LM loss semantics unchanged.
- Non-reentrant gradient-checkpointing helper for MoE training.
- Upcycling/equivalence validation helpers.
- MoE profiling runner and parameter-inventory helpers.
- Stage 6 release manifest/freezer/verifier types.
- CLI commands and Stage 6 configs.

## Main Stage 6 configs

- `configs/modeling/moe_upcycle_v1.yaml`
- `configs/training/moe_cpt_objective_v1.yaml`
- `configs/training/moe_optimizer_v1.yaml`
- `configs/training/taxmoe_v1.yaml`
- `configs/benchmarks/rtx3060ti_moe_v1.yaml`
- `configs/validation/moe_equivalence_v1.yaml`

## Useful commands

Validate the MoE configuration:

```powershell
taxmoe train check-moe configs/modeling/moe_upcycle_v1.yaml
```

Upcycle a local dense/TaxDense Hugging Face checkpoint:

```powershell
taxmoe model upcycle artifacts/models/releases/TaxDense-0.6B-v0.1 configs/modeling/moe_upcycle_v1.yaml --output artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1 --device cpu
```

Inspect the resulting sparse artifact:

```powershell
taxmoe model inspect-moe artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1
```

Validate the conversion against the dense parent:

```powershell
taxmoe model validate-upcycle artifacts/models/releases/TaxDense-0.6B-v0.1 artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1
```

Run one GPU profiling case:

```powershell
taxmoe benchmark moe-case artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1 --sequence-length 512 --microbatch-size 1 --precision fp16 --gradient-checkpointing --optimizer-config configs/training/moe_optimizer_v1.yaml
```

Run tests:

```powershell
python -m pytest
```

## Important production gate

The Stage 6 code and freeze/release machinery are present, but the final `TaxMoE-4E-Top2-v0.1` release should only be frozen after the real RTX 3060 Ti profiling, continued MoE training, routing-health review, and candidate evaluation are complete.

## Added prerequisite execution pipeline

If `TaxDense-0.6B-v0.1` does not yet exist, follow `PIPELINE_RUNBOOK.md`. The bundle now includes:

- `taxmoe inputs build` for verified IRS PDF → CPT records → Qwen token-only Arrow cache.
- `taxmoe train dense` for Stage 5 dense continued pretraining.
- `taxmoe model freeze-taxdense` for promoting the dense candidate.
- A clear error when `taxmoe model upcycle` is given a missing local parent path.


## Full-model gate and executable MoE training

Stage 6 now includes an end-to-end full-model equivalence gate and an executable MoE CPT runner. Supplying `--cache-manifest` to `taxmoe model validate-upcycle` compares full logits and LM loss on a deterministic real CPT example in FP32/CPU. `taxmoe train moe` trains the sparse checkpoint with the canonical `LM + 0.01 * balance` objective, preserves all source tokens through deterministic sequence blocking, and records routing-health evidence for each selected layer/expert.


## Final Stage 6 freeze

`taxmoe model freeze-taxmoe` freezes a completed `taxmoe train moe` run into `TaxMoE-4E-Top2-v0.1`. The command verifies the completed run, candidate weight identity, init-artifact lineage, frozen TaxDense parent, routing-health hash, architecture, and MoE configuration before copying the model.

A formal release requires hardware, post-training evaluation, and reproducibility evidence. Those evidence files are copied into `evidence/` inside the release and are covered by the release manifest hash. `--allow-unvalidated` is available only for an explicit Stage 7 development parent and writes an `UNVALIDATED_DEVELOPMENT_RELEASE.txt` marker.

```powershell
taxmoe model freeze-taxmoe artifacts/training/RUN-MOE-CPT-v0.1 --version 0.1 --hardware-profile artifacts/training/RUN-MOE-CPT-v0.1/hardware_profile_1024.json --allow-unvalidated

taxmoe model verify-moe-release artifacts/models/releases/TaxMoE-4E-Top2-v0.1
```
