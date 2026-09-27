# TaxMoE executable pipeline: CPT → TaxDense → MoE

This runbook covers the missing executable bridge that was added after Stage 6 integration.

## 0. Refresh the editable install

The CPT extractor now uses `pypdf`, which is declared in `pyproject.toml`.

```powershell
pip install -e .
```

## 1. Build the Stage 4 authoritative CPT token cache

This command verifies every PDF SHA-256 in `sources/registry/source_manifest_2025.yaml`, extracts text page-by-page with `pypdf`, creates deterministic CPT records, tokenizes them with the pinned Qwen tokenizer, and writes Arrow train/validation shards.

```powershell
taxmoe inputs build `
  --source-manifest sources/registry/source_manifest_2025.yaml `
  --tokenization-config configs/modeling/tokenization_v1.yaml `
  --output artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1
```

Verify the cache:

```powershell
taxmoe inputs verify-cache `
  artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json
```

The current registry contains the 2025 W-2, 1040, 1099-B, Form 8949, and Schedule D PDFs. This is sufficient to exercise the pipeline, but it is still a small CPT corpus; add IRS instructions/publications before treating the resulting TaxDense model as a production-quality tax foundation.

## 2. Run a one-step Stage 5 smoke training test

```powershell
taxmoe train dense `
  artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json `
  --config configs/training/taxdense_v1.yaml `
  --output artifacts/training/RUN-DENSE-SMOKE `
  --device cuda `
  --microbatch-size 1 `
  --max-steps 1 `
  --gradient-checkpointing
```

If this passes, delete or keep the smoke run separately. Do not freeze it as the canonical TaxDense model.

## 3. Run the Stage 5 dense CPT recipe

With the currently small registered CPT corpus, the default training budget is at most two effective passes.

```powershell
taxmoe train dense `
  artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json `
  --config configs/training/taxdense_v1.yaml `
  --output artifacts/training/RUN-DENSE-CPT-v0.1 `
  --device cuda `
  --microbatch-size 1 `
  --gradient-checkpointing
```

The run writes an HF-compatible candidate to:

```text
artifacts/training/RUN-DENSE-CPT-v0.1/final_model/
```

## 4. Freeze/promote TaxDense

A formal freeze expects real Stage 5 evaluation and reproducibility evidence. For development-only Stage 6 wiring, you can explicitly opt into an unvalidated development release:

```powershell
taxmoe model freeze-taxdense `
  artifacts/training/RUN-DENSE-CPT-v0.1 `
  --version 0.1 `
  --allow-unvalidated
```

This creates:

```text
artifacts/models/releases/TaxDense-0.6B-v0.1/
```

The directory contains `UNVALIDATED_DEVELOPMENT_RELEASE.txt` when created with `--allow-unvalidated`. Replace that development release later with a properly evaluated/reproducible release rather than treating it as canonical evidence.

## 5. Stage 6 upcycling

```powershell
taxmoe model upcycle `
  artifacts/models/releases/TaxDense-0.6B-v0.1 `
  configs/modeling/moe_upcycle_v1.yaml `
  --output artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1 `
  --device cpu
```

Inspect and validate:

```powershell
taxmoe model inspect-moe `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1

taxmoe model validate-upcycle `
  artifacts/models/releases/TaxDense-0.6B-v0.1 `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1
```

## 6. Tests

```powershell
python -m pytest
```

The updated bundle passes 49 tests in the build environment.

## Precision note for Stage 5

The default dense CPT recipe uses BF16 full-parameter training on CUDA. The runner checks `torch.cuda.is_bf16_supported()` before starting. This avoids combining FP16 trainable parameters with PyTorch GradScaler, which is invalid because GradScaler requires FP32 master parameters for FP16 AMP. If `TRAIN-BF16-UNSUPPORTED` is reported, do not simply switch the model back to FP16 parameters; profile an FP32-master FP16 run or use a BF16-capable CUDA environment/device.

## Stage 6 gradient-checkpointing profile fix

For MoE profiling on RTX 3060 Ti, use BF16 with non-reentrant gradient checkpointing:

```powershell
taxmoe benchmark moe-case `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1-rebuild2 `
  --sequence-length 256 `
  --microbatch-size 1 `
  --precision bf16 `
  --gradient-checkpointing `
  --optimizer-config configs/training/moe_optimizer_v1.yaml
```

The MoE forward keeps router tensor operations graph-stable between the original forward and checkpoint recomputation, while auxiliary balance statistics are stored only on the original training forward. The benchmark loads BF16 parameters for BF16 profiling and FP32 master parameters for FP16 AMP profiling.

## Stage 6 — Full-model equivalence gate

After the initialized MoE passes local equivalence, run the complete Dense-vs-MoE gate on a real Stage 4 CPT example. The command loads both models in FP32 on CPU so the result measures the architectural transformation rather than BF16 routing-rounding noise.

```powershell
taxmoe model validate-upcycle `
  artifacts/models/releases/TaxDense-0.6B-v0.1 `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1-rebuild3 `
  --cache-manifest artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json `
  --sequence-length 128
```

The release gate requires `full_model_logits.passed: true`, `lm_loss_passed: true`, and an empty `failures` list.

## Stage 6 — MoE CPT training

The RTX 3060 Ti profile established the practical starting envelope at BF16, sequence length 1024, microbatch 1, and non-reentrant gradient checkpointing. Long cached CPT records are split into deterministic consecutive 1024-token blocks; tails are preserved rather than truncated.

Run a one-step smoke test first:

```powershell
taxmoe train moe `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1-rebuild3 `
  artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json `
  --config configs/training/taxmoe_v1.yaml `
  --output artifacts/training/RUN-MOE-SMOKE `
  --device cuda `
  --microbatch-size 1 `
  --sequence-length 1024 `
  --max-steps 1 `
  --gradient-checkpointing
```

If that passes, run the Stage 6 CPT candidate:

```powershell
taxmoe train moe `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1-rebuild3 `
  artifacts/model_inputs/TaxMoE-Inputs-Qwen3-v0.1/cache_manifest.json `
  --config configs/training/taxmoe_v1.yaml `
  --output artifacts/training/RUN-MOE-CPT-v0.1 `
  --device cuda `
  --microbatch-size 1 `
  --sequence-length 1024 `
  --gradient-checkpointing
```

The run writes `metrics.jsonl`, `routing_health.json`, `training_result.json`, and an HF-compatible sparse `final_model/` containing `moe_config.json`. Router metrics include per-layer balance loss, entropy, assignment fraction, probability mass, and top-1 share for every expert.


## Stage 6 — Freeze the trained TaxMoE release

After the full-model equivalence gate, the full `RUN-MOE-CPT-v0.1` training run, routing-health review, and checkpoint reload all pass, freeze the trained sparse model.

### Capture the 1024-token hardware evidence

The benchmark command can now write its JSON result directly to a file:

```powershell
taxmoe benchmark moe-case `
  artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1-rebuild3 `
  --sequence-length 1024 `
  --microbatch-size 1 `
  --precision bf16 `
  --gradient-checkpointing `
  --optimizer-config configs/training/moe_optimizer_v1.yaml `
  --output artifacts/training/RUN-MOE-CPT-v0.1/hardware_profile_1024.json
```

A formal freeze also requires a post-training evaluation evidence file and a reproducibility evidence file. The freezer hashes those files and embeds copies inside the frozen release.

```powershell
taxmoe model freeze-taxmoe `
  artifacts/training/RUN-MOE-CPT-v0.1 `
  --version 0.1 `
  --hardware-profile artifacts/training/RUN-MOE-CPT-v0.1/hardware_profile_1024.json `
  --evaluation-evidence artifacts/training/RUN-MOE-CPT-v0.1/evaluation_report.json `
  --reproducibility-evidence artifacts/training/RUN-MOE-CPT-v0.1/reproducibility_report.json
```

For Stage 7 development only, when formal evaluation/reproducibility evidence has not yet been produced, explicitly create a development release:

```powershell
taxmoe model freeze-taxmoe `
  artifacts/training/RUN-MOE-CPT-v0.1 `
  --version 0.1 `
  --hardware-profile artifacts/training/RUN-MOE-CPT-v0.1/hardware_profile_1024.json `
  --allow-unvalidated
```

This writes `UNVALIDATED_DEVELOPMENT_RELEASE.txt` into the release and records which formal evidence is missing. It creates:

```text
artifacts/models/releases/TaxMoE-4E-Top2-v0.1/
```

Verify the frozen release before using it as a Stage 7 parent:

```powershell
taxmoe model verify-moe-release `
  artifacts/models/releases/TaxMoE-4E-Top2-v0.1
```

The verifier checks the frozen manifest hash and every embedded model/evidence file hash.
