from __future__ import annotations

import copy
import json
from pathlib import Path

import typer

app = typer.Typer(help="Stage 5/6 model inspection, MoE upcycling, validation, and release commands.")


def _require_local_parent_if_pathlike(value: str) -> None:
    p = Path(value)
    pathlike = value.startswith(".") or value.startswith("/") or "\\" in value or value.count("/") > 1
    if pathlike and not p.exists():
        raise typer.BadParameter(
            f"Local parent model directory does not exist: {value}. "
            "Build/train the parent first or provide a valid Hugging Face repo id."
        )



@app.command("inspect")
def inspect(config: str, output: str = "artifacts/models/base/QwenDense-0.6B"):
    from taxmoe.modeling.model_loader import ModelLoadConfig, build_manifest, load_model
    cfg = ModelLoadConfig.from_yaml(config)
    model, c = load_model(cfg)
    manifest, report = build_manifest(cfg, model, c)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / "model_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out / "architecture_report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"passed": True, "base_model_fingerprint": manifest.base_model_fingerprint, "output": str(out)}, indent=2))


@app.command("upcycle")
def upcycle(parent: str, config: str, output: str = "artifacts/models/upcycled/TaxMoE-4E-Top2-Init-v0.1", device: str = "cpu"):
    """Convert a local/HF dense causal LM into the Stage 6 exact-copy sparse MoE init artifact."""
    from transformers import AutoModelForCausalLM
    from taxmoe.moe.config import MoEUpcycleConfig
    from taxmoe.moe.upcycle import upcycle_model
    from taxmoe.modeling.taxmoe_loader import save_taxmoe_model

    _require_local_parent_if_pathlike(parent)
    cfg = MoEUpcycleConfig.from_yaml(config)
    model = AutoModelForCausalLM.from_pretrained(parent)
    model.to(device)
    report = upcycle_model(model, cfg)
    out = save_taxmoe_model(model, output, cfg, report=report)
    typer.echo(json.dumps({"passed": True, "output": str(out), "architecture_hash": report.architecture_hash, "parameter_count": report.converted_parameter_count}, indent=2))


@app.command("inspect-moe")
def inspect_moe(path: str):
    from taxmoe.modeling.taxmoe_loader import load_taxmoe_model
    from taxmoe.moe.upcycle import iter_sparse_moe
    model, cfg = load_taxmoe_model(path)
    modules = sorted(iter_sparse_moe(model), key=lambda x: x.layer_index)
    payload = {
        "passed": True,
        "target_layers": [m.layer_index for m in modules],
        "num_moe_layers": len(modules),
        "num_experts": cfg.num_experts,
        "moe_top_k": cfg.top_k,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "router_parameter_count": sum(p.numel() for m in modules for p in m.router.parameters()),
        "expert_parameter_count": sum(p.numel() for m in modules for p in m.experts.parameters()),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("validate-upcycle")
def validate_upcycle(
    parent: str,
    moe_path: str,
    atol: float = typer.Option(1e-5, "--atol"),
    rtol: float = typer.Option(1e-5, "--rtol"),
    cache_manifest: str | None = typer.Option(None, "--cache-manifest", help="When supplied, run full-model logits + LM-loss equivalence on a real CPT cache example."),
    sequence_length: int = typer.Option(128, "--sequence-length", min=2, help="Maximum real-CPT sequence length for the full-model gate."),
    full_atol: float = typer.Option(1e-3, "--full-atol"),
    full_rtol: float = typer.Option(1e-4, "--full-rtol"),
    loss_atol: float = typer.Option(1e-5, "--loss-atol"),
):
    """Validate dense->MoE structure, expert copies, local math, and optionally full-model equivalence."""
    import torch
    from transformers import AutoModelForCausalLM
    from taxmoe.modeling.taxmoe_loader import load_taxmoe_model
    from taxmoe.moe.validation import build_cache_validation_batch, validate_upcycle as run_validation

    _require_local_parent_if_pathlike(parent)
    # Full-model equivalence is a mathematical gate, so use FP32/CPU to avoid
    # BF16 routing-rounding noise and to keep the 8GB training GPU out of the
    # comparison. The local-only path retains the lighter default loading.
    if cache_manifest:
        dense = AutoModelForCausalLM.from_pretrained(parent, dtype=torch.float32)
        moe, cfg = load_taxmoe_model(moe_path, torch_dtype=torch.float32)
        batch = build_cache_validation_batch(cache_manifest, sequence_length=sequence_length)
    else:
        dense = AutoModelForCausalLM.from_pretrained(parent)
        moe, cfg = load_taxmoe_model(moe_path)
        batch = None
    report = run_validation(
        dense,
        moe,
        cfg,
        batch=batch,
        atol=atol,
        rtol=rtol,
        full_atol=full_atol,
        full_rtol=full_rtol,
        loss_atol=loss_atol,
    )
    out = Path(moe_path) / "equivalence_report.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(1)


@app.command("freeze-taxdense")
def freeze_taxdense(
    run_dir: str = typer.Argument(..., help="Completed Stage 5 dense run directory containing final_model/."),
    releases_root: str = typer.Option("artifacts/models/releases", "--releases-root"),
    version: str = typer.Option("0.1", "--version"),
    evaluation_hash: str | None = typer.Option(None, "--evaluation-hash"),
    reproducibility_hash: str | None = typer.Option(None, "--reproducibility-hash"),
    allow_unvalidated: bool = typer.Option(False, "--allow-unvalidated", help="Development only: permit a release before formal Stage 5 evaluation/reproducibility evidence exists."),
):
    """Freeze a completed Stage 5 dense candidate into TaxDense-0.6B-v<version>."""
    from taxmoe.data.cache_models import TrainingCacheManifest
    from taxmoe.ingestion.hashing import stable_hash
    from taxmoe.model_releases.freezer import TaxDenseFreezer
    from taxmoe.model_releases.models import TaxDenseReleaseManifest
    from taxmoe.modeling.model_manifest import BaseModelManifest
    from taxmoe.training.run_models import TrainingRunManifest

    root = Path(run_dir)
    candidate = root / "final_model"
    if not candidate.exists():
        raise typer.BadParameter(f"Dense candidate model directory is missing: {candidate}")
    required = [root / "run_manifest.json", root / "base_model_manifest.json", root / "training_result.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise typer.BadParameter(f"Dense run is incomplete; missing: {', '.join(missing)}")
    if (not evaluation_hash or not reproducibility_hash) and not allow_unvalidated:
        raise typer.BadParameter(
            "Formal freeze requires --evaluation-hash and --reproducibility-hash. "
            "For a development/smoke parent only, pass --allow-unvalidated."
        )

    run = TrainingRunManifest.model_validate_json((root / "run_manifest.json").read_text(encoding="utf-8"))
    base = BaseModelManifest.model_validate_json((root / "base_model_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "training_result.json").read_text(encoding="utf-8"))
    cache_path = Path(run.metadata.get("cache_manifest", ""))
    if not cache_path.exists():
        raise typer.BadParameter(f"Referenced Stage 4 cache manifest is missing: {cache_path}")
    cache = TrainingCacheManifest.model_validate_json(cache_path.read_text(encoding="utf-8"))

    evaluation_hash = evaluation_hash or stable_hash("UNVALIDATED-DEVELOPMENT", "evaluation")
    reproducibility_hash = reproducibility_hash or stable_hash("UNVALIDATED-DEVELOPMENT", "reproducibility")
    release_id = "MODELRELEASE-TAXDENSE-" + stable_hash(run.run_id, result["candidate_weight_hash"], version)[:20]
    manifest = TaxDenseReleaseManifest(
        release_id=release_id,
        version=version,
        architecture=base.architecture,
        parent_model_id=base.model_id,
        parent_model_revision=base.revision,
        parent_model_fingerprint=base.base_model_fingerprint,
        tokenizer_manifest_hash=cache.tokenizer_manifest_hash,
        input_release_id=cache.cache_id,
        input_release_hash=cache.content_hash,
        source_run_id=run.run_id,
        source_checkpoint_id=f"FINAL-STEP-{result['optimizer_steps']}",
        source_checkpoint_hash=result["candidate_weight_hash"],
        training_data_manifest_hash=cache.content_hash,
        objective_config_hash=run.objective_hash,
        optimization_config_hash=run.optimizer_config_hash,
        evaluation_hash=evaluation_hash,
        reproducibility_hash=reproducibility_hash,
        parameter_count=base.parameter_count,
        architecture_hash=result["architecture_hash"],
        weight_manifest_hash=result["candidate_weight_hash"],
    )
    destination, frozen = TaxDenseFreezer().freeze(candidate, releases_root, manifest)
    if allow_unvalidated:
        warning = destination / "UNVALIDATED_DEVELOPMENT_RELEASE.txt"
        warning.write_text(
            "This TaxDense release was created with --allow-unvalidated for development/smoke use.\n"
            "Do not treat it as the final Stage 5 frozen model until evaluation and reproducibility gates pass.\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps({
        "passed": True,
        "release_path": str(destination),
        "release_id": frozen.release_id,
        "content_hash": frozen.release_content_hash,
        "validated": not allow_unvalidated,
    }, indent=2))


@app.command("freeze-taxmoe")
def freeze_taxmoe(
    run_dir: str = typer.Argument(..., help="Completed Stage 6 MoE CPT run directory containing final_model/."),
    releases_root: str = typer.Option("artifacts/models/releases", "--releases-root"),
    version: str = typer.Option("0.1", "--version"),
    taxdense_parent: str | None = typer.Option(None, "--taxdense-parent", help="Frozen TaxDense parent release. Defaults to the parent_release in the Stage 6 training config."),
    hardware_profile: str | None = typer.Option(None, "--hardware-profile", help="Passing Stage 6 benchmark JSON (normally the 1024-token RTX 3060 Ti case)."),
    evaluation_evidence: str | None = typer.Option(None, "--evaluation-evidence", help="Passing post-training evaluation JSON/report."),
    reproducibility_evidence: str | None = typer.Option(None, "--reproducibility-evidence", help="Reproducibility evidence file/report."),
    allow_unvalidated: bool = typer.Option(False, "--allow-unvalidated", help="Development only: permit freeze without formal hardware/evaluation/reproducibility evidence."),
):
    """Freeze a completed Stage 6 MoE candidate into TaxMoE-4E-Top2-v<version>."""
    import tempfile
    import yaml

    from taxmoe.ingestion.hashing import sha256_file, stable_hash
    from taxmoe.model_releases.moe import TaxMoEReleaseManifest
    from taxmoe.model_releases.moe_freezer import TaxMoEFreezer
    from taxmoe.model_releases.verifier import verify_taxdense_release
    from taxmoe.moe.config import MoEUpcycleConfig
    from taxmoe.moe.manifests import MoEUpcycleReport
    from taxmoe.model_releases.artifact_hashing import model_artifact_hash
    from taxmoe.training.run_models import RunStatus, TrainingRunManifest

    root = Path(run_dir)
    candidate = root / "final_model"
    required = [
        candidate / "moe_config.json",
        root / "run_manifest.json",
        root / "training_result.json",
        root / "routing_health.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise typer.BadParameter(f"Stage 6 run is incomplete; missing: {', '.join(missing)}")

    run = TrainingRunManifest.model_validate_json((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run.status != RunStatus.COMPLETED:
        raise typer.BadParameter(f"Stage 6 run is not completed: {run.status}")
    result = json.loads((root / "training_result.json").read_text(encoding="utf-8"))
    routing = json.loads((root / "routing_health.json").read_text(encoding="utf-8"))
    if not bool(result.get("passed")):
        raise typer.BadParameter("Stage 6 training_result.json does not report passed=true")
    if not bool(result.get("routing_health_passed")) or not bool(routing.get("passed")):
        raise typer.BadParameter("Stage 6 routing-health gate did not pass")

    candidate_hash = model_artifact_hash(candidate)
    if candidate_hash != result.get("candidate_weight_hash"):
        raise typer.BadParameter(
            "Stage 6 candidate model hash does not match training_result.json; "
            "do not freeze a modified checkpoint."
        )
    routing_hash = stable_hash(routing)
    if routing_hash != result.get("routing_health_hash"):
        raise typer.BadParameter("routing_health.json hash does not match training_result.json")

    training_config_path = Path(str(run.metadata.get("training_config", "")))
    if not training_config_path.is_absolute():
        training_config_path = Path.cwd() / training_config_path
    if not training_config_path.exists():
        raise typer.BadParameter(f"Stage 6 training config is missing: {training_config_path}")
    training_config_raw = yaml.safe_load(training_config_path.read_text(encoding="utf-8")) or {}
    training_config_hash = stable_hash(training_config_raw)

    if taxdense_parent is None:
        configured_parent = training_config_raw.get("model", {}).get("parent_release")
        if not configured_parent:
            raise typer.BadParameter("No TaxDense parent was supplied and the training config has no model.parent_release")
        taxdense_parent = str(configured_parent)
    parent_manifest = verify_taxdense_release(taxdense_parent)
    parent_unvalidated = Path(taxdense_parent) / "UNVALIDATED_DEVELOPMENT_RELEASE.txt"
    if parent_unvalidated.exists() and not allow_unvalidated:
        raise typer.BadParameter(
            "Formal TaxMoE freeze cannot use an unvalidated-development TaxDense parent. "
            "Freeze a formally validated TaxDense parent first, or use --allow-unvalidated for Stage 7 development."
        )

    init_artifact = Path(str(run.metadata.get("init_artifact", "")))
    if not init_artifact.is_absolute():
        init_artifact = Path.cwd() / init_artifact
    if not init_artifact.exists():
        raise typer.BadParameter(f"Referenced MoE init artifact is missing: {init_artifact}")
    init_hash = model_artifact_hash(init_artifact)
    if init_hash != result.get("init_artifact_hash") or init_hash != run.metadata.get("init_artifact_hash"):
        raise typer.BadParameter("MoE init artifact hash does not match Stage 6 run lineage")
    upcycle_report_path = init_artifact / "upcycle_report.json"
    if not upcycle_report_path.exists():
        raise typer.BadParameter(f"MoE init upcycle report is missing: {upcycle_report_path}")
    upcycle_report = MoEUpcycleReport.model_validate_json(upcycle_report_path.read_text(encoding="utf-8"))
    if not upcycle_report.passed:
        raise typer.BadParameter("MoE init upcycle report did not pass")
    init_equivalence = init_artifact / "equivalence_report.json"
    if init_equivalence.exists():
        equivalence_payload = json.loads(init_equivalence.read_text(encoding="utf-8"))
        if not bool(equivalence_payload.get("passed")):
            raise typer.BadParameter("MoE init equivalence report did not pass")
        if equivalence_payload.get("full_model_logits") is None or equivalence_payload.get("lm_loss_passed") is not True:
            if not allow_unvalidated:
                raise typer.BadParameter("Formal freeze requires the full-model logits + LM-loss equivalence gate")
    elif not allow_unvalidated:
        raise typer.BadParameter(f"Formal freeze requires init equivalence evidence: {init_equivalence}")

    moe_cfg = MoEUpcycleConfig.model_validate_json((candidate / "moe_config.json").read_text(encoding="utf-8"))
    if sorted(moe_cfg.target_layers) != sorted(upcycle_report.target_layers):
        raise typer.BadParameter("Candidate MoE target layers do not match init upcycle report")
    if moe_cfg.num_experts != upcycle_report.num_experts or moe_cfg.top_k != upcycle_report.top_k:
        raise typer.BadParameter("Candidate MoE expert/top-k configuration does not match init upcycle report")

    evidence_inputs = {
        "hardware": Path(hardware_profile) if hardware_profile else None,
        "evaluation": Path(evaluation_evidence) if evaluation_evidence else None,
        "reproducibility": Path(reproducibility_evidence) if reproducibility_evidence else None,
    }
    for name, path in evidence_inputs.items():
        if path is not None and not path.is_file():
            raise typer.BadParameter(f"{name.title()} evidence file does not exist: {path}")
        if path is not None and path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(f"{name.title()} evidence JSON is invalid: {path}") from exc
            if payload.get("passed") is False or payload.get("status") in {"error", "oom", "numerical_failure", "unsupported"}:
                raise typer.BadParameter(f"{name.title()} evidence does not report a passing result: {path}")

    missing_evidence = [name for name, path in evidence_inputs.items() if path is None]
    if missing_evidence and not allow_unvalidated:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing_evidence)
        raise typer.BadParameter(
            "Formal TaxMoE freeze requires hardware, post-training evaluation, and reproducibility evidence. "
            f"Missing: {flags}. For Stage 7 development only, pass --allow-unvalidated."
        )

    evidence_hashes = {}
    for name, path in evidence_inputs.items():
        if path is not None:
            evidence_hashes[name] = sha256_file(path)
        else:
            evidence_hashes[name] = stable_hash("UNVALIDATED-DEVELOPMENT", name, run.run_id, version)

    release_id = "MODELRELEASE-TAXMOE-" + stable_hash(
        run.run_id, candidate_hash, routing_hash, version
    )[:20]
    manifest = TaxMoEReleaseManifest(
        release_id=release_id,
        version=version,
        taxdense_parent_id=parent_manifest.release_id,
        taxdense_parent_hash=parent_manifest.release_content_hash,
        moe_init_id=init_artifact.name,
        moe_init_hash=init_hash,
        input_release_id=run.input_release_id,
        input_release_hash=run.input_release_hash,
        source_run_id=run.run_id,
        source_checkpoint_id=f"FINAL-STEP-{result['optimizer_steps']}",
        source_checkpoint_hash=candidate_hash,
        num_experts=moe_cfg.num_experts,
        moe_top_k=moe_cfg.top_k,
        target_layers=sorted(moe_cfg.target_layers),
        router_type="hidden_state",
        dispatch_semantics_version=moe_cfg.dispatch.backend,
        balance_loss_version=moe_cfg.balance_loss.version,
        training_config_hash=training_config_hash,
        optimization_config_hash=result["optimization_config_hash"],
        hardware_envelope_hash=evidence_hashes["hardware"],
        evaluation_hash=evidence_hashes["evaluation"],
        routing_health_hash=routing_hash,
        reproducibility_hash=evidence_hashes["reproducibility"],
        parameter_count=upcycle_report.converted_parameter_count,
        architecture_hash=result["architecture_hash"],
        weight_manifest_hash=candidate_hash,
    )

    with tempfile.TemporaryDirectory(prefix="taxmoe-release-") as temp_dir:
        temp = Path(temp_dir)
        evidence_summary = temp / "release_evidence.json"
        evidence_summary.write_text(
            json.dumps(
                {
                    "release_mode": "development_unvalidated" if allow_unvalidated else "formal",
                    "source_run_id": run.run_id,
                    "source_checkpoint_hash": candidate_hash,
                    "routing_health_hash": routing_hash,
                    "hardware_envelope_hash": evidence_hashes["hardware"],
                    "evaluation_hash": evidence_hashes["evaluation"],
                    "reproducibility_hash": evidence_hashes["reproducibility"],
                    "missing_formal_evidence": missing_evidence,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        evidence_files = {
            "evidence/run_manifest.json": root / "run_manifest.json",
            "evidence/training_result.json": root / "training_result.json",
            "evidence/routing_health.json": root / "routing_health.json",
            "evidence/release_evidence.json": evidence_summary,
            "evidence/init_upcycle_report.json": upcycle_report_path,
        }
        for name in ("optimization.resolved.json", "objective.resolved.json", "moe_config.resolved.json"):
            path = root / name
            if path.exists():
                evidence_files[f"evidence/{name}"] = path
        if init_equivalence.exists():
            evidence_files["evidence/init_equivalence_report.json"] = init_equivalence
        for name, path in evidence_inputs.items():
            if path is not None:
                evidence_files[f"evidence/{name}{path.suffix or '.dat'}"] = path
        if allow_unvalidated:
            warning = temp / "UNVALIDATED_DEVELOPMENT_RELEASE.txt"
            warning.write_text(
                "This TaxMoE release was created with --allow-unvalidated for Stage 7 development.\n"
                "It is not the final formal Stage 6 release until hardware, post-training evaluation, "
                "and reproducibility evidence are supplied and the model is frozen again without that flag.\n",
                encoding="utf-8",
            )
            evidence_files["UNVALIDATED_DEVELOPMENT_RELEASE.txt"] = warning

        destination, frozen = TaxMoEFreezer().freeze(
            candidate,
            releases_root,
            manifest,
            evidence_files=evidence_files,
        )

    typer.echo(json.dumps({
        "passed": True,
        "release_path": str(destination),
        "release_id": frozen.release_id,
        "content_hash": frozen.release_content_hash,
        "validated": not allow_unvalidated,
        "routing_health_passed": True,
        "missing_formal_evidence": missing_evidence,
    }, indent=2))


@app.command("verify-release")
def verify_release(path: str):
    from taxmoe.model_releases.verifier import verify_taxdense_release
    m = verify_taxdense_release(path)
    typer.echo(json.dumps({"passed": True, "release_id": m.release_id, "content_hash": m.release_content_hash}, indent=2))


@app.command("verify-moe-release")
def verify_moe_release(path: str):
    from taxmoe.model_releases.verifier import verify_taxmoe_release
    m = verify_taxmoe_release(path)
    typer.echo(json.dumps({"passed": True, "release_id": m.release_id, "content_hash": m.release_content_hash}, indent=2))
