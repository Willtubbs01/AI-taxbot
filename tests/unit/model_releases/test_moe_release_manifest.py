from taxmoe.model_releases.moe import TaxMoEReleaseManifest


def test_taxmoe_release_manifest_tracks_stage6_identity():
    m = TaxMoEReleaseManifest(
        release_id="R1", taxdense_parent_id="D", taxdense_parent_hash="dh",
        moe_init_id="I", moe_init_hash="ih", input_release_id="IN", input_release_hash="inh",
        source_run_id="RUN", source_checkpoint_id="C", source_checkpoint_hash="ch",
        training_config_hash="th", optimization_config_hash="oh", hardware_envelope_hash="hh",
        evaluation_hash="eh", routing_health_hash="rh", reproducibility_hash="rrh",
        parameter_count=123, architecture_hash="ah", weight_manifest_hash="wh",
    )
    assert m.num_experts == 4
    assert m.moe_top_k == 2
    assert m.target_layers == [3, 7, 11, 15, 19, 23, 27]

from pathlib import Path

from taxmoe.model_releases.moe_freezer import TaxMoEFreezer
from taxmoe.model_releases.verifier import verify_taxmoe_release


def _release_manifest():
    return TaxMoEReleaseManifest(
        release_id="R-FREEZE",
        taxdense_parent_id="D",
        taxdense_parent_hash="dh",
        moe_init_id="I",
        moe_init_hash="ih",
        input_release_id="IN",
        input_release_hash="inh",
        source_run_id="RUN",
        source_checkpoint_id="FINAL-STEP-6",
        source_checkpoint_hash="ch",
        training_config_hash="th",
        optimization_config_hash="oh",
        hardware_envelope_hash="hh",
        evaluation_hash="eh",
        routing_health_hash="rh",
        reproducibility_hash="rrh",
        parameter_count=123,
        architecture_hash="ah",
        weight_manifest_hash="wh",
    )


def test_taxmoe_freezer_embeds_release_evidence(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "config.json").write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "routing.json"
    evidence.write_text('{"passed": true}\n', encoding="utf-8")

    dest, frozen = TaxMoEFreezer().freeze(
        candidate,
        tmp_path / "releases",
        _release_manifest(),
        evidence_files={"evidence/routing.json": evidence},
    )

    assert (dest / "evidence" / "routing.json").read_text(encoding="utf-8") == evidence.read_text(encoding="utf-8")
    assert any(f.path == "evidence/routing.json" for f in frozen.model_files)
    checked = verify_taxmoe_release(dest)
    assert checked.release_id == frozen.release_id
