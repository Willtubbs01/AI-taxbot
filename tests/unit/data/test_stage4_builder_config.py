from pathlib import Path

from taxmoe.data.stage4_builder import _for_cpt_tokenization, _load_tokenization_config


def test_cpt_builder_disables_sft_only_assistant_mask_requirement():
    root = Path(__file__).resolve().parents[3]
    shared = _load_tokenization_config(root / "configs/modeling/tokenization_v1.yaml")

    assert shared.require_assistant_mask is True

    cpt = _for_cpt_tokenization(shared)

    assert cpt.require_assistant_mask is False
    # The shared config object must remain unchanged for the later SFT path.
    assert shared.require_assistant_mask is True
    assert cpt.model == shared.model
    assert cpt.sequence == shared.sequence
    assert cpt.cpt_loss == shared.cpt_loss
