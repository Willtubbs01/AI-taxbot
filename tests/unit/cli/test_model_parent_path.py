import pytest
import typer

from taxmoe.cli.model import _require_local_parent_if_pathlike


def test_missing_local_model_path_gets_clear_error(tmp_path):
    missing = tmp_path / "artifacts" / "models" / "TaxDense"
    with pytest.raises(typer.BadParameter, match="does not exist"):
        _require_local_parent_if_pathlike(str(missing))


def test_huggingface_repo_id_is_not_rejected_as_local_path():
    _require_local_parent_if_pathlike("Qwen/Qwen3-0.6B")
