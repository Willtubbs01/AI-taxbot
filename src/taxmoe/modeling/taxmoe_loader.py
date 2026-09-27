from __future__ import annotations

import json
from pathlib import Path

import torch

from taxmoe.moe.config import MoEUpcycleConfig
from taxmoe.moe.upcycle import upcycle_model


def _validate_state_dict_result(model, missing, unexpected) -> None:
    missing = list(missing or [])
    unexpected = list(unexpected or [])
    allowed_missing = set()
    if bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        # Hugging Face safe serialization may omit the duplicate output-head
        # tensor when it is tied to the input embeddings. Qwen3 uses this.
        allowed_missing.add("lm_head.weight")
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise ValueError(
            f"MOE-INTEGRATION-STATE-DICT-MISMATCH:missing={bad_missing[:3]} unexpected={unexpected[:3]}"
        )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if "lm_head.weight" in missing and bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if inp is None or out is None or inp.weight.data_ptr() != out.weight.data_ptr():
            raise ValueError("MOE-INTEGRATION-TIED-WEIGHT-RESTORE-FAILED")


def _load_state_dict_from_pretrained_dir(model, root: Path) -> None:
    single_safe = root / "model.safetensors"
    safe_index = root / "model.safetensors.index.json"
    single_bin = root / "pytorch_model.bin"
    bin_index = root / "pytorch_model.bin.index.json"
    if single_safe.exists():
        from safetensors.torch import load_file
        state = load_file(str(single_safe), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        _validate_state_dict_result(model, missing, unexpected)
        return
    if safe_index.exists() or bin_index.exists():
        from transformers.modeling_utils import load_sharded_checkpoint
        result = load_sharded_checkpoint(model, str(root), strict=False, prefer_safe=True)
        _validate_state_dict_result(model, getattr(result, "missing_keys", None), getattr(result, "unexpected_keys", None))
        return
    if single_bin.exists():
        state = torch.load(single_bin, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        _validate_state_dict_result(model, missing, unexpected)
        return
    raise FileNotFoundError(f"No model weights found in {root}")


def save_taxmoe_model(model, output_dir: str | Path, moe_config: MoEUpcycleConfig, *, report=None, safe_serialization: bool = True) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("MOE-SAVE-PRETRAINED-UNSUPPORTED")
    model.save_pretrained(out, safe_serialization=safe_serialization)
    (out / "moe_config.json").write_text(moe_config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if report is not None:
        payload = report.model_dump_json(indent=2) if hasattr(report, "model_dump_json") else json.dumps(report, indent=2, default=str)
        (out / "upcycle_report.json").write_text(payload + "\n", encoding="utf-8")
    return out


def load_taxmoe_model(path: str | Path, *, device: str | None = None, torch_dtype=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    root = Path(path)
    moe_path = root / "moe_config.json"
    if not moe_path.exists():
        raise FileNotFoundError("MOE-CONFIG-MISSING")
    moe_config = MoEUpcycleConfig.model_validate_json(moe_path.read_text(encoding="utf-8"))
    base_config = AutoConfig.from_pretrained(root)
    kwargs = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_config(base_config, **kwargs)
    upcycle_model(model, moe_config)
    _load_state_dict_from_pretrained_dir(model, root)
    if device:
        model.to(device)
    return model, moe_config
