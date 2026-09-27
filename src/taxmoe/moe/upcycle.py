from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from taxmoe.ingestion.hashing import stable_hash

from .config import MoEUpcycleConfig
from .experts import ExpertBank
from .hashing import module_sha256, tensor_sha256
from .manifests import LayerUpcycleReport, MoEUpcycleReport
from .module import SparseMoE
from .router import HiddenStateRouter


@dataclass
class LayerPath:
    path: str
    layers: object


def find_transformer_layers(model) -> LayerPath:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return LayerPath(path=path, layers=obj)
        except AttributeError:
            continue
    raise ValueError("MODEL-LAYERS-NOT-FOUND")


def iter_sparse_moe(model):
    for module in model.modules():
        if isinstance(module, SparseMoE):
            yield module


def _module_device_dtype(module: nn.Module):
    for p in module.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


def _hidden_size(model, mlp: nn.Module) -> int:
    value = getattr(getattr(model, "config", None), "hidden_size", None)
    if value:
        return int(value)
    for name in ("gate_proj", "up_proj", "fc1"):
        proj = getattr(mlp, name, None)
        if proj is not None and hasattr(proj, "in_features"):
            return int(proj.in_features)
    for p in mlp.parameters():
        if p.ndim == 2:
            return int(p.shape[-1])
    raise ValueError("MOE-UPCYCLING-HIDDEN-SIZE-NOT-FOUND")


def _parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _target_prefixes(layer_path: str, targets: list[int]) -> tuple[str, ...]:
    return tuple(f"{layer_path}.{i}.mlp." for i in targets)


def _preserved_hashes(model, prefixes: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for name, tensor in model.state_dict().items():
        if not name.startswith(prefixes):
            out[name] = tensor_sha256(tensor)
    return out


def upcycle_model(model: nn.Module, config: MoEUpcycleConfig) -> MoEUpcycleReport:
    found = find_transformer_layers(model)
    layers = found.layers
    targets = sorted(config.target_layers)
    if any(i >= len(layers) for i in targets):
        raise ValueError("MOE-UPCYCLING-TARGET-LAYER-INVALID")
    if any(isinstance(layers[i].mlp, SparseMoE) for i in targets):
        raise ValueError("MOE-UPCYCLING-ALREADY-CONVERTED")

    parent_count = _parameter_count(model)
    prefixes = _target_prefixes(found.path, targets)
    preserved_before = _preserved_hashes(model, prefixes)
    reports: list[LayerUpcycleReport] = []
    expected_count = parent_count

    for layer_index in targets:
        layer = layers[layer_index]
        source = getattr(layer, "mlp", None)
        if source is None:
            raise ValueError(f"MOE-UPCYCLING-TARGET-LAYER-MISSING:{layer_index}")
        source_hash = module_sha256(source)
        source_count = _parameter_count(source)
        device, dtype = _module_device_dtype(source)
        bank = ExpertBank.from_dense(source, num_experts=config.num_experts)
        hidden_size = _hidden_size(model, source)
        router = HiddenStateRouter(
            hidden_size,
            config.num_experts,
            config.top_k,
            bias=config.router.bias,
            init_std=config.router.init_std,
            seed=config.router.seed,
            layer_index=layer_index,
            device=device,
            dtype=dtype,
        )
        sparse = SparseMoE(layer_index=layer_index, experts=bank, router=router)
        expert_hashes = bank.expert_hashes()
        if any(h != source_hash for h in expert_hashes):
            raise ValueError(f"MOE-UPCYCLING-EXPERT-COPY-FAIL:{layer_index}")
        router_count = _parameter_count(router)
        expected_count += (config.num_experts - 1) * source_count + router_count
        layer.mlp = sparse
        reports.append(
            LayerUpcycleReport(
                layer_index=layer_index,
                source_mlp_class=type(source).__name__,
                source_mlp_hash=source_hash,
                source_parameter_count=source_count,
                expert_hashes=expert_hashes,
                expert_parameter_count=sum(_parameter_count(e) for e in bank.experts),
                router_parameter_count=router_count,
                router_seed=router.layer_seed,
                router_hash=module_sha256(router),
            )
        )

    converted_count = _parameter_count(model)
    if converted_count != expected_count:
        raise ValueError(f"MOE-UPCYCLING-PARAMETER-COUNT-MISMATCH:{converted_count}:{expected_count}")
    preserved_after = _preserved_hashes(model, prefixes)
    changed = sorted(name for name, h in preserved_before.items() if preserved_after.get(name) != h)
    missing = sorted(set(preserved_before) - set(preserved_after))
    unexpected = sorted(set(preserved_after) - set(preserved_before))
    unexpected_changes = changed + [f"missing:{x}" for x in missing] + [f"added:{x}" for x in unexpected]
    if unexpected_changes:
        raise ValueError(f"MOE-UPCYCLING-UNEXPECTED-STATE-CHANGE:{unexpected_changes[:3]}")

    architecture_hash = stable_hash(
        "taxmoe_qwen3",
        config.model_dump(mode="json"),
        [(r.layer_index, r.source_mlp_class, r.source_parameter_count, r.router_parameter_count) for r in reports],
    )
    return MoEUpcycleReport(
        target_layers=targets,
        num_experts=config.num_experts,
        top_k=config.top_k,
        parent_parameter_count=parent_count,
        converted_parameter_count=converted_count,
        expected_parameter_count=expected_count,
        preserved_tensor_count=len(preserved_before),
        unexpected_preserved_changes=unexpected_changes,
        layers=reports,
        architecture_hash=architecture_hash,
        passed=True,
    )
