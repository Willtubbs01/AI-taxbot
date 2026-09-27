from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    if t.numel():
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    return state_dict_sha256(module.state_dict())


def named_tensor_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {name: tensor_sha256(t) for name, t in module.state_dict().items()}
