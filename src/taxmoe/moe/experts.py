from __future__ import annotations

import copy

import torch
from torch import nn

from .hashing import module_sha256


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return dict(module.named_buffers())


def verify_exact_clone(source: nn.Module, clone: nn.Module) -> None:
    if type(source) is not type(clone):
        raise ValueError("MOE-EXPERT-COPY-CLASS-MISMATCH")
    sp, cp = _named_parameters(source), _named_parameters(clone)
    if list(sp) != list(cp):
        raise ValueError("MOE-EXPERT-COPY-NAME-MISMATCH")
    for name in sp:
        a, b = sp[name], cp[name]
        if a.shape != b.shape:
            raise ValueError(f"MOE-EXPERT-COPY-SHAPE-MISMATCH:{name}")
        if a.dtype != b.dtype:
            raise ValueError(f"MOE-EXPERT-DTYPE-MISMATCH:{name}")
        if a.requires_grad != b.requires_grad:
            raise ValueError(f"MOE-EXPERT-REQUIRES-GRAD-MISMATCH:{name}")
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            raise ValueError(f"MOE-EXPERT-COPY-VALUE-MISMATCH:{name}")
        if a.data_ptr() == b.data_ptr():
            raise ValueError(f"MOE-EXPERT-STORAGE-ALIAS:{name}")
    sb, cb = _named_buffers(source), _named_buffers(clone)
    if list(sb) != list(cb):
        raise ValueError("MOE-EXPERT-BUFFER-NAME-MISMATCH")
    for name in sb:
        if not torch.equal(sb[name].detach().cpu(), cb[name].detach().cpu()):
            raise ValueError(f"MOE-EXPERT-BUFFER-VALUE-MISMATCH:{name}")


def clone_dense_mlp(source: nn.Module) -> nn.Module:
    clone = copy.deepcopy(source)
    verify_exact_clone(source, clone)
    return clone


class ExpertBank(nn.Module):
    def __init__(self, experts: list[nn.Module]):
        super().__init__()
        if not experts:
            raise ValueError("MOE-EXPERT-INVALID-COUNT")
        self.experts = nn.ModuleList(experts)
        self._verify_sibling_storage()

    @classmethod
    def from_dense(cls, dense_mlp: nn.Module, *, num_experts: int) -> "ExpertBank":
        if num_experts < 1:
            raise ValueError("MOE-EXPERT-INVALID-COUNT")
        experts = [clone_dense_mlp(dense_mlp) for _ in range(num_experts)]
        bank = cls(experts)
        source_hash = module_sha256(dense_mlp)
        if any(module_sha256(e) != source_hash for e in bank.experts):
            raise ValueError("MOE-EXPERT-COPY-HASH-MISMATCH")
        return bank

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _verify_sibling_storage(self) -> None:
        params = [dict(e.named_parameters()) for e in self.experts]
        if not params:
            return
        names = list(params[0])
        if any(list(p) != names for p in params[1:]):
            raise ValueError("MOE-EXPERT-COPY-NAME-MISMATCH")
        for name in names:
            ptrs = [p[name].data_ptr() for p in params]
            if len(ptrs) != len(set(ptrs)):
                raise ValueError(f"MOE-EXPERT-STORAGE-ALIAS:{name}")

    def expert_hashes(self) -> list[str]:
        return [module_sha256(e) for e in self.experts]

    def named_expert_parameters(self):
        for expert_id, expert in enumerate(self.experts):
            for name, parameter in expert.named_parameters():
                yield expert_id, name, parameter
