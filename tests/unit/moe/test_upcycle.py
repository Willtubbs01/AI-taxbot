import copy
from types import SimpleNamespace

import torch
from torch import nn

from taxmoe.moe.config import MoEUpcycleConfig
from taxmoe.moe.module import SparseMoE
from taxmoe.moe.upcycle import upcycle_model
from taxmoe.moe.validation import validate_upcycle


class TinyMLP(nn.Module):
    def __init__(self, h=8, inter=12):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module(); self.model.layers = nn.ModuleList([TinyBlock() for _ in range(4)])
        self.config = SimpleNamespace(hidden_size=8)


def test_upcycle_replaces_only_target_layers_and_is_equivalent():
    dense = TinyModel()
    moe = copy.deepcopy(dense)
    cfg = MoEUpcycleConfig(target_layers=[1, 3], num_experts=4, top_k=2)
    report = upcycle_model(moe, cfg)
    assert report.passed
    assert isinstance(moe.model.layers[1].mlp, SparseMoE)
    assert not isinstance(moe.model.layers[0].mlp, SparseMoE)
    assert report.converted_parameter_count == report.expected_parameter_count
    validation = validate_upcycle(dense, moe, cfg, atol=1e-5, rtol=1e-5)
    assert validation.passed, validation.failures


def test_bf16_upcycle_validation_uses_fp32_reference_math():
    dense = TinyModel().to(dtype=torch.bfloat16)
    moe = copy.deepcopy(dense)
    cfg = MoEUpcycleConfig(target_layers=[1, 3], num_experts=4, top_k=2)
    report = upcycle_model(moe, cfg)
    assert report.passed

    # Native BF16 Top-2 weighted accumulation can differ by roughly one BF16
    # ULP even though the experts are exact copies. The validator should test
    # the mathematical upcycling identity in FP32 and therefore pass.
    validation = validate_upcycle(dense, moe, cfg, atol=1e-5, rtol=1e-5)
    assert validation.passed, validation.failures
    assert all(r.comparison.max_abs < 1e-5 for r in validation.local_equivalence)

class TinyCausalModel(nn.Module):
    def __init__(self, vocab=24, h=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=h)
        self.embed = nn.Embedding(vocab, h)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(2)])
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = h + layer.mlp(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_full_model_equivalence_gates_logits_and_lm_loss():
    dense = TinyCausalModel()
    moe = copy.deepcopy(dense)
    cfg = MoEUpcycleConfig(target_layers=[0, 1], num_experts=4, top_k=2)
    upcycle_model(moe, cfg)
    ids = torch.randint(0, 24, (1, 12))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    report = validate_upcycle(
        dense,
        moe,
        cfg,
        batch=batch,
        atol=1e-5,
        rtol=1e-5,
        full_atol=1e-5,
        full_rtol=1e-5,
        loss_atol=1e-6,
    )
    assert report.passed, report.failures
    assert report.full_model_logits is not None and report.full_model_logits.passed
    assert report.lm_loss_passed is True
    assert report.lm_loss_delta is not None and report.lm_loss_delta <= 1e-6
