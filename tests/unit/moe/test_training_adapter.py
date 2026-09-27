from types import SimpleNamespace

import torch
from torch import nn

from taxmoe.moe.config import BalanceLossConfig, MoEUpcycleConfig
from taxmoe.moe.upcycle import upcycle_model
from taxmoe.training.moe_adapter import MoECausalLMAdapter


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
    def forward(self, x):
        return x + self.mlp(x)


class TinyCausal(nn.Module):
    def __init__(self, vocab=20, h=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=h, use_cache=False, vocab_size=vocab)
        self.embed = nn.Embedding(vocab, h)
        self.model = nn.Module(); self.model.layers = nn.ModuleList([TinyBlock() for _ in range(2)])
        self.lm_head = nn.Linear(h, vocab, bias=False)
    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def test_moe_adapter_adds_balance_and_router_gets_gradient():
    model = TinyCausal()
    upcycle_model(model, MoEUpcycleConfig(target_layers=[0], num_experts=4, top_k=2))
    batch = {
        "input_ids": torch.randint(0, 20, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
    }
    batch["labels"] = batch["input_ids"].clone()
    adapter = MoECausalLMAdapter(BalanceLossConfig(coefficient=0.01))
    out = adapter.compute_loss(model, batch)
    assert "moe_balance" in out.auxiliary_losses
    assert "router/0/entropy" in out.auxiliary_losses
    assert "router/0/assignment_e0" in out.auxiliary_losses
    assert "router/0/prob_mass_e0" in out.auxiliary_losses
    assert "router/0/top1_e0" in out.auxiliary_losses
    assert out.total_loss.item() >= out.lm_loss.item()
    out.total_loss.backward()
    router = model.model.layers[0].mlp.router
    assert router.hidden_proj.weight.grad is not None
    assert torch.isfinite(router.hidden_proj.weight.grad).all()


def test_moe_adapter_rejects_ngram_input():
    model = TinyCausal()
    upcycle_model(model, MoEUpcycleConfig(target_layers=[0]))
    ids = torch.randint(0, 20, (1, 4))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone(), "ngram_feature_ids": torch.zeros(1, 4, 24, dtype=torch.long)}
    try:
        MoECausalLMAdapter().compute_loss(model, batch)
    except ValueError as exc:
        assert "UNEXPECTED-NGRAM" in str(exc)
    else:
        raise AssertionError("expected n-gram rejection")

class TinyCheckpointCausal(TinyCausal):
    def __init__(self, vocab=20, h=8):
        super().__init__(vocab=vocab, h=h)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {}
        if kwargs.get("use_reentrant", False):
            raise AssertionError("test requires non-reentrant checkpointing")
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        from torch.utils.checkpoint import checkpoint
        h = self.embed(input_ids)
        for layer in self.model.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def test_moe_adapter_nonreentrant_checkpoint_backward_is_graph_stable():
    from taxmoe.training.moe_adapter import enable_moe_gradient_checkpointing

    model = TinyCheckpointCausal()
    upcycle_model(model, MoEUpcycleConfig(target_layers=[0], num_experts=4, top_k=2))
    enable_moe_gradient_checkpointing(model)
    model.train()
    ids = torch.randint(0, 20, (2, 6))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    out = MoECausalLMAdapter(BalanceLossConfig(coefficient=0.01)).compute_loss(model, batch)
    out.total_loss.backward()
    router = model.model.layers[0].mlp.router
    assert router.hidden_proj.weight.grad is not None
    assert torch.isfinite(router.hidden_proj.weight.grad).all()
