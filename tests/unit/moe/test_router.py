import torch

from taxmoe.moe.router import HiddenStateRouter, TopKRoutingPolicy


def test_topk_policy_selects_and_normalizes():
    policy = TopKRoutingPolicy(4, 2)
    logits = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    out = policy(logits)
    assert out.expert_indices.tolist() == [[[1, 3]]]
    assert torch.allclose(out.expert_weights.sum(-1), torch.ones(1, 1))
    assert torch.allclose(out.expert_weights[0, 0], torch.softmax(torch.tensor([4.0, 3.0]), dim=-1))


def test_router_initialization_is_layer_deterministic():
    a = HiddenStateRouter(8, 4, 2, seed=123, layer_index=3)
    b = HiddenStateRouter(8, 4, 2, seed=123, layer_index=3)
    c = HiddenStateRouter(8, 4, 2, seed=123, layer_index=7)
    assert torch.equal(a.hidden_proj.weight, b.hidden_proj.weight)
    assert not torch.equal(a.hidden_proj.weight, c.hidden_proj.weight)
    x = torch.randn(2, 5, 8)
    result = a(x, return_aux=True)
    assert result.aux.logits.dtype == torch.float32
    assert result.decision.expert_indices.shape == (2, 5, 2)
