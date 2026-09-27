import torch
from torch import nn

from taxmoe.moe.dispatch import dispatch_to_experts, reference_dispatch
from taxmoe.moe.experts import ExpertBank
from taxmoe.moe.router import RoutingDecision


class ScaleExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
    def forward(self, x):
        return x * self.scale


def test_dispatch_matches_reference_with_nonidentical_experts():
    bank = ExpertBank([ScaleExpert(1), ScaleExpert(2), ScaleExpert(3), ScaleExpert(4)])
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    indices = torch.tensor([[[0, 3], [2, 1], [1, 3]]])
    weights = torch.tensor([[[0.75, 0.25], [0.4, 0.6], [0.5, 0.5]]])
    decision = RoutingDecision(indices, weights)
    actual, stats = dispatch_to_experts(x, bank, decision, return_stats=True)
    expected = reference_dispatch(x, bank, decision)
    assert torch.allclose(actual, expected)
    assert int(stats.assignment_count.sum()) == 6
    assert torch.allclose(stats.weighted_mass.sum(), torch.tensor(3.0))


def test_identical_experts_reproduce_dense_function():
    source = ScaleExpert(2.5)
    bank = ExpertBank.from_dense(source, num_experts=4)
    x = torch.randn(2, 4, 6)
    indices = torch.tensor([[[0, 1], [2, 3], [1, 2], [3, 0]], [[1, 3], [0, 2], [2, 3], [0, 1]]])
    weights = torch.rand(2, 4, 2)
    weights = weights / weights.sum(-1, keepdim=True)
    actual = dispatch_to_experts(x, bank, RoutingDecision(indices, weights))
    assert torch.allclose(actual, source(x), atol=1e-6, rtol=1e-6)
    ptrs = [dict(e.named_parameters())["scale"].data_ptr() for e in bank.experts]
    assert len(ptrs) == len(set(ptrs))
