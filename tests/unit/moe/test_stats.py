import torch

from taxmoe.moe.stats import compute_router_stats


def test_router_stats_top2():
    probs = torch.tensor([[[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.6, 0.2]]])
    idx = torch.tensor([[[0, 1], [2, 3]]])
    stats = compute_router_stats(probs, idx, torch.ones(1, 2), num_experts=4, top_k=2)
    assert stats.valid_token_count == 2
    assert torch.allclose(stats.assignment_fraction, torch.full((4,), 0.25))
    assert stats.pair_counts == {"0-1": 1, "2-3": 1}
