import torch

from taxmoe.moe.balance import aggregate_balance_losses, load_balance_loss


def test_uniform_top2_balance_is_one():
    probs = torch.full((1, 4, 4), 0.25, requires_grad=True)
    indices = torch.tensor([[[0, 1], [2, 3], [0, 2], [1, 3]]])
    mask = torch.ones(1, 4, dtype=torch.long)
    result = load_balance_loss(probs, indices, mask, num_experts=4, top_k=2)
    assert torch.allclose(result.loss, torch.tensor(1.0))
    assert torch.allclose(result.assignment_fraction.sum(), torch.tensor(1.0))
    assert torch.allclose(result.probability_mass.sum(), torch.tensor(1.0))


def test_padding_does_not_affect_balance():
    logits = torch.randn(1, 5, 4, requires_grad=True)
    probs = torch.softmax(logits, -1)
    indices = torch.topk(logits, 2, -1).indices
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    a = load_balance_loss(probs, indices, mask, num_experts=4, top_k=2).loss
    probs2 = probs.clone(); probs2[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    indices2 = indices.clone(); indices2[:, 3:] = torch.tensor([0, 1])
    b = load_balance_loss(probs2, indices2, mask, num_experts=4, top_k=2).loss
    assert torch.allclose(a, b)
    assert aggregate_balance_losses([a, b]).shape == ()
