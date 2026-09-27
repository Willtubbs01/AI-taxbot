from __future__ import annotations

from taxmoe.moe.upcycle import iter_sparse_moe


def parameter_inventory(model) -> dict[str, int]:
    modules = list(iter_sparse_moe(model))
    expert_ids = {id(p) for m in modules for p in m.experts.parameters()}
    router_ids = {id(p) for m in modules for p in m.router.parameters()}
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for p in model.parameters() if id(p) in expert_ids)
    router = sum(p.numel() for p in model.parameters() if id(p) in router_ids)
    base = total - expert - router
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "expert_parameters": expert,
        "router_parameters": router,
        "base_parameters": base,
        "moe_layers": len(modules),
        "experts_total": sum(m.num_experts for m in modules),
    }
