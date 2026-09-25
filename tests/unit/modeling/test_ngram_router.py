import torch
from taxmoe.modeling.ngram_router import TaxMoERouter
from taxmoe.modeling.router_config import MoERouterConfig,NGramRouterConfig

def test_zero_g_recovers_hidden_router():
    cfg=MoERouterConfig(hidden_dim=8,num_experts=4,top_k=2,ngram=NGramRouterConfig(enabled=True,input_dim=3,initial_scale=.01))
    r=TaxMoERouter(cfg)
    h=torch.randn(2,5,8);g=torch.zeros(2,5,3)
    assert torch.allclose(r(h,g),r.hidden_router(h))

def test_missing_ngram_is_error_when_enabled():
    cfg=MoERouterConfig(hidden_dim=8,num_experts=4,ngram=NGramRouterConfig(enabled=True,input_dim=3))
    r=TaxMoERouter(cfg)
    try:r(torch.randn(1,2,8),None)
    except ValueError as e:assert 'FEATURES-MISSING' in str(e)
    else:assert False
