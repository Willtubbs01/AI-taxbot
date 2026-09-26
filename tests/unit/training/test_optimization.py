import torch
from taxmoe.training.optimization import OptimizationConfig,split_decay_parameters
class M(torch.nn.Module):
    def __init__(self): super().__init__(); self.l=torch.nn.Linear(4,4); self.norm=torch.nn.LayerNorm(4)
def test_groups():
    (d,n),(dn,nn)=split_decay_parameters(M()); assert any('l.weight' in x for x in dn) and any('norm' in x for x in nn)
def test_config(): assert OptimizationConfig().gradient_accumulation_steps==8
