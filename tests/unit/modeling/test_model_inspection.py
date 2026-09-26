import torch
from taxmoe.modeling.model_inspection import inspect_architecture
class MLP(torch.nn.Module):
    def __init__(self): super().__init__(); self.gate_proj=torch.nn.Linear(8,16,bias=False); self.up_proj=torch.nn.Linear(8,16,bias=False); self.down_proj=torch.nn.Linear(16,8,bias=False)
    def forward(self,x): return self.down_proj(self.up_proj(x))
class L(torch.nn.Module):
    def __init__(self): super().__init__(); self.mlp=MLP()
class Inner(torch.nn.Module):
    def __init__(self): super().__init__(); self.layers=torch.nn.ModuleList([L() for _ in range(4)])
class M(torch.nn.Module):
    def __init__(self): super().__init__(); self.model=Inner(); self.e=torch.nn.Embedding(10,8); self.h=torch.nn.Linear(8,10,bias=False)
    def get_input_embeddings(self): return self.e
    def get_output_embeddings(self): return self.h
def test_inspection():
    r=inspect_architecture(M(),candidate_layers=(1,3)); assert r.layer_count==4 and r.candidate_layers_valid and r.layers[1].moe_candidate
