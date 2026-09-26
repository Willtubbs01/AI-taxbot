from __future__ import annotations
import hashlib, json
from typing import Any
import torch
from .model_manifest import ArchitectureReport, LayerInspection

def _numel(module): return sum(p.numel() for p in module.parameters())

def _find_layers(model):
    paths = ["model.layers", "transformer.h", "gpt_neox.layers"]
    cur=None
    for path in paths:
        obj=model
        try:
            for part in path.split('.'): obj=getattr(obj,part)
            cur=obj; break
        except AttributeError: pass
    if cur is None: raise ValueError('MODEL-LAYERS-NOT-FOUND')
    return cur

def inspect_architecture(model, candidate_layers=(3,7,11,15,19,23,27)) -> ArchitectureReport:
    layers=_find_layers(model); out=[]
    for i,layer in enumerate(layers):
        mlp=getattr(layer,'mlp',None)
        if mlp is None: raise ValueError(f'MODEL-MLP-NOT-FOUND:{i}')
        shapes={}
        for name in ('gate_proj','up_proj','down_proj','fc1','fc2'):
            mod=getattr(mlp,name,None)
            if mod is not None and hasattr(mod,'weight'): shapes[name]=list(mod.weight.shape)
        out.append(LayerInspection(layer_index=i,layer_class=type(layer).__name__,mlp_class=type(mlp).__name__,mlp_path=f'model.layers.{i}.mlp',parameter_count=_numel(mlp),projection_shapes=shapes,moe_candidate=i in candidate_layers))
    tied=None
    try: tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight
    except Exception: pass
    return ArchitectureReport(model_class=type(model).__name__,parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),layer_count=len(layers),layers=out,candidate_layers=list(candidate_layers),candidate_layers_valid=all(0<=i<len(layers) for i in candidate_layers),tied_word_embeddings=tied)

def config_hash(config) -> str:
    data=config.to_dict() if hasattr(config,'to_dict') else dict(config)
    raw=json.dumps(data,sort_keys=True,separators=(',',':'),default=str).encode()
    return hashlib.sha256(raw).hexdigest()
