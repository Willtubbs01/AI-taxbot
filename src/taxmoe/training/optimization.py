from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml, torch
from pydantic import Field, model_validator
from taxmoe.schemas.common import TaxMoEModel
class OptimizationConfig(TaxMoEModel):
    precision:Literal['fp16','bf16','fp32']='fp16'
    optimizer:Literal['adamw']='adamw'
    learning_rate:float=8e-6
    weight_decay:float=0.1
    betas:tuple[float,float]=(0.9,0.95)
    eps:float=1e-8
    gradient_clip_norm:float=1.0
    scheduler:Literal['cosine','linear','constant']='cosine'
    warmup_ratio:float=0.05
    gradient_accumulation_steps:int=8
    @model_validator(mode='after')
    def valid(self):
        if self.learning_rate<=0: raise ValueError('OPT-LR-INVALID')
        if not 0<=self.warmup_ratio<1: raise ValueError('OPT-WARMUP-INVALID')
        if self.gradient_accumulation_steps<1: raise ValueError('OPT-ACCUM-INVALID')
        return self
    @classmethod
    def from_yaml(cls,path):
        raw=yaml.safe_load(Path(path).read_text(encoding='utf-8')); return cls.model_validate(raw.get('optimization',raw))

def split_decay_parameters(model):
    decay=[]; no_decay=[]; decay_names=[]; no_names=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lname=name.lower()
        if p.ndim<2 or lname.endswith('bias') or 'norm' in lname: no_decay.append(p); no_names.append(name)
        else: decay.append(p); decay_names.append(name)
    return (decay,no_decay),(decay_names,no_names)

def build_optimizer(model,cfg:OptimizationConfig):
    (decay,no_decay),names=split_decay_parameters(model)
    groups=[{'params':decay,'weight_decay':cfg.weight_decay},{'params':no_decay,'weight_decay':0.0}]
    return torch.optim.AdamW(groups,lr=cfg.learning_rate,betas=cfg.betas,eps=cfg.eps),names

def build_scheduler(optimizer,cfg:OptimizationConfig,total_steps:int):
    from transformers import get_scheduler
    warmup=int(total_steps*cfg.warmup_ratio)
    return get_scheduler(cfg.scheduler,optimizer=optimizer,num_warmup_steps=warmup,num_training_steps=max(total_steps,1))
