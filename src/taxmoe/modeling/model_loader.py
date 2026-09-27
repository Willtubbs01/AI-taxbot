from __future__ import annotations
from pathlib import Path
import hashlib, json, platform
from pydantic import BaseModel, ConfigDict
import yaml
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from taxmoe.ingestion.hashing import stable_hash
from .model_inspection import inspect_architecture, config_hash
from .model_manifest import BaseModelManifest

class ModelLoadConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_id:str='Qwen/Qwen3-0.6B'
    revision:str
    trust_remote_code:bool=False
    dtype:str='auto'
    low_cpu_mem_usage:bool=True
    expected_model_type:str|None='qwen3'

    @classmethod
    def from_yaml(cls,path):
        raw=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        section=raw.get('model',raw)
        if 'loading' in raw:
            section={**section,**raw['loading']}
        if 'architecture' in raw and 'expected_model_type' in raw['architecture']:
            section['expected_model_type']=raw['architecture']['expected_model_type']
        return cls.model_validate(section)

def resolve_dtype(name:str):
    return {'auto':'auto','fp16':torch.float16,'float16':torch.float16,'bf16':torch.bfloat16,'bfloat16':torch.bfloat16,'fp32':torch.float32,'float32':torch.float32}[name.lower()]

def load_config(cfg:ModelLoadConfig):
    if not cfg.revision or 'REPLACE_' in cfg.revision: raise ValueError('MODEL-REVISION-NOT-PINNED')
    c=AutoConfig.from_pretrained(cfg.model_id,revision=cfg.revision,trust_remote_code=cfg.trust_remote_code)
    if cfg.expected_model_type and getattr(c,'model_type',None)!=cfg.expected_model_type: raise ValueError(f'MODEL-TYPE-MISMATCH:{getattr(c,"model_type",None)}')
    return c

def load_model(cfg:ModelLoadConfig, *, device:str|None=None):
    config=load_config(cfg)
    kwargs=dict(revision=cfg.revision,trust_remote_code=cfg.trust_remote_code,low_cpu_mem_usage=cfg.low_cpu_mem_usage)
    dt=resolve_dtype(cfg.dtype)
    if dt!='auto': kwargs['dtype']=dt
    model=AutoModelForCausalLM.from_pretrained(cfg.model_id,**kwargs)
    if device: model.to(device)
    return model,config

def build_manifest(cfg:ModelLoadConfig, model, config, tokenizer_manifest_hash:str|None=None, weight_files:dict[str,str]|None=None):
    report=inspect_architecture(model)
    env={'python':platform.python_version(),'torch':torch.__version__}
    try:
        import transformers; env['transformers']=transformers.__version__
    except Exception: pass
    payload=dict(model_id=cfg.model_id,revision=cfg.revision,config_sha256=config_hash(config),tokenizer_manifest_hash=tokenizer_manifest_hash,weight_files=weight_files or {})
    fp=stable_hash(payload)
    return BaseModelManifest(model_id=cfg.model_id,revision=cfg.revision,architecture=type(model).__name__,model_class=type(model).__name__,config_class=type(config).__name__,model_type=getattr(config,'model_type',''),vocab_size=int(getattr(config,'vocab_size',0)),hidden_size=int(getattr(config,'hidden_size',0)),intermediate_size=int(getattr(config,'intermediate_size',0)),num_hidden_layers=int(getattr(config,'num_hidden_layers',0)),num_attention_heads=int(getattr(config,'num_attention_heads',0)),num_key_value_heads=getattr(config,'num_key_value_heads',None),max_position_embeddings=getattr(config,'max_position_embeddings',None),hidden_act=getattr(config,'hidden_act',None),rms_norm_eps=getattr(config,'rms_norm_eps',None),tie_word_embeddings=getattr(config,'tie_word_embeddings',None),rope_theta=getattr(config,'rope_theta',None),parameter_count=report.parameter_count,trainable_parameter_count=report.trainable_parameter_count,config_sha256=config_hash(config),tokenizer_manifest_hash=tokenizer_manifest_hash,weight_files=weight_files or {},environment=env,base_model_fingerprint=fp),report
