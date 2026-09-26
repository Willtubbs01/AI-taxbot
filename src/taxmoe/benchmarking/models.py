from __future__ import annotations
from enum import Enum
from pathlib import Path
import yaml
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class BenchmarkStatus(str,Enum):
    PASS='pass';OOM='oom';NUMERICAL_FAILURE='numerical_failure';UNSUPPORTED='unsupported';ERROR='error'
class BenchmarkConfig(TaxMoEModel):
    warmup_steps:int=2
    measured_steps:int=5
    sequence_lengths:list[int]=Field(default_factory=lambda:[512,1024,2048,4096])
    microbatch_sizes:list[int]=Field(default_factory=lambda:[1,2,4])
    precision:list[str]=Field(default_factory=lambda:['fp16','bf16'])
    gradient_checkpointing:list[bool]=Field(default_factory=lambda:[False,True])
    attention_backends:list[str]=Field(default_factory=lambda:['sdpa','eager'])
    @classmethod
    def from_yaml(cls,path):
        raw=yaml.safe_load(Path(path).read_text(encoding='utf-8')); return cls.model_validate(raw.get('benchmark',raw))
class DenseBenchmarkResult(TaxMoEModel):
    precision:str; sequence_length:int; microbatch_size:int; gradient_checkpointing:bool; attention_backend:str='default'; optimizer_mode:str='none'; status:BenchmarkStatus
    peak_allocated_bytes:int|None=None; peak_reserved_bytes:int|None=None; step_time_seconds:float|None=None; tokens_per_second:float|None=None; loss_mean:float|None=None; finite:bool|None=None; error:str|None=None
class HardwareEnvelope(TaxMoEModel):
    precision:str; max_safe_microbatch_by_length:dict[int,int|None]=Field(default_factory=dict); recommended_microbatch_by_length:dict[int,int|None]=Field(default_factory=dict); checkpointing_required_by_length:dict[int,bool]=Field(default_factory=dict); notes:list[str]=Field(default_factory=list)
