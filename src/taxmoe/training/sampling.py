from __future__ import annotations
import random

def deterministic_order(n:int,seed:int,pass_index:int):
    idx=list(range(n)); random.Random((seed<<16)^pass_index).shuffle(idx); return idx

def effective_passes(consumed_tokens:int,unique_tokens:int): return consumed_tokens/unique_tokens if unique_tokens else 0.0
