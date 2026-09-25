from __future__ import annotations
import math,torch

class TaxMoEGenerationCollator:
    def __init__(self,pad_token_id:int,max_features_per_token:int=24,pad_to_multiple_of:int=8,include_ngram_features:bool=False):
        self.pad=pad_token_id;self.k=max_features_per_token;self.mult=pad_to_multiple_of;self.include_ngram=include_ngram_features
    def __call__(self,examples):
        max_len=max(len(x['prompt_input_ids']) for x in examples); L=int(math.ceil(max_len/self.mult)*self.mult) if self.mult else max_len; B=len(examples)
        ids=torch.full((B,L),self.pad,dtype=torch.long);attn=torch.zeros((B,L),dtype=torch.long);ng=torch.zeros((B,L,self.k),dtype=torch.long) if self.include_ngram else None
        for i,x in enumerate(examples):
            seq=x['prompt_input_ids'];n=len(seq);start=L-n
            ids[i,start:]=torch.as_tensor(seq);attn[i,start:]=1
            if ng is not None:
                a=torch.as_tensor(x['prompt_ngram_feature_ids'],dtype=torch.long)
                if a.shape!=(n,self.k):raise ValueError('generation ngram shape mismatch')
                ng[i,start:,:]=a
        out={'input_ids':ids.contiguous(),'attention_mask':attn.contiguous()}
        if ng is not None:out['ngram_feature_ids']=ng.contiguous()
        return out
