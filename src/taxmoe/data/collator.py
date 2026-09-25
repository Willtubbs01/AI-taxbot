from __future__ import annotations
import math
import torch

class TaxMoETrainingCollator:
    def __init__(self,pad_token_id:int,label_pad_id:int=-100,ngram_pad_id:int=0,max_features_per_token:int=24,pad_to_multiple_of:int=8,include_ngram_features:bool=False):
        self.pad_token_id=pad_token_id; self.label_pad_id=label_pad_id; self.ngram_pad_id=ngram_pad_id; self.k=max_features_per_token; self.mult=pad_to_multiple_of; self.include_ngram=include_ngram_features
    def __call__(self,examples):
        if not examples: raise ValueError('empty batch')
        max_len=max(len(x['input_ids']) for x in examples)
        L=int(math.ceil(max_len/self.mult)*self.mult) if self.mult else max_len
        B=len(examples)
        ids=torch.full((B,L),self.pad_token_id,dtype=torch.long)
        labels=torch.full((B,L),self.label_pad_id,dtype=torch.long)
        attn=torch.zeros((B,L),dtype=torch.long)
        ng=torch.full((B,L,self.k),self.ngram_pad_id,dtype=torch.long) if self.include_ngram else None
        for i,x in enumerate(examples):
            n=len(x['input_ids'])
            if len(x['labels'])!=n:raise ValueError('CACHE-LABEL-LENGTH-MISMATCH')
            ids[i,:n]=torch.as_tensor(x['input_ids'],dtype=torch.long)
            labels[i,:n]=torch.as_tensor(x['labels'],dtype=torch.long)
            attn[i,:n]=1
            if ng is not None:
                rows=x.get('ngram_feature_ids')
                if rows is None:raise ValueError('CACHE-NGRAM-RECORD-MISSING')
                a=torch.as_tensor(rows,dtype=torch.long)
                if a.shape!=(n,self.k):raise ValueError('CACHE-NGRAM-WIDTH-MISMATCH')
                ng[i,:n,:]=a
        out={'input_ids':ids.contiguous(),'attention_mask':attn.contiguous(),'labels':labels.contiguous()}
        if ng is not None:out['ngram_feature_ids']=ng.contiguous()
        return out
