from __future__ import annotations
import torch

def generate_dense(model, tokenizer, batch, *, max_new_tokens=512):
    model.eval()
    with torch.no_grad():
        out=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,pad_token_id=getattr(tokenizer,'pad_token_id',None),eos_token_id=getattr(tokenizer,'eos_token_id',None))
    prompt_len=batch['input_ids'].shape[1]
    return tokenizer.batch_decode(out[:,prompt_len:],skip_special_tokens=True),out
