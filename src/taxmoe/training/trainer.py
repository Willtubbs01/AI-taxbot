from __future__ import annotations
import math, torch
from .objectives import DenseCausalLMAdapter
from .progress import commit_step
class DenseCPTTrainer:
    def __init__(self,model,optimizer,scheduler,optimization_config,*,unique_train_tokens:int,device='cuda',adapter=None):
        self.model=model; self.optimizer=optimizer; self.scheduler=scheduler; self.cfg=optimization_config; self.unique_train_tokens=unique_train_tokens; self.device=device; self.adapter=adapter or DenseCausalLMAdapter(); self.scaler=torch.amp.GradScaler('cuda',enabled=self.cfg.precision=='fp16' and str(device).startswith('cuda'))
    def _to_device(self,b): return {k:(v.to(self.device) if hasattr(v,'to') else v) for k,v in b.items() if k!='record_ids'}
    def train_optimizer_step(self,microbatches,progress):
        self.model.train(); self.optimizer.zero_grad(set_to_none=True); ex=toks=sup=0; loss_total=0.0
        dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(self.cfg.precision)
        for batch in microbatches:
            batch=self._to_device(batch); ex+=batch['input_ids'].shape[0]; toks+=int(batch['attention_mask'].sum()); sup+=int((batch['labels']!=-100).sum())
            ctx=torch.autocast('cuda',dtype=dtype) if dtype and str(self.device).startswith('cuda') else torch.autocast('cpu',enabled=False)
            with ctx: loss=self.adapter.compute_loss(self.model,batch).total_loss/self.cfg.gradient_accumulation_steps
            if not torch.isfinite(loss): raise FloatingPointError('TRAIN-NONFINITE-LOSS')
            self.scaler.scale(loss).backward(); loss_total+=float(loss.detach().cpu())
        self.scaler.unscale_(self.optimizer); grad=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.gradient_clip_norm))
        if not math.isfinite(grad): raise FloatingPointError('TRAIN-NONFINITE-GRADIENT')
        self.scaler.step(self.optimizer); self.scaler.update(); self.scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
        progress=commit_step(progress,examples=ex,tokens=toks,supervised_tokens=sup,unique_train_tokens=self.unique_train_tokens)
        return progress,{'loss':loss_total,'grad_norm':grad,'tokens':toks,'examples':ex,'lr':self.optimizer.param_groups[0]['lr']}
