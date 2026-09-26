from __future__ import annotations
import math, torch
from .models import TeacherForcedResult

def evaluate_teacher_forced(model,batches,device=None):
    model.eval(); results=[]
    with torch.no_grad():
        for batch in batches:
            ids=batch['input_ids']; labels=batch['labels']; mask=(labels!=-100)
            if device:
                ids=ids.to(device); labels=labels.to(device); am=batch['attention_mask'].to(device)
            else: am=batch['attention_mask']
            out=model(input_ids=ids,attention_mask=am,labels=labels); n=int(mask.sum().item()); mean=float(out.loss.detach().cpu());
            record_ids=batch.get('record_ids') or [f'batch:{len(results)}']
            results.append(TeacherForcedResult(record_id=str(record_ids[0]),supervised_token_count=n,loss_sum=mean*n,mean_loss=mean,finite=math.isfinite(mean)))
    return results

def aggregate_teacher_forced(results):
    tokens=sum(r.supervised_token_count for r in results); loss=sum(r.loss_sum for r in results)/tokens if tokens else float('nan')
    return {'supervised_tokens':tokens,'mean_loss':loss,'perplexity':math.exp(loss) if math.isfinite(loss) and loss<50 else float('inf')}
