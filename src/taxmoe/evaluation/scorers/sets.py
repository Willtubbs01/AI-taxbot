from __future__ import annotations

def set_metrics(predicted, gold):
    p=set(predicted or []); g=set(gold or []); inter=len(p&g)
    if not p and not g: precision=recall=f1=1.0
    else:
        precision=inter/len(p) if p else 0.0; recall=inter/len(g) if g else 0.0; f1=(2*precision*recall/(precision+recall)) if precision+recall else 0.0
    return {'precision':precision,'recall':recall,'f1':f1,'exact':float(p==g)}
