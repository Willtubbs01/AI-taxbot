from __future__ import annotations
import statistics

def summarize_seed_metric(values):
    vals=[float(v) for v in values]
    return {'n':len(vals),'mean':statistics.mean(vals) if vals else None,'std':statistics.stdev(vals) if len(vals)>1 else 0.0,'min':min(vals) if vals else None,'max':max(vals) if vals else None}
def direction_consistency(baseline,values,higher_is_better=True):
    return sum((v>baseline) if higher_is_better else (v<baseline) for v in values),len(values)
