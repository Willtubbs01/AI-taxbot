from __future__ import annotations

def metric_delta(baseline,candidate,direction='higher'):
    raw=candidate-baseline
    return raw if direction=='higher' else -raw

def paired_counts(baseline_correct,candidate_correct):
    out={'unchanged_correct':0,'regression':0,'fix':0,'unchanged_wrong':0}
    for b,c in zip(baseline_correct,candidate_correct):
        if b and c: out['unchanged_correct']+=1
        elif b and not c: out['regression']+=1
        elif not b and c: out['fix']+=1
        else: out['unchanged_wrong']+=1
    return out
