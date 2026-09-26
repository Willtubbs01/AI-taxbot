from __future__ import annotations

def differing_paths(a,b,prefix=''):
    diffs=[]; keys=set(a)|set(b)
    for k in sorted(keys):
        p=f'{prefix}.{k}' if prefix else k
        if k not in a or k not in b: diffs.append(p); continue
        x,y=a[k],b[k]
        if isinstance(x,dict) and isinstance(y,dict): diffs.extend(differing_paths(x,y,p))
        elif x!=y: diffs.append(p)
    return diffs

def validate_one_variable_change(baseline,variant,allowed_prefix):
    diffs=differing_paths(baseline,variant)
    bad=[p for p in diffs if not (p==allowed_prefix or p.startswith(allowed_prefix+'.'))]
    if bad: raise ValueError('ABLATION-MULTIPLE-VARIABLES:'+','.join(bad))
    return diffs
