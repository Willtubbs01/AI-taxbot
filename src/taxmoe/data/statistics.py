from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class LengthStatistics:
    count:int
    total:int
    mean:float
    median:float
    p90:int
    p95:int
    p99:int
    maximum:int

def _pct(xs,p):
    if not xs:return 0
    i=max(0,min(len(xs)-1,math.ceil(p*len(xs))-1))
    return xs[i]

def length_statistics(lengths):
    xs=sorted(int(x) for x in lengths)
    if not xs:return LengthStatistics(0,0,0.0,0.0,0,0,0,0)
    n=len(xs); med=(xs[(n-1)//2]+xs[n//2])/2
    return LengthStatistics(n,sum(xs),sum(xs)/n,med,_pct(xs,.90),_pct(xs,.95),_pct(xs,.99),xs[-1])
