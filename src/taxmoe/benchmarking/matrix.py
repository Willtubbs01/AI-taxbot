from __future__ import annotations
from .models import BenchmarkConfig,HardwareEnvelope,BenchmarkStatus

def safe_envelope(results, precision):
    rows=[r for r in results if r.precision==precision]
    lengths=sorted({r.sequence_length for r in rows}); maxs={}; recs={}; ck={}
    for L in lengths:
        ok=[r for r in rows if r.sequence_length==L and r.status==BenchmarkStatus.PASS]
        if not ok: maxs[L]=recs[L]=None; ck[L]=False; continue
        maxb=max(r.microbatch_size for r in ok); maxs[L]=maxb
        safe=[r for r in ok if r.peak_reserved_bytes is None or True]
        recs[L]=max(1,maxb//2) if maxb>1 else 1
        ck[L]=all(r.gradient_checkpointing for r in ok if r.microbatch_size==maxb)
    return HardwareEnvelope(precision=precision,max_safe_microbatch_by_length=maxs,recommended_microbatch_by_length=recs,checkpointing_required_by_length=ck)
