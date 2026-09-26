from taxmoe.benchmarking.models import DenseBenchmarkResult,BenchmarkStatus
from taxmoe.benchmarking.matrix import safe_envelope
def test_safe_envelope():
    rows=[DenseBenchmarkResult(precision='fp16',sequence_length=512,microbatch_size=b,gradient_checkpointing=False,status=BenchmarkStatus.PASS) for b in (1,2,4)]
    e=safe_envelope(rows,'fp16'); assert e.max_safe_microbatch_by_length[512]==4 and e.recommended_microbatch_by_length[512]==2
