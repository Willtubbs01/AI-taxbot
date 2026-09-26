from __future__ import annotations
import time, torch
from .models import DenseBenchmarkResult,BenchmarkStatus

def _amp_dtype(name): return torch.bfloat16 if name=='bf16' else torch.float16

def run_dense_case(model, vocab_size:int, precision:str, seq_len:int, batch_size:int, gradient_checkpointing:bool=False, measured_steps:int=5, warmup_steps:int=2, optimizer=None, device='cuda'):
    if not torch.cuda.is_available(): return DenseBenchmarkResult(precision=precision,sequence_length=seq_len,microbatch_size=batch_size,gradient_checkpointing=gradient_checkpointing,status=BenchmarkStatus.UNSUPPORTED,error='CUDA unavailable')
    try:
        if gradient_checkpointing and hasattr(model,'gradient_checkpointing_enable'): model.gradient_checkpointing_enable()
        elif hasattr(model,'gradient_checkpointing_disable'): model.gradient_checkpointing_disable()
        if hasattr(model,'config'): model.config.use_cache=False
        model.train(); ids=torch.randint(0,vocab_size,(batch_size,seq_len),device=device); mask=torch.ones_like(ids); labels=ids.clone()
        scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
        torch.cuda.reset_peak_memory_stats()
        losses=[]
        for step in range(warmup_steps+measured_steps):
            if optimizer: optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0=time.perf_counter()
            with torch.autocast('cuda',dtype=_amp_dtype(precision)):
                out=model(input_ids=ids,attention_mask=mask,labels=labels); loss=out.loss
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            scaler.scale(loss).backward()
            if optimizer: scaler.step(optimizer); scaler.update()
            torch.cuda.synchronize(); dt=time.perf_counter()-t0
            if step>=warmup_steps: losses.append((float(loss.detach().cpu()),dt))
        avg_t=sum(x[1] for x in losses)/len(losses); avg_l=sum(x[0] for x in losses)/len(losses)
        return DenseBenchmarkResult(precision=precision,sequence_length=seq_len,microbatch_size=batch_size,gradient_checkpointing=gradient_checkpointing,optimizer_mode='optimizer' if optimizer else 'none',status=BenchmarkStatus.PASS,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),step_time_seconds=avg_t,tokens_per_second=batch_size*seq_len/avg_t,loss_mean=avg_l,finite=True)
    except torch.cuda.OutOfMemoryError as e:
        return DenseBenchmarkResult(precision=precision,sequence_length=seq_len,microbatch_size=batch_size,gradient_checkpointing=gradient_checkpointing,status=BenchmarkStatus.OOM,error=str(e))
    except FloatingPointError as e:
        return DenseBenchmarkResult(precision=precision,sequence_length=seq_len,microbatch_size=batch_size,gradient_checkpointing=gradient_checkpointing,status=BenchmarkStatus.NUMERICAL_FAILURE,error=str(e),finite=False)
    except Exception as e:
        return DenseBenchmarkResult(precision=precision,sequence_length=seq_len,microbatch_size=batch_size,gradient_checkpointing=gradient_checkpointing,status=BenchmarkStatus.ERROR,error=f'{type(e).__name__}:{e}')
