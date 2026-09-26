from __future__ import annotations
import platform, torch

def collect_environment():
    out={'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info()
        out.update(gpu_name=props.name,total_vram=int(total),free_vram_start=int(free),cuda_runtime=torch.version.cuda,bf16_supported=bool(getattr(torch.cuda,'is_bf16_supported',lambda:False)()))
    try:
        import transformers; out['transformers']=transformers.__version__
    except Exception: pass
    return out
