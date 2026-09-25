from __future__ import annotations
import time
import torch

@torch.no_grad()
def benchmark_encoder(encoder,feature_ids,iterations:int=50):
    device=feature_ids.device
    if device.type=='cuda':torch.cuda.synchronize(device)
    for _ in range(5):encoder(feature_ids)
    if device.type=='cuda':torch.cuda.synchronize(device)
    t0=time.perf_counter()
    for _ in range(iterations):encoder(feature_ids)
    if device.type=='cuda':torch.cuda.synchronize(device)
    return {'iterations':iterations,'seconds':time.perf_counter()-t0,'milliseconds_per_forward':1000*(time.perf_counter()-t0)/iterations}

def padding_utilization(lengths,padded_length):
    if not lengths or padded_length<=0:return 0.0
    return sum(lengths)/(len(lengths)*padded_length)
