"""Per-worker run measurements, only while the existing serialized GPU lock is held."""
from __future__ import annotations
import time,threading,contextlib,platform,sys
from pathlib import Path
from core.runtime_env import SHARED_ROOT,atomic_json

@contextlib.contextmanager
def measure_job(job,cfg):
    import torch
    record={'schema':1,'status':'running','load_policy':cfg.get('quantized_loading','reject'),
        'native_int8_benchmark':False,'memory_scope':'PyTorch CUDA allocator in this worker; RSS process sample, not system-wide peaks',
        'rss_sample_interval_seconds':0.05,'torch_version':torch.__version__,'python_version':sys.version.split()[0]}
    start=time.perf_counter();stop=threading.Event();samples=[0]
    try:
        import psutil
        proc=psutil.Process()
        def sample():
            while not stop.is_set():
                try:samples[0]=max(samples[0],proc.memory_info().rss)
                except (OSError,psutil.Error):pass
                stop.wait(.05)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
    except ImportError:thread=None
    cuda=torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        record['gpu']=torch.cuda.get_device_name(0)
    try:
        yield record;record['status']='completed'
    except BaseException:
        record['status']='failed';raise
    finally:
        if cuda:
            try:
                torch.cuda.synchronize()
                record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception:record['cuda_metrics_error']=True
        record['worker_seconds']=time.perf_counter()-start
        stop.set()
        if thread:thread.join(1)
        record['peak_rss_bytes']=samples[0] or None
        jid=job.get('job_id','')
        if len(jid)==32 and all(c in '0123456789abcdef' for c in jid):
            try:atomic_json(SHARED_ROOT/'cache/qwen-metrics'/(jid+'.json'),record)
            except OSError:pass
