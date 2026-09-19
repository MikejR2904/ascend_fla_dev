"""Synchronized existing-FP32 / native-BF16 / existing-FP32 API timings."""
import statistics
import time
import torch


def measure(api,ref,checks,bd,to_device):
    rows=[]
    for length in (1024,4096):
        bf=ref.make_inputs(dict(id='perf',seed=10000+length,parameters=dict(B=1,T=length,H=8)))
        fp=ref.fp32_inputs(bf)
        bfdev=to_device(bf);fpdev=to_device(fp)
        def invoke(data):return api.chunk_precond_kda(**data,output_final_state=True,block_dim=bd)
        for data in (fpdev,bfdev):
            for _ in range(3):invoke(data)
        torch.npu.synchronize()
        rounds=[]
        for index in range(3):
            result={}
            for label,data in (('fp32_before',fpdev),('bf16',bfdev),('fp32_after',fpdev)):
                samples=[]
                for _ in range(5):
                    torch.npu.synchronize();start=time.perf_counter();got=invoke(data);torch.npu.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
                result[label]=dict(milliseconds=samples,median_ms=statistics.median(samples))
            baseline=(result['fp32_before']['median_ms']+result['fp32_after']['median_ms'])/2
            result['bf16_speedup_vs_fp32']=baseline/result['bf16']['median_ms'];rounds.append(result)
        got={n:x.cpu() for n,x in zip(('o','final_state','final_A_state'),invoke(bfdev))}
        check=checks.compare(got,checks.references(bf));assert check['passed'],check
        rows.append(dict(T=length,B=1,H=8,block_dim=bd,rounds=rounds,correctness=check))
    return dict(passed=True,baseline='Existing FP32 public PKDA, BF16-rounded inputs widened only in verifier; no Torch CPU timing baseline',timing='same device, synchronized API wall time including numeric validation and allocations',cases=rows)
