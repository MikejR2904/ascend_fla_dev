"""Synchronized existing-FP32 / native-BF16 / existing-FP32 API timings."""
import hashlib
import types
from pathlib import Path
import statistics
import time
import torch


def measure(api,ref,checks,bd,to_device):
    before_path=Path(__file__).parent/'evidence/fp32-entry-before.py'
    before=types.ModuleType('ascend_fla.ops._bf04_perf_before')
    before.__file__=api.__file__;before.__package__='ascend_fla.ops'
    exec(compile(before_path.read_text(),str(before_path),'exec'),before.__dict__)
    before._module=api._module;before._compiled=api._compiled
    rows=[]
    for length in (1024,4096):
        bf=ref.make_inputs(dict(id='perf',seed=10000+length,parameters=dict(B=1,T=length,H=8)))
        fp=ref.fp32_inputs(bf)
        bfdev=to_device(bf);fpdev=to_device(fp)
        def invoke(data):
            operator=before if data is fpdev else api
            return operator.chunk_precond_kda(**data,output_final_state=True,block_dim=bd)
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
            print('FP32_BF16_PERF_ROUND',length,index+1,result,flush=True)
        got={n:x.cpu() for n,x in zip(('o','final_state','final_A_state'),invoke(bfdev))}
        check=checks.compare(got,checks.references(bf));assert check['passed'],check
        # Also retain the user's Torch-NPU eager baseline. All of this is
        # benchmark code, outside the audited public implementation.
        import importlib
        naive=importlib.import_module(checks.__package__+'.oracles').fla_naive()
        def torch_baseline():return naive(**bfdev,output_final_state=True)
        base=torch_baseline();torch.npu.synchronize()
        base_check=checks.compare(dict(zip(('o','final_state','final_A_state'),(x.cpu() for x in base))),checks.references(bf))
        assert base_check['passed'],base_check
        torch_rounds=[]
        for index in range(3):
            row={}
            for label,fn,count in (('torch_npu_before',torch_baseline,2),('bf16',lambda:invoke(bfdev),3),('torch_npu_after',torch_baseline,2)):
                samples=[]
                for _ in range(count):
                    torch.npu.synchronize();start=time.perf_counter();fn();torch.npu.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
                row[label]=dict(milliseconds=samples,median_ms=statistics.median(samples))
            baseline=(row['torch_npu_before']['median_ms']+row['torch_npu_after']['median_ms'])/2
            row['bf16_speedup_vs_torch_npu']=baseline/row['bf16']['median_ms'];torch_rounds.append(row)
            print('TORCH_NPU_PERF_ROUND',length,index+1,row,flush=True)
        rows.append(dict(T=length,B=1,H=8,block_dim=bd,rounds=rounds,correctness=check,
                         torch_npu_rounds=torch_rounds,torch_npu_correctness=base_check))
    return dict(passed=True,baseline='Recorded pre-BF04 FP32 public wrapper and unchanged five FP32 kernels; BF16-rounded inputs widened only in verifier',baseline_source_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest(),additional_baseline='Pinned FLA naive recurrence on Torch NPU eager, actual BF16 inputs and FP32 recurrence',timing='same device, synchronized API wall time including numeric validation and allocations',cases=rows)
