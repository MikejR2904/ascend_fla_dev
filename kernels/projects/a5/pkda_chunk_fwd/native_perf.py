"""Same-device synchronized end-to-end timing against pinned Torch NPU naive."""
import statistics,time,torch

def measure(api,ref,oracle,compare,contract,bd,write):
 rows=[]
 naive=oracle.fla_naive()
 def timer(fn,repeat):
  values=[]
  for _ in range(repeat):
   torch.npu.synchronize();start=time.perf_counter();result=fn();torch.npu.synchronize()
   values.append((time.perf_counter()-start)*1000)
  return dict(samples_ms=values,median_ms=statistics.median(values)),result
 for T in (1024,4096):
  case=dict(id=f'perf_t{T}_h8',seed=8196,parameters=dict(B=1,T=T,H=8,N=T//64))
  data=ref.make_inputs(case);gold=oracle.oracle(data)
  inputs={n:t.npu() for n,t in data.items()}
  baseline=lambda:naive(**inputs,output_final_state=True)
  candidate=lambda:api.chunk_precond_kda(**inputs,output_final_state=True,block_dim=bd)
  # Compile and setup already completed; warm each full implementation once.
  _,base_out=timer(baseline,1);_,got=timer(candidate,1)
  compare(dict(zip(ref.OUTPUTS,(t.cpu() for t in base_out))),gold,contract)
  compare(dict(zip(ref.OUTPUTS,(t.cpu() for t in got))),gold,contract)
  rounds=[]
  for n in range(3):
   before,_=timer(baseline,2);native,_=timer(candidate,3);after,_=timer(baseline,2)
   midpoint=(before['median_ms']+after['median_ms'])/2
   rounds.append(dict(round=n+1,torch_npu_before=before,pkda_public=native,torch_npu_after=after,ratio_pkda_over_baseline=native['median_ms']/midpoint))
   print('PERF_ROUND',T,n+1,native['median_ms'],midpoint,flush=True)
   write(f'perf-t{T}',dict(case=case,block_dim=bd,rounds=rounds,complete=len(rounds)==3,warmup=1,baseline_repeat=2,candidate_repeat=3,synchronized=True,baseline='pinned FLA naive PKDA recurrence on Torch NPU; Python eager; same FP32 mathematical contract',scope='end-to-end public API including validation and allocations, excluding input generation/transfers, CPU goldens and vendor compile; no speed threshold'))
  rows.append(dict(T=T,block_dim=bd,rounds=rounds))
 return rows
