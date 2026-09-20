"""Diagnostic only: device events, Python dispatch, hot-path IO and clean wall time."""
from pathlib import Path
import collections,importlib,json,os,sys,time
root=Path(os.environ['FMT02_NATIVE_ROOT']);unit=root/'repo/kernels/projects/a5/kda_layout'
sys.path.insert(0,str(unit));out=root/'receipts-batched/perf-attribution-v2-bd4';assert not out.exists()
sys.argv=['verify_native.py','--block-dim','4','--mode','compile','--output',str(out)]
import verify_native
verify_native.main()
import torch
import native_support as check
from ascend_fla.ops.kda import chunk,chunk_bwd
import ascend_fla.runtime.compile as compiler
import ascend_fla.runtime.binding as binding
auto=importlib.import_module('ascend_fla.ops.kda.autograd')
before,before_bwd,_=check.baseline(chunk,chunk_bwd,auto)
real,_=check.reference_modules();torch.npu.set_device(0)
rows=[]
for tokens in (1024,4096):
 x=real.make_inputs(B=1,H=32,HV=32,C=tokens//64,span=46,want_grads=True)
 dev={n:t.npu() for n,t in x.items()};data={n:dev[n] for n in ('q','k','v','g','beta')};data['initial_state']=dev['h0']
 options=dict(block_dim=4,layout_device='npu');caches=chunk.chunk_kda_fwd_with_caches(**data,**options)[2]
 bd={n:dev[n] for n in ('q','k','v','do','dht')};bd.update(beta=dev['beta'].bfloat16(),caches=caches)
 paths={
 'plain_forward':(lambda:before.chunk_kda_fwd(**data,output_final_state=True,**options),lambda:chunk.chunk_kda_fwd(**data,output_final_state=True,**options)),
 'cached_forward':(lambda:before.chunk_kda_fwd_with_caches(**data,**options),lambda:chunk.chunk_kda_fwd_with_caches(**data,**options)),
 'backward_existing_caches':(lambda:before_bwd.chunk_kda_bwd(**bd,**options),lambda:chunk_bwd.chunk_kda_bwd(**bd,**options))}
 for scope,(old,new) in paths.items():
  old();new();torch.npu.synchronize()
  # The clean end-to-end timing has no monkeypatch, profiler or per-op sync.
  rounds=[]
  for i in range(3):
   samples=[]
   for name,fn in [('old_before',old),('candidate',new),('old_after',old)]:
    torch.npu.synchronize();start=time.perf_counter();result=fn();torch.npu.synchronize()
    samples.append(dict(phase=name,milliseconds=(time.perf_counter()-start)*1000));del result
   rounds.append(samples)
  # Event tracing records into the same stream, one final sync only. Host
  # dispatch measurement excludes event.record() and device completion.
  events=[];original=compiler.CompiledKernel.__call__
  def profiled(op,inputs,scalars,outputs):
   start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
   start.record();clock=time.perf_counter();result=original(op,inputs,scalars,outputs);host_ms=(time.perf_counter()-clock)*1000;end.record()
   events.append((start,end,dict(operator=op.op.op_name,signature=op.signature,host_dispatch_ms=host_ms,
     input_bytes=sum(t.numel()*t.element_size() for t in inputs.values()),output_bytes=sum(t.numel()*t.element_size() for t in outputs.values()))))
   return result
  compiler.CompiledKernel.__call__=profiled
  try:
   torch.npu.synchronize();start=time.perf_counter();result=new();torch.npu.synchronize();event_wall=(time.perf_counter()-start)*1000
  finally:compiler.CompiledKernel.__call__=original
  costs=[]
  for begin,end,row in events:row['device_event_ms']=begin.elapsed_time(end);costs.append(row)
  counters=collections.Counter();patches=[]
  def counted(owner,name):
   previous=getattr(owner,name)
   def call(*a,**kw):counters[name]+=1;return previous(*a,**kw)
   patches.append((owner,name,previous));setattr(owner,name,call)
  for owner,names in [(compiler,('_kernel_source','_signature','compile_kernel')),(binding,('register_custom_opp_path',)),(Path,('read_text','read_bytes','rglob'))]:
   for name in names:counted(owner,name)
  try:
   for _ in range(3):new()
   torch.npu.synchronize()
  finally:
   for owner,name,previous in reversed(patches):setattr(owner,name,previous)
  row=dict(tokens=tokens,scope=scope,clean_rounds=rounds,clean_sync='only before/after complete public call; no diagnostic instrumentation',
   event_rows=costs,event_instrumented_wall_ms=event_wall,hot_three_call_counters=dict(counters),
   layout_cache_info=chunk._layout_runtime().prepare.cache_info()._asdict())
  rows.append(row);(out/'attribution.json').write_text(json.dumps(dict(device='dev-B',block_dim=4,rows=rows,complete=False),indent=2)+'\n')
  print('ATTRIBUTION_PASS',tokens,scope,flush=True)
(out/'attribution.json').write_text(json.dumps(dict(device='dev-B',block_dim=4,rows=rows,complete=True),indent=2)+'\n')
