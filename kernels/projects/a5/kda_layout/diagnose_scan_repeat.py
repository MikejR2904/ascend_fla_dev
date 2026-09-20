"""Located autograd byte mismatch; same source/input, all vendors first."""
from pathlib import Path
import importlib,json,os,sys,hashlib
root=Path(os.environ['FMT02_NATIVE_ROOT']);unit=root/'repo/kernels/projects/a5/kda_layout'
sys.path.insert(0,str(unit))
out=root/'receipts/scan-probe-bd4';assert not out.exists()
sys.argv=['verify_native.py','--block-dim','4','--mode','compile','--output',str(out)]
import verify_native
verify_native.main()
import torch
import native_support as check
from ascend_fla.ops.kda import chunk,chunk_bwd
auto=importlib.import_module('ascend_fla.ops.kda.autograd')
torch.npu.set_device(0)
before,before_bwd,before_auto=check.baseline(chunk,chunk_bwd,auto)
real,_=check.reference_modules()
x=real.make_inputs(B=2,H=2,HV=4,C=3,span=46,want_grads=True)
names=('q','k','v','g','beta','h0')
def run(module):
 leaves={n:x[n].npu().requires_grad_(True) for n in names}
 original=module.chunk_kda_bwd;snapshots={}
 def capture(**kwargs):
  snapshots['caches']=check.cpu(kwargs['caches'])
  snapshots['inputs']=check.cpu({n:kwargs[n] for n in ('q','k','v','do','dht','beta')})
  from ascend_fla.runtime.compile import CompiledKernel
  original_call=CompiledKernel.__call__
  def scan_call(op,inputs,scalars,outputs):
   if 'dh0' not in outputs:return original_call(op,inputs,scalars,outputs)
   snapshots['scan_inputs']=check.cpu(inputs)
   result=original_call(op,inputs,scalars,outputs);torch.npu.synchronize()
   snapshots['scan_outputs']=check.cpu(outputs)
   snapshots['scan_op']=(op,inputs,scalars,outputs)
   return result
  CompiledKernel.__call__=scan_call
  try:grads=original(**kwargs);torch.npu.synchronize()
  finally:CompiledKernel.__call__=original_call
  snapshots['bf16']=check.cpu(grads)
  snapshots['torch_widen']=grads['dh0'].float().cpu()
  return grads
 module.chunk_kda_bwd=capture
 try:
  with check.instrument(audit=False) as (_,launches):
   o,ht=module.chunk_kda(*(leaves[n] for n in ('q','k','v','g','beta')),initial_state=leaves['h0'],output_final_state=True,block_dim=4,layout_device='npu')
   gradients=torch.autograd.grad((o,ht),[leaves[n] for n in names],grad_outputs=(x['do'].npu(),x['dht'].float().npu()))
  torch.npu.synchronize()
 finally:module.chunk_kda_bwd=original
 snapshots['returned']=check.cpu(dict(o=o,final_state=ht,**dict(zip(names,gradients))))
 snapshots['launches']=launches
 return snapshots
results={n:run(m) for n,m in [('candidate1',auto),('before1',before_auto),('candidate2',auto),('before2',before_auto)]}
op,inputs,scalars,outputs=results['candidate1'].pop('scan_op')
replays=[]
for i in range(12):
 outputs={n:torch.empty_like(t) for n,t in outputs.items()}
 with check.instrument(audit=False):op(inputs,scalars,outputs)
 torch.npu.synchronize();replays.append(check.cpu(outputs))
for row in results.values():row.pop('scan_op',None)
torch.save(dict(inputs=x,results=results,scan_replays=replays),out/'actual-tensors.pt')
def compare(a,b):
 a=a.contiguous();b=b.contiguous();bitsa=a.view(torch.int32 if a.dtype==torch.float32 else torch.int16).reshape(-1);bitsb=b.view(bitsa.dtype).reshape(-1)
 indices=(bitsa!=bitsb).nonzero().reshape(-1)
 av=a.reshape(-1);bv=b.reshape(-1)
 return dict(exact=not len(indices),count=len(indices),finite=bool(a.isfinite().all() and b.isfinite().all()),max_abs=float((a.float()-b.float()).abs().max()),samples=[dict(index=int(i),a=float(av[i]),b=float(bv[i]),a_bits=int(bitsa[i]),b_bits=int(bitsb[i])) for i in indices[:24]])
report={}
for name,r in results.items():
 report[name]=dict(native_vs_cpu_widen=compare(r['returned']['h0'],r['bf16']['dh0'].float()),torch_vs_cpu_widen=compare(r['torch_widen'],r['bf16']['dh0'].float()))
for a,b in [('candidate1','before1'),('candidate1','candidate2'),('before1','before2')]:
 report[a+'_'+b]=dict(returned={n:compare(results[a]['returned'][n],results[b]['returned'][n]) for n in ('h0',)},bf16=compare(results[a]['bf16']['dh0'],results[b]['bf16']['dh0']),inputs=check.exact(results[a]['inputs'],results[b]['inputs']))
for a,b in [('candidate1','before1'),('candidate1','candidate2'),('before1','before2')]:
 report[a+'_'+b]['caches']=check.exact(results[a]['caches'],results[b]['caches'])
 report[a+'_'+b]['scan_inputs']=check.exact(results[a]['scan_inputs'],results[b]['scan_inputs'])
 report[a+'_'+b]['scan_outputs']={n:compare(results[a]['scan_outputs'][n],results[b]['scan_outputs'][n]) for n in results[a]['scan_outputs']}
report['replays']=[{n:compare(v[n],replays[0][n]) for n in v} for v in replays]
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
