"""PM-requested original public-path hash distribution, without internal barriers."""
from pathlib import Path
import collections,importlib,json,os,sys
root=Path(os.environ['FMT02_NATIVE_ROOT']);unit=root/'repo/kernels/projects/a5/kda_layout'
sys.path.insert(0,str(unit));out=root/'receipts/old-public-repeats-bd4';assert not out.exists()
sys.argv=['verify_native.py','--block-dim','4','--mode','compile','--output',str(out)]
import verify_native
verify_native.main()
import torch,native_support as check
from ascend_fla.ops.kda import chunk,chunk_bwd
auto=importlib.import_module('ascend_fla.ops.kda.autograd')
torch.npu.set_device(0)
before,before_bwd,before_auto=check.baseline(chunk,chunk_bwd,auto)
real,_=check.reference_modules();x=real.make_inputs(B=2,H=2,HV=4,C=3,span=46,want_grads=True)
names=('q','k','v','g','beta','h0');oracle=check.gradients(x,real.kda_recurrent_ref)['dh0']
rows=[];tensors={}
for label,module in [('before',before_auto),('candidate',auto)]:
 for repeat in range(12):
  leaves={n:x[n].npu().requires_grad_(True) for n in names}
  with check.instrument(audit=False) as (_,launches):
   with check.Audit():
    o,ht=module.chunk_kda(*(leaves[n] for n in ('q','k','v','g','beta')),initial_state=leaves['h0'],output_final_state=True,block_dim=4,layout_device='npu')
   gradients=torch.autograd.grad((o,ht),[leaves[n] for n in names],grad_outputs=(x['do'].npu(),x['dht'].float().npu()))
  torch.npu.synchronize()
  got=check.cpu(dict(o=o,final_state=ht,**dict(zip(names,gradients))))
  tensors[f'{label}-{repeat}']=got['h0']
  row=dict(path=label,repeat=repeat,sha256={n:check.digest(t) for n,t in got.items()},
           input_unchanged=all(check.digest(leaves[n])==check.digest(x[n]) for n in names),
           h0_cpu_fp32=check.metrics(got['h0'],oracle,.05))
  rows.append(row);print(json.dumps(row),flush=True)
torch.save(dict(inputs=x,h0=tensors,cpu_fp32=oracle),out/'actual-h0.pt')
summary=dict(source='exact predecessor public autograd; original poison pattern; no added internal synchronization',
 rows=rows,h0_hash_distribution={label:dict(collections.Counter(r['sha256']['h0'] for r in rows if r['path']==label)) for label in ('before','candidate')})
(out/'repeats.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['h0_hash_distribution']),flush=True)
