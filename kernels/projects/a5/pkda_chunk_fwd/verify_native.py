"""Actual in-process NPU acceptance; one block_dim per process.

Run under the machine controller's held device lock after health/occupancy and
accepted Python/library/CANN identity checks. Machine configuration stays
external. This script generates inputs and both independent goldens on CPU.
"""
import argparse, hashlib, json, platform, sys, time, traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--block-dim',type=int,choices=(1,2,3,4),required=True);p.add_argument('--mode',choices=['compile','full','suite','tails','perf'],default='full');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
unit_root=Path(__file__).resolve().parent
out=a.output;out.mkdir(parents=True,exist_ok=True)
def write(name,data):
 path=out/(name+'.json');temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
import torch, torch_npu, ascriptor
from ascend_fla.ops import pkda_chunk_fwd as api
refstages=api._module('ref.stages');pipeline=api._module('kernels.pipeline');oracles=api._module('ref.oracles');ref=api._module('ref.reference')
sys.path.insert(0,str(unit_root));from _unit_runner import compare_outputs
contract=json.loads((unit_root/'contract.json').read_text())
source_files=[*unit_root.glob('kernels/*.py'),*unit_root.glob('ref/*.py'),Path(api.__file__), *[unit_root/n for n in ('verify_native.py','native_checks.py','native_perf.py')]]
write('environment',dict(python=platform.python_version(),torch=torch.__version__,torch_npu=torch_npu.__version__,ascriptor_version=ascriptor.__version__,source_sha256={str(f.relative_to(unit_root)) if f.is_relative_to(unit_root) else 'ascend_fla/ops/pkda_chunk_fwd.py':hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}))
torch.set_num_threads(1)
from ascend_fla.runtime.compile import compile_kernel
compiled={};builds=[]
for entry in pipeline.entries():
 print('COMPILE_START',entry.name,'bd',a.block_dim,flush=True);start=time.monotonic()
 op=compile_kernel(entry,device='a5',block_dim=a.block_dim,backend='cce');compiled[entry.name]=op
 files={str(f.relative_to(op.vendor_dir)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(op.vendor_dir.rglob('*')) if f.is_file() and f.suffix in ('.so','.o','.json')}
 builds.append(dict(entry=entry.name,signature=op.signature,seconds=time.monotonic()-start,vendor_files=files))
 write('compile',dict(complete=len(builds)==5,block_dim=a.block_dim,entries=builds));print('COMPILE_PASS',entry.name,builds[-1]['seconds'],flush=True)
api.prepare(block_dim=a.block_dim)
assert len(compiled)==5
if a.mode=='compile':sys.exit(0)
torch.npu.set_device(0)
rows=[]
def cpu(t):return t.detach().cpu().contiguous()
def execute_case(case):
 print('CASE_START',case['id'],case['parameters'],flush=True)
 inputs=ref.make_inputs(case)
 from native_checks import prepare_case
 inputs=prepare_case(inputs,case);ref.validate_inputs(inputs)
 expected_stages=refstages.reference_stages(inputs);expected={n:expected_stages[n] for n in ref.OUTPUTS};naive=oracles.oracle(inputs)
 dev={n:t.npu() if isinstance(t,torch.Tensor) else t for n,t in inputs.items()}
 stages={};timings={}
 def launch(entry,sources,outputs,scalars):
  for t in outputs.values():t.fill_(float('nan'))
  torch.npu.synchronize();start=time.monotonic();op=compiled[entry.name]
  op(sources,{n:scalars[n] for n in op.scalar_names},outputs);torch.npu.synchronize()
  timings[entry.name]=time.monotonic()-start
  print('STAGE_EXECUTED',case['id'],entry.name,timings[entry.name],flush=True)
  return outputs
 stages=pipeline.run(dev,launch,retain_stages=True)
 got={n:cpu(t) for n,t in stages.items()}
 metrics=oracles.metrics(got,expected_stages)
 hashes={n:hashlib.sha256(t.numpy().tobytes()).hexdigest() for n,t in got.items()}
 public=api.chunk_precond_kda(**dev,output_final_state=True,block_dim=a.block_dim);torch.npu.synchronize()
 public={n:cpu(t) for n,t in zip(ref.OUTPUTS,public)}
 row=dict(case=case,block_dim=a.block_dim,input_sha256={n:hashlib.sha256(t.numpy().tobytes()).hexdigest() for n,t in inputs.items() if isinstance(t,torch.Tensor)},stage_metrics=metrics,stage_sha256=hashes,stage_seconds=timings,vs_chunk=oracles.metrics(public,expected),vs_fla_naive=oracles.metrics(public,naive),passed=False)
 write(case['id'],row)
 try:
  compare_outputs(got,expected_stages,contract,stage=True)
  compare_outputs(public,expected,contract);compare_outputs(public,naive,contract)
  assert all(torch.equal(public[n],got[n]) for n in public), 'public vs staged path differs'
  assert all(torch.equal(cpu(dev[n]),t) for n,t in inputs.items() if isinstance(t,torch.Tensor)), 'input mutation'
 except BaseException:
  torch.save(dict(inputs=inputs,stages=got,public=public,expected=expected_stages,naive=naive),out/(case['id']+'-failure.pt'));raise
 row['passed']=True;write(case['id'],row);rows.append(row);write('summary',dict(passed=False,complete=False,cases=rows))
 print('CASE_PASS',case['id'],json.dumps(row['vs_fla_naive']),flush=True)
cases=[next(c for c in contract['cases'] if c['id']=='t4096_h8')]
if a.mode=='suite':
 from native_checks import extra_cases,public_boundaries
 cases.extend(c for c in contract['cases'] if c['id']!='t4096_h8')
 cases.extend(extra_cases())
if a.mode=='tails':
 from native_checks import tail_cases
 cases.extend(tail_cases())
try:
 for case in cases:execute_case(case)
 if a.mode=='suite':
  boundaries=public_boundaries(api,ref,oracles,compare_outputs,contract,a.block_dim)
  write('public-boundaries',dict(passed=True,cases=boundaries));print('PUBLIC_BOUNDARIES_PASS',len(boundaries),flush=True)
 if a.mode=='perf':
  from native_perf import measure
  perf=measure(api,ref,oracles,compare_outputs,contract,a.block_dim,write)
  write('performance',dict(passed=True,cases=perf))
 write('summary',dict(passed=True,complete=True,cases=rows))
 print('NATIVE_PASS',len(rows),'bd',a.block_dim,flush=True)
except BaseException as e:
 write('failure',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),completed=[r['case']['id'] for r in rows]));raise
