"""After full native acceptance: bounded output ownership/layout diagnostics."""
from pathlib import Path
import argparse,importlib,json,sys
import torch
from ascend_fla.ops import pkda_chunk_fwd as api
public=api._bf16_module();package=public.__package__
ref=importlib.import_module(package+'.ref.reference');checks=importlib.import_module(package+'.ref.checks')
entry=importlib.import_module(package+'.kernels.output').pkda_bf16_output
unit=Path(public.__file__).parent;sys.path.insert(0,str(unit))
from _unit_runner import launch_kernel
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=True)
case=dict(id='output_nd2nz_reuse_tail',seed=701,parameters=dict(B=1,T=129,H=1))
data=ref.make_inputs(case);refs=checks.references(data);stages=refs['stages']
sources=tuple(stages[n] for n in ('qn','gc','score','states','delta'))
rows=[]
for launcher in ('sim','pipesim'):
 output=torch.full((1,129,1,128),float('nan'),dtype=torch.bfloat16)
 options=dict(device='a5',backend='cce',block_dim=1,launcher=launcher,timeout=120,sim_processes='fork',board=None,out_dir=str(root/'output-probe'/launcher))
 print('START',launcher,flush=True)
 got=launch_kernel(entry,(*sources,output,1,129,1,3),options)
 actual=dict(refs['B'],o=got);result=checks.compare(actual,refs)
 rows.append(dict(stage=launcher,case=case,block_dim=1,result=result,evidence=options['_execution_evidence']))
 (root/'output-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert result['passed'],result
 print('PASS',launcher,result['errors']['A']['o'],flush=True)
