"""Bounded source-file models, gated on a completed full native workload."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-full-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    native=json.loads(args.native_full_receipt.read_text())
    assert native['passed'] and native['case']['id']=='full_kimi_t4096'
    assert len(native['bitwise'])==19 and all(r['passed'] for r in native['bitwise'].values())
    environment=json.loads((args.native_full_receipt.parent/'environment.json').read_text())
    sha=hashlib.sha256((root/'kernels/move.py').read_bytes()).hexdigest()
    assert environment['source_sha256']['kernels/move.py']==sha

    import torch
    import unit
    import runtime
    from _unit_runner import compare_outputs,launch_kernel,fingerprint

    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads((root/'contract.json').read_text())
    ids={'cast_bf16_bf16','cast_bf16_f32','cast_f32_bf16','cast_f32_f32',
         'strided_bf16_bf16','strided_f32_bf16','broadcast_f32_f32',
         'broadcast_bf16_f32','zero_bf16','zero_f32'}
    cases=[copy.deepcopy(c) for c in contract['cases'] if c['id'] in ids]
    gate=copy.deepcopy(next(c for c in cases if c['id']=='cast_f32_bf16'))
    gate['id']='bounded_gate_multiply';gate['parameters']['multiply']=True;cases.append(gate)
    rows=[]
    for launcher in ('sim','pipesim'):
        for case in cases:
            inputs=unit.make_inputs(case)
            expected=unit.reference(inputs);before=fingerprint(inputs)
            options=dict(device='a5',backend='cce',launcher=launcher,block_dim=1,
                         sim_processes='threads',timeout=30.,board=None,
                         out_dir=str(args.output/(launcher+'-'+case['id'])))
            actual=unit.execute(inputs,options)
            comparison=compare_outputs(actual,expected,contract)
            assert fingerprint(inputs)==before
            row=dict(id=case['id'],stage=launcher,block_dim=1,parameters=case['parameters'],
                     comparison=comparison,evidence=options['_execution_evidence'],input_unchanged=True,passed=True)
            rows.append(row)
            (args.output/(launcher+'-'+case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        for source_dtype in ('bf16','f32'):
            for destination_dtype in ('bf16','f32'):
                # Two nontrivial outer axes, repeated work and a 64-value row;
                # 256 values retain permutation and UB reuse without a full grid.
                source=torch.arange(256,dtype=torch.float32).reshape(2,2,64).to(unit.DTYPES[source_dtype])
                expected={'destination':source.transpose(0,1).contiguous().to(unit.DTYPES[destination_dtype]).view(1,256)}
                output=torch.full((1,256),float('nan'),dtype=unit.DTYPES[destination_dtype])
                scalars=[256,256,1,2,2,64,0,0,64,128,1,*runtime.tile_scalars((1,1,2,2,64)).values()]
                key=source_dtype+'_'+destination_dtype
                if key=='f32_bf16':scalars += [0,1.]
                options=dict(device='a5',backend='cce',launcher=launcher,block_dim=1,
                             sim_processes='threads',timeout=30.,board=None,
                             out_dir=str(args.output/(launcher+'-permutation-'+key)))
                before=fingerprint(source)
                actual={'destination':launch_kernel(runtime.kernels()[key],
                            (source.view(1,256),output,*scalars),options)}
                comparison=compare_outputs(actual,expected,contract)
                assert fingerprint(source)==before
                row=dict(id='permutation_'+key,stage=launcher,block_dim=1,N=256,
                         comparison=comparison,evidence=options['_execution_evidence'],input_unchanged=True,passed=True)
                rows.append(row)
                (args.output/(launcher+'-permutation-'+key+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    summary=dict(passed=True,scope='bounded 128..1024 element vector leaves; no full-workload or performance claim',
                 native_full_sha256=hashlib.sha256(args.native_full_receipt.read_bytes()).hexdigest(),
                 kernel_sha256=sha,cases=rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(rows))))


if __name__=='__main__':main()
