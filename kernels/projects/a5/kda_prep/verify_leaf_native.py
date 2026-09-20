"""Native dtype/range/tail matrix, after the first full Kimi workload."""
import argparse
import hashlib
import json
import math
import os
import platform
import types
import time
from pathlib import Path

import torch
import torch_npu
import ascriptor

from ascend_fla.ops.kda import chunk,chunk_bwd,fused_recurrent,prepare
from ascend_fla.runtime.compile import compile_kernel
import native_environment
import native_cases
import unit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--decode-block-dim',type=int,choices=(1,2,4,8,16,28))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bd=args.block_dim;dbd=args.decode_block_dim or (4 if bd==3 else bd)
    assert os.environ.get('BF07_EXTERNAL_DEVICE_LOCK')=='1'
    assert platform.python_version()=='3.12.14' and torch.__version__=='2.12.0+cu130' and torch_npu.__version__=='2.12.0'
    assert Path(ascriptor.__file__).is_relative_to(Path(os.environ['BF07_NATIVE_ROOT'])/'library')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    environment=native_environment.collect()
    environment['driver_source_sha256']={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                        for p in (__file__,unit.__file__,native_cases.__file__)}
    environment['source_sha256']={str(p.relative_to(unit.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in unit.ROOT.rglob('*.py')}

    def clean(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [clean(v) for v in value]
        return value

    def write(name,value):
        p=args.output/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(clean(dict(environment=environment,**value)),indent=2,allow_nan=False)+'\n');tmp.replace(p)

    runtime=chunk._prep_runtime()
    plan=[('prep_chunk',e,bd) for e in runtime.kernels('chunk').values()]
    plan += [('prep_decode',e,dbd) for e in runtime.kernels('decode').values()]
    plan += [('layout',e,bd) for e in chunk._layout_runtime().kernels().values()]
    plan += [('forward',e,bd) for e in chunk.kda_fwd_kernels().values()]
    plan += [('backward',e,bd) for e in chunk_bwd.kda_bwd_kernels().values()]
    plan += [('decode',fused_recurrent._native_kernel(d),dbd) for d in (torch.bfloat16,torch.float32)]
    assert len(plan)==50 and sum(f=='backward' for f,_,_ in plan)==9
    builds=[]
    for family,entry,dim in plan:
        print('COMPILE_START',family,entry.name,dim,flush=True);start=time.monotonic()
        op=compile_kernel(entry,device='a5',block_dim=dim,backend='cce')
        builds.append(dict(family=family,entry=entry.name,block_dim=dim,signature=op.signature,seconds=time.monotonic()-start))
        write('compile',dict(complete=len(builds)==50,entries=builds))
    prepare(block_dim=bd,backward=True,decode=True,decode_block_dim=dbd)
    torch.npu.set_device(0)
    precision=unit.module('precision');budgets=json.loads((unit.ROOT/'budgets.json').read_text())
    baseline_path=unit.ROOT/'baseline/chunk.py'
    baseline_receipt=json.loads((unit.ROOT/'baseline/source.json').read_text())
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==baseline_receipt['files']['chunk.py']['sha256']
    predecessor=types.ModuleType('ascend_fla.ops.kda._bf07_leaf_predecessor')
    predecessor.__package__='ascend_fla.ops.kda';predecessor.__file__=chunk.__file__
    exec(compile(baseline_path.read_text(),str(baseline_path),'exec'),predecessor.__dict__)
    environment['predecessor_sha256']=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    rows=[]
    for namespace,dim in (('chunk',bd),('decode',dbd)):
        for case in native_cases.leaf_cases():
            case_id=namespace+'_'+case['id'];print('CASE_START',case_id,flush=True)
            values=case['inputs'];source=values['source'];n=source.numel()
            guarded={};inputs={};original={}
            for name,value in values.items():
                cpu=torch.full((value.numel()+256,),-7.,dtype=value.dtype)
                cpu[128:-128]=value.reshape(-1)
                guarded[name]=cpu.npu();inputs[name]=guarded[name][128:-128].view(1,-1)
                original[name]=precision.digest(cpu)
            dtype=unit.DTYPES[case['types'].split('_')[1]] if case['kind']=='norm' else torch.float32
            output_cpu=torch.full((n+256,),13.,dtype=dtype);output_cpu[128:-128]=float('nan')
            output=output_cpu.npu();destination=output[128:-128].view(1,n)
            scalars={'N':n}
            if case['kind']=='gate':scalars.update(HV=source.shape[-2],Channels=source.shape[-2]*128,BT=source.shape[0]*source.shape[1])
            runtime.prepare('a5',dim,namespace)[case['kind']+'_'+case['types']](inputs,scalars,{'destination':destination})
            torch.npu.synchronize()
            actual=destination.cpu();whole=output.cpu()
            preserved={name:precision.digest(value.cpu())==original[name] for name,value in guarded.items()}
            canaries=bool(torch.equal(whole[:128],output_cpu[:128]) and torch.equal(whole[-128:],output_cpu[-128:]))
            unit_inputs=dict(values=values,parameters=dict(kind=case['kind'],types=case['types']))
            expected=unit.reference(unit_inputs);old=unit.host_reference(unit_inputs)
            high=precision.metrics(actual,expected['destination'])
            host=precision.metrics(actual,old['destination'])
            semantic=native_cases.semantic_comparison(actual,old['destination'])
            # Execute the byte-pinned predecessor on the same NPU. CPU FP32
            # remains the independent semantic reference; device differences
            # are reported separately and never redefine the frozen budget.
            original_device={name:value.view(values[name].shape) for name,value in inputs.items()}
            source_device=original_device['source']
            old_kwargs={}
            if case['kind']=='norm':
                old_kwargs.update(use_qk_l2norm_in_kernel=True,qk_dtype=dtype)
                old_index=0
            elif case['kind']=='gate':
                old_kwargs.update(use_gate_in_kernel=True,A_log=original_device['alog'],dt_bias=original_device['bias'])
                old_index=2
            else:
                old_kwargs.update(use_beta_sigmoid_in_kernel=True)
                old_index=3
            with torch.no_grad():
                old_native=predecessor._prepare_inputs(source_device,source_device,source_device,source_device,**old_kwargs)[old_index]
            torch.npu.synchronize();old_native=old_native.cpu().reshape(1,-1)
            native_semantic=native_cases.semantic_comparison(actual,old_native)
            old_native_vs_cpu=native_cases.semantic_comparison(old_native,old['destination'])
            if case['numerical']:
                result=unit.compare(unit_inputs,{'destination':actual},expected,budgets)
                numerical_pass=result['passed']
            else:
                numerical_pass=None
            masks=all(semantic[k] for k in ('nan_mask_equal','positive_infinity_mask_equal','negative_infinity_mask_equal'))
            # Range mismatches are explicit unresolved records, never swept into
            # the ordinary error floor or accepted by an additive tolerance.
            semantic_exact=masks and semantic['finite_differing_values']==0 and semantic['finite_sign_differences']==0
            endpoint_qualification=None
            if not case['numerical']:
                a=actual.reshape(-1);high_value=expected['destination'].reshape(-1);cpu_value=old['destination'].reshape(-1)
                tiny_value=torch.finfo(torch.float32).tiny
                normal=(torch.isfinite(a)&torch.isfinite(high_value)&torch.isfinite(cpu_value)
                        &(cpu_value.abs()>=tiny_value)&(high_value.abs()>=tiny_value))
                normal_metrics=None;normal_pass=None
                if bool(normal.any()):
                    normal_metrics=precision.metrics(a[normal],high_value[normal])
                    cpu_metrics=precision.metrics(a[normal],cpu_value[normal])
                    limit=budgets['limits'][case['kind']+':'+case['types']]
                    normal_pass=(normal_metrics['relative_l2'] is not None
                        and normal_metrics['relative_l2']<=limit['relative_l2']
                        and normal_metrics['max_relative_nonzero']<=limit['max_relative_nonzero'])
                    if a.dtype==torch.bfloat16:
                        normal_pass=normal_pass and all(m['rounded_reference_over_one_ulp']==0 for m in (normal_metrics,cpu_metrics))
                hard_masks=all(comparison[k] for comparison in (semantic,native_semantic) for k in
                    ('nan_mask_equal','positive_infinity_mask_equal','negative_infinity_mask_equal'))
                hard_signs=semantic['finite_sign_differences']==0 and native_semantic['finite_sign_differences']==0
                endpoint_qualification=dict(owner_comment='https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5749276712',
                    normal_finite_count=int(normal.sum()),normal_finite_to_fp64=normal_metrics,normal_finite_pass=normal_pass,
                    masks_match_both_predecessors=hard_masks,signs_match_both_predecessors=hard_signs,
                    passed=hard_masks and hard_signs and normal_pass is not False,
                    scope='Recorded native underflow/overflow/saturation; no CPU-subnormal bitwise requirement. Nonzero normal finite members retain frozen budgets. Empty normal subsets make no numerical-accuracy claim. Cross-bd/namespace hashes are separately required.')
            row=dict(case={k:v for k,v in case.items() if k!='inputs'},namespace=namespace,block_dim=dim,
                     input_shapes={name:list(t.shape) for name,t in values.items()},
                     input_sha256={name:precision.digest(t) for name,t in values.items()},
                     actual_sha256=precision.digest(actual),to_fp64=high,to_predecessor_cpu_fp32=host,
                     semantic=semantic,to_predecessor_npu=precision.metrics(actual,old_native),
                     predecessor_npu_semantic=native_semantic,predecessor_npu_vs_cpu=old_native_vs_cpu,endpoint_exact=semantic_exact,endpoint_qualification=endpoint_qualification,input_unchanged=preserved,
                     output_canaries_unchanged=canaries,numerical_pass=numerical_pass,
                     passed=all(preserved.values()) and canaries and (numerical_pass if case['numerical'] else endpoint_qualification['passed']))
            write('cases/'+case_id,row)
            if not row['passed']:
                torch.save(dict(inputs=values,actual=actual,predecessor=old,predecessor_npu=old_native,fp64=expected),args.output/(case_id+'.private.pt'))
            rows.append(dict(id=case_id,numerical=case['numerical'],passed=row['passed'],actual_sha256=row['actual_sha256']))
            write('summary',dict(complete=False,passed=False,cases=rows))
            print('CASE_RESULT',case_id,row['passed'],flush=True)
    write('summary',dict(complete=True,passed=all(r['passed'] for r in rows),cases=rows,
          note='PM5749276712 endpoint qualification: masks/signs and normal finite budgets enforced; native range behavior recorded. Cross-bd/namespace hashes remain a separate gate.'))
    print('LEAF_DONE',len(rows),sum(r['passed'] for r in rows),flush=True)
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
