"""Native returned-output acceptance; caller owns machine binding and lock.

CPU A oracles may overlap subsequent validation with bounded CPU workers.
They receive only CPU tensors; timing measurements use a separate program.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import torch

from ascend_fla.ops.gdn_chunk_bwd import prepare,chunk_gdn_bwd,_compiled,_pipeline
from ref.calibrate import inputs
from ref.oracle import autograd
from ref.reference import NAMES,acceptable,metrics
from ref.stages import reference_stages


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def all_metrics(a,b):
    result={}
    for n in b:
        assert a[n].shape == b[n].shape and a[n].dtype == b[n].dtype
        x,y=a[n].detach().cpu().double(),b[n].double()
        result[n]=dict(relative_l2=((x-y).norm()/y.norm().clamp_min(1e-30)).item(),
                       max_abs=(x-y).abs().max().item(),finite=bool(torch.isfinite(x).all()))
    return result


def cpu_a(values):
    assert torch.get_num_threads() == 1
    assert all(x.device.type=='cpu' for x in values)
    return autograd(*values)


def selected_cases(args):
    if args.case:
        contract=json.loads((Path(__file__).parent/'contract.json').read_text())
        chosen=[c for c in contract['cases'] if c['block_dim']==args.block_dim and ('all' in args.case or c['id'] in args.case)]
        if not chosen:raise ValueError('no cases selected for this block_dim')
        if 'all' not in args.case and set(args.case)!={c['id'] for c in chosen}:raise ValueError('unknown/wrong-block-dim case')
        chosen.sort(key=lambda c: c['id'] != f'full_r1_both_bd{args.block_dim}')
        return chosen
    return [dict(id='first_full',seed=91000,parameters=dict(B=args.B,T=args.T,H=args.H,HV=args.HV,mode=args.mode,kind=args.kind))]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--block-dim',type=int,choices=(1,2),default=2)
    p.add_argument('--T',type=int,default=4096);p.add_argument('--B',type=int,default=1)
    p.add_argument('--H',type=int,default=8);p.add_argument('--HV',type=int,default=8)
    p.add_argument('--mode',choices=('do','dht','both'),default='both');p.add_argument('--kind',default='random')
    p.add_argument('--case',action='append');p.add_argument('--oracle-workers',type=int,choices=(1,2,4),default=1)
    args=p.parse_args();cases=selected_cases(args)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    import torch_npu
    assert torch.npu.device_count()==1,'external configuration must expose exactly one device'
    torch.npu.set_device(0)
    print('BUILD_START',flush=True);prepare(block_dim=args.block_dim);print('BUILD_DONE',flush=True)
    compiled=dict(zip((e.name for e in _pipeline().entries()),_compiled(args.block_dim)))
    def launch(entry,sources,outputs,scalars):
        for t in outputs.values():t.fill_(float('nan'))
        op=compiled[entry.name];op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
        return outputs
    report=dict(stage='native_inprocess',block_dim=args.block_dim,oracle_workers=args.oracle_workers,
                torch=torch.__version__,torch_npu=torch_npu.__version__,cases=[])
    def finish(job):
        future,host,public,b,row,dtype=job
        a=future.result()
        row.update(fp32_A=metrics(host,a),fp32_B=metrics(host,b),public_A=metrics(public,a),public_B=metrics(public,b))
        report['cases'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
        print('NUMBERS',json.dumps(row),flush=True)
        assert acceptable(row['fp32_A']) and acceptable(row['fp32_B'])
        limit=1e-4
        assert acceptable(row['public_A'],limit) and acceptable(row['public_B'],limit)
    pending=deque()
    with ThreadPoolExecutor(max_workers=args.oracle_workers) as executor:
        for case in cases:
            par=case['parameters']
            for dtype in (torch.float32,):
                values=list(inputs(par['B'],par['T'],par['H'],par['HV'],par['kind'],seed=case['seed']))
                if par['mode']=='do':values[6].zero_()
                if par['mode']=='dht':values[5].zero_()
                cpu=dict(zip(('q','k','v','g','beta','do','dht'),values))
                # CPU-only workers execute the exact literal A; no precomputed fixture.
                future=executor.submit(cpu_a,values)
                gpu={n:t.npu() for n,t in cpu.items()}
                before={n:digest(t) for n,t in gpu.items()}
                print('EXECUTE',case['id'],str(dtype),json.dumps(par),flush=True)
                started=time.monotonic();actual=_pipeline().run(gpu,launch,retain_stages=True);torch.npu.synchronize()
                print('RETURNED',case['id'],time.monotonic()-started,flush=True)
                host={n:t.cpu() for n,t in actual.items()};hashes={n:digest(t) for n,t in host.items()}
                expected=reference_stages(cpu)
                composition=all_metrics(host,expected)
                print('COMPOSITION',case['id'],json.dumps(composition),flush=True)
                assert acceptable(composition,1e-4),composition
                def independent(entry,sources,outputs,scalars):
                    known={**gpu,'dout':gpu['do']}
                    known.update({n:expected[n].npu() for n in sources if n not in known})
                    return launch(entry,{n:known[n] for n in sources},outputs,scalars)
                leaves=_pipeline().run(gpu,independent,retain_stages=True);torch.npu.synchronize()
                leaf=all_metrics(leaves,expected)
                print('INDEPENDENT_LEAF',case['id'],json.dumps(leaf),flush=True)
                assert acceptable(leaf,1e-4),leaf
                public_input=dict(gpu)
                public_before={n:digest(t) for n,t in public_input.items()}
                call_inputs=dict(public_input)
                if par['mode']=='do':call_inputs['dht']=None
                if par['mode']=='dht':call_inputs['do']=None
                returned=chunk_gdn_bwd(**call_inputs,block_dim=args.block_dim);torch.npu.synchronize()
                public={n:t.cpu() for n,t in zip(NAMES,returned)}
                for n,t in public.items():
                    expected_dtype = dtype if n in ('dq','dk','dv') else torch.float32
                    assert t.shape == expected[n].shape and t.dtype == expected_dtype
                immutable=(all(digest(t)==before[n] for n,t in gpu.items()) and
                           all(digest(t)==public_before[n] for n,t in public_input.items()))
                assert immutable
                row=dict(case=case['id'],seed=case['seed'],parameters=par,dtype=str(dtype),composition=composition,
                         leaf=leaf,inputs_unchanged=immutable,public_schema={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in public.items()},stage_hashes=hashes,public_hashes={n:digest(t) for n,t in public.items()})
                pending.append((future,{n:host[n] for n in NAMES},public,{n:expected[n] for n in NAMES},row,dtype))
                # Bound CPU VJP memory and outstanding reference work.
                if len(pending)>=args.oracle_workers:finish(pending.popleft())
        while pending:finish(pending.popleft())
    report['passed']=True;out.write_text(json.dumps(report,indent=2)+'\n');print('NATIVE_PASS',flush=True)

if __name__=='__main__':main()
