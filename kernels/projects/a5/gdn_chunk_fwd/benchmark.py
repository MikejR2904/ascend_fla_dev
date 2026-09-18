"""Native in-process correctness grid and synchronized GDN baseline/profile.

Run each block dimension or source candidate in a separate process. CPU
references are generated at run time; no saved golden tensors are consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(got, ref):
    got = got.cpu().float()
    delta = got - ref.float()
    norm = ref.float().norm().item()
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': delta.norm().item()/norm if norm else (0. if not delta.any() else float('inf'))}


def timing(fn, warmup, repeat):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000)
    return {'median_us': statistics.median(samples), 'min_us': min(samples),
            'max_us': max(samples), 'samples_us': samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block-dim', type=int, choices=(1,2), default=2)
    p.add_argument('--case', default='all')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--fla-naive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch_npu
    import ascriptor
    from ascend_fla.ops.gdn_chunk_fwd import _compiled, _pipeline, chunk_gdn, prepare
    refs=load('gdn_bench_ref',ROOT/'ref/reference.py')
    sr=load('gdn_bench_stage_ref',ROOT/'ref/stages.py')
    fla=load('gdn_bench_fla',args.fla_naive)
    contract=json.loads((ROOT/'contract.json').read_text())
    cases=[c for c in contract['cases'] if args.case=='all' or c['id']==args.case]
    if not cases: raise ValueError('no matching case')
    torch.set_num_threads(4)
    prepare(block_dim=args.block_dim)
    compiled=dict(zip((e.name for e in _pipeline().entries()),_compiled(args.block_dim)))
    checkpoint = {'scope':'A5 native in-process CCE; synchronized host-inclusive latency',
                  'block_dim':args.block_dim,'warmup':args.warmup,'repeat':args.repeat,
                  'kernel_sha256':hashlib.sha256((ROOT/'kernels/stages.py').read_bytes()).hexdigest(),
                  'versions':{'torch':torch.__version__,'torch_npu':torch_npu.__version__},'cases':[]}
    checkpoint['device']={'soc':torch.npu.get_device_name(0)}
    for label,env,suffix in [('compiler','ASCEND_HOME_PATH','compiler/version.info'),('opp','ASCEND_OPP_PATH','version.info')]:
        base=os.environ.get(env)
        file=Path(base)/suffix if base else None
        checkpoint['device'][label]=file.read_text() if file and file.exists() else 'unavailable'
    opp=os.environ.get('ASCEND_OPP_PATH')
    inventory=Path(opp)/'built-in/op_impl/ai_core/tbe/kernel' if opp else None
    checkpoint['device']['opp_packages']=sorted(x.name for x in inventory.iterdir()) if inventory and inventory.exists() else []
    checkpoint['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in str(p)}
    wrapper=ROOT.parents[3]/'ascend_fla/ops/gdn_chunk_fwd.py'
    checkpoint['public_wrapper_sha256']=hashlib.sha256(wrapper.read_bytes()).hexdigest()
    lib_root=Path(ascriptor.__file__).parent
    library_hash=hashlib.sha256()
    for p in sorted(lib_root.rglob('*.py')):
        library_hash.update(str(p.relative_to(lib_root)).encode()+b'\0'+p.read_bytes())
    checkpoint['library_python_tree_sha256']=library_hash.hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for case in cases:
        cpu=refs.make_inputs(case)
        expected=refs.reference(cpu)
        stage_expected=sr.reference_stages(cpu)
        expanded=_pipeline().expand_inputs(cpu)
        fo,fs=fla.naive_recurrent_gated_delta_rule(*(expanded[n] for n in ('q','k','v','beta','g')),output_final_state=True)
        recurrent=refs.grouped_recurrent(cpu)
        npu={n:x.to('npu') for n,x in cpu.items()}
        calls=[]
        def launch(entry,sources,outputs,scalars):
            op=compiled[entry.name]
            scalars={n:scalars[n] for n in op.scalar_names}
            for x in outputs.values(): x.fill_(float('nan'))
            op(sources,scalars,outputs)
            calls.append((entry.name,op,sources,outputs,scalars))
            return outputs
        got=_pipeline().run(npu,launch)
        torch.npu.synchronize()
        row={'case':case['id'],'seed':case['seed'],'parameters':case['parameters'],'stages':{},'oracles':{},
             'input_sha256':{n:hashlib.sha256(x.numpy().tobytes()).hexdigest() for n,x in cpu.items()}}
        for n,ref in stage_expected.items():
            row['stages'][n]=metric(got[n],ref)
            torch.testing.assert_close(got[n].cpu(),ref,atol=2e-5,rtol=2e-4)
            assert row['stages'][n]['relative_l2']<=1e-4,(case['id'],n,row['stages'][n])
        # Check every leaf with independently generated CPU upstreams as well
        # as checking the actual composition above.
        upstream=dict(expanded,**stage_expected)
        row['independent_leaves']={}
        for name,op,sources,outputs,scalars in calls:
            leaf_inputs={n:upstream[n].contiguous().to('npu') for n in sources}
            leaf_outputs={n:torch.full_like(x,float('nan')) for n,x in outputs.items()}
            op(leaf_inputs,scalars,leaf_outputs)
            torch.npu.synchronize()
            for n,x in leaf_outputs.items():
                rr=stage_expected[n]
                row['independent_leaves'][n]=metric(x,rr)
                torch.testing.assert_close(x.cpu(),rr,atol=2e-5,rtol=2e-4)
                assert row['independent_leaves'][n]['relative_l2']<=1e-4
            del leaf_inputs,leaf_outputs
        for label, oracle in (('block_solve',expected),('grouped_recurrent',recurrent),('fla_naive_expanded' if cpu['v'].shape[2]!=cpu['q'].shape[2] else 'fla_naive',{'o':fo,'final_state':fs})):
            row['oracles'][label]={n:metric(got[n],r) for n,r in oracle.items()}
            for n,r in oracle.items():
                torch.testing.assert_close(got[n].cpu(),r,atol=2e-5,rtol=2e-4)
            assert all(m['relative_l2']<=1e-4 for m in row['oracles'][label].values())
        row['sha256']={n:hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for n,x in got.items()}
        for dtype in (torch.float32,torch.bfloat16):
            qkv=[npu[n].to(dtype) for n in ('q','k','v')]
            pub=chunk_gdn(*qkv,npu['g'],npu['beta'],block_dim=args.block_dim,output_final_state=True)
            inp=dict(cpu,**{n:cpu[n].to(dtype).float() for n in ('q','k','v')})
            rr=refs.reference(inp)
            key=str(dtype)
            row[key]={n:metric(x,rr[n]) for n,x in zip(('o','final_state'),pub)}
            row[key+'_sha256']={n:hashlib.sha256(x.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n,x in zip(('o','final_state'),pub)}
            torch.testing.assert_close(pub[0].cpu().float(),rr['o'],atol=2e-5,
                                       rtol=1e-2 if dtype==torch.bfloat16 else 2e-4)
            torch.testing.assert_close(pub[1].cpu(),rr['final_state'],atol=2e-5,rtol=2e-4)
            assert row[key]['o']['relative_l2'] <= (5e-3 if dtype==torch.bfloat16 else 1e-4)
            assert row[key]['final_state']['relative_l2']<=1e-4
        row['group_replication_bytes']=sum(expanded[n].numel()*expanded[n].element_size() for n in ('q','k') if expanded[n] is not cpu[n])
        row['retained_stage_workspace_bytes']=sum(x.numel()*x.element_size() for n,x in got.items() if n not in ('o','final_state'))
        if args.profile:
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            before=torch.npu.memory_allocated()
            memory_result=chunk_gdn(npu['q'],npu['k'],npu['v'],npu['g'],npu['beta'],block_dim=args.block_dim,output_final_state=True)
            torch.npu.synchronize()
            row['public_peak_allocated_bytes']=torch.npu.max_memory_allocated()-before
            del memory_result
            row['stage_timing']={}
            for name,op,sources,outputs,scalars in calls:
                row['stage_timing'][name]=timing(lambda op=op,s=sources,o=outputs,a=scalars:op(s,a,o),args.warmup,args.repeat)
            row['public_timing']=timing(lambda:chunk_gdn(npu['q'],npu['k'],npu['v'],npu['g'],npu['beta'],block_dim=args.block_dim,output_final_state=True),args.warmup,args.repeat)
        checkpoint['cases'].append(row)
        args.output.write_text(json.dumps(checkpoint,indent=2)+'\n')
        print(json.dumps({'case':case['id'],'oracles':row['oracles'],'median_us':row.get('public_timing',{}).get('median_us')}),flush=True)


if __name__=='__main__':main()
