"""Fresh native A/B, poisoned leaves/composition, old FP32 and host-op audit."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ascend_fla.ops import gdn_chunk_bwd as public
from ref.bf16 import INPUTS, NAMES, make_inputs, fp32_inputs, reference_stages, comparison, acceptable
from ref import oracle


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Audit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operators=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.operators.append(str(func))
        return func(*args,**(kwargs or {}))


def load_baseline(root):
    """Original public function, sharing byte-identical unchanged FP32 entries.

    Sharing compile/entry caches avoids duplicate registrations of the same
    CANN operator. Original wrapper code and its allocation/dispatch are intact.
    """
    current=Path(public.__file__).resolve().parents[2]
    relative=Path('kernels/projects/a5/gdn_chunk_bwd')
    files=[p for p in (root/relative).rglob('*.py')]
    assert files
    for path in files:
        assert path.read_bytes()==(current/path.relative_to(root)).read_bytes(),path
    path=root/'ascend_fla/ops/gdn_chunk_bwd.py'
    spec=importlib.util.spec_from_file_location('ascend_fla.ops._bf02_original_public',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._pipeline=public._pipeline
    module._compiled=public._compiled
    return module,dict(wrapper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       unchanged_fp32_python_files=len(files),
                       sharing='Identical original FP32 entry/compile caches only; original public function unchanged')


def literal_a(cpu):
    assert torch.get_num_threads()==1 and all(x.device.type=='cpu' for x in cpu.values())
    values=fp32_inputs(cpu)
    return oracle.autograd(*(values[n] for n in INPUTS))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--block-dim',required=True,type=int,choices=(1,2))
    parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--case',action='append')
    parser.add_argument('--oracle-workers',type=int,choices=(1,2,4),default=1)
    args=parser.parse_args()
    contract=json.loads((Path(__file__).parent/'contract.json').read_text())
    cases=[c for c in contract['cases'] if c['block_dim']==args.block_dim]
    if args.case:
        wanted=set(args.case)
        if wanted!={'all'}:
            cases=[c for c in cases if c['id'] in wanted]
            assert {c['id'] for c in cases}==wanted,'unknown or wrong-bd case'
    else:
        cases=[c for c in cases if c['id']==f'full_r1_both_bd{args.block_dim}']
    assert cases
    cases.sort(key=lambda c:c['id']!=f'full_r1_both_bd{args.block_dim}')
    import torch_npu
    torch.set_num_threads(1)
    assert torch.npu.device_count()==1
    torch.npu.set_device(0)
    baseline,baseline_identity=load_baseline(args.baseline_root)
    print('BUILD_START',flush=True)
    public.prepare(block_dim=args.block_dim)
    print('BUILD_DONE',flush=True)
    pipelines={torch.float32:public._pipeline(),torch.bfloat16:public._bf16_pipeline()}
    compiled={entry.name: artifact for dtype,get in [(torch.float32,public._compiled),(torch.bfloat16,public._bf16_compiled)]
              for entry,artifact in zip(pipelines[dtype].entries(),get(args.block_dim))}
    report=dict(stage='native_inprocess_bf02',block_dim=args.block_dim,
                torch=torch.__version__,torch_npu=torch_npu.__version__,oracle_workers=args.oracle_workers,
                oracle_pin=oracle.PIN,oracle_sha256=oracle.SHA256,baseline=baseline_identity,
                public_source_sha256=hashlib.sha256(Path(public.__file__).read_bytes()).hexdigest(),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    allowed={'aten.empty.memory_format','aten.empty_strided.default','aten.zeros.default','aten.zeros_like.default',
             'aten.view.default','aten.unsqueeze.default','aten.alias.default'}

    def save():
        pending=args.output.with_suffix('.pending')
        pending.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        pending.replace(args.output)

    def finish(job):
        future,actual,expected,row,dtype=job
        a=future.result()
        row['public_A']=comparison(actual,a,dtype)
        row['public_B']=comparison(actual,expected,dtype)
        row['passed']=all(acceptable(row[n]) for n in ('public_A','public_B','composition','independent_leaf'))
        report['cases'].append(row)
        save()
        print('NATIVE_CASE',json.dumps(row,allow_nan=False),flush=True)
        assert row['passed'],row

    pending=collections.deque()
    with ThreadPoolExecutor(max_workers=args.oracle_workers) as executor:
        for case in cases:
            for dtype in (torch.bfloat16,torch.float32):
                started=time.monotonic()
                cpu=make_inputs(case,dtype)
                future=executor.submit(literal_a,cpu)
                device={n:x.npu() for n,x in cpu.items()}
                before={n:digest(x) for n,x in device.items()}
                inputs=dict(device)
                if case['parameters']['mode']=='do':inputs['dht']=None
                if case['parameters']['mode']=='dht':inputs['do']=None
                # The unchanged FP32 pipeline uses explicit zero cotangents.
                stage_inputs=inputs if dtype==torch.bfloat16 else device
                poison=[]
                def launch(entry,sources,outputs,scalars):
                    for value in outputs.values():value.fill_(float('nan'))
                    for flag,name in [('has_do','dout'),('has_dht','dht')]:
                        if flag in scalars and scalars[flag]==0:
                            sources[name].fill_(float('nan'))
                            poison.append(name)
                    op=compiled[entry.name]
                    op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
                    return outputs
                print('EXECUTE',case['id'],str(dtype),flush=True)
                got=pipelines[dtype].run(stage_inputs,launch,retain_stages=True)
                torch.npu.synchronize()
                host={n:x.cpu() for n,x in got.items()}
                expected=reference_stages(cpu)
                composition=comparison(host,expected,dtype,stages=True)
                known={**device,'dout':device['do'],**{n:x.contiguous().npu() for n,x in expected.items()}}
                def leaf(entry,sources,outputs,scalars):
                    selected={}
                    for n,value in sources.items():
                        absent=(n=='dout' and scalars.get('has_do')==0) or (n=='dht' and scalars.get('has_dht')==0)
                        selected[n]=value if absent else known[n]
                    return launch(entry,selected,outputs,scalars)
                leaves=pipelines[dtype].run(stage_inputs,leaf,retain_stages=True)
                torch.npu.synchronize()
                independent=comparison({n:x.cpu() for n,x in leaves.items()},expected,dtype,stages=True)
                audit=Audit()
                with audit:
                    result=public.chunk_gdn_bwd(**inputs,block_dim=args.block_dim)
                torch.npu.synchronize()
                actual={n:x.cpu() for n,x in zip(NAMES,result)}
                forbidden=sorted(set(audit.operators)-allowed)
                assert not forbidden,forbidden
                hashes={n:digest(x) for n,x in actual.items()}
                assert hashes=={n:digest(host[n]) for n in NAMES},'composition/public bytes'
                old_hashes=None
                if dtype==torch.float32:
                    original=baseline.chunk_gdn_bwd(**inputs,block_dim=args.block_dim)
                    torch.npu.synchronize()
                    old_hashes={n:digest(x) for n,x in zip(NAMES,original)}
                    assert hashes==old_hashes,'original/new FP32 bytes'
                assert all(digest(x)==before[n] for n,x in device.items()),'input mutation'
                if dtype==torch.bfloat16 and case['parameters']['mode']!='both':assert len(poison)==2
                row=dict(case=case,dtype=str(dtype),composition=composition,independent_leaf=independent,
                         stage_hashes={n:digest(x) for n,x in host.items()},public_hashes=hashes,
                         inputs=before,inputs_unchanged=True,absent_cotangent_poison=poison,
                         fp32_original_hashes=old_hashes,fp32_original_byte_equal=None if old_hashes is None else True,
                         host_operators=audit.operators,host_operator_counts=dict(collections.Counter(audit.operators)),
                         forbidden_operators=forbidden,seconds_before_A_wait=time.monotonic()-started)
                pending.append((future,actual,{n:expected[n] for n in NAMES},row,dtype))
                del got,leaves,host,known,result,device
                if len(pending)>=args.oracle_workers:finish(pending.popleft())
        while pending:finish(pending.popleft())
    report['passed']=True
    save()
    print('NATIVE_PASS',len(report['cases']),flush=True)


if __name__=='__main__':main()
