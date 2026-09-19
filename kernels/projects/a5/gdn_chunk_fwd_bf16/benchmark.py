"""Native actual-output, poisoned composition/leaf and public-dispatch checks."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ref.reference import cases,make_inputs,reference,metric,budget,NAMES
from ref.stages import reference_stages
from ref import oracle
from kernels import pipeline
from ascend_fla.ops import gdn_chunk_fwd as op


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Audit(TorchDispatchMode):
    def __init__(self):super().__init__();self.operators=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.operators.append(str(func));return func(*args,**(kwargs or {}))


def check(actual,expected,dtype,*,public=False):
    result={n:metric(actual[n],expected[n]) for n in expected}
    for n,value in result.items():
        limit=budget(expected[n],dtype) if n=='o' or (public and n=='final_state') else 1e-4
        value['budget']=limit
        assert value['finite'] and value['relative_l2']<=limit,(n,value)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--block-dim',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',action='append')
    p.add_argument('--baseline-report',type=Path,required=True);p.add_argument('--oracle-workers',type=int,default=1)
    args=p.parse_args()
    import torch_npu
    torch.set_num_threads(1);assert torch.npu.device_count()==1;torch.npu.set_device(0)
    op.prepare(block_dim=args.block_dim)
    compiled=dict(zip((e.name for e in op._native_pipeline().all_entries()),op._compiled(args.block_dim)))
    baseline=json.loads(args.baseline_report.read_text());assert baseline['passed'] and baseline['block_dim']==args.block_dim
    old={row['case']['id']:row for row in baseline['cases']}
    report=dict(stage='native_inprocess_bf01',block_dim=args.block_dim,
                oracle_pin=oracle.PIN,oracle_sha256=oracle.SHA256,cases=[],
                public_source_sha256=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest(),
                baseline_report_sha256=hashlib.sha256(args.baseline_report.read_bytes()).hexdigest())
    allowed={'aten.empty.memory_format','aten.empty_strided.default','aten.view.default','aten.unsqueeze.default','aten.alias.default'}
    selected=[c for c in cases() if not args.case or c['id'] in args.case]
    assert selected
    for case in selected:
        for dtype in ('bfloat16','float32'):
            started=time.time();cpu=make_inputs(dict(case,dtype=dtype))
            a=oracle.reference(cpu);b=reference(cpu);stage=reference_stages(cpu)
            device={n:x.npu() for n,x in cpu.items()};before={n:digest(x) for n,x in device.items()}
            def launch(entry,sources,outputs,scalars):
                for value in outputs.values():value.fill_(float('nan'))
                kernel=compiled[entry.name]
                kernel(sources,{n:scalars[n] for n in kernel.scalar_names},outputs)
                return outputs
            composition=pipeline.run(device,launch)
            torch.npu.synchronize();composition_cpu={n:x.cpu() for n,x in composition.items()}
            numbers_composition=check(composition_cpu,stage,cpu['q'].dtype)
            upstream={n:x.contiguous().npu() for n,x in dict(cpu,**stage).items()}
            def leaf(entry,sources,outputs,scalars):
                return launch(entry,{n:upstream[n] for n in sources},outputs,scalars)
            leaves=pipeline.run(device,leaf)
            torch.npu.synchronize();numbers_leaf=check({n:x.cpu() for n,x in leaves.items()},stage,cpu['q'].dtype)
            audit=Audit()
            with audit:
                returned=op.chunk_gdn(**device,output_final_state=True,block_dim=args.block_dim)
            torch.npu.synchronize();actual={n:x.cpu() for n,x in zip(NAMES,returned)}
            forbidden=sorted(set(audit.operators)-allowed);assert not forbidden,forbidden
            assert actual['o'].dtype==cpu['q'].dtype and actual['o'].shape==cpu['v'].shape
            assert actual['final_state'].dtype==torch.float32 and actual['final_state'].shape==b['final_state'].shape
            assert all(digest(x)==before[n] for n,x in device.items())
            hashes={n:digest(x) for n,x in actual.items()}
            assert hashes=={n:digest(composition_cpu[n]) for n in NAMES}
            same_old=None
            if dtype=='float32':
                assert before==old[case['id']]['inputs']
                same_old=hashes==old[case['id']]['public_hashes'];assert same_old
            row=dict(case=case,dtype=dtype,public_A=check(actual,a,cpu['q'].dtype,public=True),
                     public_B=check(actual,b,cpu['q'].dtype,public=True),composition=numbers_composition,
                     independent_leaf=numbers_leaf,public_hashes=hashes,
                     stage_hashes={n:digest(x) for n,x in composition_cpu.items()},
                     inputs=before,inputs_unchanged=True,fp32_original_byte_equal=same_old,
                     host_operators=audit.operators,host_operator_counts=dict(collections.Counter(audit.operators)),
                     forbidden_operators=forbidden,seconds=time.time()-started)
            report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('NATIVE_CASE',json.dumps(row),flush=True)
            del composition,composition_cpu,upstream,leaves,actual,device
    report['passed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_PASS',len(report['cases']),flush=True)


if __name__=='__main__':main()
