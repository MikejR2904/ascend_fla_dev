"""BF-04 native acceptance. One bd per process; external device lock required.

Goldens/input generation/poison are verifier work, outside the audited public
operator. Every custom vendor (BF16 and FP32) is registered before any launch.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--block-dim',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--mode',choices=('compile','full','suite','perf'),default='full')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    def write(name,data):
        path=out/(name+'.json');temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
    import torch,torch_npu,ascriptor
    from torch.utils._python_dispatch import TorchDispatchMode
    from ascend_fla.ops import pkda_chunk_fwd as api
    from ascend_fla.runtime.compile import compile_kernel
    public=api._bf16_module();pipeline=public.pipeline
    package=public.__package__
    ref=importlib.import_module(package+'.ref.reference')
    checks=importlib.import_module(package+'.ref.checks')
    root=Path(__file__).resolve().parent
    contract=json.loads((root/'contract.json').read_text())
    torch.set_num_threads(1)
    write('environment',dict(python=platform.python_version(),torch=torch.__version__,torch_npu=torch_npu.__version__,ascriptor=ascriptor.__version__,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))},public_sha256=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()))
    vendors={};builds=[]
    for family,entries in (('BF16',pipeline.entries()),('FP32',api._module('kernels.pipeline').entries())):
        for entry in entries:
            print('COMPILE_START',family,entry.name,args.block_dim,flush=True);start=time.monotonic()
            op=compile_kernel(entry,device='a5',block_dim=args.block_dim,backend='cce')
            if family=='BF16':vendors[entry.name]=op
            builds.append(dict(family=family,entry=entry.name,signature=op.signature,seconds=time.monotonic()-start,
                vendor_files={str(p.relative_to(op.vendor_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in op.vendor_dir.rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json')}))
            write('compile',dict(complete=len(builds)==11,block_dim=args.block_dim,entries=builds))
            print('COMPILE_PASS',entry.name,flush=True)
    api.prepare(block_dim=args.block_dim,dtype=torch.float32)
    api.prepare(block_dim=args.block_dim,dtype=torch.bfloat16)
    if args.mode=='compile':return
    torch.npu.set_device(0)
    write('device',dict(name=torch.npu.get_device_name(0),block_dim=args.block_dim))
    def cpu(x):return x.detach().cpu().contiguous()
    def digest(x):return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def to_device(data):return {n:(x.npu() if isinstance(x,torch.Tensor) else x) for n,x in data.items()}
    def prepare_case(case):
        data=ref.make_inputs(case)
        if case['parameters'].get('adversarial',False):
            data['g'].fill_(-1e-5);data['g'][:,::64].fill_(-154.9992)
        return data
    allowed={'aten.empty.memory_format','aten.empty_strided.default','aten.view.default',
             'aten.reshape.default','aten.expand.default','aten.zeros.default','aten.full.default'}
    validation_allowed={'aten._local_scalar_dense.default','aten.abs.default','aten.alias.default',
                        'aten.all.default','aten.any.default','aten.bitwise_or.Tensor',
                        'aten.eq.Tensor','aten.gt.Scalar','aten.isfinite.default','aten.lt.Scalar',
                        'aten.mul.Tensor','aten.ne.Scalar','aten.neg.default','aten.pow.Tensor_Scalar',
                        'aten.slice.Tensor','aten.sum.dim_IntList'}
    class Audit(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.operations=[];self.classified=[];self.phase='operator';self.control_readbacks=[]
        def __enter__(self):
            # PM5743858594: original read-only numeric validation is a separate
            # audit class. Its masks/scalars never enter the mathematical path.
            owner=api._module('ref.reference');self.owner=owner;self.validator=owner.validate_inputs
            def validate(*args,**kwargs):
                previous=self.phase;self.phase='readonly_validation'
                try:return self.validator(*args,**kwargs)
                finally:self.phase=previous
            owner.validate_inputs=validate
            self.readback=pipeline.check_status
            def readback(status):
                self.control_readbacks.append(dict(kind='control_metadata',method='aclrtMemcpy_D2H',
                    dtype=str(status.dtype),elements=status.numel(),bytes=status.numel()*status.element_size(),
                    meaning='numeric-guard codes only; no computational tensor values'))
                return self.readback(status)
            pipeline.check_status=readback
            return super().__enter__()
        def __exit__(self,*args):
            pipeline.check_status=self.readback;self.owner.validate_inputs=self.validator
            return super().__exit__(*args)
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            name=str(func);self.operations.append(name)
            category=self.phase
            if name in ('aten.zeros.default','aten.full.default'):category='constant_allocation'
            self.classified.append(dict(operator=name,category=category))
            return func(*args,**(kwargs or {}))
        def violations(self):
            return [r for r in self.classified if r['operator'] not in allowed and
                    not (r['category']=='readonly_validation' and r['operator'] in validation_allowed)]
    def call(data):
        audit=Audit()
        with audit:got=api.chunk_precond_kda(**data,output_final_state=True,block_dim=args.block_dim)
        torch.npu.synchronize()
        unexpected=audit.violations()
        if unexpected:raise AssertionError(dict(unexpected_host_operators=unexpected,all_operations=audit.operations))
        call.last_categories=audit.classified;call.last_control_readbacks=audit.control_readbacks
        return dict(zip(pipeline.OUTPUTS,(cpu(x) for x in got))),audit.operations
    rows=[]
    def execute_case(case):
        print('CASE_START',case['id'],case['parameters'],flush=True)
        data=prepare_case(case);ref.validate_inputs(data);refs=checks.references(data)
        dev=to_device(data);timings={}
        def launch(entry,sources,outputs,scalars):
            for tensor in outputs.values():tensor.fill_(float('nan'))
            torch.npu.synchronize();start=time.monotonic();op=vendors[entry.name]
            op(sources,{n:scalars[n] for n in op.scalar_names},outputs);torch.npu.synchronize()
            timings[entry.name]=time.monotonic()-start
            print('STAGE_EXECUTED',case['id'],entry.name,timings[entry.name],flush=True)
            return outputs
        stages=pipeline.run(dev,launch,retain_stages=True)
        staged={n:cpu(x) for n,x in stages.items()}
        got,operations=call(dev)
        result=checks.compare(got,refs)
        unchanged={n:digest(cpu(dev[n]))==digest(x) for n,x in data.items() if isinstance(x,torch.Tensor)}
        stage_finite={n:bool(torch.isfinite(x).all()) for n,x in staged.items()}
        stage_equal=all(digest(got[n])==digest(staged[n]) for n in got)
        row=dict(case=case,block_dim=args.block_dim,**result,input_sha256={n:digest(x) for n,x in data.items() if isinstance(x,torch.Tensor)},
                 input_unchanged=unchanged,poison_all_written=stage_finite,public_equals_staged=stage_equal,
                 stage_sha256={n:digest(x) for n,x in staged.items()},output_sha256={n:digest(x) for n,x in got.items()},
                 stage_errors={n:checks.metric(staged[n],refs['stages'][n]) for n in refs['stages']},stage_seconds=timings,host_operations=operations,host_operator_categories=call.last_categories,control_readbacks=call.last_control_readbacks)
        row['passed']=result['passed'] and all(unchanged.values()) and all(stage_finite.values()) and stage_equal
        write(case['id'],row)
        if not row['passed']:
            torch.save(dict(inputs=data,stages=staged,public=got,references=refs),out/(case['id']+'-failure.pt'))
            raise AssertionError(row)
        rows.append(row);write('summary',dict(complete=False,passed=False,cases=rows))
        print('CASE_PASS',case['id'],json.dumps(result['errors']['A']),flush=True)
    try:
        # Original full workload is always first, including diagnostic invocations.
        execute_case(next(c for c in contract['cases'] if c['id']=='t4096_h8'))
        from native_checks import fp32_full
        fp32_result=fp32_full(api,args.block_dim,Audit,allowed,to_device,digest)
        write('fp32-full-before-after',fp32_result)
        assert fp32_result['passed'],fp32_result
        print('FP32_FULL_PASS',flush=True)
        if args.mode=='suite':
            for case in contract['cases']:
                if case['id']!='t4096_h8':execute_case(case)
            extra=[dict(id=f'tail_{t}',seed=9500+t,parameters=dict(B=1,T=t,H=3)) for t in range(65,129)]
            extra += [dict(id='b2_h32',seed=9522,parameters=dict(B=2,T=130,H=32))]
            extra += [dict(id=f'span_{span}',seed=704,parameters=dict(B=1,T=192,H=2,gate_span=span)) for span in (0.,1e-4,50.,105.,155.)]
            extra += [dict(id=f'strong_weak_{seed}',seed=seed,parameters=dict(B=1,T=192,H=8,adversarial=True)) for seed in (71,72,73)]
            for case in extra:execute_case(case)
            from native_checks import boundaries
            boundary=boundaries(api,ref,checks,args.block_dim,call,to_device,digest,Audit,allowed)
            write('boundaries',dict(passed=True,cases=boundary))
        if args.mode=='perf':
            from native_perf import measure
            write('performance',measure(api,ref,checks,args.block_dim,to_device))
        write('summary',dict(passed=True,complete=True,cases=rows))
        print('NATIVE_PASS',len(rows),args.block_dim,flush=True)
    except BaseException as error:
        write('failure',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),completed=[r['case']['id'] for r in rows]))
        raise


if __name__=='__main__':main()
