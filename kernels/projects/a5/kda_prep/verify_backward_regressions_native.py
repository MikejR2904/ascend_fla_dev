"""BF-08 full Kimi training is the first custom-kernel execution in a process."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(os.environ.get('BF08_UNIT_ROOT', Path(__file__).resolve().parent)).resolve()
REPO = ROOT.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('BF08_EXTERNAL_DEVICE_LOCK') == '1'
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    import ascriptor
    from ascend_fla.ops.kda import chunk, chunk_bwd, prepare
    from ascend_fla.ops.kda import autograd as auto
    from ascend_fla.runtime.compile import compile_kernel

    assert platform.python_version() == os.environ['BF08_ACCEPTED_PYTHON']
    assert torch.__version__ == os.environ['BF08_ACCEPTED_TORCH']
    assert torch_npu.__version__ == os.environ['BF08_ACCEPTED_TORCH_NPU']
    assert ascriptor.__version__ == '0.1.0'
    native = Path(os.environ['BF08_NATIVE_ROOT'])
    assert Path(ascriptor.__file__).is_relative_to(native / 'library')
    source_manifest = json.loads((REPO.parent / 'source-manifest.json').read_text())
    for name, digest in source_manifest.items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == digest, name
    for name, digest in json.loads((native/'accepted-source-manifest.json').read_text()).items():
        assert hashlib.sha256((native/name).read_bytes()).hexdigest() == digest, name
    environment = load('_bf08_native_env', ROOT/'native_environment.py').collect()
    environment['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment['production_source_sha256'] = source_manifest
    environment['helper_sha256'] = hashlib.sha256((Path(__file__).parent/'backward_native_support.py').read_bytes()).hexdigest()
    torch.set_num_threads(1)

    def write(name, value):
        def finite_json(item):
            if isinstance(item, float) and not math.isfinite(item):
                return None
            if isinstance(item, dict):
                return {k: finite_json(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [finite_json(v) for v in item]
            return item
        target = args.output/(name+'.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix('.tmp')
        staging.write_text(json.dumps(finite_json(dict(environment=environment, **value)), indent=2, allow_nan=False)+'\n')
        staging.replace(target)

    bd = args.block_dim
    write('environment', dict(block_dim=bd, first_workload='full Kimi training, all raw flags'))
    plan = [('prep_forward', e) for e in chunk._prep_runtime().kernels('chunk').values()]
    plan += [('prep_backward', e) for e in chunk._prep_runtime().backward_kernels().values()]
    plan += [('layout', e) for e in chunk._layout_runtime().kernels().values()]
    plan += [('forward', e) for e in chunk.kda_fwd_kernels().values()]
    plan += [('backward', e) for e in chunk_bwd.kda_bwd_kernels().values()]
    assert len(plan) == 50 and sum(f == 'backward' for f, _ in plan) == 9
    builds = []
    for family, entry in plan:
        print('COMPILE_START', family, entry.name, bd, flush=True)
        start = time.monotonic()
        op = compile_kernel(entry, device='a5', block_dim=bd, backend='cce')
        builds.append(dict(family=family, entry=entry.name, block_dim=bd, signature=op.signature,
                           seconds=time.monotonic()-start,
                           vendor_files={str(p.relative_to(op.vendor_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in op.vendor_dir.rglob('*') if p.is_file() and p.suffix in ('.so', '.o', '.json')}))
        write('compile', dict(complete=len(builds) == 50, entries=builds, first_custom_launch_not_started=True))
    prepare(block_dim=bd, backward=True)
    print('ALL_50_VENDORS_READY', flush=True)
    torch.npu.set_device(0)
    check = load('_bf08_native_checks', ROOT.parent/'kda_layout/native_support.py')
    audit_module = load('_bf08_audit', ROOT/'native_audit.py')
    audit_module.REPO = REPO
    check.Audit = audit_module.Audit
    support = load('_bf08_backward_support', Path(__file__).parent/'backward_native_support.py')
    precision = load('_bf08_precision', ROOT/'ref/calibrate.py')
    calibration = load('_bf08_backward_reference', ROOT/'ref/backward_calibrate.py')
    old, _ = calibration.predecessors()
    receipt = json.loads((ROOT/'baseline_backward/source.json').read_text())
    path = ROOT/'baseline_backward/autograd.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['files']['autograd.py']['sha256']
    old_auto = types.ModuleType('ascend_fla.ops.kda._bf08_predecessor')
    old_auto.__package__ = 'ascend_fla.ops.kda'
    exec(compile(path.read_text(), str(path), 'exec'), old_auto.__dict__)
    old_auto._prepare_inputs = old._prepare_inputs
    old_auto.chunk_kda_fwd_with_caches = chunk.chunk_kda_fwd_with_caches
    old_auto.chunk_kda_fwd = chunk.chunk_kda_fwd
    old_auto.chunk_kda_bwd = chunk_bwd.chunk_kda_bwd
    fla_path = Path(os.environ['FLA_KDA_NAIVE'])
    assert hashlib.sha256(fla_path.read_bytes()).hexdigest() == '60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016'
    fla = load('_bf08_fla', fla_path)
    from ascend_fla.reference.kda import kda_recurrent_ref
    reference = load('_bf08_fwd_reference', ROOT.parent/'kda_fwd_stable/repair_reference.py')
    limits = json.loads((ROOT/'backward_budgets.json').read_text())
    flags = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    options = dict(block_dim=bd, layout_device='npu')
    raw_names = ('q', 'k', 'v', 'g', 'beta', 'A_log', 'dt_bias', 'h0')
    prep_names = ('q', 'k', 'g', 'beta')
    sys.modules['native_support'] = check
    real, _ = check.reference_modules()
    extra = load('_bf08_existing_distributions', ROOT.parent/'kda_layout/native_checks.py')
    old.__package__ = 'ascend_fla.ops.kda'
    old.__file__ = chunk.__file__
    old._compiled_chain = chunk._compiled_chain
    ctx = types.SimpleNamespace(check=check, old=old, real=real, reference=reference,
        fla=fla, precision=precision, write=write)
    rows = []
    def execute(case,x):
        label=case['id'];print('BACKWARD_CASE_START',label,flush=True)
        dev={n:t.npu() for n,t in x.items()}
        data={n:dev[n] for n in ('q','k','v','g','beta')};data['initial_state']=dev['h0']
        with torch.no_grad(),check.instrument() as (audit,launches):
            plain=auto.chunk_kda(**data,output_final_state=True,**options)
            o,state,caches=chunk.chunk_kda_fwd_with_caches(**data,**options)
            beta=chunk._layout_runtime().cast(dev['beta'],torch.bfloat16,block_dim=bd)
            grads=chunk_bwd.chunk_kda_bwd(**{n:dev[n] for n in ('q','k','v','do','dht')},beta=beta,caches=caches,**options)
        torch.npu.synchronize()
        got=check.cpu(dict(o=o,final_state=state,**{'cache_'+n:t for n,t in caches.items()},**grads))
        old_o,old_state,old_cache=ctx.old.chunk_kda_fwd_with_caches(**data,**options)
        old_grads=chunk_bwd.chunk_kda_bwd(**{n:dev[n] for n in ('q','k','v','do','dht')},beta=dev['beta'].bfloat16(),caches=old_cache,**options)
        torch.npu.synchronize();before=check.cpu(dict(o=old_o,final_state=old_state,**{'cache_'+n:t for n,t in old_cache.items()},**old_grads))
        exact=check.exact(got,before);refs={}
        for name,fwd,fn in (('independent',ctx.reference.independent_reference(x),ctx.real.kda_recurrent_ref),('fla',ctx.reference.fla_reference(x),ctx.fla.naive_recurrent_kda)):
            grad_ref=check.gradients(x,fn)
            refs[name]=dict(**{n:ctx.reference.metrics(got[n],fwd[n]) for n in ('o','final_state')},
                gradients={n:check.metrics(got[n],t,ctx.real.BUDGET[n]) for n,t in grad_ref.items()},
                per_head_chunk=[dict(chunk=c,head=h,**ctx.reference.metrics(got['o'][:,c*64:(c+1)*64,h],fwd['o'][:,c*64:(c+1)*64,h])) for c in range(case['C']) for h in range(case['HV'])])
        row=dict(case=case,block_dim=bd,bitwise=exact,input_sha256={n:check.digest(t) for n,t in x.items()},
                 input_unchanged={n:check.digest(t)==check.digest(dev[n]) for n,t in x.items()},
                 output_sha256={n:check.digest(t) for n,t in got.items()},finite={n:bool(t.isfinite().all()) for n,t in got.items()},
                 plain_cached_exact=[check.digest(a)==check.digest(b) for a,b in zip(plain,(o,state))],references=refs,
                 operations=audit.report(),unexpected=audit.unexpected(),launches=launches)
        row['passed']=(all(t['passed'] for t in exact.values()) and all(row['input_unchanged'].values()) and all(row['finite'].values())
            and all(row['plain_cached_exact']) and not audit.unexpected() and all(r[n]['passed'] for r in refs.values() for n in ('o','final_state'))
            and all(t['passed'] for r in refs.values() for t in r['gradients'].values()) and all(t['passed'] for r in refs.values() for t in r['per_head_chunk']))
        ctx.write('backward/'+label,row);rows.append(dict(id=label,passed=row['passed']))
        if not row['passed']:
            torch.save(dict(inputs=x,actual=got,before=before),args.output/(label+'.private.pt'))
            raise AssertionError(label)
        print('BACKWARD_CASE_PASS',label,flush=True)

    try:
        for case,x in extra.original_backward_inputs(chunk_bwd):execute(case,x)
        for h,hv in ((32,32),(16,32)):
            case=dict(id=f'real_t1024_h{h}_hv{hv}',B=1,H=h,HV=hv,C=16)
            execute(case,ctx.real.make_inputs(B=1,H=h,HV=hv,C=16,span=46,want_grads=True))
        for span in (0.,100.8,105.,155.,105.001,155.001):
            x=ctx.real.make_inputs(B=1,H=1,HV=1,C=2,span=46,want_grads=True)
            raw=x['g'].clone().view(1,2,64,1,128);raw.fill_(-120.)
            count=4 if span==100.8 else 5
            if span:raw[:,:,64-count:].fill_(span/count)
            a=torch.zeros(1);bias=torch.zeros(128)
            cpu_prep=ctx.old._prepare_inputs(x['q'],x['k'],raw.view_as(x['g']),x['beta'],A_log=a,dt_bias=bias,use_gate_in_kernel=True)
            native=chunk._prepare_kernel_inputs(x['q'].npu(),x['k'].npu(),raw.view_as(x['g']).npu(),x['beta'].npu(),A_log=a.npu(),dt_bias=bias.npu(),use_gate_in_kernel=True,block_dim=bd)
            native_cpu=check.cpu(native);before_span=chunk._gate_span(cpu_prep[2],2,on_cpu=True);after_span=chunk._gate_span(native_cpu[2],2,on_cpu=True)
            row=dict(requested_span=span,predecessor_cpu_span=before_span,candidate_span=after_span,paths={},prep_difference=ctx.precision.metrics(native_cpu[2],cpu_prep[2]))
            for path,limit in (('forward',155.),('cached',105.)):
                pair=[]
                for module,prep in ((ctx.old,cpu_prep),(chunk,native_cpu)):
                    data={n:t.npu() for n,t in zip(('q','k','g','beta'),prep)};data.update(v=x['v'].npu(),initial_state=x['h0'].npu())
                    function=module.chunk_kda_fwd if path=='forward' else module.chunk_kda_fwd_with_caches
                    try:
                        result=function(**data,**options,**({'output_final_state':True} if path=='forward' else {}))
                        torch.npu.synchronize();pair.append(dict(accepted=True,outputs=check.cpu(result)))
                    except ValueError as exc:pair.append(dict(accepted=False,error=str(exc)))
                expected=(before_span<=limit,after_span<=limit)
                report=dict(expected_acceptance=list(expected),actual_acceptance=[p['accepted'] for p in pair],same_decision=pair[0]['accepted']==pair[1]['accepted'])
                assert report['actual_acceptance']==list(expected)
                assert report['same_decision']
                if pair[1]['accepted']:
                    expected_input=dict(zip(('q','k','g','beta'),cpu_prep));expected_input.update(v=x['v'],h0=x['h0'])
                    refs=dict(independent=ctx.reference.independent_reference(expected_input),fla=ctx.reference.fla_reference(expected_input))
                    report['references']={label:{n:ctx.reference.metrics(pair[1]['outputs'][i],ref[n]) for i,n in enumerate(('o','final_state'))} for label,ref in refs.items()}
                    assert all(m['passed'] for ref in report['references'].values() for m in ref.values())
                row['paths'][path]=report
            row['passed']=True;ctx.write(f'gates/span{span:g}',row)
            if span<=105.:
                prepared=dict(x,**dict(zip(('q','k','g','beta'),native_cpu)))
                execute(dict(id=f'gate_span{span:g}_backward',B=1,H=1,HV=1,C=2),prepared)
            print('GATE_PASS',span,before_span,after_span,flush=True)
        write('summary',dict(complete=True,passed=True,block_dim=bd,backward_cases=rows,gate_populations=6,
            scope='Five original backward cases, two real shapes, unchanged no-flag paths, gate155/105 acceptance and backward boundaries; raw flag grid is qualified separately.'))
        print('BF08_REGRESSIONS_PASS', bd, flush=True)
    except BaseException as exc:
        write('failure',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed=rows))
        raise

if __name__ == '__main__':main()
