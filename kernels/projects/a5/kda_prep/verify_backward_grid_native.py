"""BF-08 complete supported training flag, dtype and gradient-selection grid."""
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
    parser.add_argument('--chunk-count',type=int,choices=(1,2,3))
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
    write('environment', dict(block_dim=bd, first_workload='post-full workload training grid'))
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
    # Full training has already executed for this production source. This is
    # the complete supported flag/dtype population, repeated with actual
    # requires_grad selections, including disconnected disabled parameters.
    native_grid=load('_bf08_grid_inputs',ROOT/'native_grid.py')
    population=[c for c in native_grid.cases('chunk') if args.chunk_count is None or c['C']==args.chunk_count]
    modes=[('all',raw_names)]+[(n,(n,)) for n in ('q','k','g','beta','A_log','dt_bias')]
    modes += [('parameters',('A_log','dt_bias')),('partial',('q','g','beta'))]
    write('source-identity',dict(source_manifest=source_manifest))
    environment.pop('production_source_sha256')
    environment['production_source_manifest_sha256']=hashlib.sha256((REPO.parent/'source-manifest.json').read_bytes()).hexdigest()
    environment['source_identity_receipt']='source-identity.json'
    cases=[]

    def run(data, flags, names, capture=False, old_graph=False):
        dev={n:t.npu().requires_grad_(n in names) for n,t in data.items()}
        captured={}
        actual_prep=auto._prepare_training_inputs
        actual_cached=auto.chunk_kda_fwd_with_caches
        def capture_prep(*a,**kw):
            result=actual_prep(*a,**kw);captured['prepared']=result;return result
        def capture_caches(*a,**kw):
            result=actual_cached(*a,**kw);captured['caches']=result[2];return result
        if capture:
            auto._prepare_training_inputs=capture_prep
            auto.chunk_kda_fwd_with_caches=capture_caches
        api=old_auto if old_graph else auto
        try:
            with check.instrument(audit=not old_graph) as (audit,launches):
                o,state=api.chunk_kda(*(dev[n] for n in ('q','k','v','g','beta')),
                    A_log=dev['A_log'],dt_bias=dev['dt_bias'],initial_state=dev['h0'],
                    output_final_state=True,**flags,**options)
                targets=[dev[n] for n in names]
                prepared=captured.get('prepared',()) if capture else ()
                if o.requires_grad:
                    values=torch.autograd.grad((o,state),targets+list(prepared),
                        (dev['do'],dev['dht']),allow_unused=True)
                else:
                    values=[None]*len(targets)
        finally:
            auto._prepare_training_inputs=actual_prep
            auto.chunk_kda_fwd_with_caches=actual_cached
        torch.npu.synchronize()
        got=check.cpu(dict(o=o,final_state=state,**{'d'+n:t for n,t in zip(names,values)}))
        if capture:
            got.update(check.cpu({'cache_'+n:t for n,t in captured['caches'].items()}))
        record=dict(outputs={n:check.digest(t) if t is not None else None for n,t in got.items()},
            finite={n:bool(t.isfinite().all()) for n,t in got.items() if t is not None},
            unchanged={n:check.digest(dev[n])==check.digest(t) for n,t in data.items()},
            gradients_require_dtype={n:got['d'+n] is None or got['d'+n].dtype==data[n].dtype for n in names},
            audit=audit.report(),unexpected=audit.unexpected() if not old_graph else [],
            old_host_preparation=[v for v in audit.report() if v['category'].startswith('BF08_pending')],
            launches=launches,requested_gradients=list(names),output_requires_grad=o.requires_grad)
        if capture:
            off=(not flags['use_qk_l2norm_in_kernel'],not flags['use_qk_l2norm_in_kernel'],
                 not flags['use_gate_in_kernel'],not flags['use_beta_sigmoid_in_kernel'])
            record['disabled_identity']={n:prepared[i] is dev[n] for i,n in enumerate(prep_names) if off[i]}
            record['cache_count']=len(captured['caches'])
            sensitivities=check.cpu(dict(zip(prep_names,values[len(targets):])))
        else:sensitivities=None
        return got,record,sensitivities

    for case in population:
        label=case['id'];flags=case['flags'];print('GRID_CASE_START',label,flush=True)
        data=native_grid.inputs(case);rng=torch.Generator().manual_seed(8018)
        data['do']=(torch.randn(data['v'].shape,generator=rng)*.04).bfloat16()
        data['dht']=(torch.randn(data['h0'].shape,generator=rng)*.01).bfloat16().float()
        got,all_record,sens=run(data,flags,raw_names,capture=True)
        row=dict(case=case,all_gradients=all_record,prep_gradients={},references={},selections={},passed=False)
        write('cases/'+label,row)
        old_leaves={n:data[n].clone().requires_grad_() for n in ('q','k','g','beta','A_log','dt_bias')}
        old_prep=old._prepare_inputs(*(old_leaves[n] for n in prep_names),
            A_log=old_leaves['A_log'],dt_bias=old_leaves['dt_bias'],**flags)
        old_values=torch.autograd.grad(old_prep,list(old_leaves.values()),tuple(sens[n] for n in prep_names),allow_unused=True)
        old_values=dict(zip(old_leaves,old_values))
        high={};keys={};conditions={}
        if flags['use_qk_l2norm_in_kernel']:
            for name in ('q','k'):
                high[name]=calibration.norm_reference(data[name],sens[name])
                keys[name]='norm:'+case['types'][name]+':dx'
                z=data[name].double();d=sens[name].double();s=(z*z).sum(-1,keepdim=True)+1.e-6
                conditions[name]=(d/s.sqrt()).abs()+(z*((z*d).sum(-1,keepdim=True)/(s*s.sqrt()))).abs()
                conditions[name]=conditions[name]/high[name].abs()
        if flags['use_gate_in_kernel']:
            gate_types='_'.join(case['types'][n] for n in ('g','A_log','dt_bias'))
            high_gate=calibration.gate_reference(data['g'],data['A_log'],data['dt_bias'],sens['g'])
            for output,name in (('dg','g'),('dA_log','A_log'),('ddt_bias','dt_bias')):
                high[name]=high_gate[output];keys[name]='gate:'+gate_types+':'+output
            u=data['g'].double()+data['dt_bias'].double().view(case['HV'],128)
            sp=torch.where(u>20,u,torch.logaddexp(u,torch.zeros_like(u)))
            contribution=sens['g'].double()*sp*(-data['A_log'].double().exp().view(-1,1))
            conditions['g']=torch.ones_like(high['g'])
            conditions['A_log']=contribution.abs().sum((0,1,3))/high['A_log'].abs()
            conditions['dt_bias']=high['g'].abs().sum((0,1)).reshape(-1)/high['dt_bias'].abs()
        if flags['use_beta_sigmoid_in_kernel']:
            high['beta']=calibration.beta_reference(data['beta'],sens['beta'])
            keys['beta']='beta:'+case['types']['beta']+':dbeta'
            conditions['beta']=torch.ones_like(high['beta'])
        for name,key in keys.items():
            row['prep_gradients'][name]=support.gradient_record(key,got['d'+name],high[name],old_values[name],
                precision,limits['groups'][key],args.output/'ulp-details'/label/name,
                condition=conditions[name],environment=environment)
        # Disabled flags return the exact incoming prepared sensitivity.
        row['disabled_gradients']={n:check.digest(got['d'+n])==check.digest(sens[n])
            for n in prep_names if n not in high}
        for refname,function in (('independent',kda_recurrent_ref),('fla',fla.naive_recurrent_kda)):
            expected=support.raw_reference(data,function,old._prepare_inputs,flags)
            metrics={n:reference.metrics(got[n],expected[n]) for n in ('o','final_state')}
            metrics['gradients']={n:check.metrics(got[n],expected[n],limits['end_to_end_relative_l2'][n])
                for n in ('dq','dk','dv','dg','dbeta','dh0')}
            metrics['per_head_chunk']=[dict(chunk=c,head=h,**reference.metrics(
                got['o'][:,c*64:(c+1)*64,h],expected['o'][:,c*64:(c+1)*64,h]))
                for c in range(case['C']) for h in range(case['HV'])]
            row['references'][refname]=metrics
        before,before_record,_=run(data,flags,raw_names,old_graph=True)
        row['old_npu']={n:precision.metrics(got[n],t.double()) if t is not None else None for n,t in before.items()}
        row['disabled_all_baseline_exact']=(all(check.digest(got[n])==check.digest(t) for n,t in before.items() if t is not None)
            if not any(flags.values()) else None)
        for mode,names in modes[1:]:
            observed,record,_=run(data,flags,names)
            record['same_as_all_gradients']={n:(t is None and got[n] is None) or
                (t is not None and got[n] is not None and check.digest(t)==check.digest(got[n])) for n,t in observed.items()}
            record['passed']=(all(record['same_as_all_gradients'].values()) and all(record['unchanged'].values())
                and all(record['finite'].values()) and all(record['gradients_require_dtype'].values())
                and not record['unexpected'] and not record['old_host_preparation'])
            row['selections'][mode]=record
        row['passed']=(all(all_record['finite'].values()) and all(all_record['unchanged'].values())
            and all(all_record['gradients_require_dtype'].values()) and all(all_record['disabled_identity'].values())
            and all_record['cache_count']==9 and not all_record['unexpected'] and not all_record['old_host_preparation']
            and all(m['passed'] for m in row['prep_gradients'].values()) and all(row['disabled_gradients'].values())
            and all(m[n]['passed'] for m in row['references'].values() for n in ('o','final_state'))
            and all(v['passed'] for m in row['references'].values() for v in m['gradients'].values())
            and all(v['passed'] for m in row['references'].values() for v in m['per_head_chunk'])
            and all(m['passed'] for m in row['selections'].values()) and row['disabled_all_baseline_exact'] is not False)
        write('cases/'+label,row)
        if not row['passed']:
            torch.save(dict(raw=data,actual=got,prepared_sensitivities=sens),args.output/(label+'.private.pt'))
        cases.append(dict(id=label,passed=bool(row['passed']),all_outputs=all_record['outputs'],
            selection_outputs={n:m['outputs'] for n,m in row['selections'].items()}))
        write('summary',dict(complete=False,passed=all(c['passed'] for c in cases),
            expected_cases=len(population),gradient_selections=len(modes),cases=cases))
        print('GRID_CASE_RESULT',label,row['passed'],len(cases),'/',len(population),flush=True)
    write('summary',dict(complete=True,passed=all(c['passed'] for c in cases),
        expected_cases=len(population),gradient_selections=len(modes),cases=cases))
    if not all(c['passed'] for c in cases):raise AssertionError('Backward grid contains retained failures')


if __name__=='__main__':
    main()
