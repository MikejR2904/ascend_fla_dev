"""BF-08 post-full-workload boundaries is the first custom-kernel execution in a process."""
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
    write('environment', dict(block_dim=bd, first_workload='post-full-workload boundaries, all raw flags'))
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
    import itertools
    write('source-identity',dict(source_manifest=source_manifest))
    environment.pop('production_source_sha256')
    environment['production_source_manifest_sha256']=hashlib.sha256((REPO.parent/'source-manifest.json').read_bytes()).hexdigest()
    environment['source_identity_receipt']='source-identity.json'
    records=[]
    def semantics(candidate, previous):
        a,b=candidate.detach().cpu(),previous.detach().cpu()
        return dict(nan_mask_equal=bool(torch.equal(torch.isnan(a),torch.isnan(b))),
            positive_inf_equal=bool(torch.equal(torch.isposinf(a),torch.isposinf(b))),
            negative_inf_equal=bool(torch.equal(torch.isneginf(a),torch.isneginf(b))),
            zero_mask_equal=bool(torch.equal(a==0,b==0)),
            nonfinite_candidate=int((~a.isfinite()).sum()),nonfinite_previous=int((~b.isfinite()).sum()))
    def finish(label,row,tensors):
        row['id']=label;write('cases/'+label,row)
        torch.save(tensors,args.output/(label+'.private.pt'))
        records.append(dict(id=label,passed=row['passed']))
        write('summary',dict(complete=False,passed=all(x['passed'] for x in records),cases=records))
        print('BOUNDARY_RESULT',label,row['passed'],flush=True)
    rng=torch.Generator().manual_seed(8009)
    for dtype_labels in itertools.product(calibration.DTYPES,repeat=3):
        for kind,shape in (('gaussian',(2,192,8,128)),('full_kimi',(1,4096,32,128)),('threshold',(2,64,2,128))):
            hv=shape[-2]
            if kind=='threshold':
                v=torch.tensor([-40.,-20.,-1.,0.,1.,19.999998092651367,20.,20.000001907348633,40.,80.])
                g=v.repeat((math.prod(shape)+v.numel()-1)//v.numel())[:math.prod(shape)].reshape(shape)
                bias=torch.zeros(hv*128)
            else:g=torch.randn(shape,generator=rng);bias=torch.linspace(-.2,.2,hv*128)
            a=torch.linspace(-3.,2.7,hv)
            g,a,bias=[t.to(calibration.DTYPES[d]) for t,d in zip((g,a,bias),dtype_labels)]
            gy=torch.randn(shape,generator=rng)
            label='gate_'+'_'.join(dtype_labels)+'_'+kind
            dev=[t.npu() for t in (g,a,bias,gy)]
            with check.instrument() as (audit,launches):
                actual=chunk._prep_runtime().gate_backward(*dev,block_dim=bd)
            torch.npu.synchronize();actual=[t.cpu() for t in actual]
            leaves=[t.clone().requires_grad_() for t in (g,a,bias)]
            y=old._prepare_inputs(None,None,leaves[0],None,A_log=leaves[1],dt_bias=leaves[2],use_gate_in_kernel=True)[2]
            cpu=torch.autograd.grad(y,leaves,gy)
            leaves=[t.detach().requires_grad_() for t in dev[:3]]
            y=old._prepare_inputs(None,None,leaves[0],None,A_log=leaves[1],dt_bias=leaves[2],use_gate_in_kernel=True)[2]
            native=[t.cpu() for t in torch.autograd.grad(y,leaves,dev[3])]
            high=calibration.gate_reference(g,a,bias,gy)
            u=g.double()+bias.double().view(hv,128);sp=torch.where(u>20,u,torch.logaddexp(u,torch.zeros_like(u)))
            term=gy.double()*sp*(-a.double().exp().view(-1,1))
            conditions=dict(dg=torch.ones_like(high['dg']),dA_log=term.abs().sum((0,1,3))/high['dA_log'].abs(),
                ddt_bias=high['dg'].abs().sum((0,1)).reshape(-1)/high['ddt_bias'].abs())
            rows={}
            for name,got,prior,npugrad in zip(('dg','dA_log','ddt_bias'),actual,cpu,native):
                key='gate:'+'_'.join(dtype_labels)+':'+name
                rows[name]=support.gradient_record(key,got,high[name],prior,precision,limits['groups'][key],
                    args.output/'ulp-details'/label/name,condition=conditions[name],environment=environment)
                rows[name]['old_npu_to_fp64']=precision.metrics(npugrad,high[name])
            unchanged=all(check.digest(t)==check.digest(d) for t,d in zip((g,a,bias,gy),dev))
            finish(label,dict(gradients=rows,input_unchanged=unchanged,audit=audit.report(),unexpected=audit.unexpected(),launches=launches,
                passed=all(r['passed'] for r in rows.values()) and unchanged and not audit.unexpected()),
                dict(g=g,a=a,bias=bias,gy=gy,actual=actual,old_cpu=cpu,old_npu=native,fp64=high))
    for dtype,dt in calibration.DTYPES.items():
        for value in (-120.,-104.,-100.,80.,88.,89.,100.):
            g=torch.linspace(-2.,2.,128).reshape(1,1,1,128).to(dt);a=torch.tensor([value],dtype=dt);bias=torch.zeros(128,dtype=dt)
            gy=torch.randn(g.shape,generator=rng);dev=[t.npu() for t in (g,a,bias,gy)]
            actual=[t.cpu() for t in chunk._prep_runtime().gate_backward(*dev,block_dim=bd)]
            results={}
            for target,leaves,sensitivity in (('old_cpu',[t.clone().requires_grad_() for t in (g,a,bias)],gy),
                    ('old_npu',[t.detach().requires_grad_() for t in dev[:3]],dev[3])):
                y=old._prepare_inputs(None,None,leaves[0],None,A_log=leaves[1],dt_bias=leaves[2],use_gate_in_kernel=True)[2]
                results[target]=[t.cpu() for t in torch.autograd.grad(y,leaves,sensitivity)]
            high=calibration.gate_reference(g,a,bias,gy)
            comparisons={n:dict(cpu=semantics(v,c),npu=semantics(v,d),to_fp64=precision.metrics(v,high[n]),
                old_cpu_to_fp64=precision.metrics(c,high[n]),old_npu_to_fp64=precision.metrics(d,high[n]))
                for n,v,c,d in zip(('dg','dA_log','ddt_bias'),actual,results['old_cpu'],results['old_npu'])}
            # Observations do not establish a new endpoint class. Nonfinite masks
            # must at least match the original CPU and native graphs pointwise.
            passed=all(r[device][key] for r in comparisons.values() for device in ('cpu','npu')
                for key in ('nan_mask_equal','positive_inf_equal','negative_inf_equal'))
            finish(f'gate_{dtype}_alog{value:g}',dict(comparisons=comparisons,passed=passed,
                endpoint_qualification=False,scope='Range observations; same nonfinite masks required, no new exemption or ordinary numerical qualification.'),
                dict(g=g,a=a,bias=bias,gy=gy,actual=actual,**results,fp64=high))
    rng=torch.Generator().manual_seed(8010)
    for dtype,dt in calibration.DTYPES.items():
        populations=[('gaussian',torch.randn((2,192,8),generator=rng)),('full_kimi',torch.randn((1,4096,32),generator=rng)),
            ('zero',torch.zeros(1,64,2)),('saturation',torch.tensor([-120.,-104.,-100.,-90.,-88.,-40.,-20.,-1.,0.,1.,16.,20.,40.,100.]).reshape(1,14,1))]
        for kind,source in populations:
            x=source.to(dt);gy=torch.randn(x.shape,generator=rng);xd=x.npu();gd=gy.npu()
            with check.instrument() as (audit,launches):
                probability=chunk._prep_runtime().beta(xd,block_dim=bd)
                got=chunk._prep_runtime().beta_backward(xd,probability,gd,block_dim=bd)
            torch.npu.synchronize();got=got.cpu();saved=probability.cpu()
            leaf=x.clone().requires_grad_();s=leaf.float().sigmoid();cpu,=torch.autograd.grad(s,leaf,gy)
            leaf=xd.detach().requires_grad_();sdev=leaf.float().sigmoid();native,=torch.autograd.grad(sdev,leaf,gd);native=native.cpu()
            high=calibration.beta_reference(x,gy);mask=(saved==0)|(saved==1);key='beta:'+dtype+':dbeta';label='beta_'+dtype+'_'+kind
            row=support.gradient_record(key,got,high,cpu,precision,limits['groups'][key],args.output/'ulp-details'/label,
                source=x,sensitivity=gy,environment=environment,
                endpoint_zero_mask=mask,endpoint_authority='D-PM-55(2), actual saved sigmoid saturation')
            row.update(old_npu_to_fp64=precision.metrics(native,high),saved_sigmoid_sha256=check.digest(saved),
                old_npu_endpoint_zero=bool((native[mask]==0).all()),audit=audit.report(),unexpected=audit.unexpected(),launches=launches,
                input_unchanged=check.digest(xd)==check.digest(x) and check.digest(gd)==check.digest(gy))
            row['passed']=row['passed'] and row['old_npu_endpoint_zero'] and row['input_unchanged'] and not audit.unexpected()
            finish(label,row,dict(x=x,gy=gy,actual=got,old_cpu=cpu,old_npu=native,fp64=high,saved_sigmoid=saved,old_cpu_sigmoid=s.detach(),old_npu_sigmoid=sdev.detach().cpu()))
    write('summary',dict(complete=True,passed=all(x['passed'] for x in records),cases=records,
        scope='Gate ordinary/threshold and beta ordinary/saturation; A_log range rows are located observations, not new endpoint qualification.'))
    if not all(x['passed'] for x in records):raise AssertionError('Retained backward boundary failures')

if __name__ == '__main__':main()
