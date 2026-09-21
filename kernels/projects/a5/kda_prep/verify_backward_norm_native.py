"""BF-08 norm diagnostics after the complete training workload; no endpoint exemptions."""
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
    write('environment', dict(block_dim=bd, first_workload='post-full workload norm diagnostics'))
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
    # This diagnostic follows the retained full-training hardware run. All
    # failures remain ordinary failures; exit status reports collection only.
    write('source-identity', dict(source_manifest=source_manifest))
    environment.pop('production_source_sha256')
    environment['production_source_manifest_sha256'] = hashlib.sha256(
        (REPO.parent/'source-manifest.json').read_bytes()).hexdigest()
    environment['source_identity_receipt'] = 'source-identity.json'
    collected = []

    def cases(only_parallel):
        rng = torch.Generator().manual_seed(8008)
        for dtype, dt in calibration.DTYPES.items():
            for kind, shape in (('gaussian', (2,192,2,128)), ('full_kimi', (1,4096,32,128))):
                x = torch.randn(shape, generator=rng).to(dt)
                gy = torch.randn(shape, generator=rng).bfloat16()
                if not only_parallel:
                    yield dtype, kind, x, gy
            for scale in (0.,1e-20,1e-8,1e-4,30.,1e10,1e18,1e20):
                x = (torch.randn((1,64,2,128), generator=rng)*scale).to(dt)
                gy = torch.randn(x.shape, generator=rng).bfloat16()
                if not only_parallel:
                    yield dtype, 'scale_'+str(scale), x, gy
            x = torch.randn((1,64,2,128), generator=rng).to(dt)
            if only_parallel:
                yield dtype, 'near_null_parallel_sensitivity', x, x.bfloat16()

    for dtype, kind, x, gy in __import__('itertools').chain(cases(True), cases(False)):
        label = dtype+'_'+kind
        print('NORM_CASE_START',label,flush=True)
        xd, gd = x.npu(), gy.npu()
        with check.instrument() as (audit, launches):
            actual = chunk._prep_runtime().norm_backward(xd, gd, block_dim=bd)
            forward = chunk._prep_runtime().norm(xd, torch.bfloat16, block_dim=bd)
        torch.npu.synchronize()
        actual = actual.cpu()
        old_x = x.clone().requires_grad_()
        old_y = old._prepare_inputs(old_x, old_x, None, None, use_qk_l2norm_in_kernel=True)[0]
        old_cpu, = torch.autograd.grad(old_y, old_x, gy)
        native_x = xd.detach().requires_grad_()
        native_y = old._prepare_inputs(native_x, native_x, None, None, use_qk_l2norm_in_kernel=True)[0]
        old_native, = torch.autograd.grad(native_y, native_x, gd)
        torch.npu.synchronize()
        old_native = old_native.cpu()
        forward = forward.cpu()
        forward_high = precision.high_precision_norm(x)
        high = calibration.norm_reference(x, gy)
        z, d = x.double(), gy.double()
        square = (z*z).sum(-1, keepdim=True)+1.e-6
        first = d/square.sqrt()
        second = z*((z*d).sum(-1,keepdim=True)/(square*square.sqrt()))
        scale = first.abs()+second.abs()
        condition = scale/high.abs()
        key = 'norm:'+dtype+':dx'
        record = support.gradient_record(key, actual, high, old_cpu, precision,
            limits['groups'][key], args.output/'ulp-details'/label,
            source=x, sensitivity=gy, condition=condition, environment=environment)
        record.update(id=label, shape=list(x.shape), inputs=dict(x=check.digest(x), gy=check.digest(gy)),
            old_cpu_to_fp64=precision.metrics(old_cpu, high),
            old_npu_to_fp64=precision.metrics(old_native, high),
            candidate_to_old_npu=precision.metrics(actual, old_native),
            unchanged=dict(x=check.digest(xd)==check.digest(x), gy=check.digest(gd)==check.digest(gy)),
            audit=audit.report(), unexpected=audit.unexpected(), launches=launches)
        record['same_input_forward'] = {
            name: dict(metrics=precision.metrics(value, forward_high),
                       zeros=int((value==0).sum()), finite=int(value.isfinite().sum()))
            for name,value in (('candidate',forward),('old_cpu',old_y.detach()),
                               ('old_npu',native_y.detach().cpu()))}
        record['same_input_forward']['fp64'] = dict(zeros=int((forward_high==0).sum()),
            finite=int(forward_high.isfinite().sum()), sha256=precision.digest(forward_high))
        maximum=x.float().abs().max(-1,keepdim=True).values
        bits=maximum.view(torch.int32).bitwise_and(0x7f800000)
        factor=(0x7f000000-bits).to(torch.int32).view(torch.float32).clamp_min(2.**-120)
        factor=torch.where(maximum>=16.,factor,1.)
        scaled=x.float()*factor
        scaled_exact=x.double()*factor.double()
        record['binary_scaling_observation']=dict(
            rows_scaled=int((factor!=1).sum()),
            inexact_elements=int((scaled.double()!=scaled_exact).sum()),
            scope='Exactness on this generated input; no universal underflow claim')
        finite=torch.isfinite(condition)
        error=(actual.double()-high).abs()
        scaled=error/scale
        record['conditioning']=dict(max_finite_condition=float(condition[finite].max()) if finite.any() else None,
            undefined_or_nonfinite=int((~finite).sum()),
            max_error_over_term_magnitudes=float(scaled[torch.isfinite(scaled)].max()) if torch.isfinite(scaled).any() else None,
            derivative_norm=float(high.norm()),absolute_error_norm=float(error.norm()),
            fp32_unit_roundoff_times_term_norm=float(scale.norm())*2.**-24,
            estimate_is_scale_only_not_a_proved_error_lower_bound=True)
        # An exactly parallel row has an independent stable closed form; this
        # checks cancellation in the ordinary FP64 formula too.
        if dtype=='bf16' and kind.startswith('near_null'):
            stable=z*1.e-6/(square*square.sqrt())
            delta=(high-stable).abs()
            record['fp64_formula_to_parallel_closed_form']=dict(
                relative_l2=float(delta.norm()/stable.norm()),
                max_abs=float(delta.max()),
                max_relative_nonzero=float((delta[stable!=0]/stable[stable!=0].abs()).max()))
        record['passed']=record['passed'] and all(record['unchanged'].values()) and not audit.unexpected()
        write('cases/'+label,record)
        if kind.startswith('near_null') or not record['passed']:
            torch.save(dict(x=x,gy=gy,candidate=actual,old_cpu=old_cpu,old_npu=old_native,fp64=high,
                            forward=forward,forward_old_cpu=old_y.detach(),
                            forward_old_npu=native_y.detach().cpu(),forward_fp64=forward_high),
                       args.output/(label+'.private.pt'))
        collected.append(dict(id=label,passed=record['passed'],candidate_l2=record['to_fp64']['relative_l2'],
            candidate_max_relative=record['to_fp64']['max_relative_nonzero'],
            old_cpu_l2=record['old_cpu_to_fp64']['relative_l2'],old_npu_l2=record['old_npu_to_fp64']['relative_l2']))
        write('summary',dict(collection_complete=False,passed=all(t['passed'] for t in collected),cases=collected,
            no_endpoint_exemptions=True,exit_zero_means_collection_only=True))
        print('NORM_CASE_RESULT',json.dumps(collected[-1]),flush=True)
    write('summary',dict(collection_complete=True,passed=all(t['passed'] for t in collected),cases=collected,
        no_endpoint_exemptions=True,exit_zero_means_collection_only=True))


if __name__ == '__main__':
    main()
