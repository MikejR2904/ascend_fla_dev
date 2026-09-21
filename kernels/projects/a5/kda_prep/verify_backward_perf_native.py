"""BF-08 synchronized training performance after native correctness acceptance."""
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
    write('environment', dict(block_dim=bd, first_workload='post-full-workload training performance'))
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
    from ascend_fla.reference.kda import kda_chunk_vectorized
    from ascend_fla.runtime.compile import CompiledKernel
    import statistics
    families = {r['signature']: r['family'] for r in builds}
    measurements = []
    def timed(function):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        clock = time.perf_counter()
        start.record()
        value = function()
        end.record()
        torch.npu.synchronize()
        wall = (time.perf_counter()-clock)*1000
        return dict(wall_ms=wall, whole_stream_event_ms=start.elapsed_time(end)), value
    def attribution(function):
        calls = []
        original = CompiledKernel.__call__
        def observe(op, inputs, scalars, outputs):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            clock = time.perf_counter()
            result = original(op, inputs, scalars, outputs)
            dispatch_ms = (time.perf_counter()-clock)*1000
            end.record()
            calls.append((begin, end, dict(signature=op.signature,
                family=families[op.signature], host_dispatch_ms=dispatch_ms,
                input_bytes=sum(t.numel()*t.element_size() for t in inputs.values()),
                output_bytes=sum(t.numel()*t.element_size() for t in outputs.values()))))
            return result
        CompiledKernel.__call__ = observe
        try:
            measurement, value = timed(function)
            del value
        finally:
            CompiledKernel.__call__ = original
        records = []
        for start, end, row in calls:
            row['device_event_ms'] = start.elapsed_time(end)
            records.append(row)
        return dict(**measurement, launches=records,
            custom_event_sum_ms=sum(r['device_event_ms'] for r in records),
            custom_host_dispatch_sum_ms=sum(r['host_dispatch_ms'] for r in records),
            interpretation='Separate instrumented run. Events can include stream idle time while the host dispatches; these are not isolated kernel durations or an additive decomposition of clean wall time.')
    for tokens in (1024, 4096):
        rng = torch.Generator().manual_seed(8008)
        shape = (1, tokens, 32, 128)
        raw = {n: torch.randn(shape, generator=rng).bfloat16() for n in ('q', 'k')}
        raw.update(g=(torch.randn(shape, generator=rng)*.1).bfloat16(),
            beta=torch.randn((1,tokens,32),generator=rng).bfloat16(),
            A_log=torch.linspace(-.2,.2,32),dt_bias=torch.linspace(-.3,-.1,4096),
            v=(torch.randn(shape,generator=rng)*.04).bfloat16(),
            h0=torch.randn((1,32,128,128),generator=rng)*.01,
            do=(torch.randn(shape,generator=rng)*.04).bfloat16(),
            dht=(torch.randn((1,32,128,128),generator=rng)*.01).bfloat16().float())
        device_inputs = {n:t.npu() for n,t in raw.items()}
        def training(route):
            x = {n:t.detach().requires_grad_(n in raw_names) for n,t in device_inputs.items()}
            if route == 'torch_npu':
                q,k,g,beta = old._prepare_inputs(*(x[n] for n in ('q','k','g','beta')),
                    A_log=x['A_log'],dt_bias=x['dt_bias'],**flags)
                output,state = kda_chunk_vectorized(q.float(),k.float(),x['v'].float(),g,beta,
                    initial_state=x['h0'],output_final_state=True)
            else:
                api = old_auto if route == 'baseline' else auto
                output,state = api.chunk_kda(*(x[n] for n in ('q','k','v','g','beta')),
                    A_log=x['A_log'],dt_bias=x['dt_bias'],initial_state=x['h0'],
                    output_final_state=True,**flags,**options)
            gradients = torch.autograd.grad((output,state),[x[n] for n in raw_names],(x['do'],x['dht']))
            return dict(o=output,final_state=state,**{'d'+n:t for n,t in zip(raw_names,gradients)})
        for baseline_route in ('baseline','torch_npu'):
            print('PERF_START',tokens,baseline_route,flush=True)
            for route in (baseline_route,'candidate'):
                value = training(route)
                del value
            torch.npu.synchronize()
            rounds = []
            last_baseline = None
            last_candidate = None
            for round_index in range(3):
                samples = []
                for phase,route in (('baseline_before',baseline_route),('candidate','candidate'),('baseline_after',baseline_route)):
                    measurement,value = timed(lambda:training(route))
                    returned = check.cpu(value)
                    del value
                    assert all(bool(t.isfinite().all()) for t in returned.values()),'nonfinite measured output'
                    measurement.update(phase=phase,output_sha256={n:check.digest(t) for n,t in returned.items()})
                    samples.append(measurement)
                    if route == 'candidate':last_candidate = returned
                    else:last_baseline = returned
                rounds.append(dict(round=round_index+1,samples=samples))
            old_ms = [s['wall_ms'] for row in rounds for s in (row['samples'][0],row['samples'][2])]
            new_ms = [row['samples'][1]['wall_ms'] for row in rounds]
            comparisons = {n:check.metrics(last_candidate[n],last_baseline[n],limit)
                for n,limit in dict(o=.05,final_state=.05,**limits['end_to_end_relative_l2']).items()}
            parameter_observations = {n:precision.metrics(last_candidate[n],last_baseline[n].double())
                for n in ('dA_log','ddt_bias')}
            unchanged = {n:check.digest(device_inputs[n])==check.digest(t) for n,t in raw.items()}
            row = dict(tokens=tokens,B=1,H=32,HV=32,block_dim=bd,raw_dtype='bf16',parameter_dtype='f32',
                scope='full forward plus backward training including raw preparation',
                baseline=baseline_route,rounds=rounds,warmup_per_path=1,repetitions_per_phase=1,
                baseline_median_ms=statistics.median(old_ms),candidate_median_ms=statistics.median(new_ms),
                candidate_over_baseline=statistics.median(new_ms)/statistics.median(old_ms),
                input_sha256={n:check.digest(t) for n,t in raw.items()},inputs_unchanged=unchanged,
                output_and_six_gradient_comparisons=comparisons,parameter_difference_observations=parameter_observations,
                speed_gate=None,timing='NPU synchronized wall and whole-stream event milliseconds. CPU generation/H2D/vendor compilation and post-measurement D2H checks are excluded; raw preparation and autograd are included.',
                passed=all(m['passed'] for m in comparisons.values()) and all(unchanged.values()))
            if baseline_route == 'baseline':
                row['attribution'] = {name:attribution(lambda:training(name)) for name in ('baseline','candidate')}
            measurements.append(row)
            write('measurements',dict(complete=False,rows=measurements))
            if not row['passed']:raise AssertionError('Measured training path failed its retained comparison')
            print('PERF_RESULT',tokens,baseline_route,row['candidate_over_baseline'],flush=True)
    write('measurements',dict(complete=True,rows=measurements))
    write('summary',dict(complete=True,passed=True,speed_gate=None,scopes=len(measurements)))


if __name__ == '__main__':
    main()
