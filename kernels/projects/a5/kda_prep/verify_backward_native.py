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
    case = dict(id='full_kimi_t4096_all_flags_training', B=1, T=4096, H=32, HV=32, K=128,
                raw_dtype='bf16', parameter_dtype='f32', flags=[True, True, True], all_raw_and_parameter_gradients=True)
    try:
        print('GENERATE_FULL_KIMI_TRAINING', flush=True)
        generator = torch.Generator().manual_seed(8008)
        shape = (1, 4096, 32, 128)
        raw = {n: torch.randn(shape, generator=generator).bfloat16() for n in ('q', 'k')}
        raw['g'] = (torch.randn(shape, generator=generator)*.1).bfloat16()
        raw['beta'] = torch.randn((1, 4096, 32), generator=generator).bfloat16()
        raw['A_log'] = torch.linspace(-.2, .2, 32)
        raw['dt_bias'] = torch.linspace(-.3, -.1, 4096)
        raw['v'] = (torch.randn(shape, generator=generator)*.04).bfloat16()
        raw['h0'] = torch.randn((1, 32, 128, 128), generator=generator)*.01
        raw['do'] = (torch.randn(shape, generator=generator)*.04).bfloat16()
        raw['dht'] = (torch.randn((1, 32, 128, 128), generator=generator)*.01).bfloat16().float()
        dev = {n: t.npu().requires_grad_(n in raw_names) for n, t in raw.items()}
        captured = {}
        actual_prep = auto._prepare_training_inputs
        actual_cached = auto.chunk_kda_fwd_with_caches

        def capture_prep(*a, **kw):
            result = actual_prep(*a, **kw)
            captured['prepared'] = result
            return result

        def capture_caches(*a, **kw):
            result = actual_cached(*a, **kw)
            captured['caches'] = result[2]
            return result

        auto._prepare_training_inputs = capture_prep
        auto.chunk_kda_fwd_with_caches = capture_caches
        try:
            print('FIRST_CUSTOM_WORKLOAD_FULL_TRAINING', flush=True)
            with check.instrument() as (audit, launches):
                o, state = auto.chunk_kda(*(dev[n] for n in ('q', 'k', 'v', 'g', 'beta')),
                    A_log=dev['A_log'], dt_bias=dev['dt_bias'], initial_state=dev['h0'],
                    output_final_state=True, **flags, **options)
                values = torch.autograd.grad((o, state), [dev[n] for n in raw_names]+list(captured['prepared']),
                                             (dev['do'], dev['dht']))
        finally:
            auto._prepare_training_inputs = actual_prep
            auto.chunk_kda_fwd_with_caches = actual_cached
        torch.npu.synchronize()
        audit_rows = audit.report()
        stale_exception = [r for r in audit_rows if r['category'].startswith('BF08_pending')]
        write('training-audit', dict(operations=audit_rows, unexpected=audit.unexpected(),
                                    old_host_preparation=stale_exception, launches=launches))
        got = check.cpu(dict(o=o, final_state=state, **{'d'+n: t for n, t in zip(raw_names, values[:8])},
                             **{'cache_'+n: t for n, t in captured['caches'].items()}))
        prep_values = check.cpu(captured['prepared'])
        sensitivities = check.cpu(dict(zip(prep_names, values[8:])))
        assert len(captured['caches']) == 9 and set(captured['caches']) == set(chunk.BWD_CACHE_NAMES)
        row = dict(case=case, outputs={n: check.digest(t) for n, t in got.items()},
                   unchanged={n: check.digest(dev[n]) == check.digest(t) for n, t in raw.items()},
                   finite={n: bool(t.isfinite().all()) for n, t in got.items()},
                   prep_gradients={}, cpu_references={}, passed=False)
        write('full', row)
        # Preserve actual outputs before any comparison can fail or be corrected.
        torch.save(dict(raw=raw, actual=got, prepared=prep_values, sensitivities=sensitivities),
                   args.output/'actual.private.pt')
        cpu_leaves = {n: raw[n].clone().requires_grad_() for n in ('q', 'k', 'g', 'beta', 'A_log', 'dt_bias')}
        old_prep = old._prepare_inputs(*(cpu_leaves[n] for n in prep_names),
            A_log=cpu_leaves['A_log'], dt_bias=cpu_leaves['dt_bias'], **flags)
        old_grads = torch.autograd.grad(old_prep, list(cpu_leaves.values()),
                                       tuple(sensitivities[n] for n in prep_names))
        old_grads = dict(zip(cpu_leaves, old_grads))
        high = {n: calibration.norm_reference(raw[n], sensitivities[n]) for n in ('q', 'k')}
        high.update({{'dg': 'g', 'dA_log': 'A_log', 'ddt_bias': 'dt_bias'}[n]: t
            for n, t in calibration.gate_reference(raw['g'], raw['A_log'], raw['dt_bias'], sensitivities['g']).items()})
        high['beta'] = calibration.beta_reference(raw['beta'], sensitivities['beta'])
        keys = dict(q='norm:bf16:dx', k='norm:bf16:dx', g='gate:bf16_f32_f32:dg',
                    A_log='gate:bf16_f32_f32:dA_log', dt_bias='gate:bf16_f32_f32:ddt_bias', beta='beta:bf16:dbeta')
        for name, key in keys.items():
            row['prep_gradients'][name] = support.gradient_record(key, got['d'+name], high[name], old_grads[name],
                precision, limits['groups'][key], args.output/'ulp-details'/name,
                source=raw.get(name), sensitivity=sensitivities.get(name), environment=environment)
            write('full', row)
        del high, old_grads, cpu_leaves, old_prep
        with torch.no_grad():
            plain = auto.chunk_kda(*(dev[n] for n in ('q', 'k', 'v', 'g', 'beta')),
                A_log=dev['A_log'], dt_bias=dev['dt_bias'], initial_state=dev['h0'],
                output_final_state=True, **flags, **options)
        row['plain_cached_exact'] = [check.digest(t) == check.digest(got[n]) for t, n in zip(plain, ('o', 'final_state'))]
        old_dev = {n: raw[n].npu().requires_grad_(n in raw_names) for n in raw}
        before_o, before_state = old_auto.chunk_kda(*(old_dev[n] for n in ('q', 'k', 'v', 'g', 'beta')),
            A_log=old_dev['A_log'], dt_bias=old_dev['dt_bias'], initial_state=old_dev['h0'],
            output_final_state=True, **flags, **options)
        before_grads = torch.autograd.grad((before_o, before_state), [old_dev[n] for n in raw_names],
                                           (old_dev['do'], old_dev['dht']))
        before = check.cpu(dict(o=before_o, final_state=before_state, **{'d'+n: t for n, t in zip(raw_names, before_grads)}))
        row['old_npu_difference'] = {n: precision.metrics(got[n], t.double()) for n, t in before.items()}
        torch.save(before, args.output/'old-native.private.pt')
        write('full', row)
        for label, function in (('independent', kda_recurrent_ref), ('fla', fla.naive_recurrent_kda)):
            print('CPU_FP32_RAW_TRAINING_REFERENCE', label, flush=True)
            expected = support.raw_reference(raw, function, old._prepare_inputs, flags)
            metrics = {n: reference.metrics(got[n], expected[n]) for n in ('o', 'final_state')}
            metrics['gradients'] = {n: check.metrics(got[n], expected[n], limits['end_to_end_relative_l2'][n])
                                    for n in ('dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0')}
            metrics['parameter_gradient_observations'] = {n: precision.metrics(got[n], expected[n].double())
                                                        for n in ('dA_log', 'ddt_bias')}
            metrics['per_head_chunk'] = [dict(chunk=c, head=h, **reference.metrics(
                got['o'][:, c*64:(c+1)*64, h], expected['o'][:, c*64:(c+1)*64, h]))
                for c in range(64) for h in range(32)]
            row['cpu_references'][label] = metrics
            write('full', row)
        row['passed'] = (all(row['unchanged'].values()) and all(row['finite'].values())
            and all(row['plain_cached_exact']) and not audit.unexpected() and not stale_exception
            and all(m['passed'] for m in row['prep_gradients'].values())
            and all(m[n]['passed'] for m in row['cpu_references'].values() for n in ('o', 'final_state'))
            and all(g['passed'] for m in row['cpu_references'].values() for g in m['gradients'].values())
            and all(g['passed'] for m in row['cpu_references'].values() for g in m['per_head_chunk']))
        write('full', row)
        write('summary', dict(complete=True, passed=bool(row['passed']), cases=[case], scope='full training workload only'))
        if not row['passed']:
            raise AssertionError('BF-08 full training workload failed; retain each comparison and actual tensors')
        print('FULL_BF08_NATIVE_PASS', bd, flush=True)
    except BaseException as error:
        write('failure', dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
