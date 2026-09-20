"""FMT-02 hardware acceptance: full workload before leaf or model diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--mode', choices=('compile', 'full', 'suite', 'perf'), default='full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('FMT02_EXTERNAL_DEVICE_LOCK') == '1', 'hold both shared locks'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        p = out / (name + '.json')
        temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp.replace(p)

    import torch
    import torch_npu
    import ascriptor
    from ascend_fla.ops.kda import chunk, chunk_bwd, fused_recurrent, prepare
    from ascend_fla.runtime.compile import compile_kernel
    import native_support as check

    torch.set_num_threads(1)
    assert platform.python_version() == os.environ.get('FMT02_ACCEPTED_PYTHON','3.12.13')
    assert torch.__version__ == os.environ.get('FMT02_ACCEPTED_TORCH','2.12.0+cpu')
    assert torch_npu.__version__ == os.environ.get('FMT02_ACCEPTED_TORCH_NPU','2.12.0')
    assert ascriptor.__version__ == '0.1.0'
    assert Path(ascriptor.__file__).is_relative_to(Path(os.environ['FMT02_NATIVE_ROOT']) / 'library')
    auto = importlib.import_module('ascend_fla.ops.kda.autograd')
    root = Path(__file__).resolve().parent
    fla_path = Path(os.environ['FLA_KDA_NAIVE'])
    fla_sha = hashlib.sha256(fla_path.read_bytes()).hexdigest()
    assert fla_sha == '60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016'
    write('environment', dict(python=platform.python_version(), torch=torch.__version__,
          torch_npu=torch_npu.__version__, ascriptor=ascriptor.__version__, fla_naive_sha256=fla_sha,
          library_commit='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5',
          kernels_commit='b3b3f9c16df7c4626ed3c081032a1be5a753d0b1',
          source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob('*.py'))},
          wrapper_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (chunk, chunk_bwd, auto)}))
    plan = [('layout', e) for e in chunk._layout_runtime().kernels().values()]
    plan += [('forward', e) for e in chunk.kda_fwd_kernels().values()]
    plan += [('backward', e) for e in chunk_bwd.kda_bwd_kernels().values()]
    plan += [('decode', fused_recurrent._native_kernel(d)) for d in (torch.bfloat16, torch.float32)]
    assert len(plan) == 22 and sum(f == 'backward' for f, _ in plan) == 9
    builds = []
    for family, entry in plan:
        print('COMPILE_START', family, entry.name, args.block_dim, flush=True)
        start = time.monotonic()
        op = compile_kernel(entry, device='a5', block_dim=args.block_dim, backend='cce')
        builds.append(dict(family=family, entry=entry.name, block_dim=args.block_dim,
                           signature=op.signature, seconds=time.monotonic() - start,
                           vendor_files={str(p.relative_to(op.vendor_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in op.vendor_dir.rglob('*') if p.is_file() and p.suffix in ('.so', '.o', '.json')}))
        write('compile', dict(complete=len(builds) == 22, entries=builds))
        print('COMPILE_PASS', entry.name, flush=True)
    prepare(block_dim=args.block_dim, backward=True, decode=True)
    if args.mode == 'compile':
        return
    torch.npu.set_device(0)
    write('device', dict(name=torch.npu.get_device_name(0), block_dim=args.block_dim))
    before, before_bwd, before_auto = check.baseline(chunk, chunk_bwd, auto)
    real, reference = check.reference_modules()
    fla = check.load_file('_fmt02_fla_naive', fla_path)
    rows = []

    def execute(case, supplied=None):
        print('CASE_START', case['id'], flush=True)
        x = supplied if supplied is not None else real.make_inputs(
            **{n: case[n] for n in ('B', 'H', 'HV', 'C')}, span=case.get('span', 46), want_grads=True)
        if case.get('exact_span105'):
            g = x['g'].view(case['B'], case['C'], 64, case['HV'], 128)
            g[:, :, 0].zero_(); g[:, :, 1:61].fill_(-1.5); g[:, :, 61:].fill_(-5.)
            assert chunk._gate_span(x['g'], case['C'], on_cpu=True) == 105.
        if not case.get('state', True):
            x['h0'].zero_()
        dev = {n: t.npu() for n, t in x.items()}
        data = {n: dev[n] for n in ('q', 'k', 'v', 'g', 'beta')}
        data['initial_state'] = dev['h0'] if case.get('state', True) else None
        options = dict(block_dim=args.block_dim, layout_device='npu')
        print('FULL_WORKLOAD_FORWARD_CACHED_BACKWARD', flush=True)
        # First custom launch belongs to the complete original Kimi workload.
        # Each public output and intermediate GM output is poisoned at launch.
        with check.instrument() as (audit, launches):
            plain = chunk.chunk_kda_fwd(**data, output_final_state=True, **options)
            o, ht, caches = chunk.chunk_kda_fwd_with_caches(**data, **options)
            beta = chunk._layout_runtime().cast(dev['beta'], torch.bfloat16, block_dim=args.block_dim)
            got_grads = chunk_bwd.chunk_kda_bwd(
                **{n: dev[n] for n in ('q', 'k', 'v', 'do', 'dht')}, beta=beta, caches=caches, **options)
        torch.npu.synchronize()
        write(case['id'] + '-audit', dict(operations=audit.report(), unexpected=audit.unexpected(), launches=launches))
        assert not audit.unexpected(), audit.unexpected()
        got = check.cpu(dict(o=o, final_state=ht, **{'cache_' + n: t for n, t in caches.items()}, **got_grads))
        old_plain = before.chunk_kda_fwd(**data, output_final_state=True, **options)
        old_o, old_ht, old_cache = before.chunk_kda_fwd_with_caches(**data, **options)
        old_grads = before_bwd.chunk_kda_bwd(
            **{n: dev[n] for n in ('q', 'k', 'v', 'do', 'dht')}, beta=dev['beta'].bfloat16(), caches=old_cache, **options)
        torch.npu.synchronize()
        old = check.cpu(dict(o=old_o, final_state=old_ht, **{'cache_' + n: t for n, t in old_cache.items()}, **old_grads))
        exact = check.exact(got, old)
        exact.update(check.exact(dict(plain_o=plain[0], plain_state=plain[1]),
                                 dict(plain_o=old_plain[0], plain_state=old_plain[1])))
        assert set(caches) == set(chunk.BWD_CACHE_NAMES) and len(caches) == 9 and len(got_grads) == 6
        unchanged = {n: check.digest(dev[n]) == check.digest(t) for n, t in x.items()}
        row = dict(case=case, block_dim=args.block_dim, bitwise=exact, input_unchanged=unchanged,
                   input_sha256={n:check.digest(t) for n,t in x.items()},
                   plain_cached_exact=[check.digest(a) == check.digest(b) for a, b in zip(plain, (o, ht))],
                   all_outputs_finite=all(bool(t.isfinite().all()) for t in got.values()),
                   cpu_references={}, passed=False)
        write(case['id'], row)
        if not all(r['passed'] for r in exact.values()):
            torch.save(dict(inputs=x, actual=got, before=old), out / (case['id'] + '-failure.pt'))
            raise AssertionError('baseline/candidate bytes differ: ' + case['id'])
        print('CPU_FP32_REFERENCES', case['id'], flush=True)
        oracle_x = {n: x[n] for n in ('q', 'k', 'v', 'g', 'beta', 'h0')}
        references = dict(independent=reference.independent_reference(oracle_x), fla=reference.fla_reference(oracle_x))
        for label, ref in references.items():
            measures = {n: reference.metrics(got[n], ref[n]) for n in ('o', 'final_state')}
            measures['per_head_chunk'] = [dict(chunk=c, head=h,
                **reference.metrics(got['o'][:, c*64:(c+1)*64, h], ref['o'][:, c*64:(c+1)*64, h]))
                for c in range(case['C']) for h in range(case['HV'])]
            fn = real.kda_recurrent_ref if label == 'independent' else fla.naive_recurrent_kda
            gradient_ref = check.gradients(x, fn)
            measures['gradients'] = {n: check.metrics(got[n], value, real.BUDGET[n]) for n, value in gradient_ref.items()}
            row['cpu_references'][label] = measures
            write(case['id'], row)
        row['passed'] = (all(unchanged.values()) and all(row['plain_cached_exact']) and row['all_outputs_finite']
            and all(r['passed'] for r in exact.values())
            and all(m[n]['passed'] for m in row['cpu_references'].values() for n in ('o', 'final_state'))
            and all(g['passed'] for m in row['cpu_references'].values() for g in m['gradients'].values())
            and all(h['passed'] for m in row['cpu_references'].values() for h in m['per_head_chunk']))
        write(case['id'], row)
        rows.append(dict(case=case, passed=row['passed'], receipt=case['id']+'.json',
                         receipt_sha256=hashlib.sha256((out/(case['id']+'.json')).read_bytes()).hexdigest()))
        write('summary', dict(complete=False, passed=False, cases=rows))
        assert row['passed'], case['id']
        print('CASE_PASS', case['id'], flush=True)

    try:
        execute(dict(id='full_kimi_t4096', B=1, H=32, HV=32, C=64, state=True))
        import native_checks
        write('leaf', native_checks.leaf(chunk._layout_runtime(), args.block_dim))
        write('autograd_views', native_checks.autograd_views(auto, before_auto, real, args.block_dim))
        write('gates', native_checks.gates(chunk, before, real, args.block_dim))
        write('decode_audit', native_checks.decode_audit(fused_recurrent, real, args.block_dim))
        if args.mode == 'suite':
            for case, source in native_checks.original_backward_inputs(chunk_bwd):
                execute(case, source)
            for c in (1, 2, 3):
                for group in (1, 2, 4, 8):
                    for state in (False, True):
                        execute(dict(id=f'grid_c{c}_g{group}_s{int(state)}', B=2, H=2,
                                     HV=2*group, C=c, state=state))
            execute(dict(id='kimi_t1024', B=1, H=32, HV=32, C=16, state=True))
            execute(dict(id='gradient_span105', B=1, H=1, HV=1, C=2, state=True, exact_span105=True))
            for case in reference.grid_cases():
                source = reference.make_inputs(case)
                generator = torch.Generator().manual_seed(20602)
                source['do'] = (torch.randn(source['v'].shape,generator=generator)*.04).bfloat16()
                source['dht'] = (torch.randn(source['h0'].shape,generator=generator)*.01).bfloat16()
                execute(dict(case,id='public_'+case['id'],state=True),source)
            write('stable_contracts', native_checks.stable_contracts(chunk, before, real, reference, args.block_dim))
        if args.mode == 'perf':
            from native_perf import measure
            write('performance', measure(chunk, before, chunk_bwd, before_bwd, real, args.block_dim))
        write('summary', dict(complete=True, passed=True, scope=args.mode, cases=rows))
        print('NATIVE_PASS', args.mode, args.block_dim, len(rows), flush=True)
    except BaseException as error:
        write('failure', dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                             completed=[r['case']['id'] for r in rows if r['passed']]))
        raise


if __name__ == '__main__':
    main()
