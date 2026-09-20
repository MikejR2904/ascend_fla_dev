"""BF-06 NPU acceptance; one block_dim per process and an external lock.

Runtime-generated CPU FP32 references are independent of the custom kernels.
The 17 required vendors, including all nine backward entries, precede execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-dim', type=int, choices=(1, 2, 4, 8, 16, 28), required=True)
    parser.add_argument('--mode', choices=('compile', 'full', 'suite'), default='full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('BF06_EXTERNAL_DEVICE_LOCK') == '1', 'hold both shared locks'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        p = out / (name + '.json')
        temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(p)

    import torch
    import torch_npu
    import ascriptor
    from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
    from ascend_fla.ops.kda import fused_recurrent as api
    from ascend_fla.ops.kda import chunk, chunk_bwd, prepare
    from ascend_fla.runtime.compile import compile_kernel
    import research

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    write('environment', dict(python=platform.python_version(), torch=torch.__version__,
          torch_npu=torch_npu.__version__, ascriptor=ascriptor.__version__,
          source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob('*.py'))},
          public_sha256=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
          library_commit='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5',
          kernels_commit='b3b3f9c16df7c4626ed3c081032a1be5a753d0b1'))
    chunk_bd = min(args.block_dim, 4)
    plan = [('decode', api._native_kernel(d), args.block_dim) for d in (torch.float32, torch.bfloat16)]
    plan += [('original_decode', api.kda_fused_recurrent_kernel(), args.block_dim)]
    plan += [('forward', e, chunk_bd) for e in chunk.kda_fwd_kernels().values()]
    plan += [('backward', e, chunk_bd) for e in chunk_bwd.kda_bwd_kernels().values()]
    assert len(plan) == 17 and sum(f == 'backward' for f, _, _ in plan) == 9
    builds = []
    old_op = None
    for family, entry, bd in plan:
        print('COMPILE_START', family, entry.name, bd, flush=True)
        start = time.monotonic()
        op = compile_kernel(entry, device='a5', block_dim=bd, backend='cce')
        if family == 'original_decode':
            old_op = op
        builds.append(dict(family=family, entry=entry.name, block_dim=bd,
                           signature=op.signature, seconds=time.monotonic() - start,
                           vendor_files={str(p.relative_to(op.vendor_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in op.vendor_dir.rglob('*')
                                         if p.is_file() and p.suffix in ('.so', '.o', '.json')}))
        write('compile', dict(complete=len(builds) == len(plan), entries=builds))
        print('COMPILE_PASS', entry.name, flush=True)
    prepare(block_dim=chunk_bd, backward=True, decode=True, decode_block_dim=args.block_dim)
    if args.mode == 'compile':
        return

    torch.npu.set_device(0)
    write('device', dict(name=torch.npu.get_device_name(0), block_dim=args.block_dim))
    fla = research.load_fla(os.environ['BF06_FLA_NAIVE'])

    def cpu(x):
        return x.detach().cpu().contiguous()

    def to_device(data):
        return {n: (x.npu() if isinstance(x, torch.Tensor) else x) for n, x in data.items()}

    class Audit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    def public(data):
        selected = api._compiled('a5', args.block_dim, data['v'].dtype)
        original_compiled = api._compiled
        poison_count = []

        def poison_call(ins, scalars, outputs):
            # Validation instrumentation only; every actual public output is
            # poisoned immediately before the real compiled kernel consumes it.
            with _disable_current_modes():
                for x in outputs.values():
                    x.fill_(float('nan'))
                poison_count.append(len(outputs))
            return selected(ins, scalars, outputs)

        api._compiled = lambda *a: poison_call
        try:
            with Audit() as audit:
                got = api.fused_recurrent_kda(**data, block_dim=args.block_dim)
        finally:
            api._compiled = original_compiled
        torch.npu.synchronize()
        assert poison_count == [2]
        unexpected = set(audit.operations) - {'aten.empty.memory_format', 'aten.view.default'}
        assert not unexpected, (unexpected, audit.operations)
        return tuple(cpu(x) for x in got), audit.operations

    def original_fp32(data):
        # Original public wrapper's FP32 preparation and output layout, with its
        # unchanged read-only kernel. This baseline intentionally includes host work.
        q, k, v, g, beta = (data[n] for n in ('q', 'k', 'v', 'g', 'beta'))
        b, t, h, _ = q.shape
        hv = v.shape[2]
        groups = hv // h
        def bhv(x):
            return x.to(torch.float32).permute(0, 2, 1, 3).contiguous()
        qs = bhv(q.repeat_interleave(groups, dim=2) * data['scale']) if groups > 1 else bhv(q * data['scale'])
        kk = bhv(k.repeat_interleave(groups, dim=2)) if groups > 1 else bhv(k)
        bb = beta.float().permute(0, 2, 1).contiguous().view(b, hv, 1, t)
        state = data['initial_state']
        if state is None:
            state = torch.zeros(b, hv, 128, 128, dtype=torch.float32, device='cpu').to(q.device)
        oo = torch.empty(b, hv, t, 128, dtype=torch.float32, device=q.device)
        ss = torch.empty(b, hv, 128, 128, dtype=torch.float32, device=q.device)
        old_op(dict(qs=qs, k=kk, v=bhv(v), g=bhv(g), beta=bb, initial_state=state),
               dict(B=b, HV=hv, T=t, head_dim=128, value_dim=128), dict(o=oo, final_state=ss))
        result = oo.permute(0, 2, 1, 3).contiguous(), ss
        torch.npu.synchronize()
        return tuple(cpu(x) for x in result)

    rows = []

    def execute(p, *, fp32=False):
        label = p['id'] + ('_fp32' if fp32 else '_bf16')
        print('CASE_START', label, flush=True)
        data = research.make_inputs(p)
        if fp32:
            for n in ('q', 'k', 'v'):
                data[n] = data[n].float()
        refs = research.references(data, fla)
        dev = to_device(data)
        got, ops = public(dev)
        if fp32:
            comparison = {n: dict(o=research.metrics(got[0], r[0]), final_state=research.metrics(got[1], r[1]))
                          for n, r in refs.items()}
            for row in comparison.values():
                row['passed'] = all(m['finite'] and m['relative_l2'] <= 1e-5 for m in row.values())
            old = original_fp32(dev)
            exact = [research.digest(a) == research.digest(b) for a, b in zip(got, old)]
        else:
            comparison = research.compare(got, refs)
            exact = None
        unchanged = {n: research.digest(cpu(dev[n])) == research.digest(x)
                     for n, x in data.items() if isinstance(x, torch.Tensor)}
        row = dict(case=p, dtype='float32' if fp32 else 'bfloat16', block_dim=args.block_dim,
                   comparison=comparison, original_fp32_bitwise=exact, input_unchanged=unchanged,
                   host_operations=ops, poison_all_written=[bool(torch.isfinite(x).all()) for x in got],
                   input_sha256={n: research.digest(x) for n, x in data.items() if isinstance(x, torch.Tensor)},
                   output_sha256=dict(zip(('o', 'final_state'), map(research.digest, got))))
        row['passed'] = (all(c['passed'] for c in comparison.values()) and all(unchanged.values())
                         and all(row['poison_all_written']) and (exact is None or all(exact)))
        write(label, row)
        rows.append(row)
        write('summary', dict(complete=False, passed=False, cases=rows))
        if not row['passed']:
            torch.save(dict(inputs=data, actual=got, reference=refs), out / (label + '-failure.pt'))
            raise AssertionError(row)
        print('CASE_PASS', label, json.dumps(comparison['A']), flush=True)

    try:
        full = dict(id='full_b2_t16_h4_g8', B=2, T=16, H=4, G=8, state=True)
        execute(full)
        execute(full, fp32=True)
        if args.mode == 'suite':
            for p in research.cases():
                execute(p)
            for p in research.cases():
                if p['T'] in (1, 2, 16) and p['id'].startswith(('grid_', 'real_')):
                    execute(p, fp32=True)
        write('summary', dict(complete=True, passed=True, cases=rows))
        print('NATIVE_PASS', args.block_dim, len(rows), flush=True)
    except BaseException as error:
        write('failure', dict(type=type(error).__name__, error=str(error),
                             traceback=traceback.format_exc(), completed=[r['case']['id'] for r in rows if r['passed']]))
        raise


if __name__ == '__main__':
    main()
