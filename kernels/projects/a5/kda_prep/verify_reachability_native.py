"""Actual public behavior at the measured raw-gate endpoint cross-products."""
import argparse
from pathlib import Path

import torch

from ascend_fla.ops.kda import autograd, fused_recurrent
from native_context import Context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ctx = Context(args.output, 4, 4, __file__)
    generator = torch.Generator().manual_seed(7019)
    rows = []
    for route, tokens in (('chunk', 64), ('decode', 1)):
        q, k = (torch.randn(1, tokens, 1, 128, generator=generator) for _ in range(2))
        q, k = (x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k))
        base = dict(q=q.bfloat16(), k=k.bfloat16(),
                    v=(torch.randn(1, tokens, 1, 128, generator=generator) * .04).bfloat16(),
                    beta=torch.full((1, tokens, 1), .3),
                    initial_state=torch.randn(1, 1, 128, 128, generator=generator) * .01,
                    dt_bias=torch.zeros(128))
        for alog in (-3., -.2, 2.7, 80., 89., 100.):
            for value in (-104., -100., -90., -88., -87., -40., -20., -16.):
                cpu = dict(base, g=torch.full((1, tokens, 1, 128), value), A_log=torch.tensor([alog]))
                device = {n: t.npu() for n, t in cpu.items()}
                before = {n: ctx.check.digest(t) for n, t in device.items()}
                results = {}
                apis = (('candidate', autograd.chunk_kda if route == 'chunk' else fused_recurrent.fused_recurrent_kda),
                        ('predecessor_npu', ctx.old_auto.chunk_kda if route == 'chunk' else ctx.old_decode.fused_recurrent_kda))
                for label, api in apis:
                    kwargs = dict(device, use_gate_in_kernel=True, output_final_state=True, block_dim=4)
                    if route == 'chunk':
                        kwargs['layout_device'] = 'npu'
                    try:
                        with torch.no_grad():
                            output, state = api(**kwargs)
                        torch.npu.synchronize()
                        tensors = ctx.check.cpu(dict(o=output, final_state=state))
                        results[label] = dict(behavior='returned', outputs={
                            n: dict(dtype=str(t.dtype), sha256=ctx.check.digest(t),
                                    finite=int(t.isfinite().sum()), nan=int(t.isnan().sum()),
                                    positive_infinity=int(t.isposinf().sum()), negative_infinity=int(t.isneginf().sum()),
                                    elements=t.numel()) for n, t in tensors.items()})
                    except ValueError as exc:
                        results[label] = dict(behavior='rejected', exception='ValueError', message=str(exc))
                preserved = {n: ctx.check.digest(t) == before[n] for n, t in device.items()}
                assert all(preserved.values())
                rows.append(dict(route=route, tokens=tokens, A_log=alog, u=value,
                                 within_fla_default_alog_range=0 <= alog <= 2.772588722239781,
                                 public_default_checks=True, behaviors=results, input_unchanged=preserved))
                ctx.write('public-reachability', dict(complete=False, cases=rows,
                    scope='Observed endpoint behavior only; no numerical pass line and no input-domain change.'))
                print('PUBLIC_ENDPOINT', route, alog, value, results['candidate']['behavior'], flush=True)
    ctx.write('public-reachability', dict(complete=True, cases=rows,
        scope='Observed endpoint behavior only; no numerical pass line and no input-domain change.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
