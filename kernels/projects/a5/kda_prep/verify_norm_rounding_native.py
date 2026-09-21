"""Measure ordinary/epsilon-dominated normalization rounding on the same NPU."""
import argparse
import itertools
from pathlib import Path

import torch

from ascend_fla.ops.kda import chunk
from native_context import Context
import native_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ctx = Context(args.output, 4, 4, __file__)
    rows = []
    for chunks, raw, target, population in itertools.product((1, 2, 3), native_grid.TYPES,
                                                           native_grid.TYPES, ('ordinary', 'nearzero')):
        case = dict(B=1, C=chunks, H=1, HV=1, seed=7011,
                    flags=dict(zip(native_grid.FLAG_NAMES, (True, False, False))),
                    types=dict(q=raw, k=raw, v=target, g='f32', beta='f32', A_log='f32', dt_bias='f32'),
                    boundary=population)
        values = native_grid.inputs(case) if population == 'ordinary' else native_grid.boundary_inputs(case)
        dtype = native_grid.TYPES[target]
        for name in ('q', 'k'):
            source = values[name]
            device = source.npu()
            actual = chunk._prep_runtime().norm(device, dtype, block_dim=4, namespace='decode').cpu()
            candidate_f32 = chunk._prep_runtime().norm(device, torch.float32, block_dim=4, namespace='decode').cpu()
            old_cpu = ctx.old._prepare_inputs(source, source, None, None,
                        use_qk_l2norm_in_kernel=True, qk_dtype=dtype)[0]
            old_npu = ctx.old._prepare_inputs(device, device, None, None,
                        use_qk_l2norm_in_kernel=True, qk_dtype=dtype)[0].cpu()
            cpu_squares = source.float().square().sum(-1, keepdim=True)
            cpu_denominator = (cpu_squares + 1e-6).sqrt()
            native_squares = device.float().square().sum(-1, keepdim=True)
            native_denominator = (native_squares + 1e-6).sqrt()
            native_quotient = (device.float() / native_denominator).cpu()
            native_squares, native_denominator = native_squares.cpu(), native_denominator.cpu()
            cpu_quotient = source.float() / cpu_denominator
            high = ctx.precision.high_precision_norm(source)
            different = actual != old_cpu
            samples = []
            for index in different.nonzero()[:8]:
                at = tuple(index.tolist())
                row_at = at[:-1] + (0,)
                samples.append(dict(index=list(at), raw=float(source[at]), candidate=float(actual[at]),
                    predecessor_cpu=float(old_cpu[at]), predecessor_npu=float(old_npu[at]),
                    candidate_f32_variant=float(candidate_f32[at]), cpu_quotient=float(cpu_quotient[at]),
                    native_torch_quotient=float(native_quotient[at]), fp64=float(high[at]),
                    cpu_denominator=float(cpu_denominator[row_at]), native_torch_denominator=float(native_denominator[row_at]),
                    cpu_square_sum=float(cpu_squares[row_at]), native_torch_square_sum=float(native_squares[row_at])))
            comparison = ctx.precision.metrics(actual, old_cpu)
            rows.append(dict(population=population, C=chunks, input_dtype=raw, output_dtype=target, tensor=name,
                input_sha256=ctx.check.digest(source), actual_sha256=ctx.check.digest(actual),
                predecessor_cpu_sha256=ctx.check.digest(old_cpu), predecessor_npu_sha256=ctx.check.digest(old_npu),
                elements=source.numel(), differing_from_cpu=int(different.sum()),
                differing_from_npu=int((actual != old_npu).sum()), to_cpu=comparison,
                to_fp64=ctx.precision.metrics(actual, high),
                f32_variant_cpu_cast_matches_output=ctx.check.digest(candidate_f32.to(dtype)) == ctx.check.digest(actual),
                cpu_epsilon_dominates_all_rows=bool((cpu_squares + 1e-6 == torch.tensor(1e-6)).all()),
                native_epsilon_dominates_all_rows=bool((native_squares + 1e-6 == torch.tensor(1e-6)).all()),
                samples=samples, input_unchanged=ctx.check.digest(device) == ctx.check.digest(source)))
            assert rows[-1]['input_unchanged']
            ctx.write('rounding-boundaries', dict(complete=False, cases=rows,
                scope='Located rounding diagnostics. Candidate FP32 variant and Torch primitive denominators are separate observations; the candidate internal denominator is not captured.'))
    ctx.write('rounding-boundaries', dict(complete=True, cases=rows,
        scope='Located rounding diagnostics; no new tolerance or formula. Both implementations divide by sqrt(sum(x*x)+1e-6).'))
    print('NORM_ROUNDING_COLLECTED', len(rows), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
