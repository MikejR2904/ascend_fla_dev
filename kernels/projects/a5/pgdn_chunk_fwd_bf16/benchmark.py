"""Native PGDN leaves/composition, original FP32 comparison and host audit."""
import argparse
import collections
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ascend_fla.ops import pgdn_chunk_fwd as public
from ref import oracle
from ref.reference import OUTPUTS, make_inputs, reference_stages, metric, budget


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_baseline(root):
    current = Path(public.__file__).resolve().parents[2]
    relative = Path('kernels/projects/a5/pgdn_chunk_fwd')
    files = [p for p in (root / relative).rglob('*.py')]
    assert files
    for path in files:
        assert path.read_bytes() == (current / path.relative_to(root)).read_bytes()
    path = root / 'ascend_fla/ops/pgdn_chunk_fwd.py'
    spec = importlib.util.spec_from_file_location('ascend_fla.ops._bf03_original_public', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Identical FP32 entries share compilation; the original wrapper is intact.
    module._pipeline = public._pipeline
    module._compiled = public._compiled
    return module, dict(wrapper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        unchanged_fp32_python_files=len(files))


class Audit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.phase = 'production'
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.events.append(dict(phase=self.phase, operator=str(func)))
        return func(*args, **(kwargs or {}))

    def validation(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            assert self.phase == 'production'
            self.phase = 'validation'
            try:
                return function(*args, **kwargs)
            finally:
                self.phase = 'production'
        return wrapped


def check(actual, expected, dtype, *, public_outputs=False):
    assert set(actual) == set(expected)
    result = {}
    for name, target in expected.items():
        value = actual[name]
        assert value.shape == target.shape
        assert value.dtype == (dtype if name == 'o' else torch.float32)
        limit = budget(target, dtype, name) if public_outputs or name == 'o' else 1e-4
        row = dict(**metric(value, target), budget=limit)
        assert row['finite'] and row['relative_l2'] <= limit, (name, row)
        if not public_outputs:
            torch.testing.assert_close(value.float(), target, atol=2e-5,
                                       rtol=.01 if name == 'o' and dtype == torch.bfloat16 else 2e-4)
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block-dim', required=True, type=int, choices=(1, 2))
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--oracle-workers', type=int, choices=(1,), default=1)
    args = parser.parse_args()
    contract = json.loads((Path(__file__).parent / 'contract.json').read_text())
    cases = [c for c in contract['cases'] if c['block_dim'] == args.block_dim]
    wanted = set(args.case or [f'pgdn340m_bd{args.block_dim}'])
    if wanted != {'all'}:
        cases = [c for c in cases if c['id'] in wanted]
        assert {c['id'] for c in cases} == wanted
    cases.sort(key=lambda c: c['id'] != f'pgdn340m_bd{args.block_dim}')
    import torch_npu
    torch.set_num_threads(1)
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    baseline, baseline_identity = load_baseline(args.baseline_root)
    print('BUILD_START', flush=True)
    public.prepare(block_dim=args.block_dim)
    print('BUILD_DONE', flush=True)
    pipelines = {torch.float32: public._pipeline(), torch.bfloat16: public._bf16_pipeline()}
    entries = (*pipelines[torch.float32].entries(), *pipelines[torch.bfloat16].entries())
    compiled = dict(zip((entry.name for entry in entries), public._compiled(args.block_dim)))
    report = dict(stage='native_inprocess_bf03', block_dim=args.block_dim,
                  oracle_pin=oracle.PIN, oracle_sha256=oracle.SHA256, baseline=baseline_identity,
                  torch=torch.__version__, torch_npu=torch_npu.__version__, cases=[])
    allowed = {'aten.empty.memory_format', 'aten.empty_strided.default', 'aten.view.default',
               'aten.unsqueeze.default', 'aten.alias.default', 'aten.zeros.default'}
    for case in cases:
        rounded_case = None
        for dtype_name in ('bfloat16', 'float32'):
            started = time.monotonic()
            cpu = make_inputs(dict(case, dtype=dtype_name))
            dtype = cpu['q'].dtype
            a = oracle.reference(cpu)
            stages = reference_stages(cpu)
            b = {name: stages[name] for name in OUTPUTS}
            device = {name: value.npu() for name, value in cpu.items()}
            before = {name: digest(value) for name, value in device.items()}
            pipeline = pipelines[dtype]
            def launch(entry, sources, outputs, scalars):
                for value in outputs.values():
                    value.fill_(float('nan'))
                kernel = compiled[entry.name]
                kernel(sources, {name: scalars[name] for name in kernel.scalar_names}, outputs)
                return outputs
            composition = pipeline.run(device, launch)
            torch.npu.synchronize()
            composed = {name: value.cpu() for name, value in composition.items()}
            composition_numbers = check(composed, stages, dtype)
            upstream = {name: value.contiguous().npu() for name, value in dict(cpu, **stages).items()}
            def leaf(entry, sources, outputs, scalars):
                return launch(entry, {name: upstream[name] for name in sources}, outputs, scalars)
            leaves = pipeline.run(device, leaf)
            torch.npu.synchronize()
            leaf_numbers = check({name: value.cpu() for name, value in leaves.items()}, stages, dtype)
            inputs = {name: value for name, value in device.items() if name != 'initial_state'}
            audit = Audit()
            original_validate = public._validate
            public._validate = audit.validation(original_validate)
            try:
                with audit:
                    returned = public.chunk_pgdn(**inputs, output_final_state=True, block_dim=args.block_dim)
            finally:
                public._validate = original_validate
            torch.npu.synchronize()
            actual = {name: value.cpu() for name, value in zip(OUTPUTS, returned)}
            forbidden = [row for row in audit.events if row['phase'] == 'production' and row['operator'] not in allowed]
            assert not forbidden, forbidden
            hashes = {name: digest(value) for name, value in actual.items()}
            assert hashes == {name: digest(composed[name]) for name in OUTPUTS}
            unchanged = all(digest(value) == before[name] for name, value in device.items())
            assert unchanged
            same_old = None
            if dtype == torch.float32:
                original = baseline.chunk_pgdn(**inputs, output_final_state=True, block_dim=args.block_dim)
                torch.npu.synchronize()
                same_old = hashes == {name: digest(value) for name, value in zip(OUTPUTS, original)}
                assert same_old
            row = dict(case=case, dtype=dtype_name, public_A=check(actual, a, dtype, public_outputs=True),
                       public_B=check(actual, b, dtype, public_outputs=True), composition=composition_numbers,
                       independent_leaf=leaf_numbers, public_hashes=hashes,
                       stage_hashes={name: digest(value) for name, value in composed.items()}, inputs=before,
                       inputs_unchanged=unchanged, fp32_original_byte_equal=same_old,
                       host_events=audit.events, host_counts={phase: dict(collections.Counter(
                           r['operator'] for r in audit.events if r['phase'] == phase))
                           for phase in ('validation', 'production')}, forbidden_operators=forbidden,
                       seconds=time.monotonic() - started)
            if 'norm' in case['parameters']:
                if dtype == torch.bfloat16:
                    rounded_case = dict(native=actual, A=a, B=b,
                                        actual_q_first_component=cpu['q'][0, 0, 0, 0].item())
                else:
                    # This compares different input values; it is not device error
                    # and does not replace either path's own fixed A/B checks.
                    row['bf16_vs_unrounded_fp32'] = dict(
                        actual_bf16_q_first_component=rounded_case['actual_q_first_component'],
                        native_including_output_rounding={n: metric(rounded_case['native'][n], actual[n])
                                                          for n in OUTPUTS},
                        oracle_A_input_quantization={n: metric(rounded_case['A'][n], a[n]) for n in OUTPUTS},
                        oracle_B_input_quantization={n: metric(rounded_case['B'][n], b[n]) for n in OUTPUTS})
            report['cases'].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('NATIVE_CASE', json.dumps(row, allow_nan=False), flush=True)
            del composition, composed, upstream, leaves, actual, device, returned, inputs
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('NATIVE_PASS', len(report['cases']), flush=True)


if __name__ == '__main__':
    main()
