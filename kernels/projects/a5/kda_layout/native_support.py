"""Acceptance helpers; none is imported by the production wrapper."""
from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import json
import traceback
import types
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline(chunk, backward, autograd):
    """Execute the pinned, unedited predecessor wrappers against the same vendors."""
    receipt = json.loads((ROOT / 'baseline/source.json').read_text())
    modules = []
    for name, actual in zip(('chunk', 'chunk_bwd', 'autograd'), (chunk, backward, autograd)):
        path = ROOT / 'baseline' / (name + '.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256'][path.name]
        module = types.ModuleType('ascend_fla.ops.kda._fmt02_before_' + name)
        module.__package__ = 'ascend_fla.ops.kda'
        # Root discovery addresses the identical readonly source dependencies.
        module.__file__ = actual.__file__
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        modules.append(module)
    fwd, bwd, auto = modules
    fwd._compiled_chain = chunk._compiled_chain
    bwd._compiled_chain = backward._compiled_chain
    bwd._resolve_layout = fwd._resolve_layout
    auto.chunk_kda_fwd = fwd.chunk_kda_fwd
    auto.chunk_kda_fwd_with_caches = fwd.chunk_kda_fwd_with_caches
    auto.chunk_kda_bwd = bwd.chunk_kda_bwd
    return modules


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {n: cpu(t) for n, t in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(t) for t in value)
    return value


def digest(value):
    value = cpu(value)
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def exact(actual, expected):
    assert set(actual) == set(expected)
    return {n: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=digest(a),
                    before_sha256=digest(expected[n]),
                    passed=a.shape == expected[n].shape and a.dtype == expected[n].dtype
                    and digest(a) == digest(expected[n])) for n, a in actual.items()}


def metrics(a, e, budget):
    a, e = cpu(a).float(), cpu(e).float()
    error = a - e
    relative = float(error.norm() / e.norm().clamp_min(1e-30))
    floor = float((e.bfloat16().float() - e).norm() / e.norm().clamp_min(1e-30))
    finite = bool(a.isfinite().all() and e.isfinite().all())
    return dict(relative_l2=relative, max_abs=float(error.abs().max()), finite=finite,
                budget=budget, bf16_rounding_floor=floor,
                exceeds_3F=relative > 3 * floor, passed=finite and relative <= budget)


class Audit(TorchDispatchMode):
    """Record all operations; only the explicitly registered source spans are exempt."""
    allowed = {'aten.empty.memory_format', 'aten.empty_strided.default', 'aten.view.default',
               'aten._unsafe_view.default', 'aten.as_strided.default', 'aten.slice.Tensor',
               'aten.select.int', 'aten.detach.default', 'aten.alias.default',
               'aten.unsqueeze.default', 'aten.squeeze.dim', 'aten.expand.default'}

    def __init__(self):
        super().__init__()
        self.rows = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        frames = traceback.extract_stack()
        scope = [f for f in frames if '/ascend_fla/ops/kda/' in f.filename
                 or '/kda_layout/runtime.py' in f.filename]
        origin = scope[-1] if scope else None
        category = 'operator'
        if any(f.name == '_scan_states' for f in scope):
            category = 'DPM42_scan_states_whole_exception_not_compliant'
        elif any(f.name in ('_gate_span', '_check_gate_range', '_check_input_domain') for f in scope):
            category = 'readonly_validation_DPM42_provisional'
        elif origin and origin.name == 'chunk_kda_bwd' and str(func) == 'aten.neg.default' and origin.line == 'dw = -d_vh':
            category = 'DPM42_dw_negation_exception_not_compliant'
        elif origin and origin.name == 'chunk_kda_fwd_with_caches' and str(func) == 'aten.log2.default':
            category = 'DPM42_upstream_log2_exception_not_compliant'
        file = str(Path(origin.filename).relative_to(REPO)) if origin else '<runtime bridge>'
        self.rows[(category, file, origin.lineno if origin else 0, str(func))] += 1
        return func(*args, **(kwargs or {}))

    def report(self):
        return [dict(category=c, file=f, line=l, operator=o, count=n)
                for (c, f, l, o), n in sorted(self.rows.items())]

    def unexpected(self):
        return [r for r in self.report() if r['category'] == 'operator' and r['operator'] not in self.allowed]


@contextlib.contextmanager
def instrument(*, poison=True, audit=True):
    """Poison actual GM destinations immediately before their real native launch."""
    from ascend_fla.runtime.compile import CompiledKernel
    original = CompiledKernel.__call__
    recorder = Audit()
    launches = []

    def call(op, inputs, scalars, outputs):
        with _disable_current_modes():
            if poison:
                for tensor in outputs.values():
                    tensor.fill_(float('nan'))
            launches.append(dict(operator=op.op.op_name, signature=op.signature,
                                 outputs=list(outputs), poisoned=poison))
        return original(op, inputs, scalars, outputs)

    CompiledKernel.__call__ = call
    try:
        with recorder if audit else contextlib.nullcontext():
            yield recorder, launches
    finally:
        CompiledKernel.__call__ = original


def reference_modules():
    return (load_file('_fmt02_real_shapes', REPO / 'benchmarks/verify_real_shapes.py'),
            load_file('_fmt02_repair_reference', ROOT.parent / 'kda_fwd_stable/repair_reference.py'))


def gradients(x, function):
    """CPU FP32 exact recurrence with checkpointing, never a kernel checkpoint."""
    from torch.utils.checkpoint import checkpoint
    names = ('q', 'k', 'v', 'beta', 'g', 'h0')
    leaves = {n: x[n].float().clone().requires_grad_(True) for n in names}
    state, outputs = leaves['h0'], []
    for start in range(0, x['q'].shape[1], 64):
        s = slice(start, start + 64)
        def segment(q, k, v, g, beta, state):
            return function(q, k, v, g, beta, initial_state=state, output_final_state=True)
        o, state = checkpoint(segment, *(leaves[n][:, s] for n in ('q', 'k', 'v', 'g', 'beta')),
                              state, use_reentrant=False)
        outputs.append(o)
    loss = (torch.cat(outputs, 1) * x['do'].float()).sum() + (state * x['dht'].float()).sum()
    values = torch.autograd.grad(loss, [leaves[n] for n in names])
    return dict(zip(('dq', 'dk', 'dv', 'dbeta', 'dg', 'dh0'), values))
