"""Native decode ABI and computational host-op regression checks.

These CPU proxy checks do not substitute for the separate NPU acceptance.
"""
import importlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

decode = importlib.import_module('ascend_fla.ops.kda.fused_recurrent')


def inputs(dtype, state=True):
    return dict(q=torch.randn(2, 2, 2, 128).to(dtype),
                k=torch.randn(2, 2, 2, 128).to(dtype),
                v=torch.randn(2, 2, 8, 128).to(dtype),
                g=torch.randn(2, 2, 8, 128), beta=torch.rand(2, 2, 8),
                initial_state=torch.randn(2, 8, 128, 128) if state else None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('state', [False, True])
@pytest.mark.parametrize('return_state', [False, True])
def test_default_public_call_has_native_pointers_and_only_allocation(
        monkeypatch, dtype, state, return_state):
    data = inputs(dtype, state)
    snapshots = {n: x.clone() for n, x in data.items() if x is not None}
    captured = {}

    def kernel(ins, scalars, outs):
        captured.update(inputs=ins, scalars=scalars, outputs=outs)
        # The real aclnn call does not perform Torch arithmetic. Fill only the
        # proxy's actual returned buffers, outside the audited host operation.
        with _disable_current_modes():
            outs['o'].fill_(0.25)
            outs['final_state'].fill_(-0.5)

    def compiled(device, bd, selected_dtype):
        assert (device, bd, selected_dtype) == ('a5', 28, dtype)
        return kernel

    class Audit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(decode, '_compiled', compiled)
    with Audit() as audit:
        out, final = decode.fused_recurrent_kda(
            **data, scale=.3, output_final_state=return_state, block_dim=28)
    assert audit.operations == ['aten.empty.memory_format'] * 2
    for name in ('q', 'k', 'v', 'g', 'beta'):
        assert captured['inputs'][name] is data[name]
    assert captured['scalars'] == dict(B=2, T=2, H=2, HV=8, has_initial=int(state), scale=.3)
    initial = captured['inputs']['initial_state']
    assert initial is (data['initial_state'] if state else captured['outputs']['final_state'])
    assert out is captured['outputs']['o']
    assert out.dtype == dtype and out.shape == (2, 2, 8, 128)
    assert torch.equal(out, torch.full_like(out, .25))
    assert final is (captured['outputs']['final_state'] if return_state else None)
    if final is not None:
        assert final.dtype == torch.float32
        assert torch.equal(final, torch.full_like(final, -.5))
    for name, before in snapshots.items():
        assert torch.equal(data[name], before)


@pytest.mark.parametrize('name,dtype', [
    ('q', torch.float32), ('k', torch.float32),
    ('v', torch.float16), ('v', torch.float64),
    ('g', torch.bfloat16), ('beta', torch.bfloat16),
    ('initial_state', torch.bfloat16),
])
def test_invalid_dtype_rejected_before_compile(monkeypatch, name, dtype):
    data = inputs(torch.bfloat16)
    data[name] = data[name].to(dtype)
    monkeypatch.setattr(decode, '_compiled', lambda *a: pytest.fail('invalid ABI reached compilation'))
    with pytest.raises(ValueError, match='dtype|FP32|BF16'):
        decode.fused_recurrent_kda(**data)


@pytest.mark.parametrize('name', ['q', 'k', 'v', 'g', 'beta', 'initial_state'])
def test_noncontiguous_input_rejected_before_launch(monkeypatch, name):
    data = inputs(torch.bfloat16)
    data[name] = data[name].transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not data[name].is_contiguous()
    monkeypatch.setattr(decode, '_compiled', lambda *a: pytest.fail('invalid layout reached compilation'))
    with pytest.raises(ValueError, match='contiguous'):
        decode.fused_recurrent_kda(**data)


def test_prepare_compiles_both_dtype_vendors_once(monkeypatch):
    compiler = importlib.import_module('ascend_fla.runtime.compile')
    calls = []
    kernels = {d: object() for d in (torch.float32, torch.bfloat16)}
    monkeypatch.setattr(decode, '_native_kernel', kernels.__getitem__)

    def compile_kernel(entry, **kwargs):
        calls.append((entry, kwargs))
        return entry

    monkeypatch.setattr(compiler, 'compile_kernel', compile_kernel)
    prep = decode._prep_runtime()
    prep.prepare.cache_clear()
    prep_kernels = set(prep.kernels('decode').values())
    assert len(prep_kernels) == 14
    decode._compiled.cache_clear()
    decode._compiled_pair.cache_clear()
    try:
        assert decode._compiled('a5', 4) is kernels[torch.float32]
        assert decode._compiled('a5', 4, torch.bfloat16) is kernels[torch.bfloat16]
        decode_calls = [row for row in calls if row[0] in kernels.values()]
        prep_calls = [row for row in calls if row[0] in prep_kernels]
        assert len(decode_calls) == 2
        assert {row[0] for row in decode_calls} == set(kernels.values())
        assert len(prep_calls) == len(prep_kernels)
        assert {row[0] for row in prep_calls} == prep_kernels
        assert len(calls) == len(prep_kernels) + len(kernels)
        assert all(kw == dict(device='a5', block_dim=4) for _, kw in calls)
    finally:
        prep.prepare.cache_clear()
        decode._compiled.cache_clear()
        decode._compiled_pair.cache_clear()
