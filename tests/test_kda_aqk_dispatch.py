"""Host routing regressions; native numerical evidence uses the public probe."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ascend_fla.ops.kda import chunk as forward
from ascend_fla.ops.kda import chunk_bwd as backward


def test_actual_sources_keep_repaired_dispatch_and_original_negative_control():
    pytest.importorskip("ascriptor")
    try:
        root = forward._kernels_root()
    except FileNotFoundError:
        pytest.skip("The accepted upstream kernel checkout is required")
    assert forward.kda_fwd_kernels()["recurrent"].name == "kda_sub45_aqk_repaired_kernel"
    stable = forward.kda_fwd_kernels("stable")
    original = forward.kda_fwd_kernels("upstream")
    assert stable["recurrent"].name == "kda_sub45_aqk_repaired_kernel"
    assert original["recurrent"].name == "kda_sub45_fused_kernel"
    assert Path(inspect.getsourcefile(stable["recurrent"].fn)).resolve() == (
        forward._stable_kernels_root() / "kernels/recurrent.py").resolve()
    assert Path(inspect.getsourcefile(original["recurrent"].fn)).resolve() == (
        root / "kernels/recurrent.py").resolve()
    assert stable["inverse"].name == original["inverse"].name

    path = forward._stable_kernels_root() / "repair_runtime.py"
    spec = importlib.util.spec_from_file_location("_aqk_control_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    baseline = runtime.selected_kernels("baseline")
    repaired = runtime.selected_kernels("repaired")
    assert baseline["recurrent"] is original["recurrent"]
    assert repaired["recurrent"].name == stable["recurrent"].name
    for stage in ("gate", "scores", "inverse", "wy"):
        assert baseline[stage] is repaired[stage] is stable[stage]
    with pytest.raises(ValueError, match="Unknown variant"):
        runtime.selected_kernels("typo")


def test_cached_forward_dependency_fits_mutex_budget_and_keeps_upstream_control():
    pytest.importorskip("ascriptor")
    from ascriptor.passes.manager import PassError
    from ascriptor.runtime.opexec import lower_kernel

    try:
        root = backward._bwd_kernels_root()
    except FileNotFoundError:
        pytest.skip("The accepted upstream kernel checkout is required")
    stable = backward.kda_bwd_kernels("stable")
    original = backward.kda_bwd_kernels("upstream")
    assert stable["inverse_mm"].name == "inverse_mm_bounded_kernel"
    assert original["inverse_mm"].name == "inverse_mm_kernel"
    assert Path(inspect.getsourcefile(stable["inverse_mm"].fn)).resolve() == (
        backward._stable_bwd_root() / "kernels/inverse_mm.py").resolve()
    assert Path(inspect.getsourcefile(original["inverse_mm"].fn)).resolve() == (
        root / "kernels/inverse_mm.py").resolve()
    lowered = lower_kernel(stable["inverse_mm"])
    counts = {fn.attrs["side"].name: fn.attrs["local_mutex_count"]
              for fn in lowered.functions if "side" in fn.attrs}
    assert counts == {"cube": 32, "vec": 15}
    with pytest.raises(PassError, match=r"cube needs 34 mutex IDs \(maximum 32\)"):
        lower_kernel(original["inverse_mm"])


class _NpuMetadata:
    """Only enough metadata to reach the public safety/dispatch boundary."""

    device = SimpleNamespace(type="npu")

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True


def _inputs(c):
    return [
        _NpuMetadata((1, 64 * c, 1, 128), torch.bfloat16),
        _NpuMetadata((1, 64 * c, 1, 128), torch.bfloat16),
        _NpuMetadata((1, 64 * c, 2, 128), torch.bfloat16),
        _NpuMetadata((1, 64 * c, 2, 128), torch.float32),
        _NpuMetadata((1, 64 * c, 2), torch.float32),
    ]


@pytest.mark.parametrize("entry", ["chunk_kda_fwd", "chunk_kda_fwd_with_caches"])
@pytest.mark.parametrize("c", [1, 3, 5])
def test_unsafe_upstream_rejected_before_layout_compile_or_launch(monkeypatch, entry, c):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe upstream call passed its public safety gate")

    for name in ("_resolve_layout", "_check_gate_range", "_compiled_chain", "_run_chain"):
        monkeypatch.setattr(forward, name, forbidden)
    monkeypatch.setattr(backward, "_compiled_chain", forbidden)
    with pytest.raises(ValueError, match=f"odd C={c}"):
        getattr(forward, entry)(*_inputs(c), block_dim=1, impl="upstream",
                                check_gate_range=False)


@pytest.mark.parametrize("entry", ["chunk_kda_fwd", "chunk_kda_fwd_with_caches"])
@pytest.mark.parametrize("impl,c", [("stable", 1), ("stable", 3), ("stable", 5),
                                   ("upstream", 2), ("upstream", 4)])
def test_public_entries_forward_explicit_impl_and_preserve_gate_path(monkeypatch, entry, impl, c):
    calls = []

    class ReachedLaunch(Exception):
        pass

    def launch(*args, **kwargs):
        calls.append(("forward", kwargs["impl"], kwargs["c"]))
        raise ReachedLaunch

    monkeypatch.setattr(forward, "_resolve_layout", lambda _: False)
    monkeypatch.setattr(forward, "_check_gate_range",
                        lambda *a, **kw: calls.append(("gate", kw["impl"], kw["path"])))
    monkeypatch.setattr(backward, "_compiled_chain",
                        lambda device, bd, selected: calls.append(("backward", selected, bd)))
    monkeypatch.setattr(forward, "_run_chain", launch)
    with pytest.raises(ReachedLaunch):
        getattr(forward, entry)(*_inputs(c), block_dim=1, impl=impl)
    cached = entry.endswith("with_caches")
    expected = [("gate", impl, "backward" if cached else "forward")]
    if cached:
        expected.append(("backward", impl, 1))
    assert calls == expected + [("forward", impl, c)]


@pytest.mark.parametrize("entry", ["chunk_kda_fwd", "chunk_kda_fwd_with_caches"])
def test_unknown_impl_is_rejected_even_when_gate_range_check_is_disabled(entry):
    with pytest.raises(ValueError, match="impl must be one of"):
        getattr(forward, entry)(None, None, None, None, None,
                                impl="typo", check_gate_range=False)
