"""九个前向检查点的正确性（需要 A5 NPU + CANN）。

``kda_bwd`` 要九个前向检查点（``g_cumsum`` / ``Aqk`` / ``Akk`` / ``w`` / ``u`` / ``qg`` /
``kg`` / ``v_new`` / ``h``），而前向 kernel 只直接给出其中六个。
:func:`ascend_fla.ops.kda.chunk.chunk_kda_fwd_with_caches` 补齐另外三个。

**这是整条反向链的地基**：检查点错了，九个反向 kernel 全部白跑，而且错法会表现成
"某个梯度偏大"这种极难定位的样子。所以逐个对 ``kda_bwd`` 单元自己的
``ref.forward.build_saved_forward`` —— 那是它声明输入时用的同一个函数。

预算：``build_saved_forward`` 按 bwd ABI 收 **bf16** 输入，而我们的前向按 fwd ABI 收
``g``/``beta``/``initial_state`` 的 **fp32**。这个差异就是 ``kda-fwd-bwd-dtype-mismatch``，
已量化在 ~1e-3 量级（见 ``benchmarks/quantify_bwd_dtype_mismatch.py``），所以预算取
2e-2 —— 足以抓住结构性错误（错位、转置、base-2 与自然底混用），又不会被这个已知的
降精度误伤。

    pytest tests/test_kda_caches_npu.py -v
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.ops.kda.chunk import BWD_CACHE_NAMES, _kernels_root, chunk_kda_fwd_with_caches  # noqa: E402

# 结构性错误在小形状上就会暴露；大形状留给 benchmark。gva 覆盖 HV>H 的分组路径。
CASES = [
    pytest.param(1, 64, 1, 1, id="single_chunk"),
    pytest.param(1, 128, 1, 1, id="multi_chunk"),
    pytest.param(1, 128, 1, 2, id="gva"),
    pytest.param(1, 256, 2, 4, id="multi_head"),
]
BUDGET = 2e-2


def _bwd_unit_root() -> pathlib.Path:
    """kda_bwd 单元的位置。与 fwd 同级，只读引用（AGENTS.md §3）。"""
    direct = os.environ.get("ASCRIPTOR_KDA_BWD")
    if direct:
        return pathlib.Path(direct)
    return _kernels_root().parent / "kda_bwd"


@pytest.fixture(scope="module")
def build_saved_forward():
    """导入 kda_bwd 单元的 ``ref.forward.build_saved_forward``。

    ``ref`` 是单元目录下的包，只依赖自身子模块（不碰 ``_unit_runner``），所以把单元
    目录放进 ``sys.path`` 就能直接用。
    """
    root = _bwd_unit_root()
    if not (root / "ref" / "forward.py").is_file():
        pytest.skip(f"找不到 kda_bwd 单元的 ref/forward.py（试了 {root}）")
    sys.path.insert(0, str(root))
    try:
        from ref.forward import build_saved_forward as fn
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"导入 ref.forward 失败：{e}")
    return fn


def _inputs(b, t, h, hv, seed=2026):
    """按 kda_fwd 的 ABI 造输入：q/k/v bf16，g/beta/initial_state fp32。"""
    gen = torch.Generator().manual_seed(seed)
    kd = vd = 128
    return dict(
        q=(torch.randn(b, t, h, kd, generator=gen) * 0.04).bfloat16(),
        k=(torch.randn(b, t, h, kd, generator=gen) * 0.04).bfloat16(),
        v=(torch.randn(b, t, hv, vd, generator=gen) * 0.04).bfloat16(),
        g=torch.empty(b, t, hv, kd).uniform_(-0.03, 0.0, generator=gen),
        beta=torch.empty(b, t, hv).uniform_(0.05, 0.5, generator=gen),
        initial_state=torch.randn(b, hv, kd, vd, generator=gen) * 0.01,
    )


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("b,t,h,hv", CASES)
def test_caches_match_bwd_unit_reference(build_saved_forward, b, t, h, hv):
    x = _inputs(b, t, h, hv)
    npu = {n: v.to("npu") for n, v in x.items()}
    _, _, caches = chunk_kda_fwd_with_caches(**npu, layout_device="auto")

    # build_saved_forward 按 bwd ABI 收 bf16 —— 这一步降精度就是被量化的那个 gap
    ref = build_saved_forward(*(x[n].bfloat16() for n in
                                ("q", "k", "v", "g", "beta", "initial_state")), 64)
    want = dict(vars(ref))

    assert set(caches) == set(BWD_CACHE_NAMES) == set(want), (
        f"检查点名字不一致：我们 {sorted(caches)} / 参考 {sorted(want)}"
    )
    errors = {}
    for name in BWD_CACHE_NAMES:
        got, exp = caches[name], want[name]
        assert tuple(got.shape) == tuple(exp.shape), (
            f"{name} 形状不符：{tuple(got.shape)} vs 参考 {tuple(exp.shape)}"
        )
        errors[name] = _rel_l2(got.cpu(), exp)
    bad = {n: e for n, e in errors.items() if not (e < BUDGET)}
    assert not bad, f"超出预算 {BUDGET} 的检查点：{bad}；全部误差 {errors}"
    print(f"\nB{b} T{t} H{h} HV{hv} 检查点相对 L2："
          + "  ".join(f"{n}={errors[n]:.2e}" for n in BWD_CACHE_NAMES))


def test_caches_share_forward_output(build_saved_forward):
    """带缓存的前向与普通前向必须逐位相同 —— 它们共用同一次 kernel 调用。"""
    from ascend_fla.ops.kda.chunk import chunk_kda_fwd

    x = {n: v.to("npu") for n, v in _inputs(1, 128, 1, 2).items()}
    o_plain, state_plain = chunk_kda_fwd(**x, output_final_state=True)
    o_cached, state_cached, _ = chunk_kda_fwd_with_caches(**x)
    assert torch.equal(o_plain.cpu(), o_cached.cpu()), "o 不是逐位相同"
    assert torch.equal(state_plain.cpu(), state_cached.cpu()), "final_state 不是逐位相同"
