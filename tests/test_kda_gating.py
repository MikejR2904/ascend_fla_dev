"""KDA 的门控测试 —— 纯 host 侧，**不需要 NPU**。

这些检查必须在没有硬件的机器上也能跑：它们守的正是"不满足就报错"这条规则
（AGENTS.md §7），而其中 block_dim 那一条如果漏掉，后果是在硬件 barrier 上
死锁而不是报错 —— 那是最不该靠有卡的机器才能发现的问题。

    pytest tests/test_kda_gating.py -v
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda.chunk import SUPPORTED_BLOCK_DIM, _check  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("bad", [0, -1, 5, 8, 28, 32])
def test_block_dim_outside_contract_is_rejected(bad):
    """block_dim 门控在碰任何张量之前就报错。"""
    with pytest.raises(ValueError, match="block_dim 只支持"):
        _check(None, None, None, None, None, None, bad)


@pytest.mark.parametrize("ok", SUPPORTED_BLOCK_DIM)
def test_block_dim_inside_contract_passes_that_check(ok):
    """契约内的值不能被 block_dim 这一条挡住（后面的张量检查报错是预期的）。"""
    with pytest.raises((ValueError, AttributeError)) as e:
        _check(None, None, None, None, None, None, ok)
    assert "block_dim" not in str(e.value)


def test_supported_block_dim_matches_ascriptor_contract():
    """别让这个常量和 ascriptor 契约各自漂移。

    契约不在本仓内（只读引用，见 AGENTS.md §3），所以本机没有 kernels 树时 skip；
    有的时候则必须一致 —— 这是"矩阵是唯一真相源"的同款约束。
    """
    from ascend_fla.ops.kda.chunk import _kernels_root

    try:
        contract = _kernels_root() / "contract.json"
    except FileNotFoundError:
        pytest.skip("本机没有 ascriptor kernels 树")
    if not contract.is_file():
        pytest.skip(f"{contract} 不存在")
    declared = json.loads(contract.read_text(encoding="utf-8"))["domain"]["block_dim"]
    assert tuple(declared) == SUPPORTED_BLOCK_DIM, (
        f"contract.json 声明 {declared}，chunk.py 写的是 {list(SUPPORTED_BLOCK_DIM)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 门控跨度的上限：代码、两个单元的 contract、以及两条链的理论值必须互相对得上。
# 这些都是纯常量核对，没卡也能跑 —— 而它们守的是一类很隐蔽的错：上限只按前向定，
# 于是跨度 94 能过检查、前向正常、反向吐 NaN（见 gaps.json 的 bwd-gate-range-overflow）。
# ──────────────────────────────────────────────────────────────────────────────

#: fp32/bf16 的两条硬线。前向失效于下溢、反向失效于上溢，所以两个常数都要用到。
UNDERFLOW_LN = 87.3368   # -ln(FLT_MIN_NORMAL)
OVERFLOW_LN = 88.7228    # ln(FLT_MAX) == ln(BF16_MAX) 到四位有效数字


def _unit_contract(name: str) -> dict:
    path = REPO / "kernels/projects/a5" / name / "contract.json"
    assert path.is_file(), f"本仓自有单元缺 contract.json：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_span_keys_match_impls():
    from ascend_fla.ops.kda.chunk import IMPLS, MAX_GATE_SPAN

    assert set(MAX_GATE_SPAN) == set(IMPLS), (
        f"MAX_GATE_SPAN 的键 {sorted(MAX_GATE_SPAN)} 与 IMPLS {sorted(IMPLS)} 不一致 ——"
        f"多一个实现却没给上限，门控就会 KeyError 而不是报出有用的错"
    )


def test_gate_span_has_both_paths():
    """每套实现都必须给出前向与反向两条链各自的上限。

    它们不是同一回事：前向的约束是**有限性**（越线吐 NaN），反向的约束是**精度**
    （到 169.8 都还有限，但 dq 在 130 处就超出契约预算 0.05）。用一个数会说谎。
    """
    from ascend_fla.ops.kda.chunk import GATE_PATHS, MAX_GATE_SPAN

    for impl, limits in MAX_GATE_SPAN.items():
        assert set(limits) == set(GATE_PATHS), (
            f"{impl} 的上限键是 {sorted(limits)}，应当是 {sorted(GATE_PATHS)}"
        )


def test_upstream_limit_guards_the_stricter_chain():
    """``upstream`` 两条链的上限都必须挡在更早失效的那条线以内。

    前向在 ``-ln(FLT_MIN_NORMAL) ≈ 87.34`` 处下溢，反向在 ``ln(FLT_MAX) ≈ 88.72`` 处上溢。
    """
    from ascend_fla.ops.kda.chunk import MAX_GATE_SPAN

    for path, limit in MAX_GATE_SPAN["upstream"].items():
        assert limit < min(UNDERFLOW_LN, OVERFLOW_LN), (
            f"upstream 的 {path} 上限 {limit} 没有挡在 "
            f"{min(UNDERFLOW_LN, OVERFLOW_LN):.2f} 以内"
        )


def test_stable_limits_are_wider_than_upstream_and_cover_default_init():
    """``stable`` 两条链都要比 upstream 宽，且都要覆盖 fla 默认初始化的跨度 ~94。

    后半条是这次改动的**目的**：默认初始化的层要能训练。反向的闸若退到 94 以下，
    默认初始化就又用不了了。
    """
    from ascend_fla.ops.kda.chunk import MAX_GATE_SPAN

    theoretical = min(2 * UNDERFLOW_LN, 2 * OVERFLOW_LN)
    for path, limit in MAX_GATE_SPAN["stable"].items():
        assert limit < theoretical, f"stable 的 {path} 上限 {limit} 超过理论值 {theoretical:.1f}"
        assert limit > MAX_GATE_SPAN["upstream"][path], (
            f"stable 的 {path} 上限不比 upstream 宽，那就白改了"
        )
        assert limit > 94, f"stable 的 {path} 上限 {limit} 覆盖不了 fla 默认初始化的跨度 ~94"
    # 反向的闸必须**不宽于**前向 —— 它守的是精度，而精度先于有限性失效
    assert MAX_GATE_SPAN["stable"]["backward"] <= MAX_GATE_SPAN["stable"]["forward"], (
        "反向上限比前向还宽，说明两条链的约束关系记错了"
    )


@pytest.mark.parametrize("unit", ["kda_fwd_stable", "kda_bwd_stable"])
def test_stable_units_declare_the_same_limit_as_the_code(unit):
    """两个单元的 contract 与 ``MAX_GATE_SPAN["stable"]`` 必须是同一个数。"""
    from ascend_fla.ops.kda.chunk import MAX_GATE_SPAN

    path = "forward" if unit.endswith("fwd_stable") else "backward"
    span = _unit_contract(unit)["domain"]["gate_span"]
    assert span["recommended_limit"] == MAX_GATE_SPAN["stable"][path], (
        f"{unit}/contract.json 声明 {span['recommended_limit']}，"
        f"chunk.py 的 {path} 上限是 {MAX_GATE_SPAN['stable'][path]}"
    )


def test_bwd_stable_overrides_exactly_the_kernels_it_declares():
    """``_STABLE_BWD_KERNELS`` 与 contract 的 ``own_kernels`` 必须一字不差。

    漏一个的后果很隐蔽：九个 kernel 里只换了一个，两个文件的锚点就不一致，
    ``finalize_pre`` 产出的因子和 ``finalize_post`` 补的因子配不上 —— 结果是**静默错**
    （不是 NaN），只能靠精度回归发现。
    """
    from ascend_fla.ops.kda.chunk_bwd import _KERNEL_MODULES, _STABLE_BWD_KERNELS

    assert set(_STABLE_BWD_KERNELS) <= set(_KERNEL_MODULES), (
        f"覆盖了不存在的 stage：{set(_STABLE_BWD_KERNELS) - set(_KERNEL_MODULES)}"
    )
    declared = set(_unit_contract("kda_bwd_stable")["dependencies"]["own_kernels"])
    assert set(_STABLE_BWD_KERNELS.values()) == declared, (
        f"chunk_bwd.py 覆盖 {sorted(_STABLE_BWD_KERNELS.values())}，"
        f"contract.json 声明 {sorted(declared)}"
    )


@pytest.mark.parametrize("unit,stems", [
    ("kda_fwd_stable", ("gate", "intra", "wy")),
    ("kda_bwd_stable", ("finalize_pre", "finalize_post")),
])
def test_stable_units_define_the_functions_they_claim(unit, stems):
    """单元里 contract 声明的 kernel 函数必须真的定义在对应文件里。

    纯文本核对，不 import（import 要 ascriptor DSL，而这个测试要在无卡机器上能跑）。
    """
    root = REPO / "kernels/projects/a5" / unit / "kernels"
    src = "\n".join((root / f"{s}.py").read_text(encoding="utf-8") for s in stems)
    for fn in _unit_contract(unit)["dependencies"]["own_kernels"]:
        assert f"def {fn}(" in src, f"{unit} 的 contract 声明了 {fn}，但 kernels/ 下没有它"


def test_bwd_stable_keeps_the_anchor_consistent_across_both_files():
    """两个文件必须用**同一个锚点** —— 这是改法能成立的前提。

    ``finalize_pre`` 产出 ``q_scaled``/``k_scaled``/``kg``，``finalize_post`` 补上配对因子；
    锚点不一致的话乘积不再是 ``exp(g_i − g_j)``，而且不会报错。
    """
    root = REPO / "kernels/projects/a5/kda_bwd_stable/kernels"
    for stem in ("finalize_pre", "finalize_post"):
        src = (root / f"{stem}.py").read_text(encoding="utf-8")
        assert "gmid <<= glast * MID" in src, f"{stem} 没有按中点算锚点"
        assert "tmp <<= g - gmid" in src, f"{stem} 的 rscale 没用中点锚"
        assert "tmp <<= gmid - g" in src, f"{stem} 的 cscale 没用中点锚"
        assert "tmp <<= g - glast" not in src, f"{stem} 还留着上游的端点锚"
    # MID 必须是 0.5 —— 取别的值虽然数学上仍同义，但两个因子不再对称，上限就不是翻倍
    for stem in ("finalize_pre", "finalize_post"):
        assert "MID = 0.5" in (root / f"{stem}.py").read_text(encoding="utf-8")


def test_chunk_kda_skips_caches_when_no_grad_is_needed():
    """``chunk_kda`` 在不需要梯度时必须走纯前向那条路 —— 纯 host 侧，不需要 NPU。

    两件事：① 省掉九个检查点（no_grad 下它们是纯浪费）；② 用前向那条更宽的闸
    （``stable`` 下 155 而不是 100）—— 推理不受反向的精度约束，被反向的闸挡住是错的。

    用 mock 盯住"调了哪一条"，而不是比结果 —— 结果本来就该相同（共用同一次 kernel 调用），
    所以比结果验不出这件事。
    """
    from unittest import mock

    import torch

    from ascend_fla.ops.kda import autograd as ag

    x = {n: torch.zeros(1) for n in "qkvgb"}
    sentinel = ("o", "state")
    with mock.patch.object(ag, "chunk_kda_fwd", return_value=sentinel) as fwd, \
            mock.patch.object(ag._ChunkKDA, "apply") as apply_:
        got = ag.chunk_kda(x["q"], x["k"], x["v"], x["g"], x["b"])
        assert got is sentinel, "不需要梯度时没有走 chunk_kda_fwd"
        assert fwd.call_count == 1 and apply_.call_count == 0

    # 有叶子要梯度时必须走 autograd.Function
    leaf = torch.zeros(1, requires_grad=True)
    with mock.patch.object(ag, "chunk_kda_fwd") as fwd, \
            mock.patch.object(ag._ChunkKDA, "apply", return_value=sentinel) as apply_:
        ag.chunk_kda(leaf, x["k"], x["v"], x["g"], x["b"])
        assert apply_.call_count == 1 and fwd.call_count == 0, "要梯度时却走了纯前向"

    # no_grad 里即使输入 requires_grad 也走纯前向
    with torch.no_grad():
        with mock.patch.object(ag, "chunk_kda_fwd", return_value=sentinel) as fwd, \
                mock.patch.object(ag._ChunkKDA, "apply") as apply_:
            ag.chunk_kda(leaf, x["k"], x["v"], x["g"], x["b"])
            assert fwd.call_count == 1 and apply_.call_count == 0, "no_grad 下仍走了 autograd"


# --------------------------------------------------------------- C=1 多头的静默错误
def test_single_chunk_multihead_gate_matches_measured_failure_pattern():
    """C=1 且 ``B*HV > block_dim`` 必须报错 —— 上游 kernel 在这里静默写错 ``o``。

    闸的边界直接照实测的"正确的头 = 每个 cube 核的最后一个头"写：安全条件是
    ``B*HV <= block_dim``（``pair_begin/pair_end`` 按 ``GetCubeIdx()/GetCubeNum()`` 切
    ``B*HV``，而 ``GetCubeNum() == block_dim``）。这里把实测过的组合逐个钉住 ——
    **这类缺陷没有 NaN、范数还正常**，一旦闸被改松就再也没有别的东西会报警。
    """
    from ascend_fla.ops.kda.chunk import _check_single_chunk_heads

    # 实测全对的组合（每核 ≤1 个头）：不该报错
    for b, hv, bd in ((1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 2, 4), (1, 1, 4)):
        _check_single_chunk_heads(b, hv, c=1, block_dim=bd)
    # 实测出错的组合（每核 ≥2 个头）：必须报错
    for b, hv, bd in ((1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1),
                      (1, 4, 2), (1, 8, 2), (1, 8, 4), (1, 16, 4), (1, 32, 4)):
        with pytest.raises(ValueError, match="静默错误"):
            _check_single_chunk_heads(b, hv, c=1, block_dim=bd)
    # C≥2 实测全对，任何 HV 都不该被拦
    for c in (2, 3, 16, 64):
        for hv in (1, 2, 8, 32):
            _check_single_chunk_heads(1, hv, c=c, block_dim=1)


def test_single_chunk_multihead_gate_is_wired_into_the_forward_check():
    """闸必须挂在 ``_check`` 上 —— 两个前向入口共用它，漏挂一个就等于没挂。"""
    import inspect

    from ascend_fla.ops.kda import chunk as m

    src = inspect.getsource(m._check)
    assert "_check_single_chunk_heads" in src, "_check 里没有调用 C=1 多头闸"
    # 两个前向入口都必须过 _check
    for fn in (m.chunk_kda_fwd, m.chunk_kda_fwd_with_caches):
        assert "_check(" in inspect.getsource(fn), f"{fn.__name__} 没有过 _check"
    # 反向**不该**装这个闸：实测 C=1/HV=8 的六项梯度全在预算内（反向不消费 o）
    from ascend_fla.ops.kda import chunk_bwd as mb
    assert "_check_single_chunk_heads" not in inspect.getsource(mb), (
        "反向装了这个闸，但实测反向不受影响 —— 多余的闸会把能用的路径拒掉"
    )


def test_c1_multihead_gap_is_recorded_as_p0():
    """静默错误必须在缺口表里，而且不能降级处理 —— 它比 NaN 糟。"""
    import json
    import pathlib

    gaps = json.loads((pathlib.Path(__file__).resolve().parent.parent
                       / "docs/matrix/gaps.json").read_text(encoding="utf-8"))
    g = {x["id"]: x for x in gaps["gaps"]}.get("c1-multihead-o-corrupt")
    assert g is not None, "gaps.json 里没有 c1-multihead-o-corrupt"
    assert g["severity"] == "P0", f"静默错误不该是 {g['severity']}"
    assert "kda" in g["applies_to"]
