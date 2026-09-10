"""kda_fwd 经 runtime 桥的端到端精度回归（需要 A5 NPU + CANN）。

无 NPU 时整文件 skip。形状取自 docs/matrix/models.json 的 test_case_shapes。

    pytest tests/test_kda_fwd_npu.py -v
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

from ascend_fla.reference.kda import kda_chunk_ref, kda_recurrent_ref  # noqa: E402

# kda_fwd contract 的比较预算：rtol=atol=0.02，max_relative_l2=0.05
BUDGET_REL_L2 = 0.05


def _inputs(b, t, h, hv, seed=2026):
    gen = torch.Generator().manual_seed(seed)
    k_dim = v_dim = 128
    return dict(
        q=(torch.randn(b, t, h, k_dim, generator=gen) * 0.04).bfloat16(),
        k=(torch.randn(b, t, h, k_dim, generator=gen) * 0.04).bfloat16(),
        v=(torch.randn(b, t, hv, v_dim, generator=gen) * 0.04).bfloat16(),
        g=torch.empty(b, t, hv, k_dim).uniform_(-0.03, 0.0, generator=gen),
        beta=torch.empty(b, t, hv).uniform_(0.05, 0.5, generator=gen),
        initial_state=torch.randn(b, hv, k_dim, v_dim, generator=gen) * 0.01,
    )


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize(("b", "t", "h", "hv"), [
    (1, 64, 1, 1),    # 单 chunk
    (1, 128, 1, 1),   # 多 chunk，覆盖 state 跨 chunk 传递
    (1, 128, 1, 2),   # GVA 分组 HV % H == 0
])
def test_chunk_kda_fwd_matches_cpu_reference(b, t, h, hv):
    from ascend_fla.ops.kda import chunk_kda_fwd

    x = _inputs(b, t, h, hv)
    o_ref, s_ref = kda_chunk_ref(**x, output_final_state=True)
    o_rec, _ = kda_recurrent_ref(**x, output_final_state=True)
    # 先确认两个 CPU oracle 自己是一致的，再拿它们判 NPU
    assert _rel_l2(o_ref, o_rec) < 1e-3, "两个 CPU oracle 不一致，先查参考实现"

    o_npu, s_npu = chunk_kda_fwd(
        *(x[n].to("npu") for n in ("q", "k", "v", "g", "beta")),
        initial_state=x["initial_state"].to("npu"), output_final_state=True,
    )
    torch.npu.synchronize()

    r_o = _rel_l2(o_npu.cpu(), o_ref)
    r_s = _rel_l2(s_npu.cpu(), s_ref)
    assert r_o < BUDGET_REL_L2, f"o 的 rel_l2={r_o:.3e} 超出预算 {BUDGET_REL_L2}"
    assert r_s < BUDGET_REL_L2, f"final_state 的 rel_l2={r_s:.3e} 超出预算 {BUDGET_REL_L2}"


def test_gate_rejects_bad_shapes():
    """门控必须报错而不是静默降级（AGENTS.md §7）。"""
    from ascend_fla.ops.kda import chunk_kda_fwd

    x = _inputs(1, 64, 1, 1)
    npu = {n: x[n].to("npu") for n in ("q", "k", "v", "g", "beta")}

    with pytest.raises(ValueError, match="64 的整数倍"):   # T 不是 64 的倍数
        bad = _inputs(1, 128, 1, 1)
        chunk_kda_fwd(*(bad[n][:, :100].to("npu") for n in ("q", "k", "v", "g", "beta")))

    with pytest.raises(ValueError, match="dtype"):          # q 用了 fp32
        chunk_kda_fwd(npu["q"].float(), npu["k"], npu["v"], npu["g"], npu["beta"])

    with pytest.raises(ValueError, match="K=V=128"):        # 头维不是 128
        q = torch.randn(1, 64, 1, 64, dtype=torch.bfloat16, device="npu")
        chunk_kda_fwd(q, q, q, npu["g"], npu["beta"])

    # block_dim 超出契约声明的 [1,2,3,4]。这一项尤其重要：超过物理核数不会报错，
    # 会在硬件 barrier 上死锁 —— 必须在 host 侧挡住。
    for bd in (0, 5, 8, 28, 32):
        with pytest.raises(ValueError, match="block_dim"):
            chunk_kda_fwd(npu["q"], npu["k"], npu["v"], npu["g"], npu["beta"], block_dim=bd)

