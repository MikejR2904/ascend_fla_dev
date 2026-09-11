"""KDA 层的端到端梯度对齐（需要 A5 NPU + CANN）。

第二期的验收项。检查的是**整层**：投影、短卷积、门控变换（``-exp(A_log)*softplus``）、
q/k 的 L2 归一化、beta 的 sigmoid、KDA 算子、``FusedRMSNormGated``、输出投影，以及
这一整条链的反向。

参考怎么造：用**同一份权重**在 CPU 上跑同一个层，只把 KDA 算子换成 fp32 的 CPU 参考
（``reference.kda.kda_recurrent_ref``，逐 token 递推）。于是两边的差异只来自算子实现
与 bf16，层本身的数学是同一份代码 —— 层里写错的公式不会被「参考也写错」抵消掉，因为
参考用的就是它。换言之这个测试守的是**算子在层里接对了**，而算子本身的正确性由
``test_kda_bwd_npu.py`` 对单元 oracle 守。

用递推版而不是向量化分块版，是因为真实层的门控跨度超出了后者的 fp32 安全范围 ——
见 ``test_layer_gate_span_exceeds_declared_domain``。

**梯度对齐测试里把 ``exp(A_log)`` 压到了 1**，为的是把"接线对不对"和"深衰减下 bf16 本来
就糙"分开测，不是因为默认初始化跑不了 —— 默认初始化（跨度 ~94）由
``test_default_init_forward_matches_recurrent_oracle`` 与
``test_default_init_backward_matches_cpu_reference`` 专门盯。

预算：参数梯度比单个算子输出更敏感（经过了投影的放大）。算子侧 ``dg`` 的契约预算是
0.25，而 ``A_log`` / ``dt_bias`` 的梯度直接由 ``dg`` 来，所以它们用同一档；其余参数
用 0.1。这些是按契约反推的，不是调到能过为止 —— 超了要去查，不要放宽。

    pytest tests/test_kda_layer_npu.py -v
"""
from __future__ import annotations

import pathlib
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.layers.kda import KimiDeltaAttention  # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref  # noqa: E402

# dg 的契约预算是 0.25，A_log/dt_bias 的梯度直接由它来
BUDGET_DEFAULT = 0.1
BUDGET_GATE = 0.25
GATE_PARAMS = ("A_log", "dt_bias", "f_proj")


def _budget(name: str) -> float:
    return BUDGET_GATE if any(p in name for p in GATE_PARAMS) else BUDGET_DEFAULT


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _cpu_reference_attn(q, k, v, g, beta, *args, **kwargs):
    """替换 ``chunk_kda`` 的 CPU fp32 参考。签名与它兼容。

    用**逐 token 递推**版而不是向量化分块版。原因：随机初始化的 KDA 层产生的 chunk 内
    门控跨度约 94（见 ``test_layer_gate_span_exceeds_declared_domain``），而向量化版要算
    ``exp(g_max - g)``，跨度超过 80 就在 fp32 上溢。递推版按 per-token 增量做
    ``exp(g_i)``（量级 ~1.5），任何跨度都安全。
    """
    o, state = kda_recurrent_ref(
        q.float(), k.float(), v.float(), g, beta,
        initial_state=kwargs.get("initial_state"),
        output_final_state=kwargs.get("output_final_state", False),
    )
    return o.to(q.dtype), state


@pytest.mark.parametrize("b,t,num_heads,num_v_heads,hidden", [
    pytest.param(1, 64, 1, 1, 256, id="smoke"),
    pytest.param(1, 128, 2, 2, 512, id="multi_chunk"),
    pytest.param(1, 128, 1, 2, 256, id="gva"),
])
def test_layer_grads_match_cpu_reference(b, t, num_heads, num_v_heads, hidden):
    torch.manual_seed(2026)
    kw = dict(hidden_size=hidden, head_dim=128, num_heads=num_heads,
              num_v_heads=num_v_heads, conv_size=4, norm_eps=1e-5)
    layer_cpu = KimiDeltaAttention(**kw).float()
    # ⚠️ 把 exp(A_log) 从 fla 初始化的 ~15 压到 1，让 chunk 内门控跨度落在 ~6 而不是 ~94。
    # **不是为了让测试通过** —— stable 实现在 94 下也能跑（见
    # test_default_init_backward_matches_cpu_reference）。这里压低是为了把两件事分开：
    # 本测试验"算子在层里接对了"（17 个参数的梯度逐个对账），深衰减下 bf16 对 log2 累积
    # 门控的舍入会让 dk/dg 的误差涨一个数量级（契约里 dk 0.15 / dg 0.25 就是为它留的），
    # 混在一起测会让"接线错了"和"深衰减本来就糙"分不开。
    with torch.no_grad():
        layer_cpu.A_log.zero_()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(b, t, hidden) * 0.5
    # 固定的上游梯度，避免 loss=o.sum() 把符号抵消掉
    seed_grad = torch.randn(b, t, hidden, generator=torch.Generator().manual_seed(7))

    x_cpu = x.clone().requires_grad_(True)
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x_cpu)
    (o_cpu.float() * seed_grad).sum().backward()

    x_npu = x.clone().to("npu").requires_grad_(True)
    o_npu, _ = layer_npu(x_npu)
    (o_npu.float() * seed_grad.to("npu")).sum().backward()

    fwd_err = _rel_l2(o_npu.detach().cpu(), o_cpu.detach())
    errors = {"[output]": fwd_err, "[dx]": _rel_l2(x_npu.grad.cpu(), x_cpu.grad)}
    for (name, p_cpu), (_, p_npu) in zip(layer_cpu.named_parameters(),
                                         layer_npu.named_parameters()):
        assert p_cpu.grad is not None, f"CPU 侧 {name} 没有梯度，参考路径没走全"
        assert p_npu.grad is not None, f"NPU 侧 {name} 没有梯度，反向链断了"
        errors[name] = _rel_l2(p_npu.grad.cpu(), p_cpu.grad)

    bad = {n: (e, _budget(n)) for n, e in errors.items() if not (e < _budget(n))}
    report = "\n  ".join(f"{n:<24} {e:.3e} / {_budget(n)}" for n, e in errors.items())
    assert not bad, f"超出预算：{bad}\n  全部：\n  {report}"
    print(f"\nB{b} T{t} H{num_heads} HV{num_v_heads} hidden{hidden}\n  {report}")


def test_layer_rejects_unsupported_config():
    """层级门控：不支持的上游开关必须报错，不静默忽略（AGENTS.md §7）。"""
    base = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1)

    with pytest.raises(ValueError, match="head_k_dim=head_v_dim=128"):
        KimiDeltaAttention(**{**base, "head_dim": 64})
    with pytest.raises(ValueError, match="head_k_dim=head_v_dim=128"):
        KimiDeltaAttention(**{**base, "expand_v": 2.0})
    with pytest.raises(ValueError, match="整数倍"):
        KimiDeltaAttention(**{**base, "num_heads": 2, "num_v_heads": 3})
    with pytest.raises(ValueError, match="allow_neg_eigval"):
        KimiDeltaAttention(**base, allow_neg_eigval=True)
    with pytest.raises(ValueError, match="safe_gate"):
        KimiDeltaAttention(**base, safe_gate=True)
    with pytest.raises(ValueError, match="lower_bound"):
        KimiDeltaAttention(**base, lower_bound=0.5)
    with pytest.raises(ValueError, match="mode"):
        KimiDeltaAttention(**base, mode="fused_recurrent")

    layer = KimiDeltaAttention(**base).to("npu")
    with pytest.raises(ValueError, match="64 的整数倍"):
        layer(torch.randn(1, 100, 256, device="npu"))
    with pytest.raises(ValueError, match="cu_seqlens|varlen"):
        layer(torch.randn(1, 64, 256, device="npu"), cu_seqlens=torch.tensor([0, 64]))


def test_default_init_needs_stable_impl():
    """默认初始化的层只能用 ``impl="stable"`` —— 钉成一个会说话的测试。

    按 fla 自己的初始化，chunk 内门控跨度约 94。实测（benchmarks 侧扫描）：

    * ``upstream``（ascriptor 原版）：跨度 ≤66.84 正常（relL2 2.9e-03），≥89.12 全 NaN。
      根因是 gate 只写 ``eg = exp(gc)``，它在 ``-ln(FLT_MIN_NORMAL) ≈ 87.3`` 下溢到 0，
      下游的 ``k/eg`` 与 ``eg_last/eg`` 变成 ``0×inf`` 与 ``0/0``。
    * ``stable``（本仓 ``kernels/projects/a5/kda_fwd_stable``）：跨度到 155.97 仍全部有限，
      relL2 稳定 2.85e-03~3.19e-03，**重叠域内与 upstream 四位有效数字相同**。

    所以：默认层在 ``upstream`` 下应当被门控**挡住并报错**（不是返回 NaN），在 ``stable``
    下应当正常工作。见 ``docs/matrix/gaps.json`` 的 ``gate-range-beyond-declared``。
    """
    torch.manual_seed(2026)
    x = torch.randn(1, 64, 256, device="npu") * 0.5
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1)

    up = KimiDeltaAttention(**kw, impl="upstream").float().to("npu")
    with pytest.raises(ValueError, match="门控跨度"):
        up(x)

    # 默认就是 stable —— 同样的跨度应当跑得通
    st = KimiDeltaAttention(**kw).float().to("npu")
    assert st.impl == "stable", "默认实现应当是 stable"
    o, _ = st(x)
    assert o.float().isfinite().all(), "stable 实现下默认初始化的层仍不是有限值"


def test_default_init_forward_matches_recurrent_oracle():
    """默认初始化（跨度 ~94）下，整层前向对逐 token 递推 oracle 的精度。

    这是 stable 实现的意义所在：上一个测试只证明"不是 NaN"，这里要求它**准**。
    预算取 0.05，与算子契约的 max_relative_l2 同档。
    """
    torch.manual_seed(2026)
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=2, conv_size=4)
    layer_cpu = KimiDeltaAttention(**kw).float()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(1, 128, 256) * 0.5
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x)
    o_npu, _ = layer_npu(x.to("npu"))
    err = _rel_l2(o_npu.detach().cpu(), o_cpu.detach())
    print(f"\n默认初始化（跨度 ~94）下整层前向相对 L2 = {err:.3e}")
    assert err < 0.05, f"相对 L2 {err:.3e} 超出 0.05"


def test_default_init_backward_matches_cpu_reference():
    """默认初始化（跨度 ~94）下**整层反向**对 CPU 参考的精度。

    为什么单独立一个：前向修好不等于能训练。ascriptor 的反向在 ``finalize_pre`` /
    ``finalize_post`` 里把成对衰减分解成 ``exp(g − g_last) · exp(g_last − g)``，前者在
    ``ln(MAX) ≈ 88.72`` 处**上溢**到 inf（输出是 bf16 GM），配对的因子同时下溢到 0，
    下游矩阵乘得 ``inf × 0 = NaN``。方向与前向那处下溢相反，是独立的一处 —— 见
    ``docs/matrix/gaps.json`` 的 ``bwd-gate-range-overflow``。

    所以这个测试盯两件事，顺序不能换：

    1. **有限性**：跨度 ~94 下六类梯度全部有限。这是 ``upstream`` 会直接挂掉的地方。
    2. **精度**：逐参数对 CPU 参考在预算内。预算用深衰减档（``dg`` 的契约预算 0.25），
       因为 bf16 对 log2 累积门控的舍入在深衰减下确实会放大 —— 那是
       ``kda-fwd-bwd-dtype-mismatch`` 已量化过的成因，不是接线错。

    ``A_log`` **不压低** —— 这正是与 ``test_layer_grads_match_cpu_reference`` 的分工。
    """
    torch.manual_seed(2026)
    kw = dict(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=2, conv_size=4)
    layer_cpu = KimiDeltaAttention(**kw).float()
    layer_npu = KimiDeltaAttention(**kw).float()
    layer_npu.load_state_dict(layer_cpu.state_dict())
    layer_npu = layer_npu.to("npu")

    x = torch.randn(1, 128, 256) * 0.5
    span = _gate_span_of(layer_cpu, x)
    assert span > 80, f"跨度只有 {span:.1f}，没到要测的深衰减区间，本测试前提失效"

    seed_grad = torch.randn(1, 128, 256, generator=torch.Generator().manual_seed(7))

    x_cpu = x.clone().requires_grad_(True)
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _cpu_reference_attn):
        o_cpu, _ = layer_cpu(x_cpu)
    (o_cpu.float() * seed_grad).sum().backward()

    x_npu = x.clone().to("npu").requires_grad_(True)
    o_npu, _ = layer_npu(x_npu)
    (o_npu.float() * seed_grad.to("npu")).sum().backward()

    # ① 有限性先判 —— NaN 下比精度没有意义，而且 rel_l2 会给出 nan 让断言信息难读
    nonfinite = {"[output]": o_npu, "[dx]": x_npu.grad}
    nonfinite |= {n: p.grad for n, p in layer_npu.named_parameters()}
    broken = [n for n, t in nonfinite.items()
              if t is None or not t.detach().cpu().float().isfinite().all()]
    assert not broken, (
        f"跨度 {span:.1f} 下这些量不是有限值：{broken}。"
        f"若 impl 是 upstream 这是预期的（bwd-gate-range-overflow）；"
        f"若 impl 是 stable 说明 kda_bwd_stable 没生效或没修对"
    )

    # ② 再判精度
    errors = {"[output]": _rel_l2(o_npu.detach().cpu(), o_cpu.detach()),
              "[dx]": _rel_l2(x_npu.grad.cpu(), x_cpu.grad)}
    for (name, p_cpu), (_, p_npu) in zip(layer_cpu.named_parameters(),
                                         layer_npu.named_parameters()):
        errors[name] = _rel_l2(p_npu.grad.cpu(), p_cpu.grad)
    bad = {n: (e, BUDGET_GATE) for n, e in errors.items() if not (e < BUDGET_GATE)}
    report = "\n  ".join(f"{n:<24} {e:.3e}" for n, e in errors.items())
    assert not bad, f"超出预算 {BUDGET_GATE}：{bad}\n  全部：\n  {report}"
    print(f"\n默认初始化（跨度 {span:.1f}）下整层反向，预算 {BUDGET_GATE}：\n  {report}")


def _gate_span_of(layer, x) -> float:
    """层在这份输入下产生的 chunk 内门控跨度。与算子的门控检查同一个定义。"""
    b, t = x.shape[0], x.shape[1]
    g = layer._gate(x, b, t)                      # _gate(hidden_states, b, t) -> [B,T,HV,KD]
    cum = g.view(b, t // 64, 64, layer.num_v_heads, layer.head_k_dim).cumsum(2)
    return (cum.amax(dim=2) - cum.amin(dim=2)).max().item()


def test_layer_gate_span_exceeds_declared_domain():
    """把"层产生的门控远超算子声明域"这件事钉成一个会说话的测试。

    ``kda_fwd`` contract 的 ``input_generation`` 声明 ``g_raw ∈ [-0.03, 0]``，即 64 个
    token 的 chunk 内跨度 ≤ 1.92。而 KDA 层按 **fla 自己的初始化**
    （``A_log = log(U(1,16))``、``dt_bias`` 为 Mamba 的 inv-softplus，``dt ∈ [0.001,0.1]``）
    产生的 per-token ``g`` 可达 ``-15 × 0.096 ≈ -1.47``，64 个 token 累计约 **-94** ——
    比声明域宽约 50 倍，且**几乎与输入尺度无关**（由两个参数的初始化决定）。

    所以这个测试不是在查 bug，而是固定住一个事实：算子被验证过的门控域不覆盖真实层的
    初始化。见 ``docs/matrix/gaps.json`` 的 ``gate-range-beyond-declared``。
    """
    torch.manual_seed(2026)
    layer = KimiDeltaAttention(hidden_size=256, head_dim=128, num_heads=1, num_v_heads=1).float()
    spans = []
    for scale in (0.5, 0.05, 0.01):
        g = layer._gate(torch.randn(1, 64, 256) * scale, 1, 64)
        cum = g.cumsum(1)
        spans.append((cum.amax(dim=1) - cum.amin(dim=1)).max().item())
    print(f"\n输入 scale 0.5/0.05/0.01 下的 chunk 内门控跨度："
          + " / ".join(f"{s:.1f}" for s in spans))
    assert min(spans) > 20, f"跨度 {spans} 远小于预期，说明初始化被改过，本测试的前提失效"
    assert max(spans) / min(spans) < 1.2, (
        f"跨度随输入尺度明显变化（{spans}），与「由初始化决定」的结论矛盾"
    )
