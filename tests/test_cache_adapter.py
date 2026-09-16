"""``ascend_fla.compat.cache`` —— fla/HF KDA cache ⇄ 本仓 cache 的适配器。

判定纪律（``AGENTS.md`` §6）：

* 全部在 **fp32** 下判。
* **报数字不报 OK**：每个断言前先 print ``max_abs_diff``。
* **判别性**：state 一律用非对称的（``S != S.transpose(-1,-2)``），并且每个"方向正确"
  的检查都配一个"方向搞错"的对照 —— 否则 K==V=128 时转置错了什么都测不出来，
  正是 ``state-layout-k-first`` 说的那个静默失败面。
* conv_state **逐 token 报数，不取平均**：漏传 conv_state 只坏前 ``W-1`` 个 token，
  平均一取就掩盖了。
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ascend_fla.compat.cache import to_fla, to_ours  # noqa: E402
from ascend_fla.modules.convolution import ShortConvolution  # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref  # noqa: E402

# Kimi-Linear 的真实头维（docs/matrix/models.json）。K == V 正是转置查不出来的那档，
# 所以主链路的精度检查就在这个形状上做。
K_DIM = V_DIM = 128
CONV_SIZE = 4


# --------------------------------------------------------------------------------------
# fla oracle：macOS 上没有 triton，``import fla.ops...`` 会炸，所以按文件路径加载 naive.py
# （``import fla`` 本身不触发 triton）。见 docs/pm/START.md §6。
# --------------------------------------------------------------------------------------
def _load_fla_kda_naive():
    fla = pytest.importorskip("fla", reason="需要 flash-linear-attention 作语义权威 oracle")
    path = pathlib.Path(fla.__file__).parent / "ops/kda/naive.py"
    if not path.exists():  # pragma: no cover - 上游挪了文件就直接说清楚
        pytest.skip(f"fla 里找不到 {path.name}（实际路径 {path}）")
    spec = importlib.util.spec_from_file_location("fla_kda_naive_for_cache_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fla_kda_layer_source() -> str:
    fla = pytest.importorskip("fla")
    path = pathlib.Path(fla.__file__).parent / "layers/kda.py"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"fla 里找不到 {path}")
    return path.read_text()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def _asymmetric_state(b: int, hv: int, d0: int, d1: int, seed: int = 0) -> torch.Tensor:
    """造一个**明确非对称**的 state，形状 ``[b, hv, d0, d1]``、fp32。

    纯随机已经几乎必然非对称，但"几乎必然"不是判据 —— 这里额外加一个只作用在下三角的
    偏置，并由调用方断言 ``S != S^T``。
    """
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(b, hv, d0, d1, generator=g, dtype=torch.float32)
    if d0 == d1:
        state = state + torch.tril(torch.full((d0, d1), 3.0), diagonal=-1)
    return state.contiguous()


def _fla_layer_state(recurrent_state=None, conv_state=None) -> dict:
    """fla 的 layer state 字典（``FLALayer.update`` 造出来的那个）。"""
    return {
        "recurrent_state": recurrent_state,
        "attn_state": None,
        "conv_state": conv_state,
        "ffn_state": None,
    }


# ======================================================================================
# 0. 布局前提：fla 的 KDA layer 确实用 state_v_first=True
# ======================================================================================
def test_fla_kda_layer_still_uses_state_v_first():
    """适配器的全部依据就是这一行。上游改了要立刻红，而不是悄悄转错方向。"""
    source = _fla_kda_layer_source()
    count = source.count("state_v_first=True")
    print(f"fla/layers/kda.py 中 state_v_first=True 出现 {count} 次（chunk + fused_recurrent）")
    assert count >= 2, (
        "fla 的 KimiDeltaAttention 不再对两条路径都传 state_v_first=True，"
        "compat/cache.py 的默认布局假设要重新核实"
    )


# ======================================================================================
# 1. 判别性：非对称 state + 转错方向必须变
# ======================================================================================
def test_state_fixture_is_asymmetric():
    state = _asymmetric_state(2, 3, K_DIM, V_DIM, seed=1)
    diff = _max_abs_diff(state, state.transpose(-1, -2))
    print(f"S vs S^T max_abs_diff = {diff:.6e}（K=V={K_DIM}，形状相同、内容不同）")
    assert diff > 1.0, "测试用的 state 必须非对称，否则整个测试没有判别力"
    assert tuple(state.shape) == tuple(state.transpose(-1, -2).shape), (
        "K==V 时转置前后形状一模一样 —— 这正是形状检查抓不住方向的原因"
    )


def test_to_ours_actually_transposes_when_state_v_first():
    fla_state = _asymmetric_state(2, 3, V_DIM, K_DIM, seed=2)
    ours = to_ours(_fla_layer_state(fla_state), head_k_dim=K_DIM, head_v_dim=V_DIM)
    same = _max_abs_diff(ours["recurrent_state"], fla_state)
    transposed = _max_abs_diff(ours["recurrent_state"], fla_state.transpose(-1, -2))
    print(f"to_ours(S_vfirst) vs S_vfirst      max_abs_diff = {same:.6e}（应当很大）")
    print(f"to_ours(S_vfirst) vs S_vfirst^T    max_abs_diff = {transposed:.6e}（应当为 0）")
    assert transposed == 0.0
    assert same > 1.0


def test_to_ours_is_identity_when_state_already_k_first():
    k_first = _asymmetric_state(2, 3, K_DIM, V_DIM, seed=3)
    ours = to_ours(
        _fla_layer_state(k_first), state_v_first=False, head_k_dim=K_DIM, head_v_dim=V_DIM
    )
    print(f"state_v_first=False 时 max_abs_diff = {_max_abs_diff(ours['recurrent_state'], k_first):.6e}")
    assert _max_abs_diff(ours["recurrent_state"], k_first) == 0.0


# ======================================================================================
# 2. 往返逐位还原 + 不共享存储
# ======================================================================================
@pytest.mark.parametrize("state_v_first", [True, False])
def test_roundtrip_is_bitwise_identical(state_v_first):
    d0, d1 = (V_DIM, K_DIM) if state_v_first else (K_DIM, V_DIM)
    fla_state = _asymmetric_state(2, 3, d0, d1, seed=4)
    conv = tuple(torch.randn(2, 16, CONV_SIZE) for _ in range(3))
    source = _fla_layer_state(fla_state, conv)

    ours = to_ours(source, state_v_first=state_v_first, head_k_dim=K_DIM, head_v_dim=V_DIM)
    back = to_fla(ours, state_v_first=state_v_first, head_k_dim=K_DIM, head_v_dim=V_DIM)

    diff = _max_abs_diff(back["recurrent_state"], fla_state)
    print(f"state_v_first={state_v_first}: to_fla(to_ours(S)) vs S max_abs_diff = {diff:.6e}")
    assert torch.equal(back["recurrent_state"], fla_state), "往返必须逐位还原"
    for tag, before, after in zip("qkv", conv, back["conv_state"]):
        conv_diff = _max_abs_diff(after, before)
        print(f"  conv_state[{tag}] 往返 max_abs_diff = {conv_diff:.6e}")
        assert torch.equal(after, before)

    # 别名语义：适配器永远返回新张量，不会出现"某个布局下是视图、另一个下是拷贝"。
    assert ours["recurrent_state"].data_ptr() != fla_state.data_ptr()
    assert back["recurrent_state"].data_ptr() != ours["recurrent_state"].data_ptr()
    for after, before in zip(back["conv_state"], conv):
        assert after.data_ptr() != before.data_ptr()


def test_roundtrip_with_empty_cache():
    ours = to_ours(_fla_layer_state(None, None))
    assert ours == {"recurrent_state": None, "conv_state": None}
    assert to_fla(ours) == {"recurrent_state": None, "conv_state": None}


# ======================================================================================
# 3. 主链路：从 fla 的 state 出发，经适配器跑一步 decode，与 fla naive 逐元素比
# ======================================================================================
def _kda_inputs(b=2, t=1, h=2, hv=4, seed=7):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    q = rnd(b, t, h, K_DIM)
    k = torch.nn.functional.normalize(rnd(b, t, h, K_DIM), p=2, dim=-1)
    v = rnd(b, t, hv, V_DIM)
    # g 是 log 空间的衰减增量，恒为负；取 [-0.2, 0) 避免把 exp 推到量程边上。
    gate = -0.2 * torch.rand(b, t, hv, K_DIM, generator=g, dtype=torch.float32)
    beta = torch.sigmoid(rnd(b, t, hv))
    return q, k, v, gate, beta


def test_one_step_decode_from_fla_state_matches_fla_naive():
    naive = _load_fla_kda_naive()
    b, h, hv = 2, 2, 4

    # ① 先让 fla 自己跑一段 prefill，拿到它的 final_state（naive 用 K-first）。
    q0, k0, v0, g0, beta0 = _kda_inputs(b=b, t=8, h=h, hv=hv, seed=11)
    _, state_k_first = naive.naive_recurrent_kda(
        q0, k0, v0, g0, beta0, initial_state=None, output_final_state=True
    )
    assert tuple(state_k_first.shape) == (b, hv, K_DIM, V_DIM)
    assert state_k_first.dtype is torch.float32

    # ② fla 的 KDA layer 用 state_v_first=True，cache 里存的是它的转置。
    fla_cached = state_k_first.transpose(-1, -2).contiguous()
    sym = _max_abs_diff(state_k_first, fla_cached)
    print(f"prefill 后 S vs S^T max_abs_diff = {sym:.6e}（非对称才有判别力）")
    assert sym > 1e-3

    # ③ 过适配器，再走本仓的 torch 参考递推一步 decode。
    ours = to_ours(_fla_layer_state(fla_cached), head_k_dim=K_DIM, head_v_dim=V_DIM)
    q1, k1, v1, g1, beta1 = _kda_inputs(b=b, t=1, h=h, hv=hv, seed=12)
    o_ours, s_ours = kda_recurrent_ref(
        q1, k1, v1, g1, beta1,
        initial_state=ours["recurrent_state"], output_final_state=True,
    )

    # ④ 参考：fla naive 从 K-first 的 state 直接续一步。
    o_ref, s_ref = naive.naive_recurrent_kda(
        q1, k1, v1, g1, beta1, initial_state=state_k_first, output_final_state=True
    )

    o_diff = _max_abs_diff(o_ours, o_ref)
    s_diff = _max_abs_diff(s_ours, s_ref)
    print(f"[方向正确] o     max_abs_diff = {o_diff:.6e} (B={b},HV={hv},K=V={K_DIM},fp32)")
    print(f"[方向正确] state max_abs_diff = {s_diff:.6e}")
    assert o_diff <= 1e-6, f"o 的 max_abs_diff {o_diff:.6e} 超出 1e-6"
    assert s_diff <= 1e-6, f"final_state 的 max_abs_diff {s_diff:.6e} 超出 1e-6"

    # ⑤ 判别性对照：把适配器那一步跳掉（直接拿 V-first 的 state 当 K-first 用）。
    #    形状一样、不报错、输出有限 —— 但内容是错的。
    o_wrong, s_wrong = kda_recurrent_ref(
        q1, k1, v1, g1, beta1, initial_state=fla_cached, output_final_state=True
    )
    o_wrong_diff = _max_abs_diff(o_wrong, o_ref)
    s_wrong_diff = _max_abs_diff(s_wrong, s_ref)
    print(f"[漏了转置] o     max_abs_diff = {o_wrong_diff:.6e}（静默失败面，形状不报错）")
    print(f"[漏了转置] state max_abs_diff = {s_wrong_diff:.6e}")
    assert torch.isfinite(o_wrong).all(), "漏转置的输出仍是有限值 —— 所以只看 nan/inf 抓不到它"
    assert o_wrong_diff > 1e-2, "判别性不足：转错方向居然没让结果变，测试白写了"
    assert s_wrong_diff > 1e-2


def test_adapted_state_feeds_a_chunk_prefill_cross_check():
    """独立算术路径的交叉校验。

    上面那条对 fla ``naive_recurrent_kda`` 的 max_abs_diff 是 **0.0** —— 本仓的
    ``kda_recurrent_ref`` 与它逐句同构，所以那条只证明了"布局对"，证明不了"算得准"。
    这里换 fla 的 ``naive_chunk_kda``（分块，算术顺序完全不同）再比一次：从适配器出来的
    state 起算 64 个 token，本仓逐 token 递推 vs fla 分块，差的量级才是浮点该有的样子。
    """
    naive = _load_fla_kda_naive()
    b, h, hv, t = 2, 2, 4, 64

    q0, k0, v0, g0, beta0 = _kda_inputs(b=b, t=8, h=h, hv=hv, seed=31)
    _, state_k_first = naive.naive_recurrent_kda(
        q0, k0, v0, g0, beta0, initial_state=None, output_final_state=True
    )
    fla_cached = state_k_first.transpose(-1, -2).contiguous()
    ours = to_ours(_fla_layer_state(fla_cached), head_k_dim=K_DIM, head_v_dim=V_DIM)

    q1, k1, v1, g1, beta1 = _kda_inputs(b=b, t=t, h=h, hv=hv, seed=32)
    o_ours, s_ours = kda_recurrent_ref(
        q1, k1, v1, g1, beta1,
        initial_state=ours["recurrent_state"], output_final_state=True,
    )
    o_ref, s_ref = naive.naive_chunk_kda(
        q1, k1, v1, g1, beta1, initial_state=state_k_first, output_final_state=True
    )

    o_diff = _max_abs_diff(o_ours, o_ref)
    s_diff = _max_abs_diff(s_ours, s_ref)
    o_l2 = ((o_ours - o_ref).norm() / o_ref.norm()).item()
    s_l2 = ((s_ours - s_ref).norm() / s_ref.norm()).item()
    print(
        f"[recurrent vs fla chunk] T={t} B={b} HV={hv} K=V={K_DIM} fp32: "
        f"o max_abs_diff = {o_diff:.6e} rel-L2 = {o_l2:.6e}; "
        f"state max_abs_diff = {s_diff:.6e} rel-L2 = {s_l2:.6e}"
    )
    assert o_l2 <= 1e-5, f"o 的相对 L2 {o_l2:.6e} 超出 1e-5"
    assert s_l2 <= 1e-5, f"final_state 的相对 L2 {s_l2:.6e} 超出 1e-5"

    # 同样配一个方向搞错的对照：没有它，上面两个数说明不了适配器做对了事。
    o_wrong, _ = kda_recurrent_ref(
        q1, k1, v1, g1, beta1, initial_state=fla_cached, output_final_state=True
    )
    wrong_l2 = ((o_wrong - o_ref).norm() / o_ref.norm()).item()
    print(f"[漏了转置] o rel-L2 = {wrong_l2:.6e}")
    assert wrong_l2 > 1e-2


def test_multi_step_decode_reports_per_token_numbers():
    """逐 token 报数（不取平均），连着走 4 步，state 每步经适配器往返一次。"""
    naive = _load_fla_kda_naive()
    b, h, hv, steps = 2, 2, 4, 4

    q0, k0, v0, g0, beta0 = _kda_inputs(b=b, t=8, h=h, hv=hv, seed=21)
    _, state_ref = naive.naive_recurrent_kda(
        q0, k0, v0, g0, beta0, initial_state=None, output_final_state=True
    )
    layer_state = _fla_layer_state(state_ref.transpose(-1, -2).contiguous())

    worst = 0.0
    for step in range(steps):
        q1, k1, v1, g1, beta1 = _kda_inputs(b=b, t=1, h=h, hv=hv, seed=100 + step)
        ours = to_ours(layer_state, head_k_dim=K_DIM, head_v_dim=V_DIM)
        o_ours, s_ours = kda_recurrent_ref(
            q1, k1, v1, g1, beta1,
            initial_state=ours["recurrent_state"], output_final_state=True,
        )
        o_ref, state_ref = naive.naive_recurrent_kda(
            q1, k1, v1, g1, beta1, initial_state=state_ref, output_final_state=True
        )
        o_diff = _max_abs_diff(o_ours, o_ref)
        s_diff = _max_abs_diff(s_ours, state_ref)
        print(f"decode step {step}: o max_abs_diff = {o_diff:.6e}, state max_abs_diff = {s_diff:.6e}")
        worst = max(worst, o_diff, s_diff)
        # 写回 fla 侧再取出来，让每一步都真的过一次双向适配器。
        layer_state = _fla_layer_state(
            to_fla({"recurrent_state": s_ours, "conv_state": None})["recurrent_state"]
        )
    print(f"decode 4 步中最差 max_abs_diff = {worst:.6e}")
    assert worst <= 1e-6


# ======================================================================================
# 4. conv_state
# ======================================================================================
def _fla_step_oracle(cache: torch.Tensor, x_t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fla 在 ``short_conv.py`` 里写死的单步语义，直接照抄当独立 oracle：

    ``cache.copy_(cache.roll(shifts=-1, dims=-1)); cache[:, :, -1] = x``
    ``y = (cache * weight).sum(-1)``

    ——**最新的 token 在最后一个下标**。本仓 ``modules/convolution.py`` 的
    ``hist[..., -w:]`` 是同一个约定，这个函数就是用来把"同一个约定"钉死的。
    """
    rolled = cache.roll(shifts=-1, dims=-1).clone()
    rolled[:, :, -1] = x_t
    y = (rolled * weight).sum(-1)
    return torch.nn.functional.silu(y), rolled


def _make_conv(d: int = 16, seed: int = 31) -> ShortConvolution:
    torch.manual_seed(seed)
    conv = ShortConvolution(d, kernel_size=CONV_SIZE, bias=False, activation="silu")
    conv = conv.to(torch.float32)
    with torch.no_grad():
        conv.weight.copy_(torch.randn_like(conv.weight))
    return conv


@pytest.mark.parametrize("prefill_t", [2, 8])
def test_conv_state_step_matches_full_sequence_per_token(prefill_t):
    """``prefill_t=2`` 就是 ``T < conv_size``（补零）那一档，``8`` 是常规档。"""
    d, decode_steps = 16, 5
    conv = _make_conv(d)
    torch.manual_seed(41)
    x = torch.randn(2, prefill_t + decode_steps, d, dtype=torch.float32)

    full, _ = conv(x, cache=None, output_final_state=False)

    y_pre, cache = conv(x[:, :prefill_t], cache=None, output_final_state=True)
    assert tuple(cache.shape) == (2, d, CONV_SIZE), f"cache 形状 {tuple(cache.shape)}"
    if prefill_t < CONV_SIZE:
        pad = cache[:, :, : CONV_SIZE - prefill_t]
        print(f"T={prefill_t} < W={CONV_SIZE}：左侧补零块 max|x| = {pad.abs().max().item():.6e}")
        assert pad.abs().max().item() == 0.0, "T < conv_size 时左侧必须补零"
        assert _max_abs_diff(cache[:, :, CONV_SIZE - prefill_t:], x[:, :prefill_t].transpose(1, 2)) == 0.0

    pre_diff = _max_abs_diff(y_pre, full[:, :prefill_t])
    print(f"prefill T={prefill_t} 输出 max_abs_diff = {pre_diff:.6e}")
    assert pre_diff <= 1e-6

    # 每一步都把 cache 经 to_fla -> to_ours 送一圈，模拟与 HF/fla cache 的交接。
    weight = conv.weight.squeeze(1)
    for step in range(decode_steps):
        t = prefill_t + step
        fla_side = to_fla({"recurrent_state": None, "conv_state": (cache, cache, cache)})
        ours = to_ours(_fla_layer_state(None, fla_side["conv_state"]))
        cache_in = ours["conv_state"][0]

        y_step, cache = conv(x[:, t:t + 1], cache=cache_in, output_final_state=True)
        step_diff = _max_abs_diff(y_step[:, 0], full[:, t])
        y_oracle, cache_oracle = _fla_step_oracle(cache_in, x[:, t], weight)
        oracle_diff = _max_abs_diff(y_step[:, 0], y_oracle)
        cache_diff = _max_abs_diff(cache, cache_oracle)
        print(
            f"token {t}: vs 整段 max_abs_diff = {step_diff:.6e}, "
            f"vs fla 单步语义 max_abs_diff = {oracle_diff:.6e}, "
            f"cache max_abs_diff = {cache_diff:.6e}"
        )
        assert step_diff <= 1e-6
        assert oracle_diff <= 1e-6
        assert cache_diff == 0.0


def test_conv_state_time_axis_direction_is_discriminative():
    """把 conv_state 的时间轴翻过来（一个很自然的"约定搞反"），输出必须变。"""
    d = 16
    conv = _make_conv(d)
    torch.manual_seed(51)
    x = torch.randn(2, 8, d, dtype=torch.float32)
    full, _ = conv(x, cache=None, output_final_state=False)
    _, cache = conv(x[:, :7], cache=None, output_final_state=True)

    y_right, _ = conv(x[:, 7:8], cache=cache, output_final_state=True)
    y_flipped, _ = conv(x[:, 7:8], cache=cache.flip(-1).contiguous(), output_final_state=True)
    right = _max_abs_diff(y_right[:, 0], full[:, 7])
    flipped = _max_abs_diff(y_flipped[:, 0], full[:, 7])
    print(f"conv 方向正确 max_abs_diff = {right:.6e}；时间轴翻转 max_abs_diff = {flipped:.6e}")
    assert right <= 1e-6
    assert flipped > 1e-2, "判别性不足：conv_state 顺序反了却看不出来"


def test_conv_state_missing_one_stream_raises():
    c = torch.randn(2, 16, CONV_SIZE)
    with pytest.raises(ValueError, match="要么都给要么都不给"):
        to_ours(_fla_layer_state(None, (c, None, c)))


# ======================================================================================
# 5. 门控：对不上就报错，错误信息里带实际值（AGENTS.md §7）
# ======================================================================================
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_non_fp32_state_raises_with_actual_dtype(dtype):
    state = torch.zeros(2, 3, V_DIM, K_DIM, dtype=dtype)
    with pytest.raises(ValueError) as excinfo:
        to_ours(_fla_layer_state(state))
    message = str(excinfo.value)
    print(f"dtype={dtype} -> {message}")
    assert "fp32" in message and str(dtype) in message


def test_wrong_ndim_raises_with_actual_shape():
    with pytest.raises(ValueError) as excinfo:
        to_ours(_fla_layer_state(torch.zeros(3, V_DIM, K_DIM)))
    message = str(excinfo.value)
    print(message)
    assert "3 维" in message and f"({3}, {V_DIM}, {K_DIM})" in message


def test_declared_head_dims_catch_wrong_layout_when_k_ne_v():
    """K != V 时，声明错布局就会被形状检查抓住（K == V 时抓不住，只能靠约定）。"""
    k_dim, v_dim = 64, 128
    v_first = _asymmetric_state(2, 3, v_dim, k_dim, seed=61)
    ok = to_ours(_fla_layer_state(v_first), head_k_dim=k_dim, head_v_dim=v_dim)
    assert tuple(ok["recurrent_state"].shape) == (2, 3, k_dim, v_dim)

    with pytest.raises(ValueError) as excinfo:
        to_ours(_fla_layer_state(v_first), state_v_first=False, head_k_dim=k_dim, head_v_dim=v_dim)
    message = str(excinfo.value)
    print(f"K={k_dim} != V={v_dim} 且布局声明错 -> {message}")
    assert f"K={k_dim}" in message and str(v_dim) in message


def test_attn_state_is_rejected_not_dropped():
    state = _asymmetric_state(2, 3, V_DIM, K_DIM, seed=62)
    source = _fla_layer_state(state)
    source["attn_state"] = (torch.zeros(2, 4, 8), torch.zeros(2, 4, 8))
    with pytest.raises(ValueError, match="attn_state"):
        to_ours(source)


def test_conv_state_batch_mismatch_raises():
    state = _asymmetric_state(2, 3, V_DIM, K_DIM, seed=63)
    conv = tuple(torch.randn(3, 16, CONV_SIZE) for _ in range(3))
    with pytest.raises(ValueError) as excinfo:
        to_ours(_fla_layer_state(state, conv))
    message = str(excinfo.value)
    print(message)
    assert "batch 是 3" in message and "对不上" in message


def test_conv_state_shape_mismatch_between_streams_raises():
    conv = (
        torch.randn(2, 16, CONV_SIZE),
        torch.randn(2, 16, CONV_SIZE),
        torch.randn(2, 32, CONV_SIZE),
    )
    with pytest.raises(ValueError, match="三路形状要一致"):
        to_ours(_fla_layer_state(None, conv))


def test_conv_state_wrong_container_raises():
    with pytest.raises(TypeError, match=r"\(q, k, v\)"):
        to_ours(_fla_layer_state(None, torch.randn(2, 16, CONV_SIZE)))


def test_missing_recurrent_state_key_lists_actual_keys():
    with pytest.raises(KeyError) as excinfo:
        to_ours({"conv_state": None})
    print(str(excinfo.value))
    assert "conv_state" in str(excinfo.value)


def test_to_fla_rejects_unknown_keys():
    ours = {
        "recurrent_state": _asymmetric_state(2, 3, K_DIM, V_DIM, seed=64),
        "conv_state": None,
        "offset": 1,
    }
    with pytest.raises(ValueError, match="offset"):
        to_fla(ours)


# ======================================================================================
# 6. Cache 容器（带 layer_idx）
# ======================================================================================
class _FakeCache:
    """只实现 ``__getitem__`` 的最小容器，形状对齐 fla 的 ``LegacyFLACache``。"""

    def __init__(self, states):
        self.states = list(states)

    def __getitem__(self, idx):
        if idx >= len(self.states):
            raise KeyError(f"Cache only has {len(self.states)} layers")
        return self.states[idx]


def test_layer_idx_selects_the_right_layer():
    layer0 = _fla_layer_state(_asymmetric_state(2, 3, V_DIM, K_DIM, seed=71))
    layer1 = _fla_layer_state(_asymmetric_state(2, 3, V_DIM, K_DIM, seed=72))
    cache = _FakeCache([layer0, layer1])
    ours = to_ours(cache, layer_idx=1, head_k_dim=K_DIM, head_v_dim=V_DIM)
    diff_right = _max_abs_diff(ours["recurrent_state"], layer1["recurrent_state"].transpose(-1, -2))
    diff_wrong = _max_abs_diff(ours["recurrent_state"], layer0["recurrent_state"].transpose(-1, -2))
    print(f"layer_idx=1 取到的层 max_abs_diff = {diff_right:.6e}；对 layer0 = {diff_wrong:.6e}")
    assert diff_right == 0.0
    assert diff_wrong > 1.0


def test_layer_idx_out_of_range_raises():
    cache = _FakeCache([_fla_layer_state(None)])
    with pytest.raises(ValueError, match="layer_idx=5"):
        to_ours(cache, layer_idx=5)


def test_cache_container_without_layer_idx_raises():
    cache = _FakeCache([_fla_layer_state(None)])
    with pytest.raises(TypeError, match="layer_idx"):
        to_ours(cache)


def test_layer_idx_with_mapping_raises():
    with pytest.raises(TypeError, match="layer_idx"):
        to_ours(_fla_layer_state(None), layer_idx=0)
