"""modules 层的语义测试 —— 纯 CPU，**不需要 NPU**。

这两个模块的错法都属于"看起来差不多"：
``FusedRMSNormGated`` 若写成 ``rmsnorm(x * act(g))`` 而不是 ``rmsnorm(x) * act(g)``，
输出仍在合理范围；``ShortConvolution`` 的因果截断若取错一端，结果也不会爆。所以这里
对着闭式公式逐项验，而不是只看形状。

    pytest tests/test_modules.py -v
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.modules import (  # noqa: E402
    FusedRMSNormGated,
    PackedShortConvolution,
    ShortConvolution,
)


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


class TestFusedRMSNormGated:
    def test_matches_closed_form(self):
        """逐项对闭式：先归一化、再乘激活过的门控。"""
        torch.manual_seed(0)
        d = 128
        m = FusedRMSNormGated(d, activation="sigmoid", eps=1e-5)
        with torch.no_grad():
            m.weight.normal_(1.0, 0.1)
        x = torch.randn(2, 7, d) * 3.0
        gate = torch.randn(2, 7, d)

        xf = x.float()
        want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
        want = want * m.weight.float() * torch.sigmoid(gate.float())
        assert _rel_l2(m(x, gate), want) < 1e-6

    def test_order_matters(self):
        """归一化 x*act(g) 与 归一化(x)*act(g) 必须不同 —— 否则这个测试没有判别力。"""
        torch.manual_seed(0)
        d = 64
        m = FusedRMSNormGated(d, activation="sigmoid")
        x, gate = torch.randn(1, 4, d) * 2, torch.randn(1, 4, d)
        g_act = torch.sigmoid(gate.float())
        wrong = (x.float() * g_act)
        wrong = wrong * torch.rsqrt(wrong.pow(2).mean(-1, keepdim=True) + m.eps)
        assert _rel_l2(m(x, gate), wrong) > 1e-2, "两种顺序给出了几乎相同的结果，测试无效"

    def test_sigmoid_vs_silu(self):
        torch.manual_seed(0)
        d = 64
        x, gate = torch.randn(1, 4, d), torch.randn(1, 4, d)
        sig = FusedRMSNormGated(d, activation="sigmoid")
        silu = FusedRMSNormGated(d, activation="silu")
        swish = FusedRMSNormGated(d, activation="swish")
        assert torch.equal(silu(x, gate), swish(x, gate)), "silu 与 swish 应等价"
        assert _rel_l2(sig(x, gate), silu(x, gate)) > 1e-2

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="activation"):
            FusedRMSNormGated(64, activation="gelu")
        with pytest.raises(ValueError, match="elementwise_affine"):
            FusedRMSNormGated(64, elementwise_affine=False, bias=True)
        m = FusedRMSNormGated(64)
        with pytest.raises(ValueError, match="最后一维"):
            m(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
        with pytest.raises(ValueError, match="同形状"):
            m(torch.randn(1, 4, 64), torch.randn(1, 5, 64))

    def test_dtype_preserved_but_computed_in_fp32(self):
        """输入 bf16 时输出也是 bf16，但中间在 fp32 算（否则平方和会丢位）。"""
        torch.manual_seed(0)
        d = 128
        m = FusedRMSNormGated(d, activation="sigmoid")
        x = (torch.randn(1, 4, d) * 50).bfloat16()   # 量级大，bf16 求和会明显丢位
        gate = torch.randn(1, 4, d).bfloat16()
        out = m(x, gate)
        assert out.dtype == torch.bfloat16
        want = m(x.float(), gate.float())
        # 只差最后一次 store 的舍入，不该差到 bf16 累加的量级
        assert _rel_l2(out, want) < 5e-3


class TestShortConvolution:
    def test_causality(self):
        """改未来的输入不能影响过去的输出。"""
        torch.manual_seed(0)
        d, t = 16, 12
        m = ShortConvolution(d, kernel_size=4, activation=None)
        x = torch.randn(1, t, d)
        y = m(x)[0]
        x2 = x.clone()
        x2[:, t // 2:] = torch.randn(1, t - t // 2, d)  # 只改后半
        y2 = m(x2)[0]
        assert torch.allclose(y[:, : t // 2], y2[:, : t // 2], atol=1e-6), "前半输出被未来影响了"
        assert not torch.allclose(y[:, t // 2:], y2[:, t // 2:], atol=1e-6), "后半没变，测试无效"

    def test_matches_manual_depthwise(self):
        """逐项对手算：y[t,c] = Σ_{j<W} w[c,j] * x[t - (W-1-j), c]。"""
        torch.manual_seed(0)
        d, t, w = 8, 10, 4
        m = ShortConvolution(d, kernel_size=w, bias=True, activation=None)
        with torch.no_grad():
            m.weight.normal_()
            m.bias.normal_()
        x = torch.randn(1, t, d)
        got = m(x)[0]

        want = torch.zeros(1, t, d)
        for ti in range(t):
            for j in range(w):
                src = ti - (w - 1 - j)
                if src >= 0:
                    want[0, ti] += m.weight[:, 0, j] * x[0, src]
        want += m.bias
        assert _rel_l2(got, want) < 1e-5

    def test_activation_applied_after_conv(self):
        torch.manual_seed(0)
        d = 8
        plain = ShortConvolution(d, 4, activation=None)
        with torch.no_grad():
            plain.weight.normal_()
        act = ShortConvolution(d, 4, activation="silu")
        act.load_state_dict(plain.state_dict())
        x = torch.randn(1, 6, d)
        assert _rel_l2(act(x)[0], F.silu(plain(x)[0])) < 1e-6

    def test_cache_roundtrip_matches_full_sequence(self):
        """分两段带 cache 跑，结果要与一次跑完整序列相同 —— decode 路径的正确性基础。"""
        torch.manual_seed(0)
        d, w = 16, 4
        m = ShortConvolution(d, kernel_size=w, activation="silu")
        with torch.no_grad():
            m.weight.normal_()
        x = torch.randn(1, 8, d)
        full = m(x)[0]

        first, cache = m(x[:, :4], output_final_state=True)
        assert tuple(cache.shape) == (1, d, w)
        second, _ = m(x[:, 4:], cache=cache)
        stitched = torch.cat([first, second], dim=1)
        assert _rel_l2(stitched, full) < 1e-5, "分段结果与整段不一致，cache 语义错了"

    def test_cache_shorter_than_kernel_is_left_padded(self):
        """T < kernel_size 且无 cache 时，新 cache 要左侧补零而不是报错。"""
        torch.manual_seed(0)
        d, w = 8, 4
        m = ShortConvolution(d, kernel_size=w, activation=None)
        _, cache = m(torch.randn(1, 2, d), output_final_state=True)
        assert tuple(cache.shape) == (1, d, w)
        assert torch.count_nonzero(cache[..., : w - 2]) == 0, "左侧没有补零"

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="activation"):
            ShortConvolution(8, 4, activation="relu")
        m = ShortConvolution(8, 4)
        with pytest.raises(ValueError, match=r"\[B,T,8\]"):
            m(torch.randn(1, 4, 16))
        with pytest.raises(ValueError, match="cache 应为"):
            m(torch.randn(1, 4, 8), cache=torch.zeros(1, 8, 3))


class TestPackedShortConvolution:
    def test_one_token_reuses_window_as_new_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T=1 decode 只构造一次 Cat；window 与新 cache 在数学上是同一个张量。"""
        torch.manual_seed(11)
        packed = PackedShortConvolution(8, 4, activation="silu")
        with torch.no_grad():
            packed.weight.normal_()
        x = torch.randn(1, 1, 24)
        cache = torch.randn(1, 24, 4)
        original_cat = torch.cat
        cat_calls: list[tuple[tuple[torch.Size, ...], int]] = []

        def counted_cat(tensors, dim=0, *, out=None):
            tensors = tuple(tensors)
            cat_calls.append((tuple(tensor.shape for tensor in tensors), dim))
            return original_cat(tensors, dim=dim, out=out)

        monkeypatch.setattr(torch, "cat", counted_cat)
        got, new_cache = packed(x, cache=cache, output_final_state=True)

        assert len(cat_calls) == 1
        assert cat_calls[0] == (((torch.Size([1, 24, 3]), torch.Size([1, 24, 1]))), -1)
        xt = x.transpose(1, 2)
        expected_cache = original_cat([cache[..., -3:], xt], dim=-1)
        expected = F.silu(
            F.conv1d(expected_cache, packed.weight, groups=24).transpose(1, 2)
        )
        assert torch.equal(new_cache, expected_cache)
        assert new_cache.is_contiguous()
        assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    def test_matches_three_streams_with_and_without_cache(self):
        torch.manual_seed(0)
        streams = [ShortConvolution(8, 4, activation="silu") for _ in range(3)]
        for stream in streams:
            with torch.no_grad():
                stream.weight.normal_()
        packed = PackedShortConvolution.from_convolutions(*streams)
        x_parts = [torch.randn(1, 7, 8) for _ in range(3)]
        x_packed = torch.cat(x_parts, dim=-1)

        separate_full = torch.cat(
            [stream(part, output_final_state=True)[0] for stream, part in zip(streams, x_parts)],
            dim=-1,
        )
        packed_full, packed_full_cache = packed(x_packed, output_final_state=True)
        assert torch.allclose(packed_full, separate_full, atol=1e-6, rtol=1e-6)

        separate_first = [
            stream(part[:, :2], output_final_state=True)
            for stream, part in zip(streams, x_parts)
        ]
        packed_first, packed_cache = packed(x_packed[:, :2], output_final_state=True)
        assert torch.allclose(
            packed_first,
            torch.cat([item[0] for item in separate_first], dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(packed_cache, torch.cat([item[1] for item in separate_first], dim=1))

        separate_second = [
            stream(part[:, 2:], cache=item[1], output_final_state=True)
            for stream, part, item in zip(streams, x_parts, separate_first)
        ]
        packed_second, packed_second_cache = packed(
            x_packed[:, 2:], cache=packed_cache, output_final_state=True
        )
        assert torch.allclose(
            packed_second,
            torch.cat([item[0] for item in separate_second], dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(
            packed_second_cache, torch.cat([item[1] for item in separate_second], dim=1)
        )
        assert torch.equal(packed_second_cache, packed_full_cache)

    def test_rejects_mismatched_streams_and_cache(self):
        q = ShortConvolution(8, 4)
        with pytest.raises(ValueError, match="规格"):
            PackedShortConvolution.from_convolutions(q, ShortConvolution(8, 3), q)
        packed = PackedShortConvolution(8, 4)
        with pytest.raises(ValueError, match="packed q/k/v"):
            packed(torch.randn(1, 2, 8))
        with pytest.raises(ValueError, match="cache 应为"):
            packed(torch.randn(1, 2, 24), cache=torch.zeros(1, 24, 3))
