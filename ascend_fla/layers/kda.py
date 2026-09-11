"""KDA 层（Kimi Delta Attention）—— 对齐 fla 的 ``KimiDeltaAttention``。

这是"模型 → 模块 → 层 → 算子"全链路里的层级。结构与参数名逐项对齐 fla，以便第三期
直接把它注入 HF/fla 的 ``KimiLinear*`` 模型定义（AGENTS.md §9 的"窄切片 + 注入"）。

**本层承担了 fla 放在 kernel 里做的三件事。** fla 的 ``KimiDeltaAttention`` 调
``chunk_kda`` 时传 ``use_qk_l2norm_in_kernel=True`` / ``use_gate_in_kernel=True`` /
``use_beta_sigmoid_in_kernel=True``，而 ascriptor 的 kda kernel 都不做，所以这里显式做：

1. ``q`` / ``k`` 沿头维 L2 归一化（``qk-l2norm-not-in-kernel``）；
2. ``g = -exp(A_log) * softplus(g_raw + dt_bias)``；
3. ``beta = sigmoid(b_proj(x))``。

这三步在 fp32 下做，再按算子 ABI 交给 kernel（q/k/v bf16，g/beta fp32）。

**本层默认用 ``impl="stable"``。** 按 fla 的初始化，``A_log`` 与 ``dt_bias`` 给出的
chunk 内门控跨度约 94，而 ascriptor 原版 kernel 在**前向与反向各有一处**撑不住：

* 前向约 87 —— gate 只写 ``eg = exp(gc)``，它下溢到 0 后下游的除法变成 ``0/0``；
* 反向约 88.7 —— ``finalize_pre/post`` 的 ``exp(g − g_last)`` **上溢**到 inf，配对的因子
  同时下溢到 0，矩阵乘得 ``inf × 0``。方向与前向相反，是独立的一处。

本仓 ``kernels/projects/a5/kda_fwd_stable`` 与 ``kda_bwd_stable`` 分别把两处改成对深衰减
稳定的形式，上限约 160。``impl`` 同时选两条链。细节见 ``docs/matrix/gaps.json`` 的
``gate-range-beyond-declared`` 与 ``bwd-gate-range-overflow``。

**不支持的上游开关**（传了就报错，不静默忽略 —— AGENTS.md §7）：
``allow_neg_eigval``、``safe_gate``、``lower_bound``、``cu_seqlens``（varlen）。

**尚未接 decode 路径**：``mode="fused_recurrent"`` 要等 ``kda_fused_recurrent``
（``gaps.json`` 的 ``fused-recurrent-missing``，第三期）。本层只做 chunk 前向/反向。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..modules.convolution import ShortConvolution
from ..modules.fused_norm_gated import FusedRMSNormGated
from ..ops.kda.autograd import chunk_kda
from ..ops.kda.chunk import HEAD_DIM, L_PER_CHUNK, VALUE_DIM

__all__ = ["KimiDeltaAttention"]


class KimiDeltaAttention(nn.Module):
    """KDA 层。参数名与 fla 的 ``KimiDeltaAttention`` 一致，可直接载同名权重。

    Args:
        hidden_size: 模型隐藏维。
        expand_v: ``head_v_dim = head_dim * expand_v``。本仓算子定尺 K=V=128，
            所以只接受让 ``head_v_dim == 128`` 的取值。
        head_dim: ``head_k_dim``。算子定尺要求 128。
        num_heads: q/k 的头数。
        num_v_heads: v 的头数，默认等于 ``num_heads``。需 ``num_v_heads % num_heads == 0``。
        use_short_conv: 是否用短因果卷积。
        conv_size / conv_bias: 短卷积的核长与偏置。
        norm_eps: 输出归一化的 eps。
        layer_idx: 层序号，仅用于 cache 寻址。
        allow_neg_eigval / safe_gate / lower_bound: **不支持**，非默认值即报错。
        block_dim: 传给底层算子的启动核组数。
        impl: 底层算子的实现，**同时作用于前向与反向**。``"stable"``（默认）用本仓的
            gate/scores/wy 与 finalize_pre/post，可用门控跨度约 160；``"upstream"`` 用
            ascriptor 原版，约 80。**本层按 fla 的默认初始化产生的跨度约 94，所以
            ``upstream`` 前向反向都会吐 NaN、``stable`` 才能用** —— 见
            ``docs/matrix/gaps.json`` 的 ``gate-range-beyond-declared``
            与 ``bwd-gate-range-overflow``。
        device / dtype: 参数的设备与 dtype。

    算子侧的硬约束（不满足在 forward 里报错）：``head_k_dim == head_v_dim == 128``、
    ``num_v_heads % num_heads == 0``、``T`` 是 64 的整数倍。
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        *,
        block_dim: int = 1,
        impl: str = "stable",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()
        if mode != "chunk":
            raise ValueError(
                f"只实现了 mode='chunk'，收到 {mode!r}；decode 路径要等 "
                "kda_fused_recurrent，见 docs/matrix/gaps.json 的 fused-recurrent-missing"
            )
        for name, value, default in (("allow_neg_eigval", allow_neg_eigval, False),
                                     ("safe_gate", safe_gate, False),
                                     ("lower_bound", lower_bound, None)):
            if value != default:
                raise ValueError(
                    f"ascriptor 的 kda kernel 不支持 {name}={value!r}（只支持 {default!r}）。"
                    "静默忽略会让支持矩阵说谎，所以这里报错。"
                )
        if kwargs:
            # 上游 config 里的多余字段照收，但不要让"传了没生效"悄悄发生
            unknown = sorted(kwargs)
            raise ValueError(f"不认识的参数 {unknown}；如需忽略请在调用方显式剔除")

        self.hidden_size = hidden_size
        self.mode = mode
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_k_dim
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.block_dim = block_dim
        self.impl = impl

        if self.head_k_dim != HEAD_DIM or self.head_v_dim != VALUE_DIM:
            raise ValueError(
                f"本仓 kda 算子定尺要求 head_k_dim=head_v_dim={HEAD_DIM}，"
                f"收到 head_k_dim={self.head_k_dim} head_v_dim={self.head_v_dim}"
                f"（head_dim={head_dim} expand_v={expand_v}）；"
                "见 docs/matrix/gaps.json 的 fixed-kv-128"
            )
        if self.num_v_heads % self.num_heads:
            raise ValueError(
                f"num_v_heads({self.num_v_heads}) 必须是 num_heads({self.num_heads}) 的整数倍"
            )

        f = dict(device=device, dtype=dtype)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False, **f)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False, **f)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False, **f)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias, "silu", **f)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias, "silu", **f)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias, "silu", **f)
        # f_proj 两段都无偏置；g_proj 的第二段**有**偏置 —— 与 fla 一致，别写反
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False, **f),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False, **f),
        )
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False, **f),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, **f),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False, **f)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid",
                                        eps=norm_eps, **f)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False, **f)

        # A_log：fla 用 log(U(1,16))，使 exp(A_log) ∈ [1,16]
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_v_heads, **f).uniform_(1, 16))
        )
        # dt_bias：标准 Mamba dt 初始化的逆 softplus。常数取 fla/Mamba 的默认值，
        # 属于初始化细节，不影响算子正确性；需要复现上游权重时直接 load_state_dict 覆盖。
        dt = torch.exp(
            torch.rand(self.gate_dim, **f) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def _gate(self, hidden_states: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """``g_raw -> -exp(A_log) * softplus(g_raw + dt_bias)``，fp32。

        ``A_log`` 是 per-head 的 ``[HV]``，要广播到 K 维；``dt_bias`` 是 ``[HV*K]``，
        按 ``[HV, K]`` 看。
        """
        hv, kd = self.num_v_heads, self.head_k_dim
        g_raw = self.f_proj(hidden_states).float().view(b, t, hv, kd)
        g = F.softplus(g_raw + self.dt_bias.float().view(hv, kd))
        return -torch.exp(self.A_log.float()).view(hv, 1) * g

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:
            hidden_states: ``[B, T, hidden_size]``。``T`` 必须是 64 的整数倍。
            initial_state: ``[B, HV, 128, 128]`` float32，**K 在前**。
                fla 的 KDA layer 用 ``state_v_first=True``（V 在前），对接它的 cache
                要转置 —— 见 ``gaps.json`` 的 ``state-layout-k-first``。
            output_final_state: 是否返回末态。
            cu_seqlens: **不支持**，传了就报错。

        Returns:
            ``(o, final_state)``，``o`` 为 ``[B, T, hidden_size]``。
        """
        if cu_seqlens is not None:
            raise ValueError(
                "ascriptor 的 kda kernel 不支持 varlen（cu_seqlens）；"
                "见 docs/matrix/gaps.json 的 no-varlen"
            )
        if kwargs:
            raise ValueError(f"forward 收到不认识的参数 {sorted(kwargs)}")
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states 应为 [B,T,{self.hidden_size}]，收到 {tuple(hidden_states.shape)}"
            )
        b, t, _ = hidden_states.shape
        if t % L_PER_CHUNK:
            raise ValueError(
                f"T 必须是 {L_PER_CHUNK} 的整数倍（算子无 tail 路径），收到 T={t}；"
                "见 docs/matrix/gaps.json 的 no-tail-path"
            )

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if self.use_short_conv:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        q = q.view(b, t, self.num_heads, self.head_k_dim)
        k = k.view(b, t, self.num_heads, self.head_k_dim)
        v = v.view(b, t, self.num_v_heads, self.head_v_dim)
        # kernel 不做 l2norm（fla 的 use_qk_l2norm_in_kernel=True 在它那边做了），
        # 所以这里显式做，并在 fp32 下做以免 bf16 的平方和丢位
        q = F.normalize(q.float(), dim=-1, eps=1e-6).bfloat16()
        k = F.normalize(k.float(), dim=-1, eps=1e-6).bfloat16()
        v = v.bfloat16()

        g = self._gate(hidden_states, b, t)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        o, final_state = chunk_kda(
            q, k, v, g, beta,
            initial_state=initial_state, output_final_state=output_final_state,
            block_dim=self.block_dim, impl=self.impl,
        )

        gate = self.g_proj(hidden_states).view(b, t, self.num_v_heads, self.head_v_dim)
        # gate 保持投影出来的 dtype（通常 fp32）——o_norm 内部一律升到 fp32 算，
        # 先降到 bf16 只会白丢精度。o_norm 的输出 dtype 跟随第一个参数，即 o 的 bf16。
        o = self.o_norm(o, gate)
        # 算子输出是 bf16，而 o_proj 的权重可能是 fp32（整层 .float() 的常见情形）。
        # nn.Linear 不做 dtype 提升，会直接报错 —— 这里显式对齐到权重的 dtype。
        o = o.reshape(b, t, self.value_dim).to(self.o_proj.weight.dtype)
        return self.o_proj(o), final_state
