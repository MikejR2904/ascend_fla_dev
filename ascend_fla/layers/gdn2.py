"""GDN-2 token mixer：typed cache、显式 core backend 与 packed inference 布局。

``core_backend="torch"`` 用纯 torch / torch_npu 调 fp32-state oracle；
``core_backend="cce"`` 调本仓 Ascriptor CCE recurrent kernel。未知 backend 直接报错，
不静默回退。

``projection_layout="packed-inference"`` 由 :meth:`GatedDeltaNet2.pack_for_inference`
生成。它把 q/k/v/b/w 五个同输入投影以及 f/output-gate 的两个低秩首层分别按输出行打包，
并把 q/k/v depthwise short-conv 与三份 cache 合成一路。数学与 checkpoint 参数均不变，
但 packed 布局是 inference-only，不能拿来训练或保存为原始 LitGPT checkpoint。
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ..modules import FusedRMSNormGated, PackedShortConvolution, ShortConvolution
from ..reference.gdn2 import gdn2_recurrent_reference

if TYPE_CHECKING:
    from ..models.gdn2 import GDN2Config

__all__ = [
    "GDN2_CORE_BACKENDS",
    "GDN2LayerCache",
    "GDN2ModelCache",
    "GatedDeltaNet2",
    "PackedGDN2Projections",
]

GDN2_CORE_BACKENDS = ("torch", "cce")
CanonicalConvState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
PackedConvState = torch.Tensor
ConvState = CanonicalConvState | PackedConvState | None


@dataclass(slots=True)
class GDN2LayerCache(Mapping[str, Any]):
    """单层固定大小 cache；同时提供只读 mapping 兼容旧调用方。"""

    recurrent_state: torch.Tensor
    conv_state: ConvState = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"cache offset 必须非负，收到 {self.offset}")

    def __getitem__(self, key: str) -> Any:
        if key == "recurrent_state":
            return self.recurrent_state
        if key == "conv_state":
            return self.conv_state
        if key == "offset":
            return self.offset
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "recurrent_state"
        yield "conv_state"
        yield "offset"

    def __len__(self) -> int:
        return 3

    @classmethod
    def coerce(
        cls,
        cache: "GDN2LayerCache | Mapping[str, Any] | None",
    ) -> "GDN2LayerCache | None":
        if cache is None or (isinstance(cache, Mapping) and not cache):
            return None
        if isinstance(cache, cls):
            return cache
        if not isinstance(cache, Mapping):
            raise TypeError(f"layer cache 必须是 GDN2LayerCache/mapping/None，收到 {type(cache)}")
        state = cache.get("recurrent_state")
        if not isinstance(state, torch.Tensor):
            raise TypeError("非空 layer cache 必须包含 tensor recurrent_state")
        return cls(
            recurrent_state=state,
            conv_state=cache.get("conv_state"),
            offset=int(cache.get("offset", 0)),
        )


GDN2ModelCache = list[GDN2LayerCache | Mapping[str, Any] | None]


class PackedGDN2Projections(nn.Module):
    """GDN-2 inference-only 投影布局，不额外保留 canonical 权重副本。"""

    def __init__(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        f_proj: nn.Sequential,
        b_proj: nn.Linear,
        w_proj: nn.Linear,
        output_gate_proj: nn.Sequential,
    ) -> None:
        super().__init__()
        direct = (q_proj, k_proj, v_proj, b_proj, w_proj)
        if any(module.bias is not None for module in direct):
            raise ValueError("packed q/k/v/b/w 首版只支持无 bias 的 checkpoint 布局")
        if len(f_proj) != 2 or len(output_gate_proj) != 2:
            raise ValueError("f_proj/output gate projection 必须各由两个 Linear 组成")
        f_low, f_out = f_proj
        gate_low, gate_out = output_gate_proj
        if not all(isinstance(module, nn.Linear) for module in (f_low, f_out, gate_low, gate_out)):
            raise TypeError("f_proj/output gate projection 的两层都必须是 nn.Linear")
        if f_low.bias is not None or gate_low.bias is not None:
            raise ValueError("packed f/g 低秩首层只支持无 bias")
        hidden_size = q_proj.in_features
        if any(module.in_features != hidden_size for module in direct):
            raise ValueError("q/k/v/b/w 投影必须共享相同 input size")
        if f_low.in_features != hidden_size or gate_low.in_features != hidden_size:
            raise ValueError("f/g 低秩首层必须与 q 投影共享 input size")
        if f_low.out_features != gate_low.out_features:
            raise ValueError("f/g 低秩首层的 output size 必须相同")

        self.hidden_size = hidden_size
        self.direct_splits = tuple(module.out_features for module in direct)
        self.qkv_size = sum(self.direct_splits[:3])
        self.low_size = f_low.out_features
        direct_requires_grad = any(module.weight.requires_grad for module in direct)
        low_requires_grad = f_low.weight.requires_grad or gate_low.weight.requires_grad
        self.direct_weight = nn.Parameter(
            torch.cat([module.weight.detach() for module in direct], dim=0).contiguous(),
            requires_grad=direct_requires_grad,
        )
        self.low_weight = nn.Parameter(
            torch.cat([f_low.weight.detach(), gate_low.weight.detach()], dim=0).contiguous(),
            requires_grad=low_requires_grad,
        )
        # 这两层输入不同，不能用普通 dense Linear 无代价地继续合并。
        self.f_out = f_out
        self.output_gate_out = gate_out

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direct = F.linear(hidden_states, self.direct_weight)
        q, k, v, b_raw, w_raw = torch.split(direct, self.direct_splits, dim=-1)
        qkv = direct[..., : self.qkv_size]
        if hidden_states.shape[:2] == (1, 1):
            # B=T=1 时两个 view 都连续，可用一次低秩投影减少一次 launch。
            low = F.linear(hidden_states, self.low_weight)
            f_low, output_gate_low = torch.split(low, self.low_size, dim=-1)
        else:
            # T>1 时 packed 输出的两个切片带 2*low_size 的行 stride；继续喂第二层会让
            # Ascend 选择不同路径，真实权重 output gate 出现 7.8125e-3 差异。权重仍只存
            # 一份 packed Parameter，但按连续 row slice 做两次原尺寸 Linear，贴近 canonical
            # 数值路径；验收仍按 relative-L2 预算，不要求逐位相同。
            f_low = F.linear(hidden_states, self.low_weight[: self.low_size])
            output_gate_low = F.linear(hidden_states, self.low_weight[self.low_size :])
        f_raw = self.f_out(f_low)
        output_gate = self.output_gate_out(output_gate_low)
        assert qkv.shape[-1] == q.shape[-1] + k.shape[-1] + v.shape[-1]
        return qkv, f_raw, b_raw, w_raw, output_gate


class GatedDeltaNet2(nn.Module):
    """GDN-2 token mixer，输入输出均为 ``[B,T,hidden]``。"""

    def __init__(
        self,
        config: "GDN2Config",
        layer_idx: int,
        *,
        core_backend: str = "torch",
        core_block_dim: int = 8,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if core_backend not in GDN2_CORE_BACKENDS:
            raise ValueError(
                f"GDN-2 core_backend 只支持 {GDN2_CORE_BACKENDS}，收到 {core_backend!r}；"
                "不能静默回退到 torch"
            )
        if core_backend == "cce" and core_block_dim != 8:
            raise ValueError(
                f"GDN-2 CCE recurrent 首版固定 core_block_dim=8，收到 {core_block_dim}"
            )
        self.config = config
        self.layer_idx = layer_idx
        self.core_backend = core_backend
        self.core_block_dim = core_block_dim
        self.projection_layout = "canonical"
        self.hidden_size = config.n_embd
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.num_heads = config.num_heads
        assert config.num_v_heads is not None
        self.num_v_heads = config.num_v_heads
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.use_short_conv = config.use_short_conv
        self.allow_neg_eigval = config.allow_neg_eigval
        self.use_qk_l2norm = config.use_qk_l2norm
        self.qk_norm_eps = config.qk_norm_eps
        assert config.attention_scale is not None
        self.attention_scale = config.attention_scale

        factory = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(config.n_embd, self.key_dim, bias=False, **factory)
        self.k_proj = nn.Linear(config.n_embd, self.key_dim, bias=False, **factory)
        self.v_proj = nn.Linear(config.n_embd, self.value_dim, bias=False, **factory)
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                self.key_dim, config.conv_size, config.conv_bias, "silu", **factory
            )
            self.k_conv1d = ShortConvolution(
                self.key_dim, config.conv_size, config.conv_bias, "silu", **factory
            )
            self.v_conv1d = ShortConvolution(
                self.value_dim, config.conv_size, config.conv_bias, "silu", **factory
            )

        self.f_proj = nn.Sequential(
            nn.Linear(config.n_embd, self.head_v_dim, bias=False, **factory),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False, **factory),
        )
        self.b_proj = nn.Linear(config.n_embd, self.key_dim, bias=False, **factory)
        self.w_proj = nn.Linear(config.n_embd, self.value_dim, bias=False, **factory)

        self.A_log = nn.Parameter(torch.zeros(self.num_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, device=device, dtype=torch.float32))

        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, self.head_v_dim, bias=False, **factory),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, **factory),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            activation="swish",
            eps=config.norm_eps,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(self.value_dim, config.n_embd, bias=False, **factory)

    @staticmethod
    def _split_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], -1, head_dim)

    def _expand_key_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_v_heads == self.num_heads:
            return x
        return x.repeat_interleave(self.num_v_heads // self.num_heads, dim=2)

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """兼容旧测试/诊断入口；正式 forward 通过显式 backend 调同一个 oracle。"""
        return gdn2_recurrent_reference(
            q,
            k,
            v,
            g,
            b,
            w,
            initial_state,
            scale=self.attention_scale,
            use_qk_l2norm=self.use_qk_l2norm,
            qk_norm_eps=self.qk_norm_eps,
        )

    def pack_for_inference(self) -> "GatedDeltaNet2":
        """原地消费 canonical 投影/卷积，转为不额外占权重副本的 packed 布局。"""
        if self.projection_layout == "packed-inference":
            return self
        if self.training:
            raise RuntimeError("pack_for_inference() 前必须先调用 eval()")
        if self.key_dim != self.value_dim:
            raise ValueError(
                "packed q/k/v short-conv 首版要求 key_dim == value_dim，"
                f"收到 {self.key_dim} != {self.value_dim}"
            )

        # ``A_log`` is immutable after packing.  Materialize its channel broadcast
        # now instead of launching exp/repeat/neg on every decoded token.  The
        # expanded row is also the direct GM input of the BF16 fused-decode kernel.
        # This is derived state, not part of the LitGPT checkpoint ABI.
        decay_rate = (
            -self.A_log.detach().float().exp().view(self.num_heads, 1)
        ).expand(self.num_heads, self.head_k_dim).contiguous()
        self.register_buffer("_decay_rate", decay_rate, persistent=False)

        self.packed_projections = PackedGDN2Projections(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.f_proj,
            self.b_proj,
            self.w_proj,
            self.g_proj,
        )
        if self.use_short_conv:
            self.packed_qkv_conv = PackedShortConvolution.from_convolutions(
                self.q_conv1d, self.k_conv1d, self.v_conv1d
            )
        for name in ("q_proj", "k_proj", "v_proj", "f_proj", "b_proj", "w_proj", "g_proj"):
            delattr(self, name)
        if self.use_short_conv:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                delattr(self, name)
        self.projection_layout = "packed-inference"
        self.requires_grad_(False)
        return self

    def _project_canonical(
        self,
        hidden_states: torch.Tensor,
        conv_state: ConvState,
        use_cache: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ConvState,
    ]:
        if conv_state is not None and not isinstance(conv_state, tuple):
            raise TypeError("canonical 布局的 conv_state 必须是三元 tuple")
        if self.use_short_conv:
            old_q, old_k, old_v = (None, None, None) if conv_state is None else conv_state
            q, new_q = self.q_conv1d(
                self.q_proj(hidden_states), cache=old_q, output_final_state=use_cache
            )
            k, new_k = self.k_conv1d(
                self.k_proj(hidden_states), cache=old_k, output_final_state=use_cache
            )
            v, new_v = self.v_conv1d(
                self.v_proj(hidden_states), cache=old_v, output_final_state=use_cache
            )
            new_conv: ConvState = None if not use_cache else (new_q, new_k, new_v)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            new_conv = None
        return (
            q,
            k,
            v,
            self.f_proj(hidden_states),
            self.b_proj(hidden_states),
            self.w_proj(hidden_states),
            self.g_proj(hidden_states),
            new_conv,
        )

    def _project_packed(
        self,
        hidden_states: torch.Tensor,
        conv_state: ConvState,
        use_cache: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ConvState,
    ]:
        if conv_state is not None and not isinstance(conv_state, torch.Tensor):
            raise TypeError("packed-inference 布局的 conv_state 必须是单个 packed tensor")
        qkv, f_raw, b_raw, w_raw, output_gate = self.packed_projections(hidden_states)
        if self.use_short_conv:
            if self._uses_fused_short_conv(hidden_states, conv_state, use_cache):
                assert isinstance(conv_state, torch.Tensor)
                qkv, new_conv = self._run_fused_short_conv(qkv, conv_state)
            else:
                qkv, new_conv = self.packed_qkv_conv(
                    qkv, cache=conv_state, output_final_state=use_cache
                )
        else:
            qkv, new_conv = F.silu(qkv), None
        q, k, v = torch.split(qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        return q, k, v, f_raw, b_raw, w_raw, output_gate, new_conv

    def _run_fused_short_conv(
        self,
        qkv: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the exact-shape packed convolution/cache CCE boundary."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "GDN-2 CCE short-conv decode only supports inference; "
                "use core_backend='torch' for training"
            )
        from ..ops.gdn2 import fused_short_conv_decode_gdn2

        return fused_short_conv_decode_gdn2(
            qkv,
            conv_state,
            self.packed_qkv_conv.weight,
            block_dim=self.core_block_dim,
        )

    def _run_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.core_backend == "torch":
            return self._scan(q, k, v, g, b, w, initial_state)
        if self.core_backend == "cce":
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "GDN-2 CCE recurrent 首版只支持 inference；训练请使用 core_backend='torch'"
                )
            from ..ops.gdn2 import fused_recurrent_gdn2

            # packed q/k/v 在 T>1 时是带更宽 row stride 的 view。CCE 算子入口坚持 contiguous
            # 契约，因此 layout adapter 在这里显式物化；T=1 decode 下这些调用都是 no-op。
            q, k, v, g, b, w = (
                tensor if tensor.is_contiguous() else tensor.contiguous()
                for tensor in (q, k, v, g, b, w)
            )
            output, final_state = fused_recurrent_gdn2(
                q,
                k,
                v,
                g,
                b,
                w,
                initial_state=initial_state,
                output_final_state=True,
                scale=self.attention_scale,
                use_qk_l2norm=self.use_qk_l2norm,
                qk_norm_eps=self.qk_norm_eps,
                block_dim=self.core_block_dim,
            )
            assert final_state is not None
            return output, final_state
        raise AssertionError(f"未处理的 GDN-2 backend {self.core_backend!r}")

    def _uses_fused_decode(self, hidden_states: torch.Tensor) -> bool:
        """Whether this call exactly matches the model-specific BF16 CCE ABI."""
        return (
            self.core_backend == "cce"
            and self.projection_layout == "packed-inference"
            and hidden_states.shape[:2] == (1, 1)
            and hidden_states.dtype == torch.bfloat16
            and self.num_heads == 16
            and self.num_v_heads == 16
            and self.head_k_dim == 128
            and self.head_v_dim == 128
            and not self.allow_neg_eigval
            and self.use_qk_l2norm
            and self.qk_norm_eps == 1e-6
            and self.attention_scale == 128**-0.5
            and self.config.norm_eps == 1e-5
            and self.o_norm.activation == "swish"
            and self.o_norm.weight is not None
            and self.o_norm.weight.dtype == torch.float32
            and self.o_norm.bias is None
            and hasattr(self, "_decay_rate")
        )

    def _uses_fused_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv_state: ConvState,
        use_cache: bool,
    ) -> bool:
        """Whether packed short-conv exactly matches the fixed BF16 CCE ABI."""
        if not isinstance(conv_state, torch.Tensor):
            return False
        convolution = self.packed_qkv_conv
        return (
            self.core_backend == "cce"
            and self.projection_layout == "packed-inference"
            and hidden_states.shape[:2] == (1, 1)
            and hidden_states.dtype == torch.bfloat16
            and self.key_dim + self.key_dim + self.value_dim == 6144
            and self.use_short_conv
            and use_cache
            and convolution.packed_size == 6144
            and convolution.kernel_size == 4
            and convolution.bias is None
            and convolution.activation in ("silu", "swish")
            and tuple(conv_state.shape) == (1, 6144, 4)
            and conv_state.dtype == torch.bfloat16
        )

    def _run_fused_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f_raw: torch.Tensor,
        b_raw: torch.Tensor,
        w_raw: torch.Tensor,
        output_gate: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the raw-gate recurrence and output norm in one CCE launch."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "GDN-2 CCE fused decode only supports inference; use core_backend='torch' for training"
            )
        from ..ops.gdn2 import fused_decode_gdn2

        return fused_decode_gdn2(
            self._split_heads(q, self.head_k_dim),
            self._split_heads(k, self.head_k_dim),
            self._split_heads(v, self.head_v_dim),
            self._split_heads(f_raw, self.head_k_dim),
            self._split_heads(b_raw, self.head_k_dim),
            self._split_heads(w_raw, self.head_v_dim),
            self._split_heads(output_gate, self.head_v_dim),
            self._decay_rate,
            self.dt_bias.view(self.num_heads, self.head_k_dim),
            self.o_norm.weight,
            initial_state=initial_state,
            block_dim=self.core_block_dim,
        )

    @staticmethod
    def _detach_conv_state(conv_state: ConvState) -> ConvState:
        if conv_state is None:
            return None
        if isinstance(conv_state, torch.Tensor):
            return conv_state.detach()
        return tuple(tensor.detach() for tensor in conv_state)  # type: ignore[return-value]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: GDN2LayerCache | Mapping[str, Any] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, GDN2LayerCache | None]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states 应为 [B,T,{self.hidden_size}]，收到 {tuple(hidden_states.shape)}"
            )
        layer_cache = GDN2LayerCache.coerce(cache)
        conv_state = None if layer_cache is None else layer_cache.conv_state
        if self.projection_layout == "canonical":
            projected = self._project_canonical(hidden_states, conv_state, use_cache)
        elif self.projection_layout == "packed-inference":
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "packed-inference 布局不可求导；请在 torch.inference_mode()/no_grad() 下运行，"
                    "训练请使用 canonical 布局"
                )
            projected = self._project_packed(hidden_states, conv_state, use_cache)
        else:  # pragma: no cover - 属性只由本类设置
            raise AssertionError(f"未知 projection_layout {self.projection_layout!r}")
        q, k, v, f_raw, b_raw, w_raw, output_gate, new_conv_state = projected

        initial_state = None if layer_cache is None else layer_cache.recurrent_state
        if self._uses_fused_decode(hidden_states):
            # T=1 packed slices are contiguous views.  The fused ABI rejects any
            # accidental striding rather than inserting a hidden layout copy.
            normalized, final_state = self._run_fused_decode(
                q,
                k,
                v,
                f_raw,
                b_raw,
                w_raw,
                output_gate,
                initial_state,
            )
            out = self.o_proj(normalized.reshape(1, 1, self.value_dim))
        else:
            decay_rate = (
                self._decay_rate
                if self.projection_layout == "packed-inference"
                else -self.A_log.float().exp()
            )
            g = (
                decay_rate.reshape(-1)
                if self.projection_layout == "packed-inference"
                else decay_rate.repeat_interleave(self.head_k_dim)
            )
            g = g * F.softplus(f_raw.float() + self.dt_bias)
            b = torch.sigmoid(b_raw)
            w = torch.sigmoid(w_raw)

            q = self._split_heads(q, self.head_k_dim)
            k = self._split_heads(k, self.head_k_dim)
            g = self._split_heads(g, self.head_k_dim)
            b = self._split_heads(b, self.head_k_dim)
            v = self._split_heads(v, self.head_v_dim)
            w = self._split_heads(w, self.head_v_dim)
            q, k, g, b = (self._expand_key_heads(x) for x in (q, k, g, b))
            if self.allow_neg_eigval:
                b = b * 2.0

            recurrent, final_state = self._run_core(q, k, v, g, b, w, initial_state)
            gate = self._split_heads(output_gate, self.head_v_dim)
            out = self.o_norm(recurrent, gate).reshape(
                hidden_states.shape[0], hidden_states.shape[1], self.value_dim
            )
            out = self.o_proj(out)

        if not use_cache:
            return out, None
        offset = (0 if layer_cache is None else layer_cache.offset) + hidden_states.shape[1]
        next_cache = GDN2LayerCache(
            recurrent_state=final_state.detach(),
            conv_state=self._detach_conv_state(new_conv_state),
            offset=offset,
        )
        return out, next_cache
