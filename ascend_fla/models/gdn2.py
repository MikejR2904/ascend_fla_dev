"""GDN-2 1.3B 的 LitGPT-compatible torch / torch_npu 模型。

本文件只保留 model/config/block 与 checkpoint 装配。token mixer、typed cache、显式 backend
和 packed inference 布局位于 :mod:`ascend_fla.layers.gdn2`；逐 token fp32 recurrence oracle
位于 :mod:`ascend_fla.reference.gdn2`。这种拆分让后续自编译算子只替换 core backend，
不会把 checkpoint ABI、投影布局与递推语义绑成一个大模块。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.gdn2 import (
    GDN2_CORE_BACKENDS,
    GDN2LayerCache,
    GDN2ModelCache,
    GatedDeltaNet2,
)
from .gdn2_checkpoint import checkpoint_state_dict

__all__ = [
    "GDN2Config",
    "GDN2ForCausalLM",
    "GDN2LayerCache",
    "GatedDeltaNet2",
    "checkpoint_state_dict",
]


@dataclass
class GDN2Config:
    """完整 GDN-2 模型所需的窄配置。

    默认值对应 ``LLM-OS-Models/gdn2-1.3B-fineweb-edu-100b``。``n_head=18`` 是原
    LitGPT 的通用 attention 字段；纯 GDN-2 模型真正的 recurrence 形状由
    ``num_heads=16`` 决定。
    """

    vocab_size: int = 32_000
    padding_multiple: int = 64
    padded_vocab_size: int | None = None
    block_size: int = 4_096
    n_layer: int = 18
    n_embd: int = 2_304
    intermediate_size: int = 6_208
    norm_eps: float = 1e-5
    bias: bool = False

    num_heads: int = 16
    num_v_heads: int | None = None
    head_dim: int = 128
    expand_v: float = 1.0
    conv_size: int = 4
    conv_bias: bool = False
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    use_qk_l2norm: bool = True
    qk_norm_eps: float = 1e-6
    attention_scale: float | None = None

    def __post_init__(self) -> None:
        if self.padded_vocab_size is None:
            self.padded_vocab_size = (
                (self.vocab_size + self.padding_multiple - 1) // self.padding_multiple
            ) * self.padding_multiple
        if self.num_v_heads is None:
            self.num_v_heads = self.num_heads
        for name in (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_embd",
            "intermediate_size",
            "num_heads",
            "num_v_heads",
            "head_dim",
            "conv_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数，收到 {getattr(self, name)!r}")
        if self.num_v_heads < self.num_heads or self.num_v_heads % self.num_heads:
            raise ValueError(
                f"num_v_heads({self.num_v_heads}) 必须大于等于且整除 num_heads({self.num_heads})"
            )
        exact_head_v_dim = self.head_dim * self.expand_v
        if int(exact_head_v_dim) != exact_head_v_dim:
            raise ValueError(
                f"head_dim * expand_v 必须是整数，收到 {self.head_dim} * {self.expand_v}"
            )
        if self.attention_scale is None:
            self.attention_scale = self.head_dim ** -0.5

    @property
    def head_k_dim(self) -> int:
        return self.head_dim

    @property
    def head_v_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        assert self.num_v_heads is not None
        return self.num_v_heads * self.head_v_dim

    @classmethod
    def gdn2_1_3b(cls, **overrides: Any) -> "GDN2Config":
        return cls(**overrides)


class RMSNorm(nn.Module):
    """与 checkpoint 的 ``FusedRMSNorm`` 参数布局相同的 torch 实现。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self._use_native_npu_inference = False

    def enable_native_npu_inference(self) -> "RMSNorm":
        """Use the installed fused NPU RMSNorm for immutable packed inference.

        The checkpoint weight stays in the model dtype.  It was already rounded
        to that dtype during loading, so the native path consumes the same
        quantized values as the old ``weight.float()`` composition.  CPU keeps
        the portable reference below; an enabled NPU path raises if its fused
        operator is unavailable rather than silently changing the benchmark.
        """
        if self.training:
            raise RuntimeError("enable_native_npu_inference() 前必须先调用 eval()")
        self._use_native_npu_inference = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_native_npu_inference and x.device.type == "npu":
            if self.weight.dtype != x.dtype:
                raise RuntimeError(
                    "packed NPU RMSNorm 要求 weight 与输入同 dtype，"
                    f"收到 {self.weight.dtype} 与 {x.dtype}"
                )
            try:
                op = torch.ops.npu.npu_rms_norm
            except AttributeError as error:  # pragma: no cover - 只在缺 NPU 扩展时触发
                raise RuntimeError(
                    "packed NPU inference 要求 torch_npu.npu_rms_norm；当前环境未注册该算子"
                ) from error
            return op(x, self.weight, self.eps)[0]
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def _promote_inference_norm_parameters(module: nn.Module) -> None:
    """Widen already-quantized norm parameters once for packed inference.

    Checkpoint loading first converts ordinary weights to the requested inference
    dtype.  Widening those exact values back to FP32 preserves the old per-token
    ``weight.float()`` semantics while turning that operation into a no-op.  This
    helper is intentionally used only after the model has entered the immutable
    packed-inference layout; doing it for trainable modules would change optimizer
    storage and mixed-precision behavior.
    """
    for name in ("weight", "bias"):
        parameter = getattr(module, name, None)
        if isinstance(parameter, nn.Parameter) and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


class SwiGLU(nn.Module):
    """LitGPT 命名的 LLaMA MLP；参数名为 ``swiglu.w{1,2,3}.weight``。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.intermediate_size = intermediate_size
        self.projection_layout = "canonical"
        self.mlp_backend = "torch"
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=bias, **factory)

    def pack_for_inference(self) -> "SwiGLU":
        """Pack same-input W1/W2 rows once without retaining duplicate weights."""
        if self.projection_layout == "packed-inference":
            return self
        if self.training:
            raise RuntimeError("SwiGLU.pack_for_inference() 前必须先调用 eval()")
        if self.w1.bias is not None or self.w2.bias is not None:
            raise ValueError("packed SwiGLU W1/W2 首版只支持无 bias")
        self.w12_weight = nn.Parameter(
            torch.cat((self.w1.weight.detach(), self.w2.weight.detach()), dim=0).contiguous(),
            requires_grad=self.w1.weight.requires_grad or self.w2.weight.requires_grad,
        )
        del self.w1
        del self.w2
        self.projection_layout = "packed-inference"
        return self

    def enable_mixed_decode(self, norm: RMSNorm) -> "SwiGLU":
        """Prepare a derived paired buffer for the fixed CCE decode path.

        The accepted prompt path retains ``w12_weight`` and the ordinary native
        RMSNorm.  The paired buffer is deliberately non-persistent: it is
        derived from checkpoint parameters after strict loading and must never
        become a second checkpoint ABI.  This first integration favors a clean
        numerical/performance A/B; removing the duplicate resident layout is a
        separate memory-layout decision after whole-model acceptance.
        """
        if self.mlp_backend == "cce":
            return self
        if self.training or norm.training:
            raise RuntimeError("mixed MLP decode 启用前必须先调用 eval()")
        if self.projection_layout != "packed-inference":
            raise RuntimeError("mixed MLP decode 要求先 pack_for_inference()")
        hidden_size = self.w12_weight.shape[1]
        if (
            hidden_size != 2304
            or self.intermediate_size != 6208
            or tuple(norm.weight.shape) != (hidden_size,)
            or norm.eps != 1e-5
        ):
            raise ValueError(
                "mixed MLP decode fixes hidden/intermediate/eps to "
                f"2304/6208/1e-5, got {hidden_size}/{self.intermediate_size}/{norm.eps}"
            )
        if self.w12_weight.dtype != torch.bfloat16 or norm.weight.dtype != torch.bfloat16:
            raise ValueError(
                "mixed MLP decode requires BF16 packed weights and norm gamma, got "
                f"{self.w12_weight.dtype}/{norm.weight.dtype}"
            )
        channels_per_item = 64
        items = self.intermediate_size // channels_per_item
        w1 = self.w12_weight[: self.intermediate_size].detach()
        w2 = self.w12_weight[self.intermediate_size :].detach()
        paired = torch.cat(
            (
                w1.reshape(items, channels_per_item, hidden_size),
                w2.reshape(items, channels_per_item, hidden_size),
            ),
            dim=1,
        ).contiguous()
        self.register_buffer("paired_w12_weight", paired, persistent=False)
        self.mlp_backend = "cce"
        return self

    def forward_with_norm(self, x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        """Use CCE for B=T=1; keep the accepted vendor composition for prefill."""
        if self.mlp_backend != "cce":
            return self.forward(norm(x))
        if x.ndim != 3:
            raise ValueError(f"mixed MLP 输入应为 [B,T,D]，收到 {tuple(x.shape)}")
        if x.shape[:2] != (1, 1):
            # The custom ABI is intentionally decode-only.  Prompt execution
            # uses the unchanged native norm + packed W12 implementation.
            return self.forward(norm(x))
        if x.device.type != "npu":
            raise ValueError(f"mixed MLP B=T=1 路径要求 NPU tensor，收到 {x.device}")
        from ..ops.gdn2 import fused_norm2_w12_swiglu_gdn2

        hidden = fused_norm2_w12_swiglu_gdn2(
            x.view(1, x.shape[-1]), norm.weight, self.paired_w12_weight
        )
        return self.w3(hidden.view(1, 1, self.intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection_layout == "canonical":
            left = self.w1(x)
            right = self.w2(x)
        elif self.projection_layout == "packed-inference":
            if x.ndim != 3:
                raise ValueError(
                    f"packed SwiGLU 要求输入为 [B,T,D]，收到 {tuple(x.shape)}"
                )
            if x.shape[:2] == (1, 1):
                both = F.linear(x, self.w12_weight)
                left, right = torch.split(both, self.intermediate_size, dim=-1)
            else:
                # The two row slices remain contiguous.  Keeping prompt/T>1 as
                # two original-size GEMMs avoids feeding strided output views to
                # the next operators while storing only one packed parameter.
                left = F.linear(x, self.w12_weight[: self.intermediate_size])
                right = F.linear(x, self.w12_weight[self.intermediate_size :])
        else:  # pragma: no cover - 属性只由本类设置
            raise AssertionError(f"未知 SwiGLU projection_layout {self.projection_layout!r}")
        return self.w3(F.silu(left) * right)


class LLaMAMLP(nn.Module):
    def __init__(self, config: GDN2Config, *, device=None, dtype=None) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            config.bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)

    def forward_with_norm(self, x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        return self.swiglu.forward_with_norm(x, norm)


class GDN2Block(nn.Module):
    def __init__(
        self,
        config: GDN2Config,
        layer_idx: int,
        *,
        core_backend: str,
        core_block_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps, device=device, dtype=dtype)
        self.attn = GatedDeltaNet2(
            config,
            layer_idx,
            core_backend=core_backend,
            core_block_dim=core_block_dim,
            device=device,
            dtype=dtype,
        )
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps, device=device, dtype=dtype)
        self.mlp = LLaMAMLP(config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cache: GDN2LayerCache | Mapping[str, Any] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, GDN2LayerCache | None]:
        attn_out, next_cache = self.attn(self.norm_1(x), cache=cache, use_cache=use_cache)
        x = x + attn_out
        if self.mlp.swiglu.mlp_backend == "cce":
            mlp_out = self.mlp.forward_with_norm(x, self.norm_2)
        else:
            mlp_out = self.mlp(self.norm_2(x))
        return x + mlp_out, next_cache


class GDN2ForCausalLM(nn.Module):
    """与发布的 LitGPT ``GPT(Config.from_name('gdn2_1.3B'))`` 同名的完整模型。"""

    projection_layouts = ("canonical", "packed-inference")
    core_backends = GDN2_CORE_BACKENDS
    mlp_backends = ("torch", "cce")

    def __init__(
        self,
        config: GDN2Config | None = None,
        *,
        core_backend: str = "torch",
        core_block_dim: int = 8,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config or GDN2Config.gdn2_1_3b()
        self.core_backend = core_backend
        self.core_block_dim = core_block_dim
        self.checkpoint_metadata: dict[str, int | float | str | bool | None] = {}
        assert self.config.padded_vocab_size is not None
        factory = {"device": device, "dtype": dtype}
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.padded_vocab_size, bias=False, **factory
        )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    self.config.padded_vocab_size, self.config.n_embd, **factory
                ),
                "h": nn.ModuleList(
                    GDN2Block(
                        self.config,
                        index,
                        core_backend=core_backend,
                        core_block_dim=core_block_dim,
                        device=device,
                        dtype=dtype,
                    )
                    for index in range(self.config.n_layer)
                ),
                "ln_f": RMSNorm(
                    self.config.n_embd, self.config.norm_eps, device=device, dtype=dtype
                ),
            }
        )

    @property
    def projection_layout(self) -> str:
        layouts = {layer.attn.projection_layout for layer in self.transformer["h"]}
        if len(layouts) != 1:
            raise RuntimeError(f"模型各层 projection layout 不一致：{sorted(layouts)}")
        return layouts.pop()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def reported_parameter_count(self) -> int:
        """发布方口径：只统计 transformer blocks，不含 embedding、lm_head 与 final norm。"""
        return sum(parameter.numel() for parameter in self.transformer["h"].parameters())

    @property
    def mlp_backend(self) -> str:
        backends = {layer.mlp.swiglu.mlp_backend for layer in self.transformer["h"]}
        if len(backends) != 1:
            raise RuntimeError(f"模型各层 MLP backend 不一致：{sorted(backends)}")
        return backends.pop()

    def pack_for_inference(self) -> "GDN2ForCausalLM":
        """把每层投影、SwiGLU W1/W2 与 short-conv 转成 inference-only packed 布局。"""
        if self.projection_layout == "packed-inference":
            return self
        if self.training:
            raise RuntimeError("pack_for_inference() 前必须先调用 eval()")
        for layer in self.transformer["h"]:
            # Block norms stay in model dtype for the installed fused NPU op.
            # ``o_norm`` is different: its weight is a direct FP32 GM input of
            # our fused-decode CCE ABI and therefore remains widened once.
            layer.norm_1.enable_native_npu_inference()
            layer.norm_2.enable_native_npu_inference()
            _promote_inference_norm_parameters(layer.attn.o_norm)
            layer.attn.pack_for_inference()
            layer.mlp.swiglu.pack_for_inference()
        self.transformer["ln_f"].enable_native_npu_inference()
        self.requires_grad_(False)
        return self

    def enable_mixed_mlp_inference(self) -> "GDN2ForCausalLM":
        """Enable the explicit CCE B=T=1 MLP boundary on a packed model."""
        if self.mlp_backend == "cce":
            return self
        if self.training:
            raise RuntimeError("enable_mixed_mlp_inference() 前必须先调用 eval()")
        if self.projection_layout != "packed-inference":
            raise RuntimeError("mixed MLP inference 要求 packed-inference 布局")
        for layer in self.transformer["h"]:
            layer.mlp.swiglu.enable_mixed_decode(layer.norm_2)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: GDN2ModelCache | None = None,
        use_cache: bool = False,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[GDN2LayerCache | None]]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids 应为 [B,T]，收到 {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids 不接受空序列")
        x = self.transformer["wte"](input_ids)
        layers = self.transformer["h"]
        if cache is None:
            cache = [None] * len(layers)
        if len(cache) != len(layers):
            raise ValueError(f"cache 应有 {len(layers)} 层，收到 {len(cache)}")

        keep_cache = use_cache or return_cache
        next_cache: list[GDN2LayerCache | None] = []
        for layer, layer_cache in zip(layers, cache):
            x, updated = layer(x, cache=layer_cache, use_cache=keep_cache)
            next_cache.append(updated)
        logits = self.lm_head(self.transformer["ln_f"](x))
        return (logits, next_cache) if return_cache else logits

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: GDN2Config | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
        mmap: bool = True,
        weights_only: bool = True,
        core_backend: str = "torch",
        core_block_dim: int = 8,
        projection_layout: str = "canonical",
        mlp_backend: str = "torch",
    ) -> "GDN2ForCausalLM":
        """以内存友好的 meta + assign 路径加载 checkpoint，并可在 H2D 前打包权重。"""
        if projection_layout not in cls.projection_layouts:
            raise ValueError(
                f"projection_layout 只支持 {cls.projection_layouts}，收到 {projection_layout!r}"
            )
        if mlp_backend not in cls.mlp_backends:
            raise ValueError(f"mlp_backend 只支持 {cls.mlp_backends}，收到 {mlp_backend!r}")
        if mlp_backend == "cce" and projection_layout != "packed-inference":
            raise ValueError("mlp_backend='cce' 要求 projection_layout='packed-inference'")
        if mlp_backend == "cce" and core_backend != "cce":
            raise ValueError("mlp_backend='cce' 首版要求 core_backend='cce'")
        if core_backend == "cce" and not str(device).startswith("npu"):
            raise ValueError("core_backend='cce' 要求把模型加载到 NPU device")
        path = Path(checkpoint_path)
        try:
            raw = torch.load(path, map_location="cpu", weights_only=weights_only, mmap=mmap)
        except RuntimeError as error:
            if not mmap or "mmap" not in str(error).lower():
                raise
            raw = torch.load(path, map_location="cpu", weights_only=weights_only, mmap=False)
        state = checkpoint_state_dict(raw)
        model = cls(
            config,
            core_backend=core_backend,
            core_block_dim=core_block_dim,
            device="meta",
        )
        model.load_state_dict(state, strict=strict, assign=True)
        if isinstance(raw, dict):
            for key in ("iter_num", "step_count", "trained_tokens", "trained_tokens_total"):
                if key not in raw:
                    continue
                value = raw.get(key)
                if isinstance(value, (int, float, str, bool)) or value is None:
                    model.checkpoint_metadata[key] = value
        del state, raw

        # 大矩阵先在 host 转推理 dtype；门控累计参数保留 fp32。
        for name, parameter in model.named_parameters():
            target_dtype = torch.float32 if name.endswith(("A_log", "dt_bias")) else dtype
            if parameter.dtype != target_dtype:
                parameter.data = parameter.data.to(dtype=target_dtype)
        model.eval()
        if projection_layout == "packed-inference":
            # 在 host 逐层消费 canonical 参数后再 H2D，避免 NPU 上出现打包临时副本。
            # canonical 与 packed 的正式验收是同 dtype 数值误差预算，不依赖逐位一致。
            model.pack_for_inference()
        if mlp_backend == "cce":
            model.enable_mixed_mlp_inference()
        if core_backend == "cce":
            # CANN 只在进程首次解析算子时读取 custom OPP path；必须在 embedding/matmul
            # 等任何整网算子执行前构建并注册全部 GDN-2 vendor tree。
            from ..ops.gdn2 import prepare

            prepare(block_dim=core_block_dim, mixed_mlp=mlp_backend == "cce")
        model.to(device=device)
        return model
