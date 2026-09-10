"""短因果卷积 —— KDA layer 用到的窄切片，对齐 fla 的 ``ShortConvolution``。

⚠️ **这不是自编译算子。** 本模块用 torch / torch_npu 的原生算子实现，目的是把 KDA
layer 拼起来、让层级梯度可以端到端验证。它在内置算子包不全的机器上不可用（需要
conv1d、silu），也不在本仓"高效率算子"的承诺范围内 —— 见 ``docs/matrix/gaps.json``
的 ``modules-are-torch-not-kernels``。

语义对齐 fla（``fla/modules/conv/short_conv.py``）：depthwise 卷积，
``groups=hidden_size``、``padding=kernel_size-1``，卷完截掉尾部多出来的
``kernel_size-1`` 个位置以保持因果，然后过激活。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ShortConvolution"]


class ShortConvolution(nn.Conv1d):
    """因果 depthwise 短卷积，可选 silu/swish 激活，支持单步解码的 cache。

    Args:
        hidden_size: 通道数。每个通道独立卷积（``groups=hidden_size``）。
        kernel_size: 卷积核长度（Kimi-Linear / Qwen3-Next 都是 4）。
        bias: 是否带偏置。
        activation: ``None`` / ``"silu"`` / ``"swish"``（后两者等价）。

    输入输出都是 ``[B, T, D]``（token-major），与 KDA 算子的公开布局一致，省掉一次
    转置；内部按 conv1d 要求临时转成 ``[B, D, T]``。
    """

    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False,
                 activation: str | None = "silu", device=None, dtype=None) -> None:
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"activation 只支持 None/silu/swish，收到 {activation!r}")
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,       # depthwise
            bias=bias,
            padding=kernel_size - 1,  # 左右都补，靠下面的截断恢复因果
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) if self.activation else x

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:
            x: ``[B, T, D]``。
            cache: ``[B, D, kernel_size]``，最近 ``kernel_size`` 个时刻的输入。
                给了就把它拼在 ``x`` 前面当左侧上下文（解码路径）。
            output_final_state: 是否返回更新后的 cache。

        Returns:
            ``(y, cache)``，``y`` 为 ``[B, T, D]``。
        """
        if x.dim() != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(f"x 应为 [B,T,{self.hidden_size}]，收到 {tuple(x.shape)}")
        b, t, _ = x.shape
        w = self.kernel_size[0]
        xt = x.transpose(1, 2)  # [B,D,T]

        if cache is None:
            # padding=w-1 会在两端各补 w-1；取前 t 个输出即"只看过去"
            y = F.conv1d(xt, self.weight, self.bias, groups=self.hidden_size,
                         padding=w - 1)[..., :t]
        else:
            if tuple(cache.shape) != (b, self.hidden_size, w):
                raise ValueError(
                    f"cache 应为 [{b},{self.hidden_size},{w}]，收到 {tuple(cache.shape)}"
                )
            # 左侧接上 cache 的后 w-1 个时刻，就不需要再补零
            y = F.conv1d(torch.cat([cache[..., -(w - 1):], xt], dim=-1) if w > 1 else xt,
                         self.weight, self.bias, groups=self.hidden_size, padding=0)

        y = self._act(y.transpose(1, 2))
        if not output_final_state:
            return y, None
        # 新 cache = 最近 w 个时刻的**输入**（不是输出）
        hist = xt if cache is None else torch.cat([cache, xt], dim=-1)
        new_cache = hist[..., -w:]
        if new_cache.shape[-1] < w:  # T < w 且无 cache：左侧补零对齐
            pad = torch.zeros(b, self.hidden_size, w - new_cache.shape[-1],
                              dtype=x.dtype, device="cpu").to(x.device)
            new_cache = torch.cat([pad, new_cache], dim=-1)
        return y, new_cache.contiguous()
