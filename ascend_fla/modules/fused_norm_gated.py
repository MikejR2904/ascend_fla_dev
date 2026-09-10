"""带门控的 RMSNorm —— KDA layer 的输出归一化，对齐 fla 的 ``FusedRMSNormGated``。

⚠️ **这不是自编译算子**，也**不是融合实现**。名字里的 Fused 是为了与 fla 对齐；这里
是 torch 原生算子拼的，归一化与门控是两趟。见 ``docs/matrix/gaps.json`` 的
``modules-are-torch-not-kernels``。

语义（对齐 ``fla/modules/fused_norm_gate.py``）::

    y = x * rsqrt(mean(x²) + eps)
    y = y * weight (+ bias)            # elementwise_affine
    out = y * act(gate)

注意**先归一化再乘门控**，不是归一化 ``x * act(gate)``。顺序错了数值会整体偏，而且
偏得不大 —— 是那种能通过"看起来差不多"检查的错，所以这里写明。

KDA 用 ``activation="sigmoid"``（fla 的 ``FusedRMSNormGated`` 默认是 ``"swish"``，
KimiDeltaAttention 显式传了 sigmoid）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FusedRMSNormGated"]

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "swish": F.silu,   # 与 silu 等价
}


class FusedRMSNormGated(nn.Module):
    """RMSNorm 后乘一个激活过的门控。

    Args:
        hidden_size: 归一化的最后一维（KDA 里是 ``head_v_dim``，按头归一化）。
        elementwise_affine: 是否带可学习的 ``weight``（fla 默认 True）。
        eps: 方差下界。
        activation: ``"sigmoid"`` / ``"silu"`` / ``"swish"``。KDA 用 sigmoid。
        bias: 是否带可学习的 ``bias``（fla 的实现里 bias 可选，默认不带）。
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = True,
                 eps: float = 1e-5, activation: str = "swish", bias: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation 只支持 {sorted(_ACTIVATIONS)}，收到 {activation!r}"
            )
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self._act = _ACTIVATIONS[activation]
        factory = dict(device=device, dtype=dtype)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, **factory))
        else:
            self.register_parameter("weight", None)
        if bias:
            if not elementwise_affine:
                raise ValueError("bias 需要 elementwise_affine=True")
            self.bias = nn.Parameter(torch.zeros(hidden_size, **factory))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[..., hidden_size]``，被归一化的一侧。
            gate: 与 ``x`` 同形状，门控的一侧。

        Returns:
            与 ``x`` 同形状同 dtype。
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"x 的最后一维应为 {self.hidden_size}，收到 {tuple(x.shape)}")
        if gate.shape != x.shape:
            raise ValueError(f"gate 应与 x 同形状 {tuple(x.shape)}，收到 {tuple(gate.shape)}")
        dtype = x.dtype
        # 归一化在 fp32 下算 —— bf16 上求平方和会丢有效位
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        if self.bias is not None:
            y = y + self.bias.float()
        return (y * self._act(gate.float())).to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"{self.hidden_size}, eps={self.eps}, activation={self.activation}, "
                f"elementwise_affine={self.weight is not None}")
