"""modules — 窄切片的通用模块。只做算子倒推用得到的，不照搬 fla.modules。

⚠️ 这里的模块都是 **torch / torch_npu 原生算子**实现的，不是本仓自编译的算子
（见 ``docs/matrix/gaps.json`` 的 ``modules-are-torch-not-kernels``）。它们存在的目的
是把层拼起来、让层级梯度能端到端验证，不在"高效率算子"的承诺范围内。
"""

from .convolution import PackedShortConvolution, ShortConvolution
from .fused_norm_gated import FusedRMSNormGated

__all__ = ["FusedRMSNormGated", "PackedShortConvolution", "ShortConvolution"]
