"""reference — 精度 oracle 与性能基线。

双 oracle：fla 的 naive.py（CPU fp32，语义权威）与 torch_npu 组合实现
（同时是性能基线）。两者的差异本身就是有用信息。
"""

from .gdn2 import gdn2_recurrent_reference

__all__ = ["gdn2_recurrent_reference"]
