"""compat — 可选的 fla 风格签名 wrapper。

fla 用 [B,T,H,K]，本仓算子用定尺 [B,H,C,64,128]。布局转换有访存代价，
所以是可选便利层而非默认路径。

目前有一件东西：:mod:`ascend_fla.compat.cache` —— fla/HF 的 KDA layer state 与本仓
``layers/kda.py`` 的 cache 字典互转（``state-layout-k-first``）。
"""
from ascend_fla.compat.cache import (
    CONV_STATE_STREAMS,
    FLA_KDA_STATE_V_FIRST,
    to_fla,
    to_ours,
)

__all__ = [
    "CONV_STATE_STREAMS",
    "FLA_KDA_STATE_V_FIRST",
    "to_fla",
    "to_ours",
]
