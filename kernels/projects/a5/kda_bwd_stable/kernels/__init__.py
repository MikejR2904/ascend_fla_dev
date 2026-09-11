"""本仓自有的 KDA 反向 kernel：成对衰减的分解锚点由 g_last 改为 g_last/2。

只有 finalize_pre / finalize_post 两个 —— 其余七个复用 ascriptor 的 a5.kda_bwd。
"""
