"""layers — 窄切片的 nn.Module 层。fla 有 43 个，这里只做用得到的。"""

from .kda import KimiDeltaAttention

__all__ = ["KimiDeltaAttention"]
