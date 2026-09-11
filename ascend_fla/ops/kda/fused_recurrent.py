"""KDA 的 decode 路径（逐 token 递推）。

**状态：探路原型。** kernel 是本仓自写的（`kernels/projects/a5/kda_fused_recurrent`），
不是 ascriptor 资产的改写 —— 那边整族没有 recurrent 单元（`gaps.json` 的
`fused-recurrent-missing`）。语义基准是 `fla.ops.kda.fused_recurrent_kda`，
本仓的 oracle 是 `reference/kda.py` 的 `kda_recurrent_ref`。

**什么时候用它、什么时候用 chunk**：这条路把 T 个 token 逐个推，state 常驻 UB，
所以代价与 T 成正比但与"对齐到 64"无关。chunk 那条把 64 个 token 批成矩阵乘，
单位 token 便宜得多，但 T 必须是 64 的倍数（`no-tail-path`）。所以：

* T=1（纯 decode）、T=2~8（投机解码）→ 本路径。chunk 在 T=1 时要补到 64，白做 64 倍。
* T≥64 且是 64 的倍数 → chunk 路径。

**ABI 与 chunk 路径的差别**（刻意的）：本 kernel 收 **BHV-major** 的
``[B,HV,T,128]`` 而不是 token-major，且全部 fp32。理由：decode 的 q/k/v/g 合计才
几 KB，而 state 是 64KB/头 —— 省 cast 和省跨步读比省那几 KB 带宽重要。
布局转换与 GQA 的头扩展在 host 侧做一次。
"""
from __future__ import annotations

import functools
import pathlib
import sys
from typing import Any

import torch

from .chunk import HEAD_DIM, VALUE_DIM, _load_kernel

#: 本单元声明的 block_dim。**比 chunk 路径的 4 宽** —— 这个 kernel 只用向量核、
#: 不碰 cube，而 ``GetVecNum() == 2 * block_dim``，物理上有 56 个向量核。
#: 上限仍要实测（超过物理核数会在硬件 barrier 死锁，见 AGENTS.md §5）。
SUPPORTED_BLOCK_DIM = (1, 2, 4, 8, 16, 28)


def _unit_root() -> pathlib.Path:
    repo = pathlib.Path(__file__).resolve().parents[3]
    root = repo / "kernels/projects/a5/kda_fused_recurrent"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(f"找不到本仓的 kda_fused_recurrent 单元（试了 {root}）")
    return root


@functools.lru_cache(maxsize=1)
def kda_fused_recurrent_kernel() -> Any:
    return _load_kernel("fr", _unit_root() / "kernels", "step",
                        "kda_fused_recurrent_kernel")


@functools.lru_cache(maxsize=None)
def _compiled(device: str, block_dim: int) -> Any:
    from ...runtime.compile import compile_kernel

    return compile_kernel(kda_fused_recurrent_kernel(), device=device,
                          block_dim=block_dim)


def _check(q, k, v, g, beta, initial_state, block_dim) -> tuple[int, int, int, int]:
    """门控。返回 ``(B, T, H, HV)``。不满足就报错，绝不静默降级（AGENTS.md §7）。"""
    if block_dim not in SUPPORTED_BLOCK_DIM:
        raise ValueError(
            f"block_dim 只支持 {SUPPORTED_BLOCK_DIM}，收到 {block_dim}；"
            f"本单元只用向量核（GetVecNum() == 2*block_dim，物理 56 个），"
            f"但超过物理核数会在硬件 barrier 死锁"
        )
    if q.dim() != 4:
        raise ValueError(f"q 应为 4 维 [B,T,H,D]，收到 {tuple(q.shape)}")
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[3]
    if kd != HEAD_DIM or vd != VALUE_DIM:
        raise ValueError(f"定尺要求 K=V={HEAD_DIM}，收到 K={kd} V={vd}")
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    if k.shape != q.shape:
        raise ValueError(f"k 的形状应与 q 相同，收到 {tuple(k.shape)} vs {tuple(q.shape)}")
    if v.shape != (b, t, hv, vd):
        raise ValueError(f"v 应为 [B,T,HV,V]，收到 {tuple(v.shape)}")
    if g.shape != (b, t, hv, kd):
        raise ValueError(f"g 应为 [B,T,HV,K]，收到 {tuple(g.shape)}")
    if beta.shape != (b, t, hv):
        raise ValueError(f"beta 应为 [B,T,HV]，收到 {tuple(beta.shape)}")
    if initial_state is not None and initial_state.shape != (b, hv, kd, vd):
        raise ValueError(
            f"initial_state 应为 [B,HV,K,V]={(b, hv, kd, vd)}（K 在前），"
            f"收到 {tuple(initial_state.shape)}；fla 的 KDA layer 用 V 在前，"
            f"见 gaps.json 的 state-layout-k-first"
        )
    # T 的上限由 kernel 里流式缓冲的行数定（step.py 的 T_MAX）。超了要分批调用 ——
    # 报错而不是自动分批：自动分批会把一次调用的语义悄悄变成多次，state 串接的
    # 正确性得另外验，那不是这层该默默做的决定。
    if t > T_MAX:
        raise ValueError(
            f"一次调用最多 {T_MAX} 个 token（kernel 内流式缓冲的行数），收到 T={t}；"
            f"更长的序列请走 chunk 路径，或自己按 {T_MAX} 分批并串接 state"
        )
    return b, t, h, hv


#: 与 ``kernels/.../step.py`` 里流式缓冲的行数一致。改那边要同时改这里 ——
#: 由 tests/test_kda_decode.py 锁住。
T_MAX = 16


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    *,
    device: str = "a5",
    block_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐 token 递推的 KDA 前向（decode）。

    Args:
        q, k: ``[B,T,H,128]``；v, g: ``[B,T,HV,128]``；beta: ``[B,T,HV]``。
            dtype 任意浮点 —— 内部统一升到 fp32（decode 的量很小，见模块文档）。
        scale: q 的缩放，默认 ``128 ** -0.5``。**在 host 侧预乘进 q**。
        initial_state: ``[B,HV,128,128]`` float32，K 在前。

    Returns:
        ``(o, final_state)``，``o`` 为 ``[B,T,HV,128]``（与输入同 dtype），
        ``final_state`` 为 ``[B,HV,128,128]`` float32 或 ``None``。
    """
    b, t, h, hv = _check(q, k, v, g, beta, initial_state, block_dim)
    out_dtype = v.dtype
    sc = HEAD_DIM ** -0.5 if scale is None else float(scale)
    groups = hv // h
    dev = q.device

    def bhv(x: torch.Tensor) -> torch.Tensor:
        """``[B,T,HV,D] -> [B,HV,T,D]``，fp32 连续。"""
        return x.to(torch.float32).permute(0, 2, 1, 3).contiguous()

    qs = bhv(q.repeat_interleave(groups, dim=2) * sc) if groups > 1 else bhv(q * sc)
    kk = bhv(k.repeat_interleave(groups, dim=2)) if groups > 1 else bhv(k)
    vv, gg = bhv(v), bhv(g)
    bb = beta.to(torch.float32).permute(0, 2, 1).contiguous().view(b, hv, 1, t)
    state0 = (initial_state if initial_state is not None
              else torch.zeros(b, hv, HEAD_DIM, VALUE_DIM, dtype=torch.float32,
                               device="cpu").to(dev))

    o_bhv = torch.empty(b, hv, t, VALUE_DIM, dtype=torch.float32, device=dev)
    final_state = torch.empty(b, hv, HEAD_DIM, VALUE_DIM, dtype=torch.float32, device=dev)
    _compiled(device, block_dim)(
        {"qs": qs, "k": kk, "v": vv, "g": gg, "beta": bb, "initial_state": state0},
        {"B": b, "HV": hv, "T": t, "head_dim": HEAD_DIM, "value_dim": VALUE_DIM},
        {"o": o_bhv, "final_state": final_state},
    )
    o = o_bhv.permute(0, 2, 1, 3).contiguous().to(out_dtype)
    return o, (final_state if output_final_state else None)
