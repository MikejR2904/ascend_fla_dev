"""GDN-2 的纯 torch fp32-state recurrence oracle。

这是模型 canonical/packed 两种布局以及后续 A5 recurrent/chunk kernel 的共同语义基准。
函数只做明确的数学递推；设备由输入 tensor 决定，因此 CPU 与 torch_npu 都可运行。
"""
from __future__ import annotations

import torch

__all__ = ["gdn2_recurrent_reference"]


def _qk_float(x: torch.Tensor, *, normalize: bool, eps: float) -> torch.Tensor:
    xf = x.float()
    if not normalize:
        return xf
    return xf * torch.rsqrt(xf.square().sum(dim=-1, keepdim=True) + eps)


def gdn2_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    use_qk_l2norm: bool,
    qk_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 token 执行 GDN-2 recurrence。

    ``q/k/g/b`` 为 ``[B,T,H,K]``，``v/w`` 为 ``[B,T,H,V]``，state 为
    ``[B,H,K,V]``。state 和所有累积计算固定使用 fp32，输出回写为 q 的 dtype。
    """
    if q.ndim != 4:
        raise ValueError(f"q 应为 [B,T,H,K]，收到 {tuple(q.shape)}")
    batch, time, heads, key_dim = q.shape
    if time == 0:
        raise ValueError("GDN-2 不接受空序列")
    if k.shape != q.shape or g.shape != q.shape or b.shape != q.shape:
        raise ValueError(
            "k/g/b 应与 q 同形状，实际为 "
            f"q={tuple(q.shape)} k={tuple(k.shape)} g={tuple(g.shape)} b={tuple(b.shape)}"
        )
    if v.ndim != 4 or v.shape[:3] != (batch, time, heads):
        raise ValueError(f"v 应为 [B,T,H,V] 且前三维匹配 q，收到 {tuple(v.shape)}")
    if w.shape != v.shape:
        raise ValueError(f"w 应与 v 同形状，收到 {tuple(w.shape)} vs {tuple(v.shape)}")

    value_dim = v.shape[-1]
    expected_state = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        state = torch.zeros(expected_state, device=q.device, dtype=torch.float32)
    else:
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(
                f"recurrent_state 应为 {expected_state}，收到 {tuple(initial_state.shape)}"
            )
        state = initial_state.to(device=q.device, dtype=torch.float32)

    outputs: list[torch.Tensor] = []
    for index in range(time):
        q_t = _qk_float(q[:, index], normalize=use_qk_l2norm, eps=qk_norm_eps) * scale
        k_t = _qk_float(k[:, index], normalize=use_qk_l2norm, eps=qk_norm_eps)
        v_t = v[:, index].float()
        g_t = g[:, index].float()
        b_t = b[:, index].float()
        w_t = w[:, index].float()

        state = state * torch.exp(g_t).unsqueeze(-1)
        erase = torch.matmul((b_t * k_t).unsqueeze(-2), state).squeeze(-2)
        update = w_t * v_t - erase
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(-2)
        out = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)
        outputs.append(out.to(q.dtype))
    return torch.stack(outputs, dim=1), state
