"""KDA 参考实现 —— 精度 oracle 与性能基线。

三个实现，用途不同：

``kda_recurrent_ref``
    逐 token 递推。语义最直白，是 chunk 路径的独立 oracle。
``kda_chunk_ref``
    分块路径，忠实移植 fla 的 ``naive_chunk_kda``（含两处 O(BT) 列循环）。
    正确性优先，**不要拿它测性能**。
``kda_chunk_vectorized``
    把那两处列循环向量化，用作 torch_npu 性能基线。数学上与 ``kda_chunk_ref``
    等价，但对 g 的动态范围有要求 —— 见该函数的说明。

ABI 与 fla 的 ``fla.ops.kda`` 一致，也与 ascriptor ``a5.kda_fwd`` 的公开张量一致：

====================  ==========================  ==========
张量                  形状                        说明
====================  ==========================  ==========
``q`` / ``k``         ``[B, T, H, K]``
``v``                 ``[B, T, HV, V]``           ``HV % H == 0``
``g``                 ``[B, T, HV, K]``           log 空间的 per-dimension 衰减增量
``beta``              ``[B, T, HV]``
``initial_state``     ``[B, HV, K, V]``
``o``                 ``[B, T, HV, V]``
``final_state``       ``[B, HV, K, V]``
====================  ==========================  ==========

``scale`` 默认 ``K ** -0.5``。ascriptor 的 ABI 没有 scale 入口（见 docs/matrix
的 ``scale-param-no-slot``），对齐时要显式传 ``scale=1.0`` 或在 host 侧预乘 q。
"""
from __future__ import annotations

import torch

__all__ = [
    "kda_recurrent_ref",
    "kda_chunk_ref",
    "kda_chunk_vectorized",
]


def _to_chunks(x: torch.Tensor, bt: int) -> torch.Tensor:
    """``[B, T, H, *rest] -> [B, H, NT, BT, *rest]``（等价 einops 的 ``b (n c) h ... -> b h n c ...``）。"""
    b, t, h, *rest = x.shape
    x = x.reshape(b, t // bt, bt, h, *rest)
    # (b, n, c, h, *rest) -> (b, h, n, c, *rest)
    perm = (0, 3, 1, 2, *range(4, 4 + len(rest)))
    return x.permute(*perm)


def _from_chunks(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, NT, BT, D] -> [B, NT*BT, H, D]``（等价 ``b h n c d -> b (n c) h d``）。"""
    b, h, n, c, d = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(b, n * c, h, d)


def _expand_qk_heads(x: torch.Tensor, groups: int, dim: int) -> torch.Tensor:
    """把 qk 头扩展到 value 头数（GVA：``HV = H * groups``）。"""
    return x.repeat_interleave(groups, dim=dim) if groups > 1 else x


def kda_recurrent_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐 token 递推的 KDA。chunk 路径的独立 oracle。

    在 fp32 下计算，返回值转回 ``v`` 的 dtype。
    """
    dtype = v.dtype
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[-1]
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    groups = hv // h
    if scale is None:
        scale = kd ** -0.5

    q, k, v, g, beta = (x.to(torch.float32) for x in (q, k, v, g, beta))
    q = _expand_qk_heads(q, groups, dim=2) * scale
    k = _expand_qk_heads(k, groups, dim=2)

    state = q.new_zeros(b, hv, kd, vd)
    if initial_state is not None:
        state = state + initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    for i in range(t):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        state = state * g_i[..., None].exp()
        # delta 规则：用当前 state 预测 v，再按 beta 修正
        delta = v_i - (k_i[..., None] * state).sum(-2)
        state = state + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, delta)
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    return o.to(dtype), (state if output_final_state else None)


def kda_chunk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """分块 KDA，忠实移植 fla 的 ``naive_chunk_kda``。

    保留两处 O(BT) 的列循环（构造 ``A`` 与 ``Aqk``）—— 它们逐列用
    ``(g - g_i).exp()`` 的差分形式，避免 ``exp(-g)`` 单独出现。正确性优先，
    **性能基线请用** :func:`kda_chunk_vectorized`。
    """
    dtype = v.dtype
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[-1]
    bt = chunk_size
    if t % bt:
        raise ValueError(f"T({t}) 必须是 chunk_size({bt}) 的整数倍")
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    groups, nt = hv // h, t // bt
    if scale is None:
        scale = kd ** -0.5

    q, k = (_to_chunks(x, bt).to(torch.float32) for x in (q, k))
    v, g, beta = (_to_chunks(x, bt).to(torch.float32) for x in (v, g, beta))
    q = _expand_qk_heads(q, groups, dim=1) * scale
    k = _expand_qk_heads(k, groups, dim=1)
    g = g.cumsum(-2)

    # ---- WY 表示：A = (I - tril(beta * K G Kᵀ, -1))⁻¹ · beta ----
    inc_mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=0)
    a = torch.zeros(*g.shape[:-1], bt, dtype=torch.float32, device=q.device)
    for i in range(bt):
        k_i, g_i = k[..., i, :], g[..., i : i + 1, :]
        a[..., i] = torch.einsum("...cd,...d->...c", k * (g - g_i).exp(), k_i)
    a = a * beta[..., None]
    a = -a.masked_fill(inc_mask, 0)
    # 前向消元求严格下三角逆
    for i in range(1, bt):
        a[..., i, :i] = a[..., i, :i].clone() + (a[..., i, :, None].clone() * a[..., :, :i].clone()).sum(-2)
    a = (a + torch.eye(bt, dtype=torch.float32, device=q.device)) * beta[..., None, :]

    w = a @ (g.exp() * k)
    u = a @ v

    state = q.new_zeros(b, hv, kd, vd)
    if initial_state is not None:
        state = state + initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    strict_mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=1)
    for n in range(nt):
        q_n, k_n, u_n, g_n, w_n = q[:, :, n], k[:, :, n], u[:, :, n], g[:, :, n], w[:, :, n]
        aqk = torch.zeros(b, hv, bt, bt, dtype=torch.float32, device=q.device)
        for j in range(bt):
            k_j, g_j = k[:, :, n, j], g[:, :, n, j : j + 1, :]
            aqk[..., j] = torch.einsum("...cd,...d->...c", q_n * (g_n - g_j).exp(), k_j)
        aqk = aqk.masked_fill(strict_mask, 0)

        v_n = u_n - w_n @ state
        o[:, :, n] = (q_n * g_n.exp()) @ state + aqk @ v_n
        g_last = g_n[:, :, -1]
        state = state * g_last.exp().unsqueeze(-1)
        state = state + ((g_n[:, :, -1:] - g_n).exp() * k_n).transpose(-1, -2) @ v_n

    return _from_chunks(o).to(dtype), (state if output_final_state else None)


def kda_chunk_vectorized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    check_gate_range: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """向量化的分块 KDA，用作 torch_npu 性能基线。

    与 :func:`kda_chunk_ref` 数学等价，但把两处列循环改写成单个 matmul：

        A[c, i] = Σ_d k[c,d]·exp(g[c,d] − g[i,d])·k[i,d]
                = Σ_d (k[c,d]·exp(g[c,d] − m[d])) · (k[i,d]·exp(m[d] − g[i,d]))

    其中 ``m`` 是该 chunk 内 g 的逐维最大值（g 经 cumsum 后沿 chunk 单调不增，
    故 ``m`` 取首行）。减去 ``m`` 让两个因子的指数都 ≤ 0，避免 ``exp(-g)`` 溢出。

    ``check_gate_range=True`` 时校验 chunk 内 g 的跨度，超出 fp32 安全范围即报错
    而不是静默产出 inf —— 本仓的门控原则是不满足就报错。
    """
    dtype = v.dtype
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[-1]
    bt = chunk_size
    if t % bt:
        raise ValueError(f"T({t}) 必须是 chunk_size({bt}) 的整数倍")
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    groups, nt = hv // h, t // bt
    if scale is None:
        scale = kd ** -0.5

    q, k = (_to_chunks(x, bt).to(torch.float32) for x in (q, k))
    v, g, beta = (_to_chunks(x, bt).to(torch.float32) for x in (v, g, beta))
    q = _expand_qk_heads(q, groups, dim=1) * scale
    k = _expand_qk_heads(k, groups, dim=1)
    g = g.cumsum(-2)

    # g 沿 chunk 单调不增（增量非正），首行即逐维最大值
    g_max = g[..., :1, :]
    span = (g_max - g[..., -1:, :]).amax()
    if check_gate_range and span > 80.0:
        raise ValueError(
            f"chunk 内 g 的跨度 {span.item():.3f} 超出向量化实现的 fp32 安全范围(80)；"
            f"请改用 kda_chunk_ref，或减小 chunk_size"
        )

    g_rel = (g - g_max).exp()          # ≤ 1
    g_inv = (g_max - g).exp()          # ≥ 1，但受 span 约束

    # A[c, i] = <k[c]·g_rel[c], k[i]·g_inv[i]>
    a = (k * g_rel) @ (k * g_inv).transpose(-1, -2)
    inc_mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=0)
    a = a * beta[..., None]
    a = -a.masked_fill(inc_mask, 0)
    for i in range(1, bt):
        a[..., i, :i] = a[..., i, :i].clone() + (a[..., i, :, None].clone() * a[..., :, :i].clone()).sum(-2)
    a = (a + torch.eye(bt, dtype=torch.float32, device=q.device)) * beta[..., None, :]

    w = a @ (g.exp() * k)
    u = a @ v

    state = q.new_zeros(b, hv, kd, vd)
    if initial_state is not None:
        state = state + initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    strict_mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=1)
    for n in range(nt):
        q_n, k_n, u_n, g_n, w_n = q[:, :, n], k[:, :, n], u[:, :, n], g[:, :, n], w[:, :, n]
        gr_n, gi_n = g_rel[:, :, n], g_inv[:, :, n]
        # Aqk[c, j] = <q[c]·g_rel[c], k[j]·g_inv[j]>
        aqk = ((q_n * gr_n) @ (k_n * gi_n).transpose(-1, -2)).masked_fill(strict_mask, 0)

        v_n = u_n - w_n @ state
        o[:, :, n] = (q_n * g_n.exp()) @ state + aqk @ v_n
        g_last = g_n[:, :, -1]
        state = state * g_last.exp().unsqueeze(-1)
        state = state + ((g_n[:, :, -1:] - g_n).exp() * k_n).transpose(-1, -2) @ v_n

    return _from_chunks(o).to(dtype), (state if output_final_state else None)
