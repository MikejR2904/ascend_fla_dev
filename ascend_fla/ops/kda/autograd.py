"""把 KDA 的前向/反向接成 ``torch.autograd.Function``。

对外入口是 :func:`chunk_kda`，签名与 fla 的 ``fla.ops.kda.chunk_kda`` **前五个位置参数
一致**（q, k, v, g, beta），方便对照；但本仓是独立算子库，不接 fla 的 dispatch
（AGENTS.md §1），所以不承诺它那一长串 ``use_*_in_kernel`` 开关。

**与 fla 的语义差异 —— 调用方必须自己做的事。** fla 的 KDA layer 调 ``chunk_kda`` 时
传 ``use_qk_l2norm_in_kernel=True`` / ``use_gate_in_kernel=True`` /
``use_beta_sigmoid_in_kernel=True``，即这三步在它的 kernel 里做。ascriptor 的 kda
kernel **都不做**，所以：

* ``q`` / ``k`` 要调用方先做 L2 归一化（见 ``docs/matrix/gaps.json`` 的
  ``qk-l2norm-not-in-kernel``）；
* ``g`` 要调用方先算成 ``-exp(A_log) * softplus(g_raw + dt_bias)``；
* ``beta`` 要调用方先过 ``sigmoid``。

``ascend_fla.layers.kda`` 按这个约定做了，所以**层级用户不需要关心**；直接调本模块
的人需要。这些都不在门控能检查的范围内（数值上合法的输入无法区分做过没做过），所以
写在这里而不是塞进 ``_check``。

反向依赖九个前向检查点，forward 里一并算出并 ``save_for_backward``。这比重算前向省
一次完整前向，代价是显存：检查点里 ``h`` 是 ``[B,C,HV,128,128]``，在
kimi_linear_layer 形状（B1/HV32/C16）下是 16MB（bf16）。
"""
from __future__ import annotations

import torch

from .chunk import BWD_CACHE_NAMES, HEAD_DIM, VALUE_DIM, chunk_kda_fwd_with_caches
from .chunk_bwd import chunk_kda_bwd

__all__ = ["chunk_kda"]


def _zeros_like_npu(shape, dtype, device) -> torch.Tensor:
    """在 NPU 上造零张量。

    ``torch.zeros(device="npu")`` 在内置算子包不全的机器上不可用（需要 ZerosLike），
    所以在 CPU 上造好再 H2D —— 见 AGENTS.md §5 的可用面表。
    """
    return torch.zeros(*shape, dtype=dtype, device="cpu").to(device)


class _ChunkKDA(torch.autograd.Function):
    """KDA 的 autograd 封装。不支持二阶导（检查点不参与建图）。"""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state,
                device, block_dim, layout_device, check_gate_range):
        o, final_state, caches = chunk_kda_fwd_with_caches(
            q, k, v, g, beta, scale, initial_state,
            device=device, block_dim=block_dim, layout_device=layout_device,
            check_gate_range=check_gate_range,
        )
        # beta 在前向 ABI 里是 fp32、反向 ABI 里是 bf16。这一步降精度的代价已量化
        # （见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），**显式**做，不当无害的类型适配。
        ctx.save_for_backward(q, k, v, beta.bfloat16(), *(caches[n] for n in BWD_CACHE_NAMES))
        ctx.bwd_options = dict(device=device, block_dim=block_dim)
        ctx.state_shape = (q.shape[0], v.shape[2], HEAD_DIM, VALUE_DIM)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, beta_bf16, *cache_list = ctx.saved_tensors
        caches = dict(zip(BWD_CACHE_NAMES, cache_list))

        # 上游梯度的 dtype/连续性都不能假定：下游算子可能升到 fp32，也可能给非连续视图
        do = do.contiguous().bfloat16()
        if dht is None:
            # final_state 没参与 loss。反向 kernel 没有"省略 dht"的入口，只能喂零。
            dht = _zeros_like_npu(ctx.state_shape, torch.bfloat16, do.device)
        else:
            dht = dht.contiguous().bfloat16()

        grads = chunk_kda_bwd(q=q, k=k, v=v, beta=beta_bf16, do=do, dht=dht,
                              caches=caches, **ctx.bwd_options)

        # 反向 ABI 全出 bf16，而 g / beta / initial_state 的前向入口是 fp32 ——
        # 升回去，否则 autograd 会因 dtype 不匹配报错。
        # 用 ctx.needs_input_grad 而不是在 forward 里记 requires_grad：前者看的是整张图，
        # 后者只看那一刻的叶子标记。顺序与 forward 的参数表一致。
        need = ctx.needs_input_grad
        return (
            grads["dq"] if need[0] else None,
            grads["dk"] if need[1] else None,
            grads["dv"] if need[2] else None,
            grads["dg"].float() if need[3] else None,
            grads["dbeta"].float() if need[4] else None,
            None,                                              # scale
            grads["dh0"].float() if need[6] else None,          # initial_state
            # output_final_state / device / block_dim / layout_device / check_gate_range
            None, None, None, None, None,
        )


def chunk_kda(
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
    layout_device: str = "auto",
    check_gate_range: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """可求导的 KDA 分块注意力。

    Args:
        q, k: ``[B, T, H, 128]`` bfloat16。**调用方需已做 L2 归一化**（见模块 docstring）。
        v: ``[B, T, HV, 128]`` bfloat16，``HV % H == 0``。
        g: ``[B, T, HV, 128]`` float32，log 空间的 per-channel 衰减**增量**。
            **调用方需已算成** ``-exp(A_log) * softplus(g_raw + dt_bias)``。
        beta: ``[B, T, HV]`` float32，**调用方需已过** ``sigmoid``。
        scale: q 的缩放，默认 ``128 ** -0.5``。
        initial_state: ``[B, HV, 128, 128]`` float32，可选、可求导。
            注意布局是 **K 在前**；fla 的 KDA layer 用 ``state_v_first=True``，
            即 V 在前 —— 对接它的 cache 要转置（见 gaps.json 的 ``state-layout-k-first``）。
        output_final_state: 是否返回末态。
        device: ascriptor 设备名。
        block_dim: 启动核组数，只接受 ``SUPPORTED_BLOCK_DIM``。
        layout_device: 见 :func:`~ascend_fla.ops.kda.chunk.chunk_kda_fwd`。
        check_gate_range: 校验 chunk 内门控跨度不超过
            :data:`~ascend_fla.ops.kda.chunk.MAX_GATE_SPAN`（默认 80）。超限时 kernel 会
            fp32 上溢吐 NaN。**注意 fla 默认初始化的 KDA 层会越过这条线**（跨度约 94），
            见 ``gaps.json`` 的 ``gate-range-beyond-declared``。

    Returns:
        ``(o, final_state)``。``o`` 为 ``[B, T, HV, 128]`` bfloat16。

    Raises:
        ValueError: 任何定尺/dtype/设备约束不满足。绝不静默降级（AGENTS.md §7）。
    """
    return _ChunkKDA.apply(q, k, v, g, beta, scale, initial_state, output_final_state,
                           device, block_dim, layout_device, check_gate_range)
