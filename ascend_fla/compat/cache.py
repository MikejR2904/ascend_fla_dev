"""KDA cache 适配器 —— fla / HF 的 layer state ⇄ 本仓 ``layers/kda.py`` 的 cache 字典。

背景（`docs/matrix/gaps.json` 的 ``state-layout-k-first``）：

* 本仓的 ``recurrent_state`` 是 ``[B, HV, K, V]``，**K 在前**，fp32。
* fla 的 ``KimiDeltaAttention`` 调 ``chunk_kda`` / ``fused_recurrent_kda`` 时都传
  ``state_v_first=True``，于是它 cache 里的 state 是 ``[B, HV, V, K]``，**V 在前**
  （见 fla ``ops/kda/fused_recurrent.py``：``state_v_first`` 为真时
  ``final_state = q.new_empty(N, HV, V, K)``）。
* Kimi-Linear 的 ``K == V == 128``，所以**转置错了不会报形状错**，decode 第一步就用错
  状态起算，输出还是有限值、量级也正常。这是个静默失败面，本模块的存在就是为了把它
  变成显式的、可测的一步。

因为形状查不出方向，本模块的做法是：**源布局必须由调用方声明**（``state_v_first``），
默认 ``True`` 对齐 fla 的 KDA layer；能查的东西一律查死（ndim / dtype / batch 一致性 /
conv 三元组 / 声明的 ``head_k_dim`` 与 ``head_v_dim``），不满足就报错并写出实际值
（``AGENTS.md`` §7：绝不静默降级、绝不"近似等价"地悄悄替换）。

两条约定，写在这里省得下一个人再翻源码：

* **state 必须是 fp32**。给别的 dtype 直接报错，不做隐式转换 —— state 是跨 chunk 的
  累积量，本仓约定 FP32（``docs/matrix`` 的 ``state-dtype-bf16``）。
* **conv_state 两侧布局相同**：都是 ``(q, k, v)`` 三个 ``[B, D, W]``，最近的 token 在
  **最后一个下标**，``T < W`` 时左侧补零（fla ``modules/conv/short_conv.py`` 的
  ``cache.roll(shifts=-1, dims=-1); cache[:, :, -1] = x``；本仓
  ``modules/convolution.py`` 的 ``hist[..., -w:]`` + 左侧 pad）。所以这一半是**校验**
  而不是重排 —— 但它照样要测，漏传 conv_state 只会坏掉前 ``W-1`` 个 token。

**本模块不 import fla**（``AGENTS.md`` §1：fla 只在测试期依赖）。适配器按上面这份约定
写死，再由 ``tests/test_cache_adapter.py`` 拿 fla 去验它。

别名语义：两个方向**都返回新的连续张量**，不与入参共享存储。写回上游要显式调另一个方向，
不会因为"有时是视图、有时是拷贝"而出现只在某个布局下成立的原地更新。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

__all__ = [
    "FLA_KDA_STATE_V_FIRST",
    "CONV_STATE_STREAMS",
    "to_ours",
    "to_fla",
]

#: fla 的 ``KimiDeltaAttention`` 固定传 ``state_v_first=True``（fla/layers/kda.py）。
FLA_KDA_STATE_V_FIRST = True

#: conv_state 是 ``(q, k, v)`` 三路，顺序不能换。
CONV_STATE_STREAMS = ("q", "k", "v")

_OURS_KEYS = ("recurrent_state", "conv_state")


def _shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(x.shape)


def _resolve_layer_state(source, layer_idx: int | None) -> Mapping:
    """把 ``Cache`` 容器或 layer state 字典归一成一个 mapping。"""
    if layer_idx is not None:
        if isinstance(source, Mapping):
            raise TypeError(
                "给了 layer_idx 就说明 source 是 Cache 容器，但收到的是 mapping "
                f"（键 {sorted(source)}）。要么去掉 layer_idx，要么传 Cache 本身。"
            )
        try:
            source = source[layer_idx]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"从 {type(source).__name__} 取 layer_idx={layer_idx} 失败：{exc}"
            ) from exc
    if not isinstance(source, Mapping):
        raise TypeError(
            f"layer state 应为 mapping（fla 的 layer state 字典），收到 {type(source).__name__}。"
            " 传的是 Cache 容器的话要同时给 layer_idx。"
        )
    return source


def _check_state_tensor(
    state: torch.Tensor,
    *,
    name: str,
    dim0_name: str,
    dim1_name: str,
    dim0: int | None,
    dim1: int | None,
) -> None:
    """state 的通用校验：类型 / ndim / dtype / 声明的两个尾维。"""
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{name} 应为 torch.Tensor，收到 {type(state).__name__}")
    if state.dim() != 4:
        raise ValueError(
            f"{name} 应为 4 维 [B, HV, {dim0_name}, {dim1_name}]，"
            f"收到 {state.dim()} 维 {_shape(state)}"
        )
    if state.dtype is not torch.float32:
        raise ValueError(
            f"{name} 必须是 fp32（state 是跨 chunk 的累积量，本仓不做隐式 dtype 转换），"
            f"收到 {state.dtype}，形状 {_shape(state)}"
        )
    if dim0 is not None and state.shape[-2] != dim0:
        raise ValueError(
            f"{name} 的倒数第二维应为 {dim0_name}={dim0}，收到 {state.shape[-2]}"
            f"（完整形状 {_shape(state)}）"
        )
    if dim1 is not None and state.shape[-1] != dim1:
        raise ValueError(
            f"{name} 的最后一维应为 {dim1_name}={dim1}，收到 {state.shape[-1]}"
            f"（完整形状 {_shape(state)}）"
        )


def _check_conv_state(conv_state, *, name: str, batch: int | None):
    """conv_state 校验：三元组、每项 ``[B, D, W]``、三路同形同 dtype。"""
    if conv_state is None:
        return None
    if isinstance(conv_state, torch.Tensor) or not isinstance(conv_state, Sequence):
        raise TypeError(
            f"{name} 应为 (q, k, v) 三项的 tuple/list，收到 {type(conv_state).__name__}"
        )
    if len(conv_state) != len(CONV_STATE_STREAMS):
        raise ValueError(
            f"{name} 应为 {CONV_STATE_STREAMS} 三项，收到 {len(conv_state)} 项"
        )
    present = [(tag, t) for tag, t in zip(CONV_STATE_STREAMS, conv_state) if t is not None]
    if not present:
        return tuple(None for _ in CONV_STATE_STREAMS)
    if len(present) != len(CONV_STATE_STREAMS):
        missing = [tag for tag, t in zip(CONV_STATE_STREAMS, conv_state) if t is None]
        raise ValueError(
            f"{name} 的三路要么都给要么都不给，缺了 {missing}"
            "（漏传一路只会坏掉前 W-1 个 token，不会报形状错）"
        )
    ref_tag, ref = present[0]
    for tag, t in present:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name}[{tag}] 应为 torch.Tensor，收到 {type(t).__name__}")
        if t.dim() != 3:
            raise ValueError(
                f"{name}[{tag}] 应为 3 维 [B, D, W]，收到 {t.dim()} 维 {_shape(t)}"
            )
        if _shape(t) != _shape(ref):
            raise ValueError(
                f"{name} 三路形状要一致：[{ref_tag}] 是 {_shape(ref)}，[{tag}] 是 {_shape(t)}"
            )
        if t.dtype is not ref.dtype:
            raise ValueError(
                f"{name} 三路 dtype 要一致：[{ref_tag}] 是 {ref.dtype}，[{tag}] 是 {t.dtype}"
            )
        if batch is not None and t.shape[0] != batch:
            raise ValueError(
                f"{name}[{tag}] 的 batch 是 {t.shape[0]}，与 recurrent_state 的 {batch} 对不上"
                f"（完整形状 {_shape(t)}）"
            )
    return tuple(t.contiguous().clone() for t in conv_state)


def _reject_unsupported(layer_state: Mapping) -> None:
    """KDA 用不到 attn_state / ffn_state；带了就报错，不静默丢。"""
    for key in ("attn_state", "ffn_state"):
        value = layer_state.get(key)
        if value is not None:
            raise ValueError(
                f"layer state 里的 {key} 非空（{type(value).__name__}），"
                "本适配器只搬 recurrent_state 与 conv_state，不会把它带过去。"
                " KDA 层不该有这个字段 —— 请确认 layer_idx 指到的是 KDA 层。"
            )


def to_ours(
    source,
    *,
    layer_idx: int | None = None,
    state_v_first: bool = FLA_KDA_STATE_V_FIRST,
    head_k_dim: int | None = None,
    head_v_dim: int | None = None,
) -> dict:
    """fla / HF 的 KDA layer state → 本仓 ``layers/kda.py`` 的 cache 字典。

    Args:
        source: fla 的 layer state 字典（``past_key_values[layer_idx]``），或者
            ``Cache`` 容器本身（这时要给 ``layer_idx``）。
        layer_idx: 给了就先从 ``source`` 里取这一层。
        state_v_first: **源** state 的布局。``True``（默认，对齐 fla 的
            ``KimiDeltaAttention``）表示源是 ``[B, HV, V, K]``；``False`` 表示源已经是
            ``[B, HV, K, V]``。K==V 时形状查不出方向，所以这一项必须由调用方声明。
        head_k_dim / head_v_dim: 给了就校验对应的维度。K != V 时这是唯一能抓住
            "布局声明错了"的检查，**接线时建议一律传**。

    Returns:
        ``{"recurrent_state": [B, HV, K, V] fp32 或 None,
        "conv_state": (q, k, v) 或 None}``，都是新的连续张量。
    """
    layer_state = _resolve_layer_state(source, layer_idx)
    if "recurrent_state" not in layer_state:
        raise KeyError(
            f"layer state 里没有 recurrent_state 键，实际的键是 {sorted(layer_state)}"
        )
    _reject_unsupported(layer_state)

    state = layer_state["recurrent_state"]
    batch = None
    if state is not None:
        if state_v_first:
            _check_state_tensor(
                state,
                name="fla recurrent_state（state_v_first=True）",
                dim0_name="V",
                dim1_name="K",
                dim0=head_v_dim,
                dim1=head_k_dim,
            )
            state = state.transpose(-1, -2)
        else:
            _check_state_tensor(
                state,
                name="fla recurrent_state（state_v_first=False）",
                dim0_name="K",
                dim1_name="V",
                dim0=head_k_dim,
                dim1=head_v_dim,
            )
        state = state.contiguous().clone()
        batch = state.shape[0]

    conv_state = _check_conv_state(
        layer_state.get("conv_state"), name="fla conv_state", batch=batch
    )
    return {"recurrent_state": state, "conv_state": conv_state}


def to_fla(
    ours: Mapping,
    *,
    state_v_first: bool = FLA_KDA_STATE_V_FIRST,
    head_k_dim: int | None = None,
    head_v_dim: int | None = None,
) -> dict:
    """本仓 cache 字典 → fla / HF 的 KDA layer state 字典（``to_ours`` 的逆）。

    Args:
        ours: 本仓 ``layers/kda.py`` 的 cache 字典，``recurrent_state`` 为
            ``[B, HV, K, V]`` fp32。
        state_v_first: **目标** 布局，含义同 :func:`to_ours`。
        head_k_dim / head_v_dim: 给了就校验。

    Returns:
        ``{"recurrent_state": …, "conv_state": …}``，可以直接喂给 fla 的
        ``Cache.update(layer_idx=…, **返回值)``。
    """
    if not isinstance(ours, Mapping):
        raise TypeError(f"ours 应为本仓的 cache 字典，收到 {type(ours).__name__}")
    unknown = set(ours) - set(_OURS_KEYS)
    if unknown:
        raise ValueError(
            f"本仓 cache 只认 {_OURS_KEYS} 两个键，多出来的 {sorted(unknown)} 不会被搬过去"
        )
    if "recurrent_state" not in ours:
        raise KeyError(f"cache 里没有 recurrent_state 键，实际的键是 {sorted(ours)}")

    state = ours["recurrent_state"]
    batch = None
    if state is not None:
        _check_state_tensor(
            state,
            name="本仓 recurrent_state",
            dim0_name="K",
            dim1_name="V",
            dim0=head_k_dim,
            dim1=head_v_dim,
        )
        batch = state.shape[0]
        if state_v_first:
            state = state.transpose(-1, -2)
        state = state.contiguous().clone()

    conv_state = _check_conv_state(
        ours.get("conv_state"), name="本仓 conv_state", batch=batch
    )
    return {"recurrent_state": state, "conv_state": conv_state}
