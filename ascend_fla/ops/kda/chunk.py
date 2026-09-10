"""KDA 分块前向 —— 把 ascriptor ``a5.kda_fwd`` 的五个 kernel 接到 runtime 桥上。

五段与 fla 的对应关系::

    gate      g_raw 的 chunk 内 cumsum -> eg          ~ ops/utils/cumsum
    scores    Aqk 与严格下三角 strict                  ~ ops/common/chunk_o + chunk_scaled_dot_kkt
    inverse   strict 的 64x64 严格下三角求逆 -> Akk     ~ ops/utils/solve_tril
    wy        WY 表示 -> w, u, qg, kg                  ~ kda/wy_fast
    recurrent chunk 递推 -> o, final_state              ~ ops/common/chunk_delta_h

公开 ABI 与 fla ``fla.ops.kda.chunk_kda`` 一致（token-major），内部自行 permute 到
kernel 的 BHCLD 布局 —— 与 ascriptor 单元 ``composition.chunked()`` 的做法相同。

硬约束（不满足直接报错，不静默降级，见 AGENTS.md §7）：
``L=64``、``K=V=128``、``T % 64 == 0``、``HV % H == 0``、q/k/v 为 bf16、g/beta/state 为 fp32。
"""
from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys
from typing import Any

import torch

__all__ = ["chunk_kda_fwd", "kda_fwd_kernels"]

L_PER_CHUNK = 64
HEAD_DIM = 128
VALUE_DIM = 128

_KERNEL_MODULES = {
    "gate": ("gate", "kda_sub1_gate_kernel"),
    "scores": ("intra", "kda_sub2_score_kernel"),
    "inverse": ("triangular_inverse", "tril_inverse64_v2_strict_bf16_kernel"),
    "wy": ("wy", "kda_sub3_wy_kernel"),
    "recurrent": ("recurrent", "kda_sub45_fused_kernel"),
}


def _kernels_root() -> pathlib.Path:
    """ascriptor kernels 仓里 ``projects/a5/kda_fwd`` 的位置。

    依次尝试 ``$ASCRIPTOR_KDA_FWD``、``$ASCRIPTOR_WORKSPACE/kernels/...``、同级
    ``../ascriptor/kernels/...``。我们只读不改（见 AGENTS.md §3）。
    """
    direct = os.environ.get("ASCRIPTOR_KDA_FWD")
    if direct:
        return pathlib.Path(direct)
    ws = os.environ.get("ASCRIPTOR_WORKSPACE")
    candidates = []
    if ws:
        candidates.append(pathlib.Path(ws) / "kernels/projects/a5/kda_fwd")
    repo = pathlib.Path(__file__).resolve().parents[3]
    candidates += [
        repo.parent / "ascriptor/kernels/projects/a5/kda_fwd",
        repo / "src/kernels/projects/a5/kda_fwd",
    ]
    for c in candidates:
        if (c / "kernels").is_dir():
            return c
    raise FileNotFoundError(
        "找不到 ascriptor 的 kda_fwd 单元；设 $ASCRIPTOR_KDA_FWD 指向它。已试："
        + ", ".join(str(c) for c in candidates)
    )


@functools.lru_cache(maxsize=1)
def kda_fwd_kernels() -> dict[str, Any]:
    """载入五个 kernel 定义。

    逐模块按文件加载，**不**经由单元的 ``kernels/__init__`` 之外的东西 ——
    ``composition.py`` 依赖单元本地的 ``_unit_runner``，那条路会把 harness 拖进来。
    """
    root = _kernels_root()
    pkg_dir = root / "kernels"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    out: dict[str, Any] = {}
    for stage, (module_name, fn_name) in _KERNEL_MODULES.items():
        spec = importlib.util.spec_from_file_location(f"_afla_kda_{module_name}", pkg_dir / f"{module_name}.py")
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"无法加载 {pkg_dir / f'{module_name}.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        out[stage] = getattr(mod, fn_name)
    return out


def _check(q, k, v, g, beta, initial_state) -> tuple[int, int, int, int]:
    """门控。返回 ``(B, H, HV, C)``。任何不满足都报错，绝不静默降级。"""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q/k/v 应为 4 维 [B,T,H,D]，收到 {q.shape} / {k.shape} / {v.shape}")
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[3]
    if kd != HEAD_DIM or vd != VALUE_DIM:
        raise ValueError(
            f"ascriptor a5.kda_fwd 定尺要求 K=V={HEAD_DIM}，收到 K={kd} V={vd}；"
            f"见 docs/matrix/gaps.json 的 fixed-kv-128"
        )
    if t % L_PER_CHUNK:
        raise ValueError(
            f"T 必须是 {L_PER_CHUNK} 的整数倍（无 tail 路径），收到 T={t}；"
            f"最近的合法值是 {t // L_PER_CHUNK * L_PER_CHUNK} 或 {(t // L_PER_CHUNK + 1) * L_PER_CHUNK}；"
            f"见 gaps.json 的 no-tail-path"
        )
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    if k.shape != q.shape:
        raise ValueError(f"k 的形状应与 q 相同，收到 {k.shape} vs {q.shape}")
    if g.shape != (b, t, hv, kd):
        raise ValueError(f"g 应为 [B,T,HV,K]={(b, t, hv, kd)}，收到 {tuple(g.shape)}")
    if beta.shape != (b, t, hv):
        raise ValueError(f"beta 应为 [B,T,HV]={(b, t, hv)}，收到 {tuple(beta.shape)}")
    for name, x, want in (("q", q, torch.bfloat16), ("k", k, torch.bfloat16), ("v", v, torch.bfloat16),
                          ("g", g, torch.float32), ("beta", beta, torch.float32)):
        if x.dtype != want:
            raise ValueError(f"{name} 的 dtype 应为 {want}（kda_fwd ABI），收到 {x.dtype}")
    if initial_state is not None:
        if initial_state.shape != (b, hv, HEAD_DIM, VALUE_DIM):
            raise ValueError(
                f"initial_state 应为 [B,HV,K,V]={(b, hv, HEAD_DIM, VALUE_DIM)}，"
                f"收到 {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state 的 dtype 应为 float32，收到 {initial_state.dtype}")
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if x.device.type != "npu":
            raise ValueError(f"{name} 应在 NPU 上，收到 device={x.device}")
    return b, h, hv, t // L_PER_CHUNK


@functools.lru_cache(maxsize=1)
def _npu_supports_d2d_copy() -> bool:
    """NPU 上的 ``permute().contiguous()``（device-to-device copy）是否可用。

    它走 torch_npu 的内置 copy 算子；内置算子包不覆盖当前 SoC 时会失败
    （见 docs/matrix/gaps.json 的 ``npu-builtin-ops-missing``）。探测一次并缓存。
    """
    try:
        probe = torch.arange(8, dtype=torch.float32).reshape(2, 4).to("npu")
        probe.permute(1, 0).contiguous()
        torch.npu.synchronize()
        return True
    except Exception:
        return False


def _to_bhcld(x: torch.Tensor, heads: int, *, on_cpu: bool) -> torch.Tensor:
    """token-major → kernel 的 BHCLD 布局。

    * 4 维 ``[B,T,heads,D] -> [B,heads,C,64,D]``
    * 3 维 ``[B,T,heads]   -> [B,heads,C,64]``

    ``on_cpu=True`` 时先 D2H、在 CPU 上重排、再 H2D —— 这条路只为绕开
    ``npu-builtin-ops-missing``，数值语义完全相同，但多两次主机往返，
    **不要用它的计时当性能数据**。
    """
    dev = x.device
    src = x.cpu() if on_cpu else x
    b, t = src.shape[0], src.shape[1]
    c = t // L_PER_CHUNK
    if src.dim() == 3:
        out = src.reshape(b, c, L_PER_CHUNK, heads).permute(0, 3, 1, 2).contiguous()
    elif src.dim() == 4:
        last = src.shape[3]
        out = src.reshape(b, c, L_PER_CHUNK, heads, last).permute(0, 3, 1, 2, 4).contiguous()
    else:
        raise ValueError(f"_to_bhcld 期望 3 或 4 维，收到 {tuple(src.shape)}")
    return out.to(dev) if on_cpu else out


def _from_bhcld(x: torch.Tensor, *, on_cpu: bool) -> torch.Tensor:
    """``[B,HV,C,64,D] -> [B,C*64,HV,D]``，与 :func:`_to_bhcld` 对称。"""
    dev = x.device
    src = x.cpu() if on_cpu else x
    b, hv, c, l, d = src.shape
    out = src.permute(0, 2, 3, 1, 4).reshape(b, c * l, hv, d).contiguous()
    return out.to(dev) if on_cpu else out


def chunk_kda_fwd(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA 分块前向。

    Args:
        q, k: ``[B, T, H, 128]`` bfloat16。
        v: ``[B, T, HV, 128]`` bfloat16，``HV % H == 0``。
        g: ``[B, T, HV, 128]`` float32，log 空间的 per-dimension 衰减**增量**
            （kernel 内部做 chunk 内 cumsum）。
        beta: ``[B, T, HV]`` float32。
        scale: q 的缩放，默认 ``128 ** -0.5``。kernel 有 f32 标量入口，不需要 host 预乘。
        initial_state: ``[B, HV, 128, 128]`` float32，可选。
        output_final_state: 是否返回末态。
        device: ascriptor 设备名。
        block_dim: 启动核组数。ascriptor 契约声明 ``[1,2,3,4]``；Ascend950PR 只有
            28 cube，超过物理核数会在硬件 barrier 死锁。
        layout_device: token-major ↔ BHCLD 的重排在哪做。``"npu"`` 最快但需要
            内置 copy 算子；``"cpu"`` 绕主机往返（数值相同，计时不可用于性能结论）；
            ``"auto"``（默认）探测一次后自行选择。

    Returns:
        ``(o, final_state)``，``o`` 为 ``[B, T, HV, 128]`` bfloat16；
        ``final_state`` 为 ``[B, HV, 128, 128]`` float32 或 ``None``。
    """
    from ...runtime.compile import compile_kernel

    b, h, hv, c = _check(q, k, v, g, beta, initial_state)
    if layout_device not in ("auto", "npu", "cpu"):
        raise ValueError(f"layout_device 只能是 auto/npu/cpu，收到 {layout_device!r}")
    on_cpu = (layout_device == "cpu") or (layout_device == "auto" and not _npu_supports_d2d_copy())
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    kern = kda_fwd_kernels()
    dev = q.device
    compiled = {name: compile_kernel(fn, device=device, block_dim=block_dim)
                for name, fn in kern.items()}

    def empty(shape, dtype):
        return torch.empty(*shape, dtype=dtype, device=dev)

    qc, kc = _to_bhcld(q, h, on_cpu=on_cpu), _to_bhcld(k, h, on_cpu=on_cpu)
    vc, gc = _to_bhcld(v, hv, on_cpu=on_cpu), _to_bhcld(g, hv, on_cpu=on_cpu)
    bc = _to_bhcld(beta, hv, on_cpu=on_cpu)
    # torch.zeros 在内置算子缺失的 SoC 上不可用，所以在 CPU 上造零再 H2D
    state0 = initial_state if initial_state is not None else torch.zeros(
        b, hv, HEAD_DIM, VALUE_DIM, dtype=torch.float32, device="cpu").to(dev)

    bhc = (b, hv, c, L_PER_CHUNK, HEAD_DIM)
    sq = (b, hv, c, L_PER_CHUNK, L_PER_CHUNK)

    # 1) gate：chunk 内 cumsum
    eg = empty(bhc, torch.float32)
    compiled["gate"]({"g_raw": gc}, {"B": b, "HV": hv, "C": c,
                                     "length_per_chunk": L_PER_CHUNK, "head_dim": HEAD_DIM}, {"eg": eg})

    # 2) scores：Aqk 与严格下三角
    aqk, strict = empty(sq, torch.bfloat16), empty(sq, torch.float32)
    compiled["scores"](
        {"q": qc, "k": kc, "eg": eg, "beta": bc},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "scale": scale},
        {"Aqk": aqk, "strict": strict},
    )

    # 3) inverse：严格下三角求逆（kernel 的 H 位传 hv）
    akk = empty(sq, torch.bfloat16)
    compiled["inverse"]({"a": strict}, {"B": b, "H": hv, "C": c}, {"inv": akk})

    # 4) wy：WY 表示
    w, u, qg, kg = (empty(bhc, torch.bfloat16) for _ in range(4))
    compiled["wy"](
        {"q": qc, "k": kc, "v": vc, "beta": bc, "Akk": akk, "eg": eg},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM},
        {"w": w, "u": u, "qg": qg, "kg": kg},
    )

    # 5) recurrent：chunk 递推
    o_c = empty(bhc, torch.bfloat16)
    final_state = empty((b, hv, HEAD_DIM, VALUE_DIM), torch.float32)
    compiled["recurrent"](
        {"q": qc, "Aqk": aqk, "kg": kg, "w": w, "u": u, "eg": eg, "initial_state": state0},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM, "scale": scale},
        {"o": o_c, "final_state": final_state},
    )

    o = _from_bhcld(o_c, on_cpu=on_cpu)
    return o, (final_state if output_final_state else None)
