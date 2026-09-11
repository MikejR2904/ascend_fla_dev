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
import math
import os
import pathlib
import sys
from typing import Any

import torch

__all__ = ["chunk_kda_fwd", "kda_fwd_kernels"]

L_PER_CHUNK = 64
HEAD_DIM = 128
VALUE_DIM = 128
# ascriptor kda_fwd contract.json 的 shapes.block_dim 声明；只有这几个值被 cases 覆盖过。
# 契约的 core_ownership 说明分区方式：gate 按向量核切 B*HV*C，scores/WY/inverse 按 cube
# 组切，融合尾部按 B*HV 头对切（两个 V=64 tile 必须留在同一组）。
SUPPORTED_BLOCK_DIM = (1, 2, 3, 4)

#: 两套前向实现。``upstream`` 原样用 ascriptor 的五个 kernel；``stable`` 把 gate / scores /
#: wy 换成本仓 ``kernels/projects/a5/kda_fwd_stable/`` 下的版本，把门控算术改成对深衰减
#: 数值稳定的形式（见那三个文件的 docstring 与 gaps.json 的 gate-range-beyond-declared）。
#: inverse 与 recurrent 两套共用 —— 它们只用绝对量，下溢到 0 本身就是正确结果。
IMPLS = ("stable", "upstream")

_UPSTREAM_KERNELS = {
    "gate": ("gate", "kda_sub1_gate_kernel"),
    "scores": ("intra", "kda_sub2_score_kernel"),
    "inverse": ("triangular_inverse", "tril_inverse64_v2_strict_bf16_kernel"),
    "wy": ("wy", "kda_sub3_wy_kernel"),
    "recurrent": ("recurrent", "kda_sub45_fused_kernel"),
}
_STABLE_KERNELS = {
    "gate": ("gate", "kda_sub1_gate_stable_kernel"),
    "scores": ("intra", "kda_sub2_score_stable_kernel"),
    "wy": ("wy", "kda_sub3_wy_stable_kernel"),
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


def _stable_kernels_root() -> pathlib.Path:
    """本仓自有单元 ``kernels/projects/a5/kda_fwd_stable`` 的位置。"""
    repo = pathlib.Path(__file__).resolve().parents[3]
    root = repo / "kernels/projects/a5/kda_fwd_stable"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(f"找不到本仓的 kda_fwd_stable 单元（试了 {root}）")
    return root


def _load_kernel(tag: str, pkg_dir: pathlib.Path, module_name: str, fn_name: str) -> Any:
    """按文件加载单个 kernel 定义。

    逐模块按文件加载，**不** import 单元的 ``kernels`` 包 —— 那里的 ``composition.py``
    依赖单元本地的 ``_unit_runner``，会把 harness 拖进来。
    """
    path = pkg_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_afla_kda_{tag}_{module_name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


@functools.lru_cache(maxsize=len(IMPLS))
def kda_fwd_kernels(impl: str = "stable") -> dict[str, Any]:
    """载入某一套实现的五个 kernel 定义。

    ``stable`` 的 gate / scores / wy 来自本仓单元，inverse / recurrent 仍用 ascriptor 的。
    """
    if impl not in IMPLS:
        raise ValueError(f"impl 只能是 {IMPLS}，收到 {impl!r}")
    up_root = _kernels_root()
    if str(up_root) not in sys.path:
        sys.path.insert(0, str(up_root))

    out: dict[str, Any] = {}
    for stage, (module_name, fn_name) in _UPSTREAM_KERNELS.items():
        out[stage] = _load_kernel("up", up_root / "kernels", module_name, fn_name)
    if impl == "stable":
        st_dir = _stable_kernels_root() / "kernels"
        for stage, (module_name, fn_name) in _STABLE_KERNELS.items():
            out[stage] = _load_kernel("st", st_dir, module_name, fn_name)
    return out


@functools.lru_cache(maxsize=None)
def _compiled_chain(device: str, block_dim: int, impl: str = "stable") -> dict[str, Any]:
    """(device, block_dim) → 已编译的 5 个 kernel。

    缓存到这一层是因为热路径不该每次前向都去查 5 次编译缓存；``compile_kernel``
    自己也有进程内缓存，但查它要算签名（见 runtime/compile.py 的 ``_sig_memo``）。
    """
    from ...runtime.compile import compile_kernel

    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in kda_fwd_kernels(impl).items()}


def _check(q, k, v, g, beta, initial_state, block_dim) -> tuple[int, int, int, int]:
    """门控。返回 ``(B, H, HV, C)``。任何不满足都报错，绝不静默降级。"""
    if block_dim not in SUPPORTED_BLOCK_DIM:
        raise ValueError(
            f"block_dim 只支持 {SUPPORTED_BLOCK_DIM}（ascriptor kda_fwd 契约声明的范围），"
            f"收到 {block_dim}；更大的值未经契约 case 覆盖，且超过物理核数会在硬件 "
            f"barrier 死锁，见 docs/matrix/gaps.json 的 block-dim-ceiling"
        )
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
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                    *((("initial_state", initial_state),) if initial_state is not None else ())):
        if x.device.type != "npu":
            raise ValueError(f"{name} 应在 NPU 上，收到 device={x.device}")
        # 必须连续：kernel 按连续 GM 布局读，而且在缺内置算子包的机器上我们既不能在
        # device 上做 contiguous()（要 d2d copy），也不能对跨步视图直接 D2H
        # （要 NPU 侧的 Slice，实测报 errno 561000 —— 见 chunk_bwd.py 里 g_last 那段）。
        # 所以这里报错而不是悄悄修正 —— 悄悄 .contiguous() 在那种机器上根本做不到。
        if not x.is_contiguous():
            raise ValueError(
                f"{name} 必须是连续张量，收到 stride={tuple(x.stride())} "
                f"shape={tuple(x.shape)}；先自己 .contiguous() 再传进来"
            )
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


#: 每套实现能承受的 chunk 内门控跨度上限。
#:
#: ``upstream``：实测跨度 ≤66.84 时前向完全正常（相对 L2 稳定 2.86e-03~2.98e-03），
#: ≥88.67 时 ``Aqk``/``strict``/``kg`` 出 NaN。根因是 gate 只写 ``eg = exp(gc)``，而它在
#: ``-ln(FLT_MIN_NORMAL) ≈ 87.3`` 处下溢到 0（fp32 非正规数被 flush），下游的 ``k/eg`` 与
#: ``eg_last/eg`` 就变成 ``0×inf`` 和 ``0/0``。取 80 留约 8% 余量，也与
#: ``reference.kda.kda_chunk_vectorized`` 的上限一致，好让两边对同一组输入都可用。
#:
#: ``stable``：本仓的 gate 额外写出 log 空间的 ``g_cumsum``，scores 按逐通道中点对称分解
#: （两个因子的指数都压到 ``±S/2``），wy 改成先减后指数。前向理论上限因此翻倍到
#: ``2 × 87.3 ≈ 174``，实测到 155.97 仍不退化。
#:
#: ⚠️ **这两个上限同时管前向与反向，取两者较小者。** 反向是独立的一处：ascriptor 的
#: ``finalize_pre`` / ``finalize_post`` 把成对衰减分解成 ``exp(g−g_last)·exp(g_last−g)``，
#: 前者在 ``ln(MAX) ≈ 88.72`` 处**上溢**（方向与前向的下溢相反），而它的输出是 bf16 GM。
#: 所以 ``upstream`` 的 80 对反向同样适用；``stable`` 的反向把锚点改成 ``g_last/2``，
#: 理论上限 ``2 × 88.72 ≈ 177``，取 160 是两条链的公共安全值。见 ``gaps.json`` 的
#: ``bwd-gate-range-overflow`` 与 ``gate-span-still-bounded``。
#:
#: **混用 impl 会让这张表说谎** —— 前向 stable + 反向 upstream 时跨度 94 能过检查却在反向
#: 吐 NaN。所以 ``prepare`` / ``chunk_kda`` 的 ``impl`` 同时选两条链，不提供分开的开关。
MAX_GATE_SPAN = {"upstream": 80.0, "stable": 160.0}


def _gate_span(g: torch.Tensor, c: int, *, on_cpu: bool) -> float:
    """g 在 chunk 内累计后的最大跨度（``max(cumsum) - min(cumsum)``）。

    必须按 **chunk 内**算 —— kernel 的 cumsum 每 64 个 token 重置，跨 chunk 的累计不参与
    那个 ``exp(m−g)``。``on_cpu=True`` 时绕主机算（缺内置算子的机器上 cumsum 不可用）。
    """
    src = g.cpu() if on_cpu else g
    b, t, hv, kd = src.shape
    cum = src.float().view(b, c, L_PER_CHUNK, hv, kd).cumsum(dim=2)
    return (cum.amax(dim=2) - cum.amin(dim=2)).max().item()


def _check_gate_range(g: torch.Tensor, c: int, *, on_cpu: bool, impl: str) -> None:
    """门控跨度超限就报错，绝不让 kernel 静默吐 NaN（AGENTS.md §7）。

    ⚠️ **这条限制比 contract 声明的输入域宽得多。** contract 的 ``input_generation`` 是
    ``g_raw ∈ [-0.03, 0]``（64 token 跨度 ≤1.92），而 fla 自己的 KDA 初始化
    （``A_log = log(U(1,16))``、``dt`` 最大 0.1）给出的跨度约 **94** —— 宽约 50 倍。
    ``upstream`` 的上限 80 撑不住它（会吐 NaN），``stable`` 的 160 可以。两个上限的由来
    与实测见 ``docs/matrix/gaps.json`` 的 ``gate-range-beyond-declared``。
    """
    limit = MAX_GATE_SPAN[impl]
    span = _gate_span(g, c, on_cpu=on_cpu)
    if span > limit:
        hint = ("换 impl=\"stable\"（本仓的数值稳定实现，上限 "
                f"{MAX_GATE_SPAN['stable']}）" if impl == "upstream" else
                "减小 g 的量级：KDA 层里即减小 exp(A_log) 或 dt")
        raise ValueError(
            f"chunk 内门控跨度 {span:.1f} 超过 impl={impl!r} 的上限 {limit}，"
            f"kernel 会在 fp32 下溢/上溢并输出 NaN。g 是 log 空间的 per-token 衰减增量；"
            f"{hint}。确知安全时可传 check_gate_range=False 跳过本检查。"
            f"详见 docs/matrix/gaps.json 的 gate-range-beyond-declared"
        )


def _resolve_layout(layout_device: str) -> bool:
    """``layout_device`` → 是否把重排绕到 CPU 上做。"""
    if layout_device not in ("auto", "npu", "cpu"):
        raise ValueError(f"layout_device 只能是 auto/npu/cpu，收到 {layout_device!r}")
    return (layout_device == "cpu") or (layout_device == "auto" and not _npu_supports_d2d_copy())


def _run_chain(q, k, v, g, beta, scale, initial_state, *, device, block_dim,
               on_cpu, b, h, hv, c, impl="stable") -> dict[str, torch.Tensor]:
    """跑完五个前向 kernel，返回**全部** BHCLD 中间量。

    ``chunk_kda_fwd`` 只要其中的 ``o`` 与 ``final_state``；``chunk_kda_fwd_with_caches``
    还要 ``eg`` / ``Aqk`` / ``Akk`` / ``w`` / ``u`` / ``qg`` / ``kg`` 去拼 kda_bwd 需要的
    九个前向检查点。抽成一处是为了两条路径**共用同一次 kernel 调用**，不会因为实现
    漂移而给出不同的中间量。
    """
    dev = q.device
    compiled = _compiled_chain(device, block_dim, impl)

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

    # 1) gate：chunk 内 cumsum。stable 版同时写出 log 空间的 g_cumsum ——
    # 下游 scores/wy 要靠它把除法变成减法（见 kernels/.../kda_fwd_stable/gate.py）
    eg = empty(bhc, torch.float32)
    gate_scalars = {"B": b, "HV": hv, "C": c,
                    "length_per_chunk": L_PER_CHUNK, "head_dim": HEAD_DIM}
    if impl == "stable":
        g_cumsum = empty(bhc, torch.float32)
        compiled["gate"]({"g_raw": gc}, gate_scalars,
                         {"g_cumsum": g_cumsum, "eg": eg})
    else:
        g_cumsum = None
        compiled["gate"]({"g_raw": gc}, gate_scalars, {"eg": eg})
    gate_in = {"g_cumsum": g_cumsum} if impl == "stable" else {"eg": eg}

    # 2) scores：Aqk 与严格下三角
    aqk, strict = empty(sq, torch.bfloat16), empty(sq, torch.float32)
    compiled["scores"](
        {"q": qc, "k": kc, "beta": bc, **gate_in},
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
        {"q": qc, "k": kc, "v": vc, "beta": bc, "Akk": akk, **gate_in},
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

    return {"eg": eg, "g_cumsum": g_cumsum, "Aqk": aqk, "strict": strict, "Akk": akk,
            "w": w, "u": u, "qg": qg, "kg": kg, "o": o_c, "final_state": final_state,
            "initial_state": state0}


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
    check_gate_range: bool = True,
    impl: str = "stable",
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
        block_dim: 启动核组数，只接受 ``SUPPORTED_BLOCK_DIM``。kernel 用
            ``GetVecIdx()/GetVecNum()`` 自行切分，而 ``GetVecNum() == 2 * block_dim``，
            所以这个值直接决定并行度 —— ``block_dim=1`` 只用到 2 个向量核。
            契约只覆盖到 4；Ascend950PR 物理上有 28 cube / 56 vec，但超过物理核数
            会在硬件 barrier 死锁。
        layout_device: token-major ↔ BHCLD 的重排在哪做。``"npu"`` 最快但需要
            内置 copy 算子；``"cpu"`` 绕主机往返（数值相同，计时不可用于性能结论）；
            ``"auto"``（默认）探测一次后自行选择。
        check_gate_range: 是否校验 chunk 内门控跨度不超过 :data:`MAX_GATE_SPAN` 里
            该实现的上限。默认开 —— 超限时 kernel 会静默吐 NaN，那比报错糟得多。
            代价是对 ``g`` 做一次 cumsum + 两次规约。
        impl: ``"stable"``（默认）用本仓 ``kernels/projects/a5/kda_fwd_stable`` 的
            gate / scores / wy，门控算术对深衰减数值稳定，可用跨度约 160；
            ``"upstream"`` 原样用 ascriptor 的五个 kernel，可用跨度约 80。
            两者数学同义，差别只在浮点表示范围。

    Returns:
        ``(o, final_state)``，``o`` 为 ``[B, T, HV, 128]`` bfloat16；
        ``final_state`` 为 ``[B, HV, 128, 128]`` float32 或 ``None``。
    """
    b, h, hv, c = _check(q, k, v, g, beta, initial_state, block_dim)
    on_cpu = _resolve_layout(layout_device)
    if check_gate_range:
        _check_gate_range(g, c, on_cpu=on_cpu, impl=impl)
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    chain = _run_chain(q, k, v, g, beta, scale, initial_state, device=device,
                       block_dim=block_dim, on_cpu=on_cpu, b=b, h=h, hv=hv, c=c, impl=impl)
    o = _from_bhcld(chain["o"], on_cpu=on_cpu)
    return o, (chain["final_state"] if output_final_state else None)


#: ``kda_bwd`` 声明的九个前向检查点。顺序无关，但**名字和形状必须完全一致** ——
#: 它的 ``validate_inputs`` 会逐个核对，多一个少一个都报错。
BWD_CACHE_NAMES = ("g_cumsum", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h")


def _scan_states(w, u, kg, eg, state0, *, b, hv, c, on_cpu=False):
    """逐 chunk 递推出 ``h``（chunk 起始状态）与 ``v_new``。

    ``kda_sub45_fused_kernel`` 内部算的就是这两个量，但它只写出 ``o`` 与
    ``final_state`` —— 见 `docs/matrix/gaps.json` 的 ``fwd-caches-not-emitted``。
    这里在 host 侧用 torch 复算一遍，**正确但慢**（C 次迭代 × 2 次 bmm，全是
    torch_npu 的小算子），只为先把反向链的正确性立住。

    ``on_cpu=True`` 时整段绕到 CPU 上算（D2H → 算 → H2D）。**这不是性能选项，是可用性
    选项**：这段用的是 Cast / bmm / stack 等内置算子，在内置算子包不覆盖当前 SoC 的机器上
    全部不可用（见 AGENTS.md §5 的可用面表）。也就是说 ``fwd-caches-not-emitted`` 除了慢，
    还让**训练路径**依赖内置算子包，而纯前向路径不依赖 —— 这一条曾让反向在只有 910 算子包
    的机器上以 ``561103`` / ``Cast ADD_TO_LAUNCHER_LIST_AICORE failed`` 失败。
    把这三项挪进 kernel 之后这个选项就可以删掉。

    递推（与 kda_bwd 的 ref/forward.py 逐行对应，fp32 累加、存储时降到 bf16）::

        h[c]      = state
        v_new[c]  = u[c] - w[c] @ state
        state     = state * exp2(g_last[c])[:, None] + kg[c]^T @ v_new[c]

    其中 ``exp2(g_last[c])`` 就是 ``eg`` 在该 chunk 末行的值（``eg == 2**g_cumsum``）。

    Returns:
        ``(h, v_new)``：``h`` 为 ``[B,C,HV,128,128]``、``v_new`` 为 BHCLD 的
        ``[B,HV,C,64,128]``，均 bfloat16。
    """
    dev = state0.device
    # 先 D2H 再算（不是先算再 D2H）—— 缺算子包的机器上 NPU 侧连 .float() 都不可用
    host = (lambda x: x.cpu()) if on_cpu else (lambda x: x)
    w, u, kg, eg = host(w), host(u), host(kg), host(eg)

    state = host(state0).float()
    h_chunks, v_new_chunks = [], []
    for ci in range(c):
        h_chunks.append(state)
        wc = w[:, :, ci].float()                       # [B,HV,64,128]
        vn = u[:, :, ci].float() - wc @ state          # [B,HV,64,128]
        v_new_chunks.append(vn)
        g_last = eg[:, :, ci, L_PER_CHUNK - 1, :]      # [B,HV,128] = exp2(g_cumsum 末行)
        state = state * g_last[..., None] + kg[:, :, ci].float().transpose(-1, -2) @ vn
    h = torch.stack(h_chunks, dim=1).bfloat16()        # [B,C,HV,128,128]
    v_new = torch.stack(v_new_chunks, dim=2).bfloat16()  # [B,HV,C,64,128]
    return (h.to(dev), v_new.to(dev)) if on_cpu else (h, v_new)


def chunk_kda_fwd_with_caches(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    device: str = "a5",
    block_dim: int = 1,
    layout_device: str = "auto",
    check_gate_range: bool = True,
    impl: str = "stable",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """前向，并额外产出 ``kda_bwd`` 需要的九个检查点。

    与 :func:`chunk_kda_fwd` **共用同一次 kernel 调用**（都走 ``_run_chain``），所以
    ``o`` 与 ``final_state`` 逐位相同。

    九个检查点里六个直接来自前向 kernel（``Aqk`` / ``Akk`` / ``w`` / ``u`` / ``qg`` /
    ``kg``），另外三个要补：

    * ``g_cumsum = log2(eg)`` —— gate kernel 只写出 ``eg = 2**g_cumsum``，不写 cumsum 本身。
    * ``h`` / ``v_new`` —— 融合 recurrent kernel 内部有，但不写出（见
      :func:`_scan_states`）。

    **这三项都是 host 侧补的，是当前反向链的性能瓶颈**，缺口记在
    ``docs/matrix/gaps.json`` 的 ``fwd-caches-not-emitted``。

    本函数**会顺便把反向链编译好**（首次调用时多花一次编译时间）。这不是顺手而为：
    CANN 只在首次算子解析时读 ``ASCEND_CUSTOM_OPP_PATH``，反向的 vendor 树若在前向
    执行之后才注册就解析不到（``rc=161001``）。见 ``runtime/binding.py`` 的
    ``register_custom_opp_path``。

    Returns:
        ``(o, final_state, caches)``。``caches`` 的键正是 :data:`BWD_CACHE_NAMES`，
        全部 bfloat16、token-major（``h`` 为 ``[B,C,HV,128,128]``），可直接喂 ``kda_bwd``。
    """
    b, h_q, hv, c = _check(q, k, v, g, beta, initial_state, block_dim)
    # 反向链要在**首次 aclnn 调用之前**注册完 vendor 树，否则它的算子解析不到
    # （见 runtime/binding.py 的 register_custom_opp_path）。要缓存的唯一理由就是
    # 接着跑反向，所以在这里一并编译好。编译有两级缓存，重复调用不花钱。
    from .chunk_bwd import _compiled_chain as _bwd_chain

    _bwd_chain(device, block_dim, impl)
    on_cpu = _resolve_layout(layout_device)
    if check_gate_range:
        _check_gate_range(g, c, on_cpu=on_cpu, impl=impl)
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    chain = _run_chain(q, k, v, g, beta, scale, initial_state, device=device,
                       block_dim=block_dim, on_cpu=on_cpu, b=b, h=h_q, hv=hv, c=c, impl=impl)

    h_states, v_new = _scan_states(chain["w"], chain["u"], chain["kg"], chain["eg"],
                                   chain["initial_state"], b=b, hv=hv, c=c, on_cpu=on_cpu)
    # 布局重排 + 降 bf16。``on_cpu`` 时两步都在 CPU 上做，最后一次 H2D —— 顺序要紧：
    # 先 D2H 再 cast，反过来会在缺内置算子包的机器上因为 Cast 失败（AGENTS.md §5）。
    dev = q.device

    def tok(x: torch.Tensor) -> torch.Tensor:
        out = _from_bhcld(x.cpu() if on_cpu else x, on_cpu=False).bfloat16()
        return out.to(dev) if on_cpu else out

    # kda_bwd 的 g_cumsum 是 **log2** 空间的（ref/forward.py 里 cumsum * RCP_LN2）。
    # stable 实现的 gate kernel 直接写出自然底的 cumsum，换底即可 —— 比 log2(eg) 更准，
    # 而且在 eg 已经下溢到 0 的深衰减下，log2(eg) 会给 -inf，那条路根本不可用。
    src_g = chain["g_cumsum"] if chain["g_cumsum"] is not None else chain["eg"]
    base = src_g.cpu() if on_cpu else src_g
    g_cum_log2 = (base * (1.0 / math.log(2.0))) if chain["g_cumsum"] is not None \
        else base.log2()
    caches = {
        "g_cumsum": tok(g_cum_log2),
        "Aqk": tok(chain["Aqk"]),
        "Akk": tok(chain["Akk"]),
        "w": tok(chain["w"]),
        "u": tok(chain["u"]),
        "qg": tok(chain["qg"]),
        "kg": tok(chain["kg"]),
        "v_new": tok(v_new),
        "h": h_states,
    }
    missing = set(BWD_CACHE_NAMES) ^ set(caches)
    if missing:
        raise RuntimeError(f"检查点名字与 kda_bwd 的声明不符，差异 {sorted(missing)}")
    o = _from_bhcld(chain["o"], on_cpu=on_cpu)
    return o, chain["final_state"], caches
