#!/usr/bin/env python3
"""A2-07 real-machine probes for the Qwen3-Next GDN ABI design doc.

Everything here is **host-observable on a real 910B3** and feeds
``docs/research/a2_gdn_abi.md``. Per ``AGENTS.md`` §6 / D-PM-30, any A2 number is
an **observation record only** until A2-11 — the doc labels them so. No kernel and
no ``ascend_fla/**`` is touched (write set: ``benchmarks/a2/**``).

Probes:
  1. torch_npu op availability for the non-GDN Qwen3-Next layers (§4), bf16 & fp32,
     with the raw error text when an op is missing/unsupported.
  2. FP32 and BF16 ``exp`` over/under-flow lines on this SoC's device path, which
     also reveals whether subnormals are flushed to zero (A5 flushes; a2 unknown) —
     this pins the §3 gate-decay underflow argument.
  3. Device properties (cube/vector cores) read back from torch_npu, to cross-check
     A2-01's measured 20/40.

Run:
    ASCEND_RT_VISIBLE_DEVICES=<card> python benchmarks/a2/probe_gdn_abi.py \
        --out benchmarks/a2/evidence/gdn_abi/probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform as _plat
import sys
import traceback

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"


def _version_info() -> dict:
    """CANN compiler / opp version.info raw text + sha256, and the opp built-in kernel
    directory listing (D-PM-34). No absolute paths recorded (repo hygiene) — only the
    file's relative tail, its raw content and hash."""
    import glob
    import hashlib
    out: dict = {}
    homes = [h for h in (os.environ.get("ASCEND_HOME_PATH"),
                         os.environ.get("ASCEND_TOOLKIT_HOME"),
                         "/usr/local/Ascend/ascend-toolkit/latest",
                         "/usr/local/Ascend/cann/latest") if h]
    for label, rel in (("compiler", "compiler/version.info"), ("toolkit", "version.info")):
        for base in homes:
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                raw = open(p, encoding="utf-8", errors="replace").read().strip()
                out[label] = {"file": rel, "raw": raw,
                              "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
                break
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if not opp:
        for base in homes:
            if os.path.isdir(os.path.join(base, "opp")):
                opp = os.path.join(base, "opp")
                break
    if opp:
        vp = os.path.join(opp, "version.info")
        if os.path.isfile(vp):
            raw = open(vp, encoding="utf-8", errors="replace").read().strip()
            out["opp"] = {"file": "opp/version.info", "raw": raw,
                          "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        kd = os.path.join(opp, "built-in/op_impl/ai_core/tbe/kernel")
        try:
            entries = sorted(os.listdir(kd))
            out["opp_builtin_kernel_dir"] = {
                "tail": "opp/built-in/op_impl/ai_core/tbe/kernel",
                "count": len(entries), "sample": entries[:24]}
        except Exception as e:
            # fall back to a shallower listing if the exact path differs by CANN version
            alt = sorted(glob.glob(os.path.join(opp, "built-in/op_impl/**/kernel"), recursive=True))
            out["opp_builtin_kernel_dir"] = {"error": f"{type(e).__name__}: {e}",
                                             "alt_matches": [a.split('/opp/')[-1] for a in alt[:8]]}
    return out


def _env() -> dict:
    props = {}
    try:
        p = torch.npu.get_device_properties(0)
        for attr in ("name", "total_memory", "cube_core_num", "vector_core_num"):
            if hasattr(p, attr):
                props[attr] = getattr(p, attr)
        props["device_name"] = torch.npu.get_device_name(0)
    except Exception as e:  # pragma: no cover
        props["error"] = f"{type(e).__name__}: {e}"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "?"),
        "uname": _plat.platform(),
        "ascend_compute_unit": os.environ.get("ASCEND_COMPUTE_UNIT", ""),
        "soc": os.environ.get("ASCEND_FLA_SOC", ""),
        "cann_version_info": _version_info(),  # raw version.info + sha256; no absolute paths
        "device_props": props,
        "note": "A2 numbers are observation records until A2-11 (AGENTS.md §6 / D-PM-30).",
    }


def _recip_subnormal_line(dt) -> dict:
    """1/x for x at / below the smallest normal — tests whether the reciprocal of a
    subnormal denominator overflows to inf on this SoC. This is what decides the real
    gate-range limit: eg=exp(-span) becomes subnormal well before it reaches 0, so
    k/eg overflows to inf at a smaller span than the eg->0 line."""
    import math
    xs = [1.18e-38, 6.05e-39, 3.0e-39, 2.23e-39, 1.5e-39, 1e-40]
    out = {}
    for x in xs:
        try:
            r = (1.0 / torch.tensor([x], dtype=dt, device=DEV)).item()
        except Exception as e:
            r = f"ERR {type(e).__name__}"
        out[repr(x)] = {"recip": r, "is_inf": isinstance(r, float) and math.isinf(r)}
    return out


def _probe_ops() -> dict:
    """Each op the rest of Qwen3-Next needs; run tiny bf16 & fp32 cases, catch raw errors."""
    import torch.nn.functional as F

    def mk(dt, *shape):
        return torch.randn(*shape, dtype=dt, device=DEV)

    cases = {
        "silu (short-conv act)": lambda dt: F.silu(mk(dt, 4, 8)),
        "sigmoid (gate)": lambda dt: torch.sigmoid(mk(dt, 4, 8)),
        "softplus (gate)": lambda dt: F.softplus(mk(dt, 4, 8)),
        "l2_normalize (q/k)": lambda dt: F.normalize(mk(dt, 4, 8), dim=-1),
        "cumsum (gate decay)": lambda dt: torch.cumsum(mk(dt, 4, 8), dim=-1),
        "conv1d depthwise (short conv)": lambda dt: F.conv1d(
            mk(dt, 1, 8, 16), mk(dt, 8, 1, 4), groups=8),
        "sdpa (full attn)": lambda dt: F.scaled_dot_product_attention(
            mk(dt, 1, 2, 8, 16), mk(dt, 1, 2, 8, 16), mk(dt, 1, 2, 8, 16)),
        "softmax (attn/router)": lambda dt: torch.softmax(mk(dt, 4, 8), dim=-1),
        "topk (MoE router)": lambda dt: torch.topk(mk(dt, 4, 8), 2, dim=-1),
        "bmm (MoE experts / GEMM)": lambda dt: torch.bmm(mk(dt, 2, 4, 8), mk(dt, 2, 8, 4)),
        "layer_norm": lambda dt: F.layer_norm(mk(dt, 4, 8), (8,)),
        "rms_norm (block/FusedRMSNormGated)": lambda dt: (
            F.rms_norm(mk(dt, 4, 8), (8,)) if hasattr(F, "rms_norm")
            else (_ for _ in ()).throw(AttributeError("F.rms_norm absent in this torch"))),
        "embedding": lambda dt: F.embedding(
            torch.tensor([0, 1], device=DEV), mk(dt, 4, 8)),
    }
    out = {}
    for name, fn in cases.items():
        row = {}
        for dt in (torch.float32, torch.bfloat16):
            try:
                r = fn(dt)
                torch.npu.synchronize()
                finite = bool(torch.isfinite(r).all()) if torch.is_tensor(r) else True
                row[str(dt).replace("torch.", "")] = "ok" if finite else "ok-but-nonfinite"
            except Exception as e:
                row[str(dt).replace("torch.", "")] = f"ERR {type(e).__name__}: {str(e)[:140]}"
        out[name] = row
    return out


def _exp_lines(dt) -> dict:
    xs = [-80, -85, -87, -87.3, -88, -88.7, -89, -95, -100, -103, -104, -110,
          80, 85, 87, 88, 88.7, 89, 90, 100]
    out = {}
    for x in xs:
        try:
            v = torch.exp(torch.tensor([float(x)], dtype=dt, device=DEV)).item()
        except Exception as e:
            v = f"ERR {type(e).__name__}"
        out[str(x)] = v
    # first x (most negative) that maps to exactly 0.0, and first positive that maps to inf
    import math
    zero_at = next((x for x in xs if isinstance(out[str(x)], float) and out[str(x)] == 0.0), None)
    inf_at = next((x for x in xs if isinstance(out[str(x)], float) and math.isinf(out[str(x)])), None)
    return {"values": out, "first_underflow_to_zero_at": zero_at, "first_overflow_to_inf_at": inf_at}


_EXP_XS = [-80, -85, -87, -87.3, -88, -88.7, -89, -95, -100, -103, -104, -110,
           80, 85, 87, 88, 88.7, 89, 90, 100]


def _ascriptor_exp_line() -> dict:
    """exp on the ascriptor-compiled a2 vector path (generated CCE), not torch_npu's
    aclnn built-in. This is the path a real GDN decay kernel takes, so it is the one
    that decides the §3/§5 subnormal-flush behaviour. Skips gracefully if ascriptor
    or the a2 build is unavailable."""
    import math
    import importlib.util
    import os as _os
    try:
        from ascend_fla.runtime.compile import compile_kernel

        _spec = importlib.util.spec_from_file_location(
            "_a2_exp_kernel", _os.path.join(_os.path.dirname(__file__), "_a2_exp_kernel.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        comp = compile_kernel(_mod.a2_exp_probe, device="a2", block_dim=1, backend="cce")
        xs = _EXP_XS + [0.0] * (64 - len(_EXP_XS))
        xt = torch.tensor([xs], dtype=torch.float32, device=DEV)
        yt = torch.empty(1, 64, dtype=torch.float32, device=DEV)
        comp({"x": xt}, {}, {"y": yt})
        torch.npu.synchronize()
        yv = yt[0].tolist()
        n = len(_EXP_XS)
        out = {str(_EXP_XS[i]): yv[i] for i in range(n)}
        zero_at = next((_EXP_XS[i] for i in range(n) if yv[i] == 0.0 and _EXP_XS[i] < 0), None)
        inf_at = next((_EXP_XS[i] for i in range(n) if math.isinf(yv[i]) and _EXP_XS[i] > 0), None)
        return {"values": out, "first_underflow_to_zero_at": zero_at,
                "first_overflow_to_inf_at": inf_at, "path": "ascriptor a2 cce vector exp"}
    except Exception as e:
        return {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/a2/evidence/gdn_abi/probe.json")
    ap.add_argument("--ascriptor-exp", action="store_true",
                    help="also compile+run an ascriptor a2 cce exp kernel (needs ascriptor + a2 build)")
    a = ap.parse_args()
    res = {"env": _env()}
    res["torch_op_support"] = _probe_ops()
    res["fp32_exp"] = _exp_lines(torch.float32)
    res["bf16_exp"] = _exp_lines(torch.bfloat16)
    res["fp32_reciprocal_subnormal"] = _recip_subnormal_line(torch.float32)
    if a.ascriptor_exp:
        res["ascriptor_fp32_exp"] = _ascriptor_exp_line()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    # human-readable summary to stdout (raw, for the evidence log)
    vi = res["env"]["cann_version_info"]
    print("=== device ===", json.dumps(res["env"]["device_props"], ensure_ascii=False))
    print("=== version.info (raw) ===")
    for k in ("compiler", "toolkit", "opp"):
        if k in vi:
            print(f"  [{k}] {vi[k]['raw']}  sha256={vi[k]['sha256'][:12]}")
    if "opp_builtin_kernel_dir" in vi:
        d = vi["opp_builtin_kernel_dir"]
        print(f"  [opp built-in kernel dir] count={d.get('count')} sample={d.get('sample', d.get('alt_matches'))}")
    print("=== 1/x for subnormal x (fp32) ===")
    for x, r in res["fp32_reciprocal_subnormal"].items():
        print(f"   1/{x} = {r['recip']}  inf={r['is_inf']}")
    print("=== torch_npu op support (fp32 / bf16) ===")
    for k, v in res["torch_op_support"].items():
        print(f"  {k:38s} f32={v.get('float32')}  bf16={v.get('bfloat16')}")
    for tag in ("fp32_exp", "bf16_exp"):
        e = res[tag]
        print(f"=== {tag}: underflow->0 at x={e['first_underflow_to_zero_at']}, "
              f"overflow->inf at x={e['first_overflow_to_inf_at']} ===")
        print("   " + "  ".join(f"{x}:{e['values'][x]}" for x in ("-87.3", "-88", "-88.7", "-89", "88.7", "89")))
    if "ascriptor_fp32_exp" in res:
        ae = res["ascriptor_fp32_exp"]
        if "skipped" in ae:
            print("=== ascriptor a2 cce exp: SKIPPED ===", ae["skipped"])
        else:
            print(f"=== ascriptor a2 cce exp: underflow->0 at x={ae['first_underflow_to_zero_at']}, "
                  f"overflow->inf at x={ae['first_overflow_to_inf_at']} ===")
            print("   " + "  ".join(f"{x}:{ae['values'][x]}" for x in ("-87.3", "-88", "-89", "88.7", "89")))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
