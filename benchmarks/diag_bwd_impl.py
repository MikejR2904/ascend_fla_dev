#!/usr/bin/env python3
"""同一个 kda_bwd 契约 case 跑指定 impl，用来把两个变量分开。

**为什么需要它。** `multi_chunk`（C=2）在「缺内置算子包机器的 CPU 绕行路径 + stable 反向 kernel」下失败，
而 `single_chunk`（C=1）通过。这一次同时动了两件事：

1. 为了让缺内置算子包的机器能跑，`_scan_states` / 检查点组装 / `g_last` 的 strided
   `contiguous()` / `dw` 的取负都加了绕 CPU 的路径（`fwd-caches-not-emitted` 的副作用）；
2. 反向 kernel 换成了本仓的 `kda_bwd_stable`。

C=1 与 C=2 的差别恰好落在第 1 条上 —— `g_cumsum[:, 63::64]` 在 C=1 时只取一行（等效连续），
C=2 时是真跨步。所以**必须**用同一条绕行路径跑两个 impl 才能定位：

* 两个 impl 都失败 → 绕行路径有 bug，与 kernel 无关；
* 只有 stable 失败 → kernel 改动在多 chunk 下不对。

用法（一个进程一个 impl —— 同名算子多 build 会互相覆盖）::

    python benchmarks/diag_bwd_impl.py multi_chunk upstream
    python benchmarks/diag_bwd_impl.py multi_chunk stable
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(f"用法：{sys.argv[0]} <case_id> <stable|upstream> [gate_multiplier]",
              file=sys.stderr)
        return 2
    case_id, impl = sys.argv[1], sys.argv[2]
    gate_override = float(sys.argv[3]) if len(sys.argv) == 4 else None

    import torch_npu  # noqa: F401

    from ascend_fla.ops.kda import chunk_kda_fwd_with_caches, prepare
    from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd
    from test_kda_bwd_npu import CASES, _load_ref, _rel_l2

    if case_id not in CASES:
        print(f"未知 case {case_id}，可选 {list(CASES)}", file=sys.stderr)
        return 2
    b, h, hv, c, gate_mul, block_dim = CASES[case_id]
    if gate_override is not None:
        gate_mul = gate_override      # 把契约 case 推到宽域，用来在深衰减下比精度而不只是有限性
    prepare("a5", block_dim, backward=True, impl=impl)

    oracle, make_inputs, Inputs = _load_ref()
    src = make_inputs(B=b, H=h, HV=hv, C=c, K=128, V=128, chunk_size=64, seed=2026)
    x = dict(vars(src))
    if gate_mul != 1.0:
        x["g"] = (x["g"].float() * gate_mul).to(torch.bfloat16)
    want = dict(vars(oracle(Inputs(**{n: x[n] for n in Inputs.__dataclass_fields__}))))

    npu = {n: x[n].to("npu") for n in ("q", "k", "v")}
    npu |= {n: x[n].float().to("npu") for n in ("g", "beta", "initial_state")}
    _, _, caches = chunk_kda_fwd_with_caches(**npu, block_dim=block_dim, impl=impl)
    got = chunk_kda_bwd(
        q=npu["q"], k=npu["k"], v=npu["v"], beta=x["beta"].to("npu"),
        do=x["do"].to("npu"), dht=x["dht"].to("npu"), caches=caches,
        block_dim=block_dim, impl=impl,
    )
    out = {"case": case_id, "impl": impl, "block_dim": block_dim, "C": c,
           "gate_multiplier": gate_mul, "errors": {}, "nonfinite": {},
           # oracle 自己在宽域下是否还有限 —— 它不成立的话下面的相对 L2 没有意义
           "oracle_nonfinite": {n: int((~t.float().isfinite()).sum())
                                for n, t in want.items() if hasattr(t, "float")}}
    g_cpu = x["g"].float() * 1.0
    cum = g_cpu.view(b, c, 64, hv, 128).cumsum(2)
    out["gate_span"] = (cum.amax(2) - cum.amin(2)).max().item() * 0.6931471805599453
    for name, exp in want.items():
        if name not in got:
            continue
        g = got[name].cpu().float()
        out["nonfinite"][name] = int((~g.isfinite()).sum())
        out["errors"][name] = _rel_l2(g, exp)
    # 检查点本身也看一眼有限性 —— 反向错了也可能是喂进去的检查点就错了
    out["caches_nonfinite"] = {n: int((~t.cpu().float().isfinite()).sum())
                               for n, t in caches.items()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
