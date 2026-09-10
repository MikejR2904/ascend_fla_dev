"""kda_bwd 九 kernel 链经 runtime 桥的梯度回归（需要 A5 NPU + CANN）。

参考是 ``kda_bwd`` 单元自己的 ``ref.oracle.oracle`` —— bf16 输入、fp32 前向公式上
autograd、输出降回 bf16。用它而不是自拟参考：那是这个单元被验证时用的同一个 oracle。

**预算取自 contract.json 的 ``comparison``，不要自拟。** 它逐输出给了不同上限：

* 默认 ``max_relative_l2 = 0.05``
* ``dk`` = 0.15 —— 契约的理由：bf16 的 log2 累积门控缓存本身会把参考实现的相对 L2
  推到 0.046~0.096，因为把 log2 门控舍到 0.25 以内再 exp2 会改变乘性导数。
* ``dg`` = 0.25 —— 同样的原因，参考实现自身就是 0.115~0.183。

也就是说 ``dk`` / ``dg`` 的宽预算是**算子精度 ABI 的一部分**，不是我们放水。
契约同时要求这个界"足以拒绝零梯度或严重错误的梯度"，所以下面另外检查非零。

    pytest tests/test_kda_bwd_npu.py -v

形状与 ``block_dim`` 直接用 contract.json 的五个 case，包括 ``gentle_decay``
（``gate_multiplier=0.03``，浅衰减）与 ``grouped_idle_cores``（``block_dim=3`` 而
工作量只有 2 个头对，故意让部分核空转）。

**每个 case 起一个子进程。** 这些 case 的 ``block_dim`` 不同，而同一算子名的多份
build 在一个进程里会互相覆盖（``runtime/binding.py`` 的 ``_claim_op_name`` 会直接
报错）。所以本文件兼作 worker：``python tests/test_kda_bwd_npu.py <case_id>`` 跑单个
case 并把误差打成 JSON，pytest 侧只负责派进程与判预算。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

torch_npu = pytest.importorskip("torch_npu", reason="需要 torch_npu")
if not torch.npu.is_available():  # pragma: no cover
    pytest.skip("没有可用的 NPU", allow_module_level=True)

from ascend_fla.ops.kda.chunk import chunk_kda_fwd_with_caches  # noqa: E402
from ascend_fla.ops.kda.chunk_bwd import _bwd_kernels_root, chunk_kda_bwd  # noqa: E402

# contract.json 的 comparison：默认 0.05，dk/dg 另有更宽的上限（见 docstring）
BUDGET = {"dq": 0.05, "dk": 0.15, "dv": 0.05, "dbeta": 0.05, "dg": 0.25, "dh0": 0.05}

# contract.json 的 cases，逐项照搬：(B, H, HV, C, gate_multiplier, block_dim)
CASES = {
    "single_chunk": (1, 1, 1, 1, 1.0, 1),
    "multi_chunk": (1, 1, 1, 2, 1.0, 1),
    "grouped_heads": (1, 1, 2, 2, 1.0, 2),
    "gentle_decay": (1, 1, 1, 2, 0.03, 1),
    "grouped_idle_cores": (1, 1, 2, 1, 1.0, 3),
}


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _load_ref():
    """导入 kda_bwd 单元的 ``ref`` 包。worker 与 pytest 两侧共用。"""
    root = _bwd_kernels_root()
    sys.path.insert(0, str(root))
    from ref.inputs import make_inputs as source_make_inputs
    from ref.oracle import oracle
    from ref.types import KDAOracleInputs
    return oracle, source_make_inputs, KDAOracleInputs


def run_case(case_id: str) -> dict:
    """跑一个 case，返回每个梯度的相对 L2 与是否退化为全零。

    worker 的主体。pytest 侧一律派子进程调它，不在自己进程里跑。
    """
    b, h, hv, c, gate_mul, block_dim = CASES[case_id]
    oracle, source_make_inputs, Inputs = _load_ref()

    # 用单元自己的 make_inputs：同分布、同 seed 语义，省得我们复刻它的分布假设
    src = source_make_inputs(B=b, H=h, HV=hv, C=c, K=128, V=128, chunk_size=64, seed=2026)
    x = dict(vars(src))
    if gate_mul != 1.0:
        x["g"] = (x["g"].float() * gate_mul).to(torch.bfloat16)

    ref = oracle(Inputs(**{n: x[n] for n in Inputs.__dataclass_fields__}))
    want = dict(vars(ref))

    # 前向按 kda_fwd 的 ABI 收 fp32 的 g/beta/initial_state；单元的输入是 bf16，
    # 这里升回 fp32 —— 信息量不变，只是匹配 fwd 的入口类型
    npu = {n: x[n].to("npu") for n in ("q", "k", "v")}
    npu |= {n: x[n].float().to("npu") for n in ("g", "beta", "initial_state")}
    _, _, caches = chunk_kda_fwd_with_caches(**npu, block_dim=block_dim)

    got = chunk_kda_bwd(
        q=npu["q"], k=npu["k"], v=npu["v"], beta=x["beta"].to("npu"),
        do=x["do"].to("npu"), dht=x["dht"].to("npu"), caches=caches, block_dim=block_dim,
    )

    out = {"case": case_id, "errors": {}, "shape_mismatch": {}, "all_zero": []}
    for name in BUDGET:
        exp = want[name]
        if tuple(got[name].shape) != tuple(exp.shape):
            out["shape_mismatch"][name] = [list(got[name].shape), list(exp.shape)]
            continue
        out["errors"][name] = _rel_l2(got[name].cpu(), exp)
        # 契约要求预算"足以拒绝零梯度"。relL2 对全零会给 1.0，但若参考本身也近零就可能
        # 漏过，所以显式查一次。
        if got[name].float().abs().max().item() == 0.0:
            out["all_zero"].append(name)
    return out


@pytest.mark.parametrize("case_id", list(CASES))
def test_bwd_matches_unit_oracle(case_id):
    """九 kernel 反向链的梯度对齐单元 oracle。每个 case 一个子进程，理由见模块 docstring。"""
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), case_id],
                       capture_output=True, text=True)
    payload = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
    assert r.returncode == 0 and payload, (
        f"worker 失败 exit={r.returncode}\n--- stdout ---\n{r.stdout[-3000:]}"
        f"\n--- stderr ---\n{r.stderr[-3000:]}"
    )
    out = json.loads(payload)
    assert not out["shape_mismatch"], f"形状不符：{out['shape_mismatch']}"
    errors = out["errors"]
    msg = "  ".join(f"{n}={errors[n]:.3e}/{BUDGET[n]}" for n in BUDGET if n in errors)
    assert not out["all_zero"], f"梯度全零：{out['all_zero']}；误差 {msg}"
    bad = {n: (e, BUDGET[n]) for n, e in errors.items() if not (e < BUDGET[n])}
    assert not bad, f"超出契约预算：{bad}；全部 {msg}"
    print(f"\n{case_id}: {msg}")


def test_bwd_gate_rejects_bad_inputs():
    """门控必须报错而不是静默降级（AGENTS.md §7）。"""
    _, source_make_inputs, _ = _load_ref()
    src = source_make_inputs(B=1, H=1, HV=1, C=1, K=128, V=128, chunk_size=64, seed=2026)
    x = dict(vars(src))
    npu = {n: x[n].to("npu") for n in ("q", "k", "v")}
    npu |= {n: x[n].float().to("npu") for n in ("g", "beta", "initial_state")}
    _, _, caches = chunk_kda_fwd_with_caches(**npu)
    base = dict(q=npu["q"], k=npu["k"], v=npu["v"], beta=x["beta"].to("npu"),
                do=x["do"].to("npu"), dht=x["dht"].to("npu"), caches=caches)

    with pytest.raises(ValueError, match="block_dim"):
        chunk_kda_bwd(**base, block_dim=5)

    with pytest.raises(ValueError, match="bfloat16"):       # beta 用了 fp32
        chunk_kda_bwd(**{**base, "beta": base["beta"].float()})

    with pytest.raises(ValueError, match="caches 必须恰好是"):
        chunk_kda_bwd(**{**base, "caches": {k: v for k, v in caches.items() if k != "h"}})

    with pytest.raises(ValueError, match=r"caches\['h'\]"):  # h 形状错
        bad = dict(caches)
        bad["h"] = caches["h"][:, :1] if caches["h"].shape[1] > 1 else caches["h"].squeeze(1)
        chunk_kda_bwd(**{**base, "caches": bad})


if __name__ == "__main__":
    # worker 模式：python tests/test_kda_bwd_npu.py <case_id>
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        print(f"用法：{sys.argv[0]} <{'|'.join(CASES)}>", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(run_case(sys.argv[1]), ensure_ascii=False))
