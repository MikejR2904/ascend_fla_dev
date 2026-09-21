"""FMT-01 host 算子审计：把一个公共算子入口跑一遍，记录它在 host 侧发出的全部 aten / npu_* 算子并分类。

通则见 `docs/pm/bf16-kernel-side.md`（D-PM-35 / D-PM-37）：公共入口里 dtype 转换与格式（布局）转换一律在自编译
kernel 内完成；host 侧只允许分配输出、不搬数据的元数据操作、只读校验（D-PM-42 暂定）、取指针与 launch。

用法（一条命令重跑）::

    # 列出注册的入口 × dtype 路径
    python benchmarks/host_op_audit.py --list

    # 真机：审计一个入口，落一份 JSON + 人读表
    python benchmarks/host_op_audit.py --case kda.chunk_kda.bf16 --output benchmarks/evidence/host_op_audit/run-1

    # 真机：审计全部注册 case（每个 case 一份 JSON，汇总一份 summary.json）
    python benchmarks/host_op_audit.py --all --output benchmarks/evidence/host_op_audit/run-1

    # 纯 CPU：只自检分类器（合成函数，不需要 NPU）
    python benchmarks/host_op_audit.py --self-check

设计要点（规格「已知陷阱」）：

* **分类表是显式的**：`CATEGORY_BY_OP` 逐个算子写死，没有"名字里带 copy 就算拷贝"这类启发式。
  表里没有的算子落到 `unclassified`，报告里单列——**宁可报未分类，也不猜**。
* **"元数据 view" 与 "拷贝" 按是否分配了新 storage 判**，不只看算子名：`contiguous` 在已连续时是 no-op，
  `reshape` 在可 view 时不拷贝。每行都记 `new_storage`（输出的 storage 是否不在输入的 storage 集合里）。
* **只观察，不改行为**：审计不改被测入口的任何东西；`TorchDispatchMode` 只在入口调用期间生效。
* `TorchDispatchMode` 只看得到 aten 层。runtime 桥用 ctypes 直调 aclnn 的那一步在这里看不到——那是我们自己的
  kernel launch，允许，报告里以 `bridge_launch_invisible` 说明；`torch_npu` 的 `npu_*` 自定义算子会经过 dispatch，单列。
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------- 分类表（显式）

#: 分配：产生新 storage 但不搬运已有数据。允许（`bf16-kernel-side.md`：分配输出；带常数填充的分配算分配）。
ALLOC = "alloc"
#: 元数据：不搬数据、不分配新 storage 的视图操作。允许。
META = "meta_view"
#: 只读校验：不产出进入计算的张量的判定。D-PM-42 暂定允许，但必须列出。
CHECK = "readonly_check"
#: 禁止：dtype 转换。
FORBIDDEN_DTYPE = "forbidden_dtype_cast"
#: 禁止：格式 / 布局转换（搬数据的拷贝、重排、复制、拼接、填充、NPU 私有格式转换）。
FORBIDDEN_FORMAT = "forbidden_layout_copy"
#: 禁止：产出进入计算的数据的算术。
FORBIDDEN_ARITH = "forbidden_arithmetic"
#: torch_npu 的自定义算子，单列（是不是违规看它做什么，报告里逐个人工判定）。
NPU_CUSTOM = "npu_custom_op"
#: 表里没有的算子。
UNCLASSIFIED = "unclassified"

FORBIDDEN = (FORBIDDEN_DTYPE, FORBIDDEN_FORMAT, FORBIDDEN_ARITH)

#: 算子 → 类别。键是 `str(func)` 去掉 "aten." 前缀后的 "op.overload"。
#: 只写我们在本仓入口里实际见到过或明确要判的算子；**没列的落 unclassified，不猜**。
CATEGORY_BY_OP: dict[str, str] = {
    # -- 分配
    "empty.memory_format": ALLOC, "empty_strided.default": ALLOC, "empty_like.default": ALLOC,
    "new_empty.default": ALLOC, "new_empty_strided.default": ALLOC,
    "zeros.default": ALLOC, "zeros_like.default": ALLOC, "new_zeros.default": ALLOC,
    "full.default": ALLOC, "full_like.default": ALLOC, "fill_.Scalar": ALLOC,
    "scalar_tensor.default": ALLOC, "lift_fresh.default": ALLOC, "lift_fresh_copy.default": ALLOC,
    "ones.default": ALLOC, "ones_like.default": ALLOC, "rand.default": ALLOC, "randn.default": ALLOC,
    "arange.default": ALLOC, "arange.start": ALLOC, "arange.start_step": ALLOC, "eye.default": ALLOC,
    # -- 元数据（是否真的没拷贝仍按 new_storage 校验）
    "view.default": META, "_unsafe_view.default": META, "as_strided.default": META,
    "slice.Tensor": META, "select.int": META, "detach.default": META, "alias.default": META,
    "unsqueeze.default": META, "squeeze.dim": META, "squeeze.default": META, "squeeze.dims": META,
    "expand.default": META, "permute.default": META, "t.default": META, "transpose.int": META,
    "reshape.default": META, "view_as_real.default": META, "unbind.int": META, "split.Tensor": META,
    "contiguous.default": META,  # 已连续时是 no-op；拷贝了会被 new_storage 抓出来（见 row["copy"]）
    # -- 只读校验（D-PM-42 暂定允许；产出的是判定而不是进入计算的张量）
    "isfinite.default": CHECK, "all.default": CHECK, "any.default": CHECK, "item.default": CHECK,
    "_local_scalar_dense.default": CHECK, "equal.default": CHECK, "allclose.default": CHECK,
    "max.default": CHECK, "min.default": CHECK, "amax.default": CHECK, "amin.default": CHECK,
    "eq.Scalar": CHECK, "ne.Scalar": CHECK, "gt.Scalar": CHECK, "lt.Scalar": CHECK,
    "ge.Scalar": CHECK, "le.Scalar": CHECK, "isnan.default": CHECK,
    "eq.Tensor": CHECK, "ne.Tensor": CHECK, "gt.Tensor": CHECK, "lt.Tensor": CHECK,
    "ge.Tensor": CHECK, "le.Tensor": CHECK, "logical_and.default": CHECK, "logical_or.default": CHECK,
    "logical_not.default": CHECK, "nonzero.default": CHECK, "argmax.default": CHECK,
    "bitwise_and.Tensor": CHECK, "bitwise_or.Tensor": CHECK, "bitwise_not.default": CHECK,
    # -- 禁止：dtype 转换
    "_to_copy.default": FORBIDDEN_DTYPE, "to.dtype": FORBIDDEN_DTYPE, "to.dtype_layout": FORBIDDEN_DTYPE,
    "_cast_Float.default": FORBIDDEN_DTYPE, "_cast_Half.default": FORBIDDEN_DTYPE,
    # -- 禁止：格式 / 布局转换（搬数据）
    "clone.default": FORBIDDEN_FORMAT, "copy_.default": FORBIDDEN_FORMAT, "copy.default": FORBIDDEN_FORMAT,
    "cat.default": FORBIDDEN_FORMAT, "stack.default": FORBIDDEN_FORMAT,
    "constant_pad_nd.default": FORBIDDEN_FORMAT, "pad.default": FORBIDDEN_FORMAT,
    "repeat.default": FORBIDDEN_FORMAT, "repeat_interleave.Tensor": FORBIDDEN_FORMAT,
    "repeat_interleave.self_int": FORBIDDEN_FORMAT, "roll.default": FORBIDDEN_FORMAT,
    "index_select.default": FORBIDDEN_FORMAT, "gather.default": FORBIDDEN_FORMAT,
    "index_put_.default": FORBIDDEN_FORMAT, "index.Tensor": FORBIDDEN_FORMAT,
    "slice_scatter.default": FORBIDDEN_FORMAT, "select_scatter.default": FORBIDDEN_FORMAT,
    "as_strided_copy.default": FORBIDDEN_FORMAT, "narrow_copy.default": FORBIDDEN_FORMAT,
    # -- 禁止：算术（产出进入计算的数据）
    "add.Tensor": FORBIDDEN_ARITH, "add.Scalar": FORBIDDEN_ARITH, "sub.Tensor": FORBIDDEN_ARITH,
    "mul.Tensor": FORBIDDEN_ARITH, "mul.Scalar": FORBIDDEN_ARITH, "div.Tensor": FORBIDDEN_ARITH,
    "div.Scalar": FORBIDDEN_ARITH, "neg.default": FORBIDDEN_ARITH, "abs.default": FORBIDDEN_ARITH,
    "exp.default": FORBIDDEN_ARITH, "exp2.default": FORBIDDEN_ARITH, "log.default": FORBIDDEN_ARITH,
    "log2.default": FORBIDDEN_ARITH, "log1p.default": FORBIDDEN_ARITH, "sqrt.default": FORBIDDEN_ARITH,
    "rsqrt.default": FORBIDDEN_ARITH, "pow.Tensor_Scalar": FORBIDDEN_ARITH, "sigmoid.default": FORBIDDEN_ARITH,
    "softplus.default": FORBIDDEN_ARITH, "silu.default": FORBIDDEN_ARITH, "tanh.default": FORBIDDEN_ARITH,
    "sum.default": FORBIDDEN_ARITH, "sum.dim_IntList": FORBIDDEN_ARITH, "mean.dim": FORBIDDEN_ARITH,
    "cumsum.default": FORBIDDEN_ARITH, "cumsum.dim": FORBIDDEN_ARITH, "linalg_vector_norm.default": FORBIDDEN_ARITH,
    "mm.default": FORBIDDEN_ARITH, "bmm.default": FORBIDDEN_ARITH, "matmul.default": FORBIDDEN_ARITH,
    "addmm.default": FORBIDDEN_ARITH, "baddbmm.default": FORBIDDEN_ARITH, "einsum.default": FORBIDDEN_ARITH,
    "clamp.default": FORBIDDEN_ARITH, "clamp_min.default": FORBIDDEN_ARITH, "where.self": FORBIDDEN_ARITH,
    "maximum.default": FORBIDDEN_ARITH, "minimum.default": FORBIDDEN_ARITH,
    # autograd 对 host 侧算术求导产生的反向算子：host 侧算术的一部分，同样禁止
    "sigmoid_backward.default": FORBIDDEN_ARITH, "softplus_backward.default": FORBIDDEN_ARITH,
    "tanh_backward.default": FORBIDDEN_ARITH, "silu_backward.default": FORBIDDEN_ARITH,
    "threshold_backward.default": FORBIDDEN_ARITH,
}


#: 只读校验是**按调用点**认的，不是按算子名：`torch.isfinite(x).all()` 在 aten 层会分解成
#: abs / ne / eq / mul / all，单看算子名无法与真正的计算区分。这里显式列出本仓里只做判定、
#: 不产出进入计算的张量的函数，**每个都写理由**——这是"允许"的例外，不是默认，要能被审。
#: 名单只收 `ascend_fla/ops/**` 里真实存在的函数；被改判的每一行都在 JSON 里带 `check_by_site: true`，
#: 汇总里有 `reclassified_by_site`（函数 × 算子 × 次数），读者可以逐条复核这条机制没有藏住真的计算。
CHECK_SITE_FUNCTIONS: dict[str, str] = {
    "_gate_span": "kda/chunk.py:467 —— 算 chunk 内门控跨度用于判定上限，返回 float 判据，不产出进入计算的张量",
    "_check_gate_range": "kda/chunk.py:479 —— 拿 _gate_span 的值与 MAX_GATE_SPAN 比，超了报错",
    "_check_input_domain": "kda/chunk.py:164 —— 判 g<=0 / 0<beta<1 / q,k 行范数，只产出 bool",
    "_validate_raw_inputs": "kda/chunk.py:47 —— raw 输入的 dtype / 形状 / 连续性检查",
    "_check": "kda/chunk_bwd.py:145 —— 反向入口的形状 / dtype / caches 校验，返回维度元组",
    "_validate": "gdn_chunk_fwd.py:58、gdn_chunk_bwd.py、pgdn_chunk_fwd.py:62 —— 各入口的 ABI 与域校验",
    "_check_options": "gdn2_chunk_fwd.py:45 —— device / block_dim 取值检查",
    "_constant": "pgdn_chunk_fwd.py:57 —— 常量参数与期望值比对",
}


def classify(op: str) -> str:
    """算子名 → 类别。`npu_*` 单列；表里没有的落 unclassified。"""
    name = op[len("aten."):] if op.startswith("aten.") else op
    if name in CATEGORY_BY_OP:
        return CATEGORY_BY_OP[name]
    bare = op.split(".")[-2] if op.count(".") >= 2 else op
    if bare.startswith("npu_") or op.startswith("npu."):
        return NPU_CUSTOM
    return UNCLASSIFIED


# ---------------------------------------------------------------------------- 记录

@dataclasses.dataclass(frozen=True)
class Row:
    """一次 dispatch 调用的一条记录（同一 (算子, 位置, 形状签名) 会计数合并）。"""
    op: str
    category: str
    site_file: str          # 仓内相对路径；入口之外发出的记 "<outside audited roots>"
    site_line: int
    site_func: str
    input_dtypes: str
    output_dtypes: str
    input_shapes: str
    output_shapes: str
    dtype_changed: bool     # 输出 dtype 不在输入 dtype 集合里
    copy: bool              # 输出分配了新 storage 且不是纯分配类算子 → 搬了数据
    contiguous_in: str
    check_by_site: bool = False   # 本行的类别是被"只读校验按调用点认"的规则改判过来的

    def key(self):
        return dataclasses.astuple(self)


class HostOpAudit:
    """`TorchDispatchMode`：记录期间发出的每个算子，按显式表分类，并把调用点回溯到被审计的源码行。"""

    def __init__(self, roots=("ascend_fla/ops",), repo: Path = REPO):
        self.roots = tuple(roots)
        self.repo = Path(repo)
        self.counts: collections.Counter = collections.Counter()
        self.first_seen: dict = {}
        self._mode = None

    # -- 上下文：用真正的 TorchDispatchMode 子类（延迟导入 torch，便于 --list 在无 torch 环境下工作）
    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        audit = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                try:
                    audit._record(func, args, kwargs, out)
                except Exception as error:  # noqa: BLE001 - 审计不能改变被测行为
                    audit.counts[("<audit-error>", UNCLASSIFIED, str(type(error).__name__), 0, "", "", "", "", "",
                                  False, False, "")] += 1
                return out

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        mode, self._mode = self._mode, None
        return mode.__exit__(*exc)

    # -- 记录一次调用
    def _record(self, func, args, kwargs, out):
        import torch

        def tensors(value, acc):
            if isinstance(value, torch.Tensor):
                acc.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    tensors(item, acc)
            elif isinstance(value, dict):
                for item in value.values():
                    tensors(item, acc)
            return acc

        ins = tensors(args, []) + tensors(kwargs, [])
        outs = tensors(out, [])
        in_storages = {t.untyped_storage().data_ptr() for t in ins if t.numel() or True}
        op = str(func)
        category = classify(op)
        site_func = str(self._site(2) or "")
        check_by_site = False
        if category in FORBIDDEN + (UNCLASSIFIED,) and self._in_check_site():
            category, check_by_site = CHECK, True  # 只读校验的分解产物：按调用点认，报告里单列
        in_dtypes = sorted({str(t.dtype).replace("torch.", "") for t in ins})
        out_dtypes = sorted({str(t.dtype).replace("torch.", "") for t in outs})
        new_storage = any(t.untyped_storage().data_ptr() not in in_storages for t in outs)
        row = Row(
            op=op,
            category=category,
            site_file=self._site(0),
            site_line=int(self._site(1) or 0),
            site_func=site_func,
            input_dtypes=",".join(in_dtypes),
            output_dtypes=",".join(out_dtypes),
            input_shapes=";".join("x".join(map(str, t.shape)) for t in ins[:4]),
            output_shapes=";".join("x".join(map(str, t.shape)) for t in outs[:4]),
            dtype_changed=bool(out_dtypes) and any(d not in in_dtypes for d in out_dtypes) and bool(in_dtypes),
            copy=bool(new_storage and category != ALLOC),
            contiguous_in=",".join("1" if t.is_contiguous() else "0" for t in ins[:4]),
            check_by_site=check_by_site,
        )
        self.counts[row.key()] += 1
        self.first_seen.setdefault(row.key(), len(self.first_seen))

    def _in_check_site(self) -> bool:
        """调用栈里是否有登记的只读校验函数（见 CHECK_SITE_FUNCTIONS）。"""
        return any(frame.name in CHECK_SITE_FUNCTIONS for frame in traceback.extract_stack())

    def _site(self, which):
        """回溯到被审计根目录（默认 ascend_fla/ops）里最内层的那一帧。"""
        frames = traceback.extract_stack()
        hit = None
        for frame in frames:
            try:
                rel = str(Path(frame.filename).resolve().relative_to(self.repo))
            except ValueError:
                continue
            if any(rel.startswith(root) for root in self.roots):
                hit = (rel, frame.lineno, frame.name)
        if hit is None:
            # 入口之外：把 autograd 引擎单独标出来——反向节点在 C++ 里跑，Python 栈上没有我们的帧，
            # 但它求的正是 host 侧那段算术的导数，所以仍然是 host 侧违规，只是归因不到行。
            engine = any("torch/autograd" in frame.filename or "torch/_tensor" in frame.filename
                         for frame in frames)
            hit = ("<autograd engine: backward of host-side math>" if engine else "<outside audited roots>", 0, "")
        return hit[which]

    # -- 报告
    def rows(self):
        out = []
        for key, count in self.counts.items():
            row = dict(zip([f.name for f in dataclasses.fields(Row)], key))
            row["count"] = count
            row["order"] = self.first_seen.get(key, 0)
            out.append(row)
        return sorted(out, key=lambda r: (r["order"],))

    def summary(self):
        by_category: collections.Counter = collections.Counter()
        for row in self.rows():
            by_category[row["category"]] += row["count"]
        rows = self.rows()
        forbidden = [r for r in rows if r["category"] in FORBIDDEN]
        # `contiguous` 等标了 META 但实际分配了新 storage 的，按"搬了数据"升级为格式转换
        upgraded = [r for r in rows if r["category"] == META and r["copy"]]
        return {
            "total_calls": sum(by_category.values()),
            "by_category": dict(sorted(by_category.items())),
            "forbidden_calls": sum(r["count"] for r in forbidden),
            "meta_that_copied": [{k: r[k] for k in ("op", "site_file", "site_line", "count")} for r in upgraded],
            "unclassified_ops": sorted({r["op"] for r in rows if r["category"] == UNCLASSIFIED}),
            # 被"按调用点认"改判为只读校验的行：函数 × 算子 × 次数，供读者复核这条机制（PM 04:25Z / REVIEW 第 4 条）
            "reclassified_by_site": sorted(
                ({"site_func": r["site_func"], "site_file": r["site_file"], "site_line": r["site_line"],
                  "op": r["op"], "count": r["count"]} for r in rows if r.get("check_by_site")),
                key=lambda x: (x["site_func"], x["op"])),
            "check_site_functions": dict(CHECK_SITE_FUNCTIONS),
            "npu_custom_ops": sorted({r["op"] for r in rows if r["category"] == NPU_CUSTOM}),
            "verdict": "clean" if not forbidden and not upgraded else "violations",
            "bridge_launch_invisible": "runtime 桥用 ctypes 直调 aclnn，不经过 aten dispatch，本工具看不到；那是自编译 kernel 的 launch，允许。",
        }


# ---------------------------------------------------------------------------- 环境与哈希

def git_head(repo: Path = REPO) -> str:
    """被审计的 main 提交。真机上跑的是 `git archive` 解出来的副本（不是 git checkout），
    所以允许用 `HOST_OP_AUDIT_COMMIT` 显式声明；两者都没有才记 unknown。"""
    declared = os.environ.get("HOST_OP_AUDIT_COMMIT")
    if declared:
        return declared
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256_of(paths) -> dict:
    out = {}
    for path in sorted({Path(p) for p in paths}):
        try:
            out[str(Path(path).resolve().relative_to(REPO))] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001
            continue
    return out


def redact(text: str) -> str:
    """去掉机器标识：仓库公开，证据里不许出现绝对路径 / 主机名 / 账号（`AGENTS.md` §5）。

    报错信息里常带 kernel 源码的绝对路径（ascriptor 的 `loc(...)`），所以写盘前统一过一遍。
    """
    if not text:
        return text
    for value, token in ((os.environ.get("ASCRIPTOR_WORKSPACE"), "<ascriptor>"), (str(REPO), "<repo>"),
                         (os.environ.get("HOME"), "<home>")):
        if value:
            text = text.replace(str(Path(value).resolve()), token).replace(str(value), token)
    # 兜底：任何残留的绝对路径都替换掉，只留最后两段（文件名与上一级目录）
    return re.sub(r"(/[\w.+-]+){2,}/([\w.+-]+/[\w.+-]+)", r"<path>/\2", text)


def environment() -> dict:
    """环境行：CANN 版本 + compiler timestamp + opp 目录 + torch / torch_npu 版本。不含主机标识。"""
    info: dict = {"python": platform.python_version(), "audited_commit": git_head(),
                  "audit_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception as error:  # noqa: BLE001
        info["torch"] = f"unavailable: {type(error).__name__}"
    try:
        import torch_npu
        info["torch_npu"] = torch_npu.__version__
    except Exception as error:  # noqa: BLE001
        info["torch_npu"] = f"unavailable: {type(error).__name__}"
    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    for name in ("ascend_toolkit_install.info", "version.info"):
        for candidate in (Path(toolkit) / name, *(Path(toolkit).glob(f"*/{name}") if toolkit else ())):
            if candidate.is_file():
                # 记下是哪个文件：同一棵树里 toolkit 的 install.info 与子包的 version.info 可能报不同版本号
                info.setdefault("cann", {})[name] = dict(
                    path=str(candidate.relative_to(Path(toolkit))) if toolkit else candidate.name,
                    toolkit_home_basename=Path(toolkit).name if toolkit else "",
                    line=[ln for ln in candidate.read_text().splitlines() if ln.strip()][:6],
                    sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
                break
    compiler = Path(toolkit) / "compiler/ccec_compiler/bin" if toolkit else None
    if compiler and compiler.is_dir():
        info["compiler_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(compiler.stat().st_mtime))
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    kernels = Path(opp) / "built-in/op_impl/ai_core/tbe/kernel" if opp else None
    if kernels and kernels.is_dir():
        info["opp_kernel_dirs"] = sorted(p.name for p in kernels.iterdir() if p.is_dir())
    info["ascend_custom_opp_path_set"] = bool(os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
    return info


# ---------------------------------------------------------------------------- 被审计的入口（case 注册表）

UNITS = REPO / "kernels/projects/a5"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unit_inputs(unit: str, parameters: dict, seed: int = 2026):
    """用单元自带的 ref/reference.py 造输入（只读引用，和该单元的测试同一条路径）。"""
    ref = _load(f"_audit_ref_{unit}", UNITS / unit / "ref/reference.py")
    return ref.make_inputs({"seed": seed, "parameters": parameters})


def _to(device, tensors: dict, dtypes: dict | None = None):
    import torch

    out = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            out[name] = value
            continue
        want = (dtypes or {}).get(name)
        # 造输入阶段的 dtype 转换在 host 上做是允许的：它在被审计入口之外，审计只覆盖入口调用期间。
        out[name] = (value if want is None else value.to(want)).contiguous().to(device)
    return out


def _kda_inputs(device, *, b=1, t=128, h=1, hv=2, raw=False, seed=2026):
    import torch

    gen = torch.Generator().manual_seed(seed)
    k_dim = v_dim = 128
    data = dict(
        q=(torch.randn(b, t, h, k_dim, generator=gen) * 0.04),
        k=(torch.randn(b, t, h, k_dim, generator=gen) * 0.04),
        v=(torch.randn(b, t, hv, v_dim, generator=gen) * 0.04),
        beta=torch.empty(b, t, hv).uniform_(0.05, 0.5, generator=gen),
        initial_state=torch.randn(b, hv, k_dim, v_dim, generator=gen) * 0.01,
    )
    if raw:  # 原始输入路径：q/k 未归一化、g 是 raw、beta 是 logit，另需 A_log / dt_bias
        data["g"] = torch.randn(b, t, hv, k_dim, generator=gen) * 0.5
        data["beta"] = torch.randn(b, t, hv, generator=gen)
        data["A_log"] = torch.log(torch.empty(hv).uniform_(1.0, 16.0, generator=gen))
        data["dt_bias"] = torch.empty(hv * k_dim).uniform_(-4.0, -3.0, generator=gen)
        dtypes = {"q": torch.bfloat16, "k": torch.bfloat16, "v": torch.bfloat16, "g": torch.float32,
                  "beta": torch.float32, "initial_state": torch.float32, "A_log": torch.float32,
                  "dt_bias": torch.float32}
    else:    # 已准备好的路径：q/k 已 L2 归一化，g <= 0，0 < beta < 1
        import torch.nn.functional as functional

        data["q"] = functional.normalize(data["q"], dim=-1)
        data["k"] = functional.normalize(data["k"], dim=-1)
        data["g"] = torch.empty(b, t, hv, k_dim).uniform_(-0.03, 0.0, generator=gen)
        dtypes = {"q": torch.bfloat16, "k": torch.bfloat16, "v": torch.bfloat16, "g": torch.float32,
                  "beta": torch.float32, "initial_state": torch.float32}
    return _to(device, data, dtypes)


def _require_grad(tensors, names):
    for name in names:
        if name in tensors:
            tensors[name].requires_grad_(True)
    return tensors


def _backward(outputs):
    """在审计范围内触发反向：反向里的 host 算子也要被记下来。"""
    primary = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    primary.float().sum().backward() if primary.requires_grad else None
    return primary


def case_kda_chunk(device, *, raw: bool, grad: bool, dtype=None):
    from ascend_fla.ops.kda import chunk_kda, prepare

    prepare(block_dim=1, backward=grad)
    data = _kda_inputs(device, raw=raw)
    names = ("q", "k", "v", "g", "beta", "initial_state") if grad else ()
    data = _require_grad(data, names)
    kwargs = dict(initial_state=data["initial_state"], output_final_state=True, block_dim=1)
    if raw:
        kwargs.update(A_log=data["A_log"], dt_bias=data["dt_bias"], use_qk_l2norm_in_kernel=True,
                      use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)

    def run():
        out = chunk_kda(data["q"], data["k"], data["v"], data["g"], data["beta"], **kwargs)
        if grad:
            _backward(out)
        return out

    return run, [Path(__import__("ascend_fla.ops.kda.autograd", fromlist=["x"]).__file__),
                 Path(__import__("ascend_fla.ops.kda.chunk", fromlist=["x"]).__file__)]


def case_kda_fwd_with_caches(device, *, impl="stable", **_):
    from ascend_fla.ops.kda import chunk_kda_fwd_with_caches, prepare

    prepare(block_dim=1, backward=True)
    data = _kda_inputs(device)

    def run():
        return chunk_kda_fwd_with_caches(data["q"], data["k"], data["v"], data["g"], data["beta"],
                                        initial_state=data["initial_state"], block_dim=1, impl=impl)

    return run, [Path(__import__("ascend_fla.ops.kda.chunk", fromlist=["x"]).__file__)]


def case_kda_bwd(device, **_):
    import torch
    from ascend_fla.ops.kda import BWD_CACHE_NAMES, chunk_kda_bwd, chunk_kda_fwd_with_caches, prepare

    prepare(block_dim=1, backward=True)
    data = _kda_inputs(device)
    result = chunk_kda_fwd_with_caches(data["q"], data["k"], data["v"], data["g"], data["beta"],
                                      initial_state=data["initial_state"], block_dim=1)
    caches = result[-1] if isinstance(result[-1], dict) else {n: result[i] for i, n in enumerate(BWD_CACHE_NAMES)}
    o = result[0]
    # kda_bwd 的 ABI：q/k/v/beta/do/dht 全 BF16（chunk_bwd.py 的 docstring 与 _validate）。
    # 这些转换在被审计入口之外做，属于造输入，不计进审计。
    beta_bf16 = data["beta"].to(torch.bfloat16)
    do = torch.randn_like(o.float()).to(torch.bfloat16)
    dht = torch.zeros_like(data["initial_state"]).to(torch.bfloat16)

    def run():
        return chunk_kda_bwd(data["q"], data["k"], data["v"], beta_bf16, do, dht, caches, block_dim=1)

    return run, [Path(__import__("ascend_fla.ops.kda.chunk_bwd", fromlist=["x"]).__file__)]


def case_kda_decode(device, *, dtype):
    import torch
    from ascend_fla.ops.kda import fused_recurrent_kda, prepare

    prepare(block_dim=1, backward=False, decode=True)
    data = _kda_inputs(device, t=1)
    qk = torch.bfloat16 if dtype == "bf16" else torch.float32
    q, k, v = (data[n].to(qk) for n in ("q", "k", "v"))

    def run():
        return fused_recurrent_kda(q, k, v, data["g"], data["beta"], initial_state=data["initial_state"],
                                  output_final_state=True)

    return run, [Path(__import__("ascend_fla.ops.kda.fused_recurrent", fromlist=["x"]).__file__)]


def case_gdn_fwd(device, *, dtype):
    import torch
    from ascend_fla.ops.gdn_chunk_fwd import chunk_gdn, prepare

    prepare(block_dim=2)
    want = torch.bfloat16 if dtype == "bf16" else torch.float32
    raw = _unit_inputs("gdn_chunk_fwd", {"B": 1, "T": 128, "H": 2, "gate_scale": 0.03})
    data = _to(device, raw, {n: want for n in ("q", "k", "v")})

    def run():
        return chunk_gdn(data["q"], data["k"], data["v"], data["g"], data["beta"],
                        initial_state=None, output_final_state=True, block_dim=2)

    return run, [Path(__import__("ascend_fla.ops.gdn_chunk_fwd", fromlist=["x"]).__file__)]


def case_gdn_bwd(device, *, dtype):
    import torch
    from ascend_fla.ops.gdn_chunk_bwd import chunk_gdn_bwd, prepare

    prepare(block_dim=2)
    want = torch.bfloat16 if dtype == "bf16" else torch.float32
    raw = _unit_inputs("gdn_chunk_fwd", {"B": 1, "T": 128, "H": 2, "gate_scale": 0.03})
    data = _to(device, raw, {n: want for n in ("q", "k", "v")})
    do = torch.randn_like(data["v"])

    def run():
        return chunk_gdn_bwd(data["q"], data["k"], data["v"], data["g"], data["beta"], do=do,
                            initial_state=None, block_dim=2)

    return run, [Path(__import__("ascend_fla.ops.gdn_chunk_bwd", fromlist=["x"]).__file__)]


def case_pgdn_fwd(device, *, dtype):
    import torch
    from ascend_fla.ops.pgdn_chunk_fwd import chunk_pgdn, prepare

    prepare(block_dim=2)
    want = torch.bfloat16 if dtype == "bf16" else torch.float32
    raw = _unit_inputs("pgdn_chunk_fwd", {"B": 1, "T": 128, "H": 2, "HV": 2, "gate_scale": 0.03})
    data = _to(device, raw, {n: want for n in ("q", "k", "v")})

    def run():
        return chunk_pgdn(data["q"], data["k"], data["v"], data["g_atk"], data["g"], data["beta_atk"],
                         data["beta"], initial_state=None, output_final_state=True, block_dim=2)

    return run, [Path(__import__("ascend_fla.ops.pgdn_chunk_fwd", fromlist=["x"]).__file__)]


def case_pkda_fwd(device, *, dtype):
    import torch
    from ascend_fla.ops.pkda_chunk_fwd import chunk_precond_kda, prepare

    want = torch.bfloat16 if dtype == "bf16" else torch.float32
    prepare(block_dim=1, dtype=want)
    # PKDA 的 ABI 要求 q/k 行 L2 范数 <= 1+1e-5。单位向量在 BF16 下舍入后会略微越界，
    # 所以造输入时按 0.99 缩一下（造输入在被审计入口之外，不计进审计）。
    raw = _unit_inputs("pkda_chunk_fwd", {"B": 1, "T": 128, "H": 2, "gate_scale": 0.1, "row_norm": 0.99})
    data = _to(device, raw, {n: want for n in ("q", "k", "v")})  # g / g_atk / beta* 保持 FP32

    def run():
        return chunk_precond_kda(data["q"], data["k"], data["v"], data["g"], data["g_atk"],
                                data["beta_atk"], data["beta"], output_final_state=True, block_dim=1)

    return run, [Path(__import__("ascend_fla.ops.pkda_chunk_fwd", fromlist=["x"]).__file__)]


def case_gdn2_fwd(device, *, dtype):
    import torch
    from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2, prepare

    prepare(block_dim=8)
    want = torch.bfloat16 if dtype == "bf16" else torch.float32
    # GDN-2 的公开域：B=1、H=1 或 16（ops/gdn2_chunk_fwd.py 的 _validate）
    raw = _unit_inputs("gdn2_chunk_fwd", {"B": 1, "T": 128, "H": 1})
    data = _to(device, raw, {n: want for n in ("q", "k", "v")})
    erase = data.get("erase_gate", data.get("b"))

    def run():
        return chunk_gdn2(data["q"], data["k"], data["v"], data["g"], erase, data["w"],
                         initial_state=data["initial_state"], output_final_state=True, block_dim=8)

    return run, [Path(__import__("ascend_fla.ops.gdn2_chunk_fwd", fromlist=["x"]).__file__)]


#: case id → (构造函数, kwargs, 备注)。id 形如 `family.entry.path`，一条命令按 id 重跑。
CASES: dict[str, tuple] = {
    "kda.chunk_kda.prepared_nograd": (case_kda_chunk, dict(raw=False, grad=False), "推理路径，raw flags 关"),
    "kda.chunk_kda.prepared_grad": (case_kda_chunk, dict(raw=False, grad=True), "训练路径（含反向），raw flags 关"),
    "kda.chunk_kda.rawflags_nograd": (case_kda_chunk, dict(raw=True, grad=False), "raw flags 开（BF-07 后的推理路径）"),
    "kda.chunk_kda.rawflags_grad": (case_kda_chunk, dict(raw=True, grad=True), "raw flags 开 + 反向（BF-08 在做）"),
    "kda.chunk_kda_fwd_with_caches.bf16": (case_kda_fwd_with_caches, {}, "带缓存前向（九个反向检查点）"),
    "kda.chunk_kda_fwd_with_caches.upstream_impl": (case_kda_fwd_with_caches, dict(impl="upstream"),
        "impl=upstream：走 D-PM-42 登记的 log2(eg) 分支"),
    "kda.chunk_kda_bwd.bf16": (case_kda_bwd, {}, "反向入口，直接调用"),
    "kda.fused_recurrent_kda.bf16": (case_kda_decode, dict(dtype="bf16"), "decode，BF16"),
    "kda.fused_recurrent_kda.fp32": (case_kda_decode, dict(dtype="fp32"), "decode，FP32"),
    "gdn.chunk_gdn.fp32": (case_gdn_fwd, dict(dtype="fp32"), None),
    "gdn.chunk_gdn.bf16": (case_gdn_fwd, dict(dtype="bf16"), "BF-02 后"),
    "gdn.chunk_gdn_bwd.fp32": (case_gdn_bwd, dict(dtype="fp32"), None),
    "gdn.chunk_gdn_bwd.bf16": (case_gdn_bwd, dict(dtype="bf16"), None),
    "pgdn.chunk_pgdn.fp32": (case_pgdn_fwd, dict(dtype="fp32"), None),
    "pgdn.chunk_pgdn.bf16": (case_pgdn_fwd, dict(dtype="bf16"), "BF-03 后"),
    "pkda.chunk_precond_kda.fp32": (case_pkda_fwd, dict(dtype="fp32"), "阴性对照：应报干净"),
    "pkda.chunk_precond_kda.bf16": (case_pkda_fwd, dict(dtype="bf16"), "BF-04 后"),
    "gdn2.chunk_gdn2.fp32": (case_gdn2_fwd, dict(dtype="fp32"), "仓主轨道：只跑不改"),
    "gdn2.chunk_gdn2.bf16": (case_gdn2_fwd, dict(dtype="bf16"), "仓主轨道：只跑不改（BF-05 在做）"),
}


# ---------------------------------------------------------------------------- 运行一个 case

def audit_case(case_id: str, *, device: str = "npu", roots=("ascend_fla/ops",)) -> dict:
    """跑一个 case，返回可 JSON 序列化的审计结果（含环境行、被审计源文件哈希、逐行记录、汇总）。"""
    if case_id not in CASES:
        raise SystemExit(f"未注册的 case: {case_id}；--list 看全部")
    builder, kwargs, note = CASES[case_id]
    started = time.monotonic()
    run, sources = builder(device, **kwargs)          # 构造输入 / prepare 在审计范围之外
    audit = HostOpAudit(roots=roots)
    error = None
    with audit:
        try:
            run()
        except Exception as exc:  # noqa: BLE001 - 跑不起来也要如实记录，不绕
            error = f"{type(exc).__name__}: {exc}"
    return {
        "case": case_id,
        "note": note,
        "device": device,
        "audited_roots": list(roots),
        "environment": environment(),
        "audited_sources_sha256": sha256_of(sources),
        "error": redact(error),
        "seconds": round(time.monotonic() - started, 2),
        # 入口报错时 verdict 不能读成"干净"：一条记录都没有就是 did-not-run，跑了一半就是 incomplete-*
        "summary": {**audit.summary(),
                    **({"verdict": "did-not-run"} if error and not audit.rows()
                       else {"verdict": "incomplete-" + audit.summary()["verdict"]} if error else {})},
        "rows": audit.rows(),
    }


def human_table(result: dict) -> str:
    lines = [f"case {result['case']}  device={result['device']}  verdict={result['summary']['verdict']}"
             f"  calls={result['summary']['total_calls']}" + (f"  ERROR {result['error']}" if result["error"] else ""),
             f"  by_category: {result['summary']['by_category']}",
             f"  {'category':<24} {'operator':<34} {'site':<44} {'copy':<5} {'dtype→':<7} count"]
    for row in result["rows"]:
        site = f"{row['site_file']}:{row['site_line']} {row['site_func']}"
        lines.append(f"  {row['category']:<24} {row['op'][:34]:<34} {site[:44]:<44} "
                     f"{'yes' if row['copy'] else '':<5} {'yes' if row['dtype_changed'] else '':<7} {row['count']}")
    for name in ("unclassified_ops", "npu_custom_ops", "meta_that_copied"):
        if result["summary"].get(name):
            lines.append(f"  {name}: {json.dumps(result['summary'][name], ensure_ascii=False)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- 自检（纯 CPU，无需 NPU）

def self_check() -> int:
    """合成函数的正反例：分类器与 copy 判定在 CPU 上就能验。"""
    import torch

    def synthetic():
        x = torch.ones(4, 8)                 # 入口外
        y = x.view(2, 16)                    # meta_view，无新 storage
        z = x.float()                        # 同 dtype：no-op，不该报 dtype 转换
        w = x.to(torch.bfloat16)             # forbidden_dtype_cast
        c = x.t().contiguous()               # forbidden：contiguous 真的拷贝了
        a = x * 2                            # forbidden_arithmetic
        e = torch.empty(4, 8)                # alloc
        return y, z, w, c, a, e

    audit = HostOpAudit(roots=("benchmarks",))
    with audit:
        synthetic()
    rows = audit.rows()
    got = {row["op"]: row for row in rows}
    checks = []

    def expect(name, condition, detail=""):
        checks.append((name, bool(condition), detail))

    ops = set(got)
    expect("记录到调用", len(rows) > 0, f"rows={len(rows)}")
    dtype_rows = [r for r in rows if r["category"] == FORBIDDEN_DTYPE]
    expect("dtype 转换被判为 forbidden_dtype_cast", dtype_rows, f"ops={sorted(ops)}")
    expect("dtype 转换的 dtype_changed=True", all(r["dtype_changed"] for r in dtype_rows),
           json.dumps([r["input_dtypes"] + "->" + r["output_dtypes"] for r in dtype_rows]))
    arith = [r for r in rows if r["category"] == FORBIDDEN_ARITH]
    expect("算术被判为 forbidden_arithmetic", arith, f"ops={[r['op'] for r in arith]}")
    alloc = [r for r in rows if r["category"] == ALLOC]
    expect("分配被判为 alloc", alloc, f"ops={[r['op'] for r in alloc]}")
    meta = [r for r in rows if r["category"] == META]
    expect("元数据 view 被判为 meta_view", meta, f"ops={[r['op'] for r in meta]}")
    expect("元数据 view 没有被判成拷贝", all(not r["copy"] for r in meta if r["op"] != "aten.contiguous.default"),
           json.dumps([[r["op"], r["copy"]] for r in meta]))
    copied = [r for r in rows if r["copy"]]
    expect("真的拷贝了的调用被标 copy=True", copied, f"ops={[r['op'] for r in copied]}")
    expect("汇总判定为 violations", audit.summary()["verdict"] == "violations", audit.summary()["verdict"])
    expect("JSON 可序列化", json.dumps({"rows": rows, "summary": audit.summary()}, ensure_ascii=False))
    # 清洁对照：只做分配与元数据的函数必须报 clean
    clean = HostOpAudit(roots=("benchmarks",))
    with clean:
        buffer = torch.empty(4, 8)
        buffer.view(8, 4).unsqueeze(0)
    expect("干净函数报 clean", clean.summary()["verdict"] == "clean", json.dumps(clean.summary()["by_category"]))
    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
    failed = [n for n, ok, _ in checks if not ok]
    print(f"self-check: {len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


# ---------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="列出注册的 case")
    parser.add_argument("--self-check", action="store_true", help="纯 CPU 自检分类器")
    parser.add_argument("--case", action="append", default=[], help="审计这个 case（可重复）")
    parser.add_argument("--all", action="store_true", help="审计全部注册 case")
    parser.add_argument("--device", default="npu", help="张量设备（默认 npu）")
    parser.add_argument("--roots", default="ascend_fla/ops", help="被审计的源码根，逗号分隔")
    parser.add_argument("--output", type=Path, help="结果目录：每个 case 一份 JSON + summary.json")
    parser.add_argument("--in-process", action="store_true",
                        help="在当前进程里跑（子进程模式内部用；多 case 时默认每个 case 一个进程）")
    args = parser.parse_args(argv)

    if args.list:
        for case_id, (_, kwargs, note) in CASES.items():
            print(f"{case_id:<42} {json.dumps(kwargs, default=str):<34} {note or ''}")
        return 0
    if args.self_check:
        return self_check()
    cases = list(CASES) if args.all else args.case
    if not cases:
        parser.error("要 --list / --self-check / --case <id> / --all 之一")
    roots = tuple(r.strip() for r in args.roots.split(",") if r.strip())
    results = []
    # **一个 case 一个进程**：runtime 桥只在首次算子解析时注册 vendor 树（ASCEND_CUSTOM_OPP_PATH 只被读一次），
    # 同一进程里换一个算子族再 prepare() 会报"已经执行过 aclnn 算子"（AGENTS.md §5）。所以多 case 一律派子进程。
    if len(cases) > 1 and not args.in_process:
        if not args.output:
            parser.error("多个 case 要 --output（每个 case 一份 JSON）")
        args.output.mkdir(parents=True, exist_ok=True)
        for case_id in cases:
            command = [sys.executable, str(Path(__file__).resolve()), "--case", case_id, "--device", args.device,
                       "--roots", args.roots, "--output", str(args.output), "--in-process"]
            print(f"=== {case_id}: {' '.join(command[1:])}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True)
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            target = args.output / f"{case_id}.json"
            if target.is_file():
                results.append(json.loads(target.read_text()))
            else:  # 子进程没落盘：如实记一条失败，不跳过
                results.append({"case": case_id, "note": CASES[case_id][2], "device": args.device,
                                "environment": environment(), "audited_sources_sha256": {},
                                "error": f"subprocess exit {completed.returncode}: "
                                         f"{(completed.stderr or completed.stdout).strip().splitlines()[-1:] or ['no output']}",
                                "summary": {"verdict": "did-not-run", "by_category": {}, "forbidden_calls": 0,
                                            "total_calls": 0, "unclassified_ops": []}, "rows": []})
                (args.output / f"{case_id}.json").write_text(json.dumps(results[-1], ensure_ascii=False, indent=1) + "\n")
    else:
        for case_id in cases:
            result = audit_case(case_id, device=args.device, roots=roots)
            print(human_table(result), flush=True)
            results.append(result)
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / f"{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    if args.output:
        summary = {"environment": environment(),
                   "cases": [{"case": r["case"], "verdict": r["summary"]["verdict"], "error": r["error"],
                              "by_category": r["summary"]["by_category"],
                              "forbidden_calls": r["summary"]["forbidden_calls"],
                              "unclassified_ops": r["summary"]["unclassified_ops"]} for r in results]}
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {len(results)} case file(s) + summary.json to {args.output}")
    return 1 if any(r["error"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
