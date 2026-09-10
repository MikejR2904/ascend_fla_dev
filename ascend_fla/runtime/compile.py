"""编译层：ascriptor kernel → 可在进程内调用的 CANN 自定义算子。

用 ascriptor 的 ``OpExec`` 完成"生成源码 → CMake/AscendC 编译 → 安装 vendor 树"
这一段（这条路已在 Ascend950PR / CANN 9.1.0 上实测走通），但**不**使用它的 harness
执行路径；产物交给 :class:`ascend_fla.runtime.binding.AclnnOp` 在本进程内直接调用。

两层缓存：

* **磁盘**：``out_dir`` 按 (kernel, device, block_dim, 标量绑定) 的签名分目录，
  ascriptor 自己的 ``.source_hash`` 让源码未变时跳过重编译。
* **进程内**：同一签名只构造一个 ``AclnnOp``，避免重复 ``dlopen`` 同一个 .so。

标量绑定（``bindings``）会进签名：ascriptor 的 PTO 后端会把整型标量特化进生成的
代码，不同形状可能对应不同产物。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import threading
from typing import Any

from .binding import AclnnOp

__all__ = ["CompiledKernel", "compile_kernel", "default_cache_root"]

_process_cache: dict[str, "CompiledKernel"] = {}
_lock = threading.Lock()


def default_cache_root() -> pathlib.Path:
    """编译产物根目录。

    优先 ``ASCEND_FLA_CACHE``；否则用 ``$TMPDIR/ascend_fla_cache``。远程机器上
    务必让它落在分配给本任务的工作区内 —— 环境脚本应当设好 ``TMPDIR``。
    """
    env = os.environ.get("ASCEND_FLA_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "ascend_fla_cache"


def _kernel_source(kernel: Any) -> str:
    """kernel 的源码文本，用于算签名。取不到时退回模块+名字。"""
    target = getattr(kernel, "fn", kernel)
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', target)}"


def _signature(kernel: Any, device: str, block_dim: int | None, bindings: dict[str, int], backend: str) -> str:
    payload = json.dumps(
        {
            "kernel": getattr(kernel, "name", getattr(kernel, "__name__", str(kernel))),
            "source": _kernel_source(kernel),
            "device": device,
            "block_dim": block_dim,
            "bindings": dict(sorted(bindings.items())),
            "backend": backend,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CompiledKernel:
    """一个已编译的 kernel：持有 aclnn 可调用对象与产物位置。"""

    def __init__(self, op: AclnnOp, vendor_dir: pathlib.Path, out_dir: pathlib.Path,
                 signature: str, spec: Any) -> None:
        self.op = op
        self.vendor_dir = vendor_dir
        self.out_dir = out_dir
        self.signature = signature
        self.spec = spec

    @property
    def input_names(self) -> list[str]:
        return self.op.input_names

    @property
    def scalar_names(self) -> list[str]:
        return self.op.scalar_names

    @property
    def output_names(self) -> list[str]:
        return self.op.output_names

    def __call__(self, inputs: dict, scalars: dict, outputs: dict) -> dict:
        return self.op(inputs, scalars, outputs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CompiledKernel({self.op.op_name}, sig={self.signature}, out_dir={self.out_dir})"


def compile_kernel(
    kernel: Any,
    *,
    device: str = "a5",
    block_dim: int | None = None,
    bindings: dict[str, int] | None = None,
    backend: str = "cce",
    cache_root: str | os.PathLike | None = None,
    force: bool = False,
) -> CompiledKernel:
    """编译一个 ascriptor kernel，返回进程内可直接调用的对象。

    Args:
        kernel: ascriptor ``@kernel`` 装饰的函数。
        device: ascriptor 设备名。``"a5"`` → 950 profile（32 cube / 64 vec）。
            注意 Ascend950PR 实际只有 28 cube / 56 vec —— ``block_dim`` 超过物理
            核数会在硬件 barrier 上死锁，详见 ascriptor boards.json 的 ``cube_cores``。
        block_dim: 启动的核组数。不传则用 kernel 自带的声明。
        bindings: 整型标量的绑定值，参与编译签名。
        backend: ``"cce"``（默认）或 ``"pto_isa"``。
        cache_root: 产物根目录，默认 :func:`default_cache_root`。
        force: 忽略进程内缓存并强制重新 build。

    Returns:
        :class:`CompiledKernel`。

    Raises:
        RuntimeError: 编译未产出 vendor 树，或 vendor 树里找不到 ``libcust_opapi.so``。
    """
    bindings = dict(bindings or {})
    sig = _signature(kernel, device, block_dim, bindings, backend)

    with _lock:
        if not force and sig in _process_cache:
            return _process_cache[sig]

    from ascriptor.runtime.opexec import OpExec  # 延迟导入：没装 ascriptor 也能 import 本模块

    name = getattr(kernel, "name", getattr(kernel, "__name__", "kernel"))
    out_dir = pathlib.Path(cache_root or default_cache_root()) / f"{name}-{sig}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = OpExec(kernel, launcher="aclnn", device=device, block_dim=block_dim,
                backend=backend, bindings=bindings, out_dir=out_dir)
    ex.build(force=force)

    vendor = getattr(ex, "_vendor", None)
    if vendor is None:
        raise RuntimeError(f"{name}: ascriptor 未产出 vendor 树（out_dir={out_dir}）")
    vendor = pathlib.Path(vendor)
    libs = sorted(vendor.rglob("libcust_opapi.so"))
    if not libs:
        raise RuntimeError(f"{name}: vendor 树里没有 libcust_opapi.so（{vendor}）")

    spec = ex.spec
    op = AclnnOp(
        op_name=spec.op,
        opapi_lib=libs[0],
        vendor_dir=vendor,
        input_names=[p["name"] for p in spec.inputs],
        scalar_names=[p["name"] for p in spec.scalars],
        scalar_dtypes=[p["dtype"] for p in spec.scalars],
        output_names=[p["name"] for p in spec.outputs],
    )
    compiled = CompiledKernel(op=op, vendor_dir=vendor, out_dir=out_dir, signature=sig, spec=spec)

    with _lock:
        _process_cache[sig] = compiled
    return compiled
