"""ctypes 绑定层：在本进程内调用 ascriptor 编译出的 CANN 自定义算子。

这是 runtime 桥的核心。ascriptor 自己的 aclnn launcher 走"写参数文件 → 跑独立
``test_aclnnop`` 进程 → 读回输出"，输出落在 CPU 上；本模块绕过 harness，直接把
torch NPU tensor 的 ``data_ptr()`` 交给 ``aclCreateTensor``，**零拷贝**。

关键事实（均在 Ascend950PR / CANN 9.1.0 上实测确认）：

* ``aclCreateTensor`` / ``aclDestroyTensor`` 在 **libnnopbase.so**，不在
  libascendcl.so —— 后者没有这个符号。
* aclnn 调用是两段式：
  ``aclnn<Op>GetWorkspaceSize(inputs…, scalars…, outputs…, &wsSize, &executor)``
  然后 ``aclnn<Op>(wsAddr, wsSize, executor, stream)``。参数顺序由 ascriptor 的
  ``HostSpec`` 决定（``spec.inputs + spec.scalars + spec.outputs``）。
* stream 指针取自 ``torch.npu.current_stream().npu_stream``。
* CANN 需要 ``ASCEND_CUSTOM_OPP_PATH`` 指向 vendor 树才能找到算子的 JSON 配置。

torch_npu 的可用面比想象的窄：在内置算子包不覆盖的 SoC 上，``torch.zeros`` 这类
要走算子的 API 会失败，但 ``torch.empty`` / H2D / D2H / ``data_ptr`` / stream 都可用
—— 这正好够本模块用，计算全在自编译的 kernel 里。
"""
from __future__ import annotations

import ctypes
import os
import threading
from typing import Any

import torch

__all__ = ["ACL_DTYPE", "SCALAR_CTYPE", "AclnnOp", "acl_libs", "current_stream_ptr", "register_custom_opp_path"]

# 取自 CANN 9.1.0 的 acl/acl_base_rt.h（实测核对，不要凭记忆填）
ACL_DTYPE: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int8: 2,
    torch.int32: 3,
    torch.uint8: 4,
    torch.int16: 6,
    torch.int64: 9,
    torch.float64: 11,
    torch.bool: 12,
    torch.bfloat16: 27,
}
ACL_FORMAT_ND = 2

# ascriptor 标量 dtype → aclnn 接口的 ctypes 类型。
#
# ⚠️ 以生成的 aclnn_*.h 为准，不要照搬 ascriptor 的 SCALAR_C（那是 kernel 侧的
# C 类型）。CANN 的 aclnn 接口层会放宽 attr 类型：
#   整型 attr（i32/i64/…）-> int64_t
#   浮点 attr（f32/f16）   -> **double**，不是 float
# 按 c_float 传 4 字节会让被调方从 8 字节 double 槽里读到垃圾值。实测表现为
# scale 变成近 0，于是只有用到 scale 的输出归零、不用的输出照常正确 —— 很隐蔽。
SCALAR_CTYPE: dict[str, Any] = {
    "i32": ctypes.c_int64, "i64": ctypes.c_int64, "u32": ctypes.c_int64,
    "b1": ctypes.c_int64, "i8": ctypes.c_int64, "u8": ctypes.c_int64,
    "i16": ctypes.c_int64, "u16": ctypes.c_int64,
    "f32": ctypes.c_double, "f16": ctypes.c_double,
}

_lock = threading.Lock()
_libs: _AclLibs | None = None


class AclError(RuntimeError):
    """aclnn 调用返回了非零状态，或符号/库缺失。"""


class _AclLibs:
    """懒加载的 ACL 库句柄。进程内只加载一次。"""

    def __init__(self) -> None:
        # libascendcl 先加载：它带 runtime 符号，libnnopbase 依赖其中一部分
        ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        self.nnop = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)

        self.create_tensor = self.nnop.aclCreateTensor
        self.create_tensor.restype = ctypes.c_void_p
        self.create_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,     # viewDims, num, dtype
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,      # stride, offset, format
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,  # storageDims, num, data
        ]
        self.destroy_tensor = self.nnop.aclDestroyTensor
        self.destroy_tensor.argtypes = [ctypes.c_void_p]
        self.destroy_tensor.restype = ctypes.c_int


def acl_libs() -> _AclLibs:
    global _libs
    with _lock:
        if _libs is None:
            _libs = _AclLibs()
        return _libs


def register_custom_opp_path(vendor_dir: str | os.PathLike) -> None:
    """把 vendor 树加进 ``ASCEND_CUSTOM_OPP_PATH``，让 CANN 找得到算子配置。

    已存在的路径保持在前，多个自定义算子包可共存（冒号分隔）。
    """
    path = str(vendor_dir)
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    entries = [e for e in current.split(":") if e]
    if path not in entries:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([*entries, path])


def current_stream_ptr() -> int:
    """当前 NPU stream 的 aclrtStream 指针。"""
    stream = torch.npu.current_stream()
    ptr = getattr(stream, "npu_stream", None)
    if ptr is None:
        raise AclError(
            "取不到 aclrtStream 指针：torch.npu.current_stream() 没有 .npu_stream；"
            f"可用属性 {[a for a in dir(stream) if not a.startswith('_')]}"
        )
    return int(ptr)


class _AclTensors:
    """一批 aclTensor 的生命周期管理 —— 退出时逐个 destroy，不泄漏。"""

    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self._tensors = tensors
        self._handles: list[int] = []

    def __enter__(self) -> list[int]:
        libs = acl_libs()
        for t in self._tensors:
            if not t.is_contiguous():
                raise AclError(f"aclTensor 要求 contiguous 输入，收到 stride={t.stride()}")
            if t.dtype not in ACL_DTYPE:
                raise AclError(f"dtype {t.dtype} 没有对应的 aclDataType；已知 {list(ACL_DTYPE)}")
            ndim = t.dim()
            dims = (ctypes.c_int64 * ndim)(*t.shape)
            strides = (ctypes.c_int64 * ndim)(*t.stride())
            handle = libs.create_tensor(
                dims, ndim, ACL_DTYPE[t.dtype], strides, 0, ACL_FORMAT_ND,
                dims, ndim, ctypes.c_void_p(t.data_ptr()),
            )
            if not handle:
                self.__exit__(None, None, None)
                raise AclError(f"aclCreateTensor 返回空（shape={tuple(t.shape)} dtype={t.dtype}）")
            self._handles.append(handle)
        return self._handles

    def __exit__(self, *exc: object) -> None:
        libs = acl_libs()
        for h in self._handles:
            libs.destroy_tensor(ctypes.c_void_p(h))
        self._handles.clear()


class AclnnOp:
    """一个已编译的 CANN 自定义算子，可在进程内直接调用。

    由 :func:`ascend_fla.runtime.compile.compile_kernel` 构造，不要手工实例化。

    Args:
        op_name: aclnn 算子名（ascriptor ``HostSpec.op``，即 kernel 名的 CamelCase）。
        opapi_lib: ``libcust_opapi.so`` 的路径。
        vendor_dir: vendor 树根，会注册进 ``ASCEND_CUSTOM_OPP_PATH``。
        input_names / scalar_names / output_names: 参数顺序，来自 HostSpec。
    """

    def __init__(
        self,
        op_name: str,
        opapi_lib: str | os.PathLike,
        vendor_dir: str | os.PathLike,
        input_names: list[str],
        scalar_names: list[str],
        output_names: list[str],
        scalar_dtypes: list[str] | None = None,
    ) -> None:
        self.op_name = op_name
        self.input_names = list(input_names)
        self.scalar_names = list(scalar_names)
        self.output_names = list(output_names)
        # ascriptor 的标量 dtype（i32/f32/…）。aclnn 接口层把整型统一成 int64_t，
        # 浮点则是 float —— 全当 int64 传会把 scale 这类参数写坏。
        self.scalar_dtypes = list(scalar_dtypes or ["i64"] * len(self.scalar_names))
        if len(self.scalar_dtypes) != len(self.scalar_names):
            raise AclError(f"{op_name}: scalar_dtypes 与 scalar_names 长度不一致")
        self._scalar_ctypes = [SCALAR_CTYPE[d] for d in self.scalar_dtypes]
        register_custom_opp_path(vendor_dir)

        self._lib = ctypes.CDLL(str(opapi_lib), mode=ctypes.RTLD_GLOBAL)
        try:
            self._get_ws = getattr(self._lib, f"aclnn{op_name}GetWorkspaceSize")
            self._run = getattr(self._lib, f"aclnn{op_name}")
        except AttributeError as e:
            raise AclError(f"{opapi_lib} 里没有 aclnn{op_name} 相关符号") from e

        n_tensors = len(self.input_names) + len(self.output_names)
        self._get_ws.restype = ctypes.c_int
        self._get_ws.argtypes = (
            [ctypes.c_void_p] * len(self.input_names)
            + list(self._scalar_ctypes)
            + [ctypes.c_void_p] * len(self.output_names)
            + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
        )
        self._run.restype = ctypes.c_int
        self._run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        self._n_tensors = n_tensors
        # workspace 按 device 复用：16MB 级别的分配不该每次调用都做一次
        self._ws_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _workspace(self, size: int, device: torch.device) -> torch.Tensor:
        key = (device.index or 0, 0)
        buf = self._ws_cache.get(key)
        if buf is None or buf.numel() < size:
            buf = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
            self._ws_cache[key] = buf
        return buf

    def __call__(
        self,
        inputs: dict[str, torch.Tensor],
        scalars: dict[str, float],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """在当前 stream 上执行算子。``outputs`` 原地写入并返回。

        调用方负责分配 ``outputs``（用 ``torch.empty(device="npu")``）。本函数不做
        同步 —— 取值或计时前自己 ``torch.npu.synchronize()``。
        """
        missing = [n for n in self.input_names if n not in inputs]
        missing += [n for n in self.output_names if n not in outputs]
        missing += [n for n in self.scalar_names if n not in scalars]
        if missing:
            raise AclError(f"{self.op_name}: 缺少参数 {missing}")

        ordered = [inputs[n] for n in self.input_names] + [outputs[n] for n in self.output_names]
        device = ordered[0].device
        for t in ordered:
            if t.device.type != "npu":
                raise AclError(f"{self.op_name}: 期望 NPU 张量，收到 device={t.device}")

        with _AclTensors(ordered) as handles:
            in_handles = [ctypes.c_void_p(h) for h in handles[: len(self.input_names)]]
            out_handles = [ctypes.c_void_p(h) for h in handles[len(self.input_names):]]
            scalar_vals = [
                ct(float(scalars[n]) if ct is ctypes.c_double else int(scalars[n]))
                for n, ct in zip(self.scalar_names, self._scalar_ctypes)
            ]

            ws_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            rc = self._get_ws(*in_handles, *scalar_vals, *out_handles,
                              ctypes.byref(ws_size), ctypes.byref(executor))
            if rc != 0:
                raise AclError(f"aclnn{self.op_name}GetWorkspaceSize 失败 rc={rc}")

            ws = self._workspace(ws_size.value, device)
            rc = self._run(ctypes.c_void_p(ws.data_ptr()), ws_size.value, executor,
                           ctypes.c_void_p(current_stream_ptr()))
            if rc != 0:
                raise AclError(f"aclnn{self.op_name} 执行失败 rc={rc}")
        return outputs

    def __repr__(self) -> str:  # pragma: no cover
        return (f"AclnnOp({self.op_name}, inputs={self.input_names}, "
                f"scalars={self.scalar_names}, outputs={self.output_names})")
