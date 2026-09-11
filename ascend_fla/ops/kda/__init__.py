"""KDA 算子族。公开 ABI 与 fla.ops.kda 一致（token-major）。

* :func:`chunk_kda` —— 可求导入口，第一选择。
* :func:`chunk_kda_fwd` / :func:`chunk_kda_fwd_with_caches` / :func:`chunk_kda_bwd`
  —— 不建图的底层入口，用于基准测试与精度比对。
* :func:`fused_recurrent_kda` —— decode 路径（逐 token，T≤16）。精度已真机验过
  （对 fp32 递推参考 1e-07 量级，state 串接逐位相同），但**每次调用约 48µs 的固定成本**
  使它暂时不适合真实解码，且**还没接进 layer** —— 见 ``gaps.json`` 的
  ``decode-call-overhead`` 与 ``fused-recurrent-missing``。

注意本族算子**不做** q/k 的 L2 归一化、门控变换、beta 的 sigmoid —— fla 把这三步放在
kernel 里（``use_*_in_kernel=True``），这里要调用方做。``ascend_fla.layers.kda`` 已按
约定做好，层级用户不必关心。详见 :mod:`ascend_fla.ops.kda.autograd` 的 docstring。
"""

from .autograd import chunk_kda
from .chunk import BWD_CACHE_NAMES, chunk_kda_fwd, chunk_kda_fwd_with_caches
from .chunk_bwd import chunk_kda_bwd
from .fused_recurrent import fused_recurrent_kda

__all__ = [
    "BWD_CACHE_NAMES",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "chunk_kda_fwd_with_caches",
    "fused_recurrent_kda",
]


def prepare(device: str = "a5", block_dim: int = 1, *, backward: bool = True,
            impl: str = "stable", decode: bool = False,
            decode_block_dim: int | None = None) -> None:
    """把本进程要用到的 kernel 全部编译好（并注册 vendor 树）。

    **为什么需要它**：CANN 只在首次算子解析时读 ``ASCEND_CUSTOM_OPP_PATH``，之后追加的
    路径它看不见，调用时以 ``rc=161001`` 失败（plog 会说成"算子包未安装"，容易误判成
    编译产物坏了）。所以一个进程要用的 kernel 必须在第一次执行之前都编译完。

    走 :func:`chunk_kda` 或 :func:`chunk_kda_fwd_with_caches` 的不用管 chunk 那条链 ——
    它们已经把反向一起编译了。**但 decode 必须显式声明**：

    >>> prepare(decode=True)            # 同一进程里既要 prefill 又要 decode

    不声明的后果是 prefill（chunk）先跑、decode 的 kernel 后编译，于是撞上上面那条
    ——**实测踩过**，报的是"已经执行过 aclnn 算子，不能再注册新的 vendor 树"。
    这不是测试环境的毛病：任何"prefill 完再逐 token 解码"的进程都会遇到。

    Args:
        device: ascriptor 设备名。
        block_dim: 启动核组数。**一个进程只能用一个值**（见 ``runtime/binding.py`` 的
            ``_claim_op_name``），扫 ``block_dim`` 要分进程。
        backward: 是否连反向的九个 kernel 一起编译。
        impl: ``"stable"``（默认）或 ``"upstream"``。前向与反向用同一个值 —— 混用会让
            门控检查的上限对不上实际会失效的那一侧。
        decode: 是否一起编译 decode 的 ``kda_fused_recurrent``。
        decode_block_dim: decode 那个 kernel 的核组数，``None`` 表示跟 ``block_dim``。
            它是**另一个算子名**，所以可以与 chunk 取不同的值；但同样一个进程一个值。
            实测它的声明域比 chunk 宽（到 28），不过总时长几乎不随它变
            —— 见 ``gaps.json`` 的 ``decode-call-overhead``。
    """
    from .chunk import _compiled_chain as _fwd_chain

    _fwd_chain(device, block_dim, impl)
    if backward:
        from .chunk_bwd import _compiled_chain as _bwd_chain

        _bwd_chain(device, block_dim, impl)
    if decode:
        from .fused_recurrent import _compiled as _decode

        _decode(device, block_dim if decode_block_dim is None else decode_block_dim)


__all__.append("prepare")
