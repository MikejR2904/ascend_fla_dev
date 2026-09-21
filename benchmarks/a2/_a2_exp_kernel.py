"""Minimal a2 vector `exp` kernel for the GDN-ABI probe (A2-07).

Isolated in its own module because the ascriptor ``@kernel`` decorator static-evaluates
the ``GM[...]`` annotations in the defining module's globals, so the facade must be a
module-level ``import *``. Imported lazily by ``probe_gdn_abi.py`` only under
``--ascriptor-exp`` (machines without ascriptor never import this file).

It computes ``exp`` over a [1,64] fp32 row on the CCE-generated vector path, to observe
the SoC's real exp under/over-flow and subnormal-flush behaviour on the code a GDN decay
kernel would actually run (vs torch_npu's aclnn built-in).
"""
from ascriptor.a2 import *  # noqa: F401,F403


@kernel(mode="vec", block_dim=1)
def a2_exp_probe(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)]):
    xb = Tensor(DT.float, [1, 64], Position.UB)
    yb = Tensor(DT.float, [1, 64], Position.UB)
    with auto_sync():
        set_mask((1 << 64) - 1, (1 << 64) - 1)
        xb[0:1, 0:64] <<= x[0:1, 0:64]
        exp(yb[0:1, 0:64], xb[0:1, 0:64])
        y[0:1, 0:64] <<= yb[0:1, 0:64]
    return y
