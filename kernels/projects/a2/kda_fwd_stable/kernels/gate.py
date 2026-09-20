"""A2 gate cumsum: ``g_cumsum`` and ``eg = exp(g_cumsum)`` from token-major ``g_raw``.

Port of this repository's ``kernels/projects/a5/kda_fwd_stable/kernels/gate.py`` to the A2 (c220) facade.
The arithmetic is unchanged: a sequential FP32 running sum down the 64 rows of each chunk, then ``exp``.
What changes is the form:

* A2 has no ``@vf`` register functions, so the running sum is one tile-vector ``add`` per row on UB and the
  exponential is a single ``exp`` over the whole [64, 128] tile.
* ``g_raw`` is read in its public token-major layout. The kernel declares it as the 2-D view ``[B*T, HV*128]`` of
  the same contiguous ``[B, T, HV, 128]`` tensor (a metadata-only ``view`` on the host, no copy) and reads each chunk
  as a strided 2-D window, so the host does no ``permute().contiguous()`` (D-PM-35/37). The two outputs are chain-internal intermediates
  and keep the chunked head-major layout ``[B, HV, C, 64, 128]`` that the downstream kernels consume.
"""

from ascriptor.a2 import *

L = 64

K_DIM = 128


@kernel(mode="vec")
def kda_sub1_gate_a2_kernel(
    g_raw: GM[f32, ('BT', 'HVK')],
    g_cumsum: GM[f32, ('B', 'HV', 'C', 64, 128)],
    eg: GM[f32, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    HV: i32,
    C: i32,
):
    cum_ub = DBuff(DT.float, [L, K_DIM], Position.UB)
    exp_ub = DBuff(DT.float, [L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)
            row0 = Var(b_idx * C * L + c_idx * L)
            col0 = Var(hv_idx * K_DIM)

            cum_ub[work][0:L, 0:K_DIM] <<= g_raw[row0:row0 + L, col0:col0 + K_DIM]
            for r in range(1, L):
                add(cum_ub[work][r:r + 1, 0:K_DIM], cum_ub[work][r:r + 1, 0:K_DIM],
                    cum_ub[work][r - 1:r, 0:K_DIM], count=K_DIM)
            g_cumsum[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= cum_ub[work][0:L, 0:K_DIM]
            exp(exp_ub[work][0:L, 0:K_DIM], cum_ub[work][0:L, 0:K_DIM], count=L * K_DIM)
            eg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= exp_ub[work][0:L, 0:K_DIM]

    return g_cumsum, eg
