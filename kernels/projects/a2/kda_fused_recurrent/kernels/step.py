"""A2 KDA token-by-token recurrence (decode / speculative decode, T <= T_MAX).

Port of this repository's ``kernels/projects/a5/kda_fused_recurrent/kernels/step.py`` to the A2 (c220) facade.
Algorithm, per value head, with the state resident in UB::

    state <- state * exp(g)            # per-K decay, broadcast along V
    delta  = v - k^T state
    state <- state + (beta k) (x) delta
    o      = (scale q)^T state         # the *updated* state

Differences from the A5 kernel, all required by A2 or by D-PM-35/37:

* No ``@vf`` on A2: each row step is a tile-vector op on one 128-wide UB row, driven by a scalar read with
  ``Var.GetValueFrom`` (``exp(g)``, ``k``, ``beta*k``, ``q`` for that row). The accumulation order is the A5
  one: per row, multiply into a temporary, then add.
* The kernel takes the public tensors as they are: token-major ``q``/``k`` ``[B*T, H*128]`` and
  ``v`` ``[B*T, HV*128]`` in BF16, ``g`` ``[B*T, HV*128]`` and ``beta`` ``[B*T, HV]`` in FP32 (2-D views of the
  contiguous ``[B, T, *, 128]`` tensors), and writes ``o`` token-major in BF16. The BF16 -> FP32 casts, the
  ``scale`` multiply and the GQA head map (``h = hv // (HV // H)``) happen here, not on the host.
"""

from ascriptor.a2 import *

K_DIM = 128

V_DIM = 128

#: Tokens per call (streaming buffer rows). Decode is 1, speculative decode 2..8.
T_MAX = 16


@kernel(mode="vec")
def kda_fused_recurrent_a2_kernel(
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    v: GM[bf16, ('BT', 'HVV')],
    g: GM[f32, ('BT', 'HVK')],
    beta: GM[f32, ('BT', 'HVB')],
    initial_state: GM[f32, ('B', 'HV', 128, 128)],
    o: GM[bf16, ('BT', 'HVV')],
    final_state: GM[f32, ('B', 'HV', 128, 128)],
    B: i32, T: i32, H: i32, HV: i32, scale: f32,
):
    state_ub = Tensor(DT.float, [K_DIM, V_DIM], Position.UB)
    qb_ub = Tensor(DT.bfloat16, [T_MAX, K_DIM], Position.UB)
    kb_ub = Tensor(DT.bfloat16, [T_MAX, K_DIM], Position.UB)
    vb_ub = Tensor(DT.bfloat16, [T_MAX, V_DIM], Position.UB)
    q_ub = Tensor(DT.float, [T_MAX, K_DIM], Position.UB)
    k_ub = Tensor(DT.float, [T_MAX, K_DIM], Position.UB)
    v_ub = Tensor(DT.float, [T_MAX, V_DIM], Position.UB)
    g_ub = Tensor(DT.float, [T_MAX, K_DIM], Position.UB)
    eg_ub = Tensor(DT.float, [T_MAX, K_DIM], Position.UB)
    o_ub = Tensor(DT.float, [T_MAX, V_DIM], Position.UB)
    ob_ub = Tensor(DT.bfloat16, [T_MAX, V_DIM], Position.UB)
    acc_k = Tensor(DT.float, [1, V_DIM], Position.UB)
    delta = Tensor(DT.float, [1, V_DIM], Position.UB)
    tmp = Tensor(DT.float, [1, V_DIM], Position.UB)

    group = HV // H
    work_count = B * HV
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    decay = Var(0.0, dtype=DT.float)
    k_val = Var(0.0, dtype=DT.float)
    q_val = Var(0.0, dtype=DT.float)
    beta_val = Var(0.0, dtype=DT.float)

    with auto_sync():
        for work in range(work_begin, work_end):
            hv_idx = Var(work % HV)
            b_idx = Var(work // HV)
            h_idx = Var(hv_idx // group)
            row0 = Var(b_idx * T)
            qk_col = Var(h_idx * K_DIM)
            v_col = Var(hv_idx * V_DIM)

            state_ub[0:K_DIM, 0:V_DIM] <<= initial_state[b_idx, hv_idx, 0:K_DIM, 0:V_DIM]
            qb_ub[0:T, 0:K_DIM] <<= q[row0:row0 + T, qk_col:qk_col + K_DIM]
            kb_ub[0:T, 0:K_DIM] <<= k[row0:row0 + T, qk_col:qk_col + K_DIM]
            vb_ub[0:T, 0:V_DIM] <<= v[row0:row0 + T, v_col:v_col + V_DIM]
            g_ub[0:T, 0:K_DIM] <<= g[row0:row0 + T, v_col:v_col + K_DIM]

            cast(q_ub[0:T, 0:K_DIM], qb_ub[0:T, 0:K_DIM], round_mode=RoundMode.NONE, count=T * K_DIM)
            muls(q_ub[0:T, 0:K_DIM], q_ub[0:T, 0:K_DIM], scale, count=T * K_DIM)
            cast(k_ub[0:T, 0:K_DIM], kb_ub[0:T, 0:K_DIM], round_mode=RoundMode.NONE, count=T * K_DIM)
            cast(v_ub[0:T, 0:V_DIM], vb_ub[0:T, 0:V_DIM], round_mode=RoundMode.NONE, count=T * V_DIM)
            exp(eg_ub[0:T, 0:K_DIM], g_ub[0:T, 0:K_DIM], count=T * K_DIM)

            for t in range(0, T):
                beta_val.GetValueFrom(beta[row0 + t:row0 + t + 1, hv_idx:hv_idx + 1])
                dup(acc_k[0:1, 0:V_DIM], 0.0, count=V_DIM)
                dup(o_ub[t:t + 1, 0:V_DIM], 0.0, count=V_DIM)

                # pass 1: state <- state * exp(g); acc_k += k * state_dec
                for kk in range(0, K_DIM):
                    decay.GetValueFrom(eg_ub[t:t + 1, kk:kk + 1])
                    k_val.GetValueFrom(k_ub[t:t + 1, kk:kk + 1])
                    muls(state_ub[kk:kk + 1, 0:V_DIM], state_ub[kk:kk + 1, 0:V_DIM], decay, count=V_DIM)
                    muls(tmp[0:1, 0:V_DIM], state_ub[kk:kk + 1, 0:V_DIM], k_val, count=V_DIM)
                    add(acc_k[0:1, 0:V_DIM], acc_k[0:1, 0:V_DIM], tmp[0:1, 0:V_DIM], count=V_DIM)

                sub(delta[0:1, 0:V_DIM], v_ub[t:t + 1, 0:V_DIM], acc_k[0:1, 0:V_DIM], count=V_DIM)

                # pass 2: state <- state_dec + (beta k) (x) delta; o += q * state_new
                for kk in range(0, K_DIM):
                    k_val.GetValueFrom(k_ub[t:t + 1, kk:kk + 1])
                    k_val.set(k_val * beta_val)
                    q_val.GetValueFrom(q_ub[t:t + 1, kk:kk + 1])
                    muls(tmp[0:1, 0:V_DIM], delta[0:1, 0:V_DIM], k_val, count=V_DIM)
                    add(state_ub[kk:kk + 1, 0:V_DIM], state_ub[kk:kk + 1, 0:V_DIM], tmp[0:1, 0:V_DIM], count=V_DIM)
                    muls(tmp[0:1, 0:V_DIM], state_ub[kk:kk + 1, 0:V_DIM], q_val, count=V_DIM)
                    add(o_ub[t:t + 1, 0:V_DIM], o_ub[t:t + 1, 0:V_DIM], tmp[0:1, 0:V_DIM], count=V_DIM)

            cast(ob_ub[0:T, 0:V_DIM], o_ub[0:T, 0:V_DIM], round_mode=RoundMode.TO_EVEN, count=T * V_DIM)
            o[row0:row0 + T, v_col:v_col + V_DIM] <<= ob_ub[0:T, 0:V_DIM]
            final_state[b_idx, hv_idx, 0:K_DIM, 0:V_DIM] <<= state_ub[0:K_DIM, 0:V_DIM]

    return o, final_state
