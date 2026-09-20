"""Native BF16 inputs, FP32 normalization and per-key-head ATK recurrence."""
from ascriptor.a5 import *

C = 64
D = 128
LOG_X = 0.4054651081081644


@vf()
def zero_atk(state: Tensor):
    value = RegList(DT.float, 2)
    value <<= 0.0
    state[0:1, 0:D] <<= value


@vf()
def atk_tile(q: Tensor, k: Tensor, write: Tensor, gate: Tensor,
             beta: Tensor, state: Tensor, scalar: Tensor):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    ar = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    r = RegList(DT.float, 2)
    denom = RegList(DT.float, 2)
    norm = Reg(DT.float)
    decay = Reg(DT.float)
    update = Reg(DT.float)
    ar <<= state[0:1, 0:D]
    for i in range(C):
        qr <<= q[i:i+1, 0:D]
        tmp <<= qr * qr
        norm <<= tmp.cadd()
        norm <<= norm.sqrt()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        qr <<= qr / norm
        q[i:i+1, 0:D] <<= qr

        kr <<= k[i:i+1, 0:D]
        tmp <<= kr * kr
        norm <<= tmp.cadd()
        norm <<= norm.sqrt()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        kr <<= kr / norm
        k[i:i+1, 0:D] <<= kr

        decay <<= gate[i:i+1, 0:1].single()
        decay <<= decay.exp()
        update <<= beta[i:i+1, 0:1].single()
        ar <<= ar * decay
        tmp <<= kr * kr
        tmp <<= tmp * update
        ar <<= ar + tmp
        r <<= ar + 1e-6
        r <<= r.ln()
        r <<= r + 0.2
        denom <<= r.abs()
        denom <<= denom + 1.0
        r <<= r / denom
        r <<= r * (-LOG_X)
        r <<= r.exp()
        tmp <<= kr * r
        write[i:i+1, 0:D] <<= tmp
    state[0:1, 0:D] <<= ar
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def widen_qk(q: Tensor, k: Tensor, qf: Tensor, kf: Tensor):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    for i in range(C):
        # Each converting load consumes 64 BF16 elements per FP32 register.
        qr <<= q[i:i+1, 0:D]
        kr <<= k[i:i+1, 0:D]
        qf[i:i+1, 0:D] <<= qr
        kf[i:i+1, 0:D] <<= kr
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pgdn_bf03_atk(
    q: GM[bf16, ("B", "T", "H", 128)], k: GM[bf16, ("B", "T", "H", 128)],
    g_atk: GM[f32, ("B", "T", "H")], beta_atk: GM[f32, ("B", "T", "H")],
    q_norm: GM[f32, ("B", "T", "H", 128)], k_read: GM[f32, ("B", "T", "H", 128)],
    k_write: GM[f32, ("B", "T", "H", 128)], final_A_state: GM[f32, ("B", "H", 128)],
    B: i32, T: i32, H: i32, N: i32,
):
    qi = Tensor(DT.bfloat16, [C, D], Position.UB)
    ki = Tensor(DT.bfloat16, [C, D], Position.UB)
    qu = Tensor(DT.float, [C, D], Position.UB)
    ku = Tensor(DT.float, [C, D], Position.UB)
    wu = Tensor(DT.float, [C, D], Position.UB)
    gu = Tensor(DT.float, [C, 8], Position.UB)
    bu = Tensor(DT.float, [C, 8], Position.UB)
    au = Tensor(DT.float, [1, D], Position.UB)
    scalar = Tensor(DT.float, [1, 8], Position.UB)
    per = CeilDiv(B * H, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * H)
    with auto_sync():
        for item in range(begin, end):
            bb = Var(item // H)
            hh = Var(item % H)
            zero_atk(au)
            for cc in range(N):
                tt = Var(cc * C)
                gm_to_ub_pad(qi, q[bb, tt:tt+C, hh, :], C, D, (H - 1) * D, 0)
                gm_to_ub_pad(ki, k[bb, tt:tt+C, hh, :], C, D, (H - 1) * D, 0)
                gu[:, 0:1] <<= g_atk[bb, tt:tt+C, hh:hh+1]
                bu[:, 0:1] <<= beta_atk[bb, tt:tt+C, hh:hh+1]
                widen_qk(qi, ki, qu, ku)
                atk_tile(qu, ku, wu, gu, bu, au, scalar)
                # Explicit strided stores retain the key-head axis in BTHD.
                ub_to_gm_pad(q_norm[bb, tt:tt+C, hh, :], qu, C, D, 0, (H - 1) * D)
                ub_to_gm_pad(k_read[bb, tt:tt+C, hh, :], ku, C, D, 0, (H - 1) * D)
                ub_to_gm_pad(k_write[bb, tt:tt+C, hh, :], wu, C, D, 0, (H - 1) * D)
            final_A_state[bb, hh:hh+1, :] <<= au[:, :]
    return q_norm, k_read, k_write, final_A_state
