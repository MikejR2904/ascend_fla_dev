"""KDA 的逐 token 递推（decode 路径）。本仓自写，不是 ascriptor 资产的改写。

**算法**（与 ascend_fla/reference/kda.py 的 kda_recurrent_ref 逐行对应）::

    state ← state · exp(g)            # 逐 K 维衰减，g 是 [K]，沿 V 广播
    delta  = v − kᵀ·state
    state ← state + β·k ⊗ delta
    o      = qᵀ·state                 # 注意用的是**更新后**的 state

**两趟扫 state 就够**，而且两趟都在 UB 内，GM 只碰一次 state：

* 第一趟：衰减 state，同时攒 ``kᵀ·state_dec``（给 delta）。
* 第二趟：做 rank-1 更新，**顺手攒 ``qᵀ·state_new``** —— o 要的就是它。

**这里绕过一个我先踩过的坑。** 最初我用代数恒等式
``o = qᵀ·state_dec + β(q·k)·delta`` 把 o 挪到第一趟，以为更省。真机实测
``o`` 的相对 L2 是 8.5e-02 而同一次的 ``final_state`` 精确到 3.2e-08 ——
错只在 o，说明问题出在那个修正项。原因：``RegList.cadd()`` 的结果**只落在 lane 0**，
不是广播到所有 lane（ascriptor 自家 kernel 里 cadd 之后一律 ``.single_value()`` 取值、
要当乘数用还得经 UB 再 ``.single()`` 读回），所以 ``delta * corr`` 只有 1/128 个 lane 是对的。
把读出挂到第二趟之后连 ``cadd`` 都不需要了 —— **那个"优化"既错又多余**。

**为什么 K 方向的规约不需要跨 lane 规约**：state 是 ``[K, V]``，规约沿 K 即沿**行**，
所以是"取一个标量广播乘一行、累加"的模式（``.single()`` + RegList 累加）。

**并行与 block_dim**：按 ``B*HV`` 个头切给向量核，**每个头整份 state 常驻一个核的 UB**，
所以核间不需要任何同步 —— 这一条是刻意的。上游 chunk 的融合尾部把 K 切给 sub-block、
于是要手写同步，而手写那份假设了 C≥2，造成了 ``c1-multihead-o-corrupt`` 那个 P0。
本 kernel 的循环只有一层（头），``auto_sync()`` 能覆盖，不手写事件。

**T 循环在 kernel 内**：state 一次读入、一次写回，与 T 无关。T=1 是纯 decode，
T=2~8 是投机解码；T 再大就该走 chunk 路径（那边把 64 个 token 批成矩阵乘）。
"""

from ascriptor.a5 import *

K_DIM = 128

V_DIM = 128

REGS_V = V_DIM // 64

REGS_K = K_DIM // 64

#: 一次调用最多几个 token —— 流式缓冲的行数。decode 是 1，投机解码 2~8。
#: `ascend_fla/ops/kda/fused_recurrent.py` 的 T_MAX 必须与它一致（由测试锁住）。
T_MAX = 16


@vf()
def kda_recurrent_step_vf(state_ub: Tensor, qs_ub: Tensor, k_ub: Tensor, v_ub: Tensor,
                          g_ub: Tensor, beta_ub: Tensor, o_ub: Tensor, t_len: Var):
    """对 ``t_len`` 个 token 推进 ``state_ub``（``[K, V]`` fp32，常驻）并写出 ``o``。

    ``qs_ub`` 是**已乘过 scale** 的 q —— scale 在 host 侧或调用方预乘，省掉内层一次乘法。
    """
    row = RegList(DT.float, REGS_V)
    acc_k = RegList(DT.float, REGS_V)
    acc_o = RegList(DT.float, REGS_V)
    delta = RegList(DT.float, REGS_V)
    tmp_v = RegList(DT.float, REGS_V)      # DSL 规定一条赋值只能一次运算，复合式要落临时

    decay = Reg(DT.float)
    g_val = Reg(DT.float)
    k_val = Reg(DT.float)
    q_val = Reg(DT.float)
    beta = Reg(DT.float)

    for t in range(t_len):
        acc_k <<= 0.0
        acc_o <<= 0.0

        # 第一趟：state ← state·exp(g)，同时攒 kᵀ·state_dec
        for kk in range(K_DIM):
            g_val <<= g_ub[t:t + 1, kk:kk + 1].single()
            decay <<= g_val.exp()
            k_val <<= k_ub[t:t + 1, kk:kk + 1].single()
            row <<= state_ub[kk:kk + 1, 0:V_DIM]
            row <<= row * decay
            state_ub[kk:kk + 1, 0:V_DIM] <<= row
            tmp_v <<= row * k_val
            acc_k <<= acc_k + tmp_v

        beta <<= beta_ub[0:1, t:t + 1].single()
        delta <<= v_ub[t:t + 1, 0:V_DIM]
        delta <<= delta - acc_k

        # 第二趟：state ← state_dec + (β·k)⊗delta，顺手攒 qᵀ·state_new
        for kk in range(K_DIM):
            k_val <<= k_ub[t:t + 1, kk:kk + 1].single()
            k_val <<= k_val * beta
            q_val <<= qs_ub[t:t + 1, kk:kk + 1].single()
            row <<= state_ub[kk:kk + 1, 0:V_DIM]
            tmp_v <<= delta * k_val
            row <<= row + tmp_v
            state_ub[kk:kk + 1, 0:V_DIM] <<= row
            tmp_v <<= row * q_val
            acc_o <<= acc_o + tmp_v

        o_ub[t:t + 1, 0:V_DIM] <<= acc_o

    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def kda_fused_recurrent_kernel(
    qs: GM[f32, ('B', 'HV', 'T', 128)],
    k: GM[f32, ('B', 'HV', 'T', 128)],
    v: GM[f32, ('B', 'HV', 'T', 128)],
    g: GM[f32, ('B', 'HV', 'T', 128)],
    beta: GM[f32, ('B', 'HV', 1, 'T')],
    initial_state: GM[f32, ('B', 'HV', 128, 128)],
    o: GM[f32, ('B', 'HV', 'T', 128)],
    final_state: GM[f32, ('B', 'HV', 128, 128)],
    B: i32, HV: i32, T: i32, head_dim: i32, value_dim: i32,
):
    """一个头一个核：state 常驻 UB，T 个 token 在核内推完。

    ABI 取 **BHV-major**（``[B, HV, T, 128]``）而不是 chunk 路径的 token-major ——
    decode 的 T 很小，布局重排由 host 侧做一次比在 kernel 里跨步读省事；
    而且这样 state 与 q/k/v/g 的头维相邻，DMA 是连续的。
    qs 为**已乘 scale 的 q**；全部 fp32 —— decode 是带宽瓶颈且量很小，
    省 cast 比省带宽重要（T=1 时 q/k/v/g 合计只有 2KB，state 才是 64KB）。
    """
    state_ub = Tensor(DT.float, [K_DIM, V_DIM], Position.UB)
    qs_ub = DBuff(DT.float, [T_MAX, K_DIM], Position.UB)
    k_ub = DBuff(DT.float, [T_MAX, K_DIM], Position.UB)
    v_ub = DBuff(DT.float, [T_MAX, V_DIM], Position.UB)
    g_ub = DBuff(DT.float, [T_MAX, K_DIM], Position.UB)
    beta_ub = DBuff(DT.float, [1, T_MAX], Position.UB)
    o_ub = DBuff(DT.float, [T_MAX, V_DIM], Position.UB)

    work_count = B * HV
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            hv_idx = Var(work % HV)
            b_idx = Var(work // HV)

            state_ub[0:K_DIM, 0:V_DIM] <<= initial_state[b_idx, hv_idx, 0:K_DIM, 0:V_DIM]
            qs_ub[work][0:T, 0:K_DIM] <<= qs[b_idx, hv_idx, 0:T, 0:K_DIM]
            k_ub[work][0:T, 0:K_DIM] <<= k[b_idx, hv_idx, 0:T, 0:K_DIM]
            v_ub[work][0:T, 0:V_DIM] <<= v[b_idx, hv_idx, 0:T, 0:V_DIM]
            g_ub[work][0:T, 0:K_DIM] <<= g[b_idx, hv_idx, 0:T, 0:K_DIM]
            beta_ub[work][0:1, 0:T] <<= beta[b_idx, hv_idx, 0:1, 0:T]

            kda_recurrent_step_vf(state_ub, qs_ub[work], k_ub[work], v_ub[work],
                                  g_ub[work], beta_ub[work], o_ub[work], T)

            o[b_idx, hv_idx, 0:T, 0:V_DIM] <<= o_ub[work][0:T, 0:V_DIM]
            final_state[b_idx, hv_idx, 0:K_DIM, 0:V_DIM] <<= state_ub[0:K_DIM, 0:V_DIM]

    return o, final_state
