#pragma once
#include "tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void a201_chain_f32_m16_n16_k16_t2_nobar_cube(GM_ADDR a0_, GM_ADDR b0_, GM_ADDR a1_, GM_ADDR b1_, GM_ADDR z_, GM_ADDR workspace)
{
    GMTensor<float> a0((__gm__ float*)a0_);
    GMTensor<float> b0((__gm__ float*)b0_);
    GMTensor<float> a1((__gm__ float*)a1_);
    GMTensor<float> b1((__gm__ float*)b1_);
    GMTensor<float> z((__gm__ float*)z_);
    QEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1, 2, 3> ev_mte2_mte1_ready_1;  // guards l1a0, l1a1, l1b0, l1b1 #59
    SEvent<PIPE_MTE1, PIPE_M, 0, 0> ev_mte1_m_ready_1;  // guards _l0b #60
    SEvent<PIPE_M, PIPE_MTE1, 0, 0> ev_m_mte1_valid_1;  // guards _l0a #61
    SEvent<PIPE_M, PIPE_FIX, 0, 0> ev_m_fix_ready_1;  // guards l0c #62
    const DBuff<uint8_t, Position::L0A> _l0a((uint64_t)(0), (uint64_t)(32768));  // #15
    const DBuff<uint8_t, Position::L0B> _l0b((uint64_t)(0), (uint64_t)(32768));  // #16
    int32_t _l0acnt = 0;  // #18
    int32_t _l0bcnt = 0;  // #19
    const Tensor<float, Position::L1> l1a0((uint64_t)(0));  // #1
    const Tensor<float, Position::L1> l1b0((uint64_t)(1024));  // #2
    const Tensor<float, Position::L1> l1a1((uint64_t)(2048));  // #3
    const Tensor<float, Position::L1> l1b1((uint64_t)(3072));  // #4
    const Tensor<float, Position::L0C> l0c((uint64_t)(0));  // #5
    gm_to_l1_nd2nz(l1a0, a0, 16, 16, 16, 16);  // #7
    ev_mte2_mte1_ready_1.set();  // #43
    gm_to_l1_nd2nz(l1b0, b0, 16, 16, 16, 16);  // #8
    ev_mte2_mte1_ready_1.set();  // #45
    gm_to_l1_nd2nz(l1a1, a1, 16, 16, 16, 16);  // #9
    ev_mte2_mte1_ready_1.set();  // #47
    gm_to_l1_nd2nz(l1b1, b1, 16, 16, 16, 16);  // #10
    ev_mte2_mte1_ready_1.set();  // #49
    const Tensor<uint8_t, Position::L0A> l0a_slot = _l0a.get(_l0acnt);  // #21
    const Tensor<float, Position::L0A> l0a_view = _l0a.get(_l0acnt).as<float>();  // #22
    const Tensor<uint8_t, Position::L0B> l0b_slot = _l0b.get(_l0bcnt);  // #23
    const Tensor<float, Position::L0B> l0b_view = _l0b.get(_l0bcnt).as<float>();  // #24
    ev_mte2_mte1_ready_1.wait();  // #44
    l1_to_l0<false>(l0a_view, l1a0, 16, 16, 16, 16);  // #25
    ev_mte2_mte1_ready_1.wait();  // #46
    l1_to_l0<false>(l0b_view, l1b0, 16, 16, 16, 16);  // #26
    ev_mte1_m_ready_1.set();  // #51
    ev_mte1_m_ready_1.wait();  // #52
    mmad(l0c, l0a_view, l0b_view, 16, 16, 16, true);  // #27
    ev_m_mte1_valid_1.set();  // #55
    _l0acnt = 1;  // #29
    _l0bcnt = 1;  // #31
    const Tensor<uint8_t, Position::L0A> l0a_slot_1 = _l0a.get(_l0acnt);  // #32
    const Tensor<float, Position::L0A> l0a_view_1 = _l0a.get(_l0acnt).as<float>();  // #33
    const Tensor<uint8_t, Position::L0B> l0b_slot_1 = _l0b.get(_l0bcnt);  // #34
    const Tensor<float, Position::L0B> l0b_view_1 = _l0b.get(_l0bcnt).as<float>();  // #35
    ev_mte2_mte1_ready_1.wait();  // #48
    ev_m_mte1_valid_1.wait();  // #56
    l1_to_l0<false>(l0a_view_1, l1a1, 16, 16, 16, 16);  // #36
    ev_mte2_mte1_ready_1.wait();  // #50
    l1_to_l0<false>(l0b_view_1, l1b1, 16, 16, 16, 16);  // #37
    ev_mte1_m_ready_1.set();  // #53
    ev_mte1_m_ready_1.wait();  // #54
    mmad(l0c, l0a_view_1, l0b_view_1, 16, 16, 16, false);  // #38
    ev_m_fix_ready_1.set();  // #57
    _l0acnt = 2;  // #40
    _l0bcnt = 2;  // #42
    ev_m_fix_ready_1.wait();  // #58
    l0c_to_gm_nz2nd(z, l0c, 16, 16, 16, 16, false, 0.0f, 0, false, false);  // #13
    return;  // #14
}
