#pragma once
#include "tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void a201_splitk_bf16_m16_n64_k32_s16_cube(GM_ADDR x_, GM_ADDR y_, GM_ADDR z_, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> x((__gm__ bfloat16_t*)x_);
    GMTensor<bfloat16_t> y((__gm__ bfloat16_t*)y_);
    GMTensor<float> z((__gm__ float*)z_);
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 0> ev_mte2_mte1_ready_1;  // guards l1x, l1y #47
    DEvent<PIPE_MTE1, PIPE_M, 0, 0, 1> ev_mte1_m_ready_1;  // guards _l0b #48
    SEvent<PIPE_M, PIPE_FIX, 0, 0> ev_m_fix_ready_1;  // guards l0c #49
    DEvent<PIPE_M, PIPE_MTE1, 2, 0, 1> ev_m_mte1_valid_1;  // guards _l0a #50
    const DBuff<uint8_t, Position::L0A> _l0a((uint64_t)(0), (uint64_t)(32768));  // #10
    const DBuff<uint8_t, Position::L0B> _l0b((uint64_t)(0), (uint64_t)(32768));  // #11
    int32_t _l0acnt = 0;  // #13
    int32_t _l0bcnt = 0;  // #14
    const Tensor<bfloat16_t, Position::L1> l1x((uint64_t)(0));  // #1
    const Tensor<bfloat16_t, Position::L1> l1y((uint64_t)(1024));  // #2
    const Tensor<float, Position::L0C> l0c((uint64_t)(0));  // #3
    gm_to_l1_nd2nz(l1x, x, 16, 32, 16, 32);  // #5
    gm_to_l1_nd2nz(l1y, y, 64, 32, 64, 32);  // #6
    ev_mte2_mte1_ready_1.set();  // #39
    ev_mte2_mte1_ready_1.wait();  // #40
    for (int32_t _subk = 0; _subk < 32; _subk += 16) {  // #38
        const Tensor<uint8_t, Position::L0A> l0a_slot = _l0a.get(_l0acnt);  // #22
        const Tensor<bfloat16_t, Position::L0A> l0a_view = _l0a.get(_l0acnt).as<bfloat16_t>();  // #23
        const Tensor<uint8_t, Position::L0B> l0b_slot = _l0b.get(_l0bcnt);  // #24
        const Tensor<bfloat16_t, Position::L0B> l0b_view = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #25
        const Tensor<bfloat16_t, Position::L1> l1x_tile = l1x[(((_subk / 16) * 256) + (_subk % 16))];  // #26
        ev_m_mte1_valid_1.wait();  // #46
        l1_to_l0<false>(l0a_view, l1x_tile, 16, 32, 16, 16);  // #27
        const Tensor<bfloat16_t, Position::L1> l1y_tile = l1y[(((_subk / 16) * 1024) + (_subk % 16))];  // #28
        l1_to_l0<false>(l0b_view, l1y_tile, 64, 32, 64, 16);  // #29
        ev_mte1_m_ready_1.set();  // #41
        const bool _subk_is0 = _subk == 0;  // #30
        ev_mte1_m_ready_1.wait();  // #42
        if (_subk_is0) {  // #33
            mmad(l0c, l0a_view, l0b_view, 16, 64, 16, true);  // #31
        } else {
            mmad(l0c, l0a_view, l0b_view, 16, 64, 16, false);  // #32
        }
        ev_m_mte1_valid_1.set();  // #45
        PipeBarrier<PIPE_M>();  // A2-01 diagnostic: settle after the split-K MMAD
        _l0acnt = _l0acnt + 1;  // #34 #35
        _l0bcnt = _l0bcnt + 1;  // #36 #37
    }
    ev_m_fix_ready_1.set();  // #43
    ev_m_fix_ready_1.wait();  // #44
    l0c_to_gm_nz2nd(z, l0c, 16, 64, 64, 16, false, 0.0f, 0, false, false);  // #8
    return;  // #9
}
