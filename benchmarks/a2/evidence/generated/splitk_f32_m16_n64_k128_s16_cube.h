#pragma once
#include "tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void a201_splitk_f32_m16_n64_k128_s16_cube(GM_ADDR x_, GM_ADDR y_, GM_ADDR z_, GM_ADDR workspace)
{
    GMTensor<float> x((__gm__ float*)x_);
    GMTensor<float> y((__gm__ float*)y_);
    GMTensor<float> z((__gm__ float*)z_);
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 0> ev_mte2_mte1_ready_1;  // guards l1x, l1y #50
    SEvent<PIPE_M, PIPE_FIX, 0, 0> ev_m_fix_ready_1;  // guards l0c #51
    DEvent<PIPE_M, PIPE_MTE1, 2, 0, 1> l0_splitk_available;  // guards _l0a, _l0b #40
    DEvent<PIPE_MTE1, PIPE_M, 0, 0, 1> l0_splitk_ready;  // guards _l0a, _l0b #41
    const DBuff<uint8_t, Position::L0A> _l0a((uint64_t)(0), (uint64_t)(32768));  // #10
    const DBuff<uint8_t, Position::L0B> _l0b((uint64_t)(0), (uint64_t)(32768));  // #11
    int32_t _l0acnt = 0;  // #13
    int32_t _l0bcnt = 0;  // #14
    const Tensor<float, Position::L1> l1x((uint64_t)(0));  // #1
    const Tensor<float, Position::L1> l1y((uint64_t)(8192));  // #2
    const Tensor<float, Position::L0C> l0c((uint64_t)(0));  // #3
    gm_to_l1_nd2nz(l1x, x, 16, 128, 16, 128);  // #5
    gm_to_l1_nd2nz(l1y, y, 64, 128, 64, 128);  // #6
    ev_mte2_mte1_ready_1.set();  // #46
    ev_mte2_mte1_ready_1.wait();  // #47
    for (int32_t _subk = 0; _subk < 128; _subk += 16) {  // #39
        const Tensor<uint8_t, Position::L0A> l0a_slot = _l0a.get(_l0acnt);  // #22
        const Tensor<float, Position::L0A> l0a_view = _l0a.get(_l0acnt).as<float>();  // #23
        const Tensor<uint8_t, Position::L0B> l0b_slot = _l0b.get(_l0bcnt);  // #24
        const Tensor<float, Position::L0B> l0b_view = _l0b.get(_l0bcnt).as<float>();  // #25
        const Tensor<float, Position::L1> l1x_tile = l1x[(((_subk / 8) * 128) + (_subk % 8))];  // #26
        l0_splitk_available.wait();  // #42
        l1_to_l0<false>(l0a_view, l1x_tile, 16, 128, 16, 16);  // #27
        const Tensor<float, Position::L1> l1y_tile = l1y[(((_subk / 8) * 128) + (_subk % 8))];  // #28
        l1_to_l0<false>(l0b_view, l1y_tile, 64, 128, 64, 16);  // #29
        l0_splitk_ready.set();  // #43
        const bool _subk_is0 = _subk == 0;  // #30
        l0_splitk_ready.wait();  // #44
        if (_subk_is0) {  // #33
            mmad(l0c, l0a_view, l0b_view, 16, 64, 16, true);  // #31
        } else {
            mmad(l0c, l0a_view, l0b_view, 16, 64, 16, false);  // #32
        }
        l0_splitk_available.set();  // #45
        PipeBarrier<PIPE_M>();  // #34
        _l0acnt = _l0acnt + 1;  // #35 #36
        _l0bcnt = _l0bcnt + 1;  // #37 #38
    }
    ev_m_fix_ready_1.set();  // #48
    ev_m_fix_ready_1.wait();  // #49
    l0c_to_gm_nz2nd(z, l0c, 16, 64, 64, 16, false, 0.0f, 0, false, false);  // #8
    return;  // #9
}
