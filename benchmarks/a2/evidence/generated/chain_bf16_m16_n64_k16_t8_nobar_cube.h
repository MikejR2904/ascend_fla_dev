#pragma once
#include "tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void a201_chain_bf16_m16_n64_k16_t8_nobar_cube(GM_ADDR a0_, GM_ADDR b0_, GM_ADDR a1_, GM_ADDR b1_, GM_ADDR a2_, GM_ADDR b2_, GM_ADDR a3_, GM_ADDR b3_, GM_ADDR a4_, GM_ADDR b4_, GM_ADDR a5_, GM_ADDR b5_, GM_ADDR a6_, GM_ADDR b6_, GM_ADDR a7_, GM_ADDR b7_, GM_ADDR z_, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> a0((__gm__ bfloat16_t*)a0_);
    GMTensor<bfloat16_t> b0((__gm__ bfloat16_t*)b0_);
    GMTensor<bfloat16_t> a1((__gm__ bfloat16_t*)a1_);
    GMTensor<bfloat16_t> b1((__gm__ bfloat16_t*)b1_);
    GMTensor<bfloat16_t> a2((__gm__ bfloat16_t*)a2_);
    GMTensor<bfloat16_t> b2((__gm__ bfloat16_t*)b2_);
    GMTensor<bfloat16_t> a3((__gm__ bfloat16_t*)a3_);
    GMTensor<bfloat16_t> b3((__gm__ bfloat16_t*)b3_);
    GMTensor<bfloat16_t> a4((__gm__ bfloat16_t*)a4_);
    GMTensor<bfloat16_t> b4((__gm__ bfloat16_t*)b4_);
    GMTensor<bfloat16_t> a5((__gm__ bfloat16_t*)a5_);
    GMTensor<bfloat16_t> b5((__gm__ bfloat16_t*)b5_);
    GMTensor<bfloat16_t> a6((__gm__ bfloat16_t*)a6_);
    GMTensor<bfloat16_t> b6((__gm__ bfloat16_t*)b6_);
    GMTensor<bfloat16_t> a7((__gm__ bfloat16_t*)a7_);
    GMTensor<bfloat16_t> b7((__gm__ bfloat16_t*)b7_);
    GMTensor<float> z((__gm__ float*)z_);
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 0> ev_mte2_mte1_ready_1;  // guards l1a0, l1a1, l1a2, l1a3, l1a4, l1a5, l1a6, l1a7, l1b0, l1b1, l1b2, l1b3, l1b4, l1b5, l1b6, l1b7 #242
    SEvent<PIPE_MTE1, PIPE_M, 0, 0> ev_mte1_m_ready_1;  // guards _l0b #243
    SEvent<PIPE_M, PIPE_MTE1, 0, 0> ev_m_mte1_valid_1;  // guards _l0a #244
    SEvent<PIPE_M, PIPE_FIX, 0, 0> ev_m_fix_ready_1;  // guards l0c #245
    const DBuff<uint8_t, Position::L0A> _l0a((uint64_t)(0), (uint64_t)(32768));  // #45
    const DBuff<uint8_t, Position::L0B> _l0b((uint64_t)(0), (uint64_t)(32768));  // #46
    int32_t _l0acnt = 0;  // #48
    int32_t _l0bcnt = 0;  // #49
    const Tensor<bfloat16_t, Position::L1> l1a0((uint64_t)(0));  // #1
    const Tensor<bfloat16_t, Position::L1> l1b0((uint64_t)(512));  // #2
    const Tensor<bfloat16_t, Position::L1> l1a1((uint64_t)(2560));  // #3
    const Tensor<bfloat16_t, Position::L1> l1b1((uint64_t)(3072));  // #4
    const Tensor<bfloat16_t, Position::L1> l1a2((uint64_t)(5120));  // #5
    const Tensor<bfloat16_t, Position::L1> l1b2((uint64_t)(5632));  // #6
    const Tensor<bfloat16_t, Position::L1> l1a3((uint64_t)(7680));  // #7
    const Tensor<bfloat16_t, Position::L1> l1b3((uint64_t)(8192));  // #8
    const Tensor<bfloat16_t, Position::L1> l1a4((uint64_t)(10240));  // #9
    const Tensor<bfloat16_t, Position::L1> l1b4((uint64_t)(10752));  // #10
    const Tensor<bfloat16_t, Position::L1> l1a5((uint64_t)(12800));  // #11
    const Tensor<bfloat16_t, Position::L1> l1b5((uint64_t)(13312));  // #12
    const Tensor<bfloat16_t, Position::L1> l1a6((uint64_t)(15360));  // #13
    const Tensor<bfloat16_t, Position::L1> l1b6((uint64_t)(15872));  // #14
    const Tensor<bfloat16_t, Position::L1> l1a7((uint64_t)(17920));  // #15
    const Tensor<bfloat16_t, Position::L1> l1b7((uint64_t)(18432));  // #16
    const Tensor<float, Position::L0C> l0c((uint64_t)(0));  // #17
    gm_to_l1_nd2nz(l1a0, a0, 16, 16, 16, 16);  // #19
    gm_to_l1_nd2nz(l1b0, b0, 64, 16, 64, 16);  // #20
    gm_to_l1_nd2nz(l1a1, a1, 16, 16, 16, 16);  // #21
    gm_to_l1_nd2nz(l1b1, b1, 64, 16, 64, 16);  // #22
    gm_to_l1_nd2nz(l1a2, a2, 16, 16, 16, 16);  // #23
    gm_to_l1_nd2nz(l1b2, b2, 64, 16, 64, 16);  // #24
    gm_to_l1_nd2nz(l1a3, a3, 16, 16, 16, 16);  // #25
    gm_to_l1_nd2nz(l1b3, b3, 64, 16, 64, 16);  // #26
    gm_to_l1_nd2nz(l1a4, a4, 16, 16, 16, 16);  // #27
    gm_to_l1_nd2nz(l1b4, b4, 64, 16, 64, 16);  // #28
    gm_to_l1_nd2nz(l1a5, a5, 16, 16, 16, 16);  // #29
    gm_to_l1_nd2nz(l1b5, b5, 64, 16, 64, 16);  // #30
    gm_to_l1_nd2nz(l1a6, a6, 16, 16, 16, 16);  // #31
    gm_to_l1_nd2nz(l1b6, b6, 64, 16, 64, 16);  // #32
    gm_to_l1_nd2nz(l1a7, a7, 16, 16, 16, 16);  // #33
    gm_to_l1_nd2nz(l1b7, b7, 64, 16, 64, 16);  // #34
    ev_mte2_mte1_ready_1.set();  // #208
    const Tensor<uint8_t, Position::L0A> l0a_slot = _l0a.get(_l0acnt);  // #51
    const Tensor<bfloat16_t, Position::L0A> l0a_view = _l0a.get(_l0acnt).as<bfloat16_t>();  // #52
    const Tensor<uint8_t, Position::L0B> l0b_slot = _l0b.get(_l0bcnt);  // #53
    const Tensor<bfloat16_t, Position::L0B> l0b_view = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #54
    ev_mte2_mte1_ready_1.wait();  // #209
    l1_to_l0<false>(l0a_view, l1a0, 16, 16, 16, 16);  // #55
    l1_to_l0<false>(l0b_view, l1b0, 64, 16, 64, 16);  // #56
    ev_mte1_m_ready_1.set();  // #210
    ev_mte1_m_ready_1.wait();  // #211
    mmad(l0c, l0a_view, l0b_view, 16, 64, 16, true);  // #57
    ev_m_mte1_valid_1.set();  // #226
    _l0acnt = 1;  // #59
    _l0bcnt = 1;  // #61
    const Tensor<uint8_t, Position::L0A> l0a_slot_1 = _l0a.get(_l0acnt);  // #62
    const Tensor<bfloat16_t, Position::L0A> l0a_view_1 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #63
    const Tensor<uint8_t, Position::L0B> l0b_slot_1 = _l0b.get(_l0bcnt);  // #64
    const Tensor<bfloat16_t, Position::L0B> l0b_view_1 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #65
    ev_m_mte1_valid_1.wait();  // #227
    l1_to_l0<false>(l0a_view_1, l1a1, 16, 16, 16, 16);  // #66
    l1_to_l0<false>(l0b_view_1, l1b1, 64, 16, 64, 16);  // #67
    ev_mte1_m_ready_1.set();  // #212
    ev_mte1_m_ready_1.wait();  // #213
    mmad(l0c, l0a_view_1, l0b_view_1, 16, 64, 16, false);  // #68
    ev_m_mte1_valid_1.set();  // #228
    _l0acnt = 2;  // #70
    _l0bcnt = 2;  // #72
    const Tensor<uint8_t, Position::L0A> l0a_slot_2 = _l0a.get(_l0acnt);  // #73
    const Tensor<bfloat16_t, Position::L0A> l0a_view_2 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #74
    const Tensor<uint8_t, Position::L0B> l0b_slot_2 = _l0b.get(_l0bcnt);  // #75
    const Tensor<bfloat16_t, Position::L0B> l0b_view_2 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #76
    ev_m_mte1_valid_1.wait();  // #229
    l1_to_l0<false>(l0a_view_2, l1a2, 16, 16, 16, 16);  // #77
    l1_to_l0<false>(l0b_view_2, l1b2, 64, 16, 64, 16);  // #78
    ev_mte1_m_ready_1.set();  // #214
    ev_mte1_m_ready_1.wait();  // #215
    mmad(l0c, l0a_view_2, l0b_view_2, 16, 64, 16, false);  // #79
    ev_m_mte1_valid_1.set();  // #230
    _l0acnt = 3;  // #81
    _l0bcnt = 3;  // #83
    const Tensor<uint8_t, Position::L0A> l0a_slot_3 = _l0a.get(_l0acnt);  // #84
    const Tensor<bfloat16_t, Position::L0A> l0a_view_3 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #85
    const Tensor<uint8_t, Position::L0B> l0b_slot_3 = _l0b.get(_l0bcnt);  // #86
    const Tensor<bfloat16_t, Position::L0B> l0b_view_3 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #87
    ev_m_mte1_valid_1.wait();  // #231
    l1_to_l0<false>(l0a_view_3, l1a3, 16, 16, 16, 16);  // #88
    l1_to_l0<false>(l0b_view_3, l1b3, 64, 16, 64, 16);  // #89
    ev_mte1_m_ready_1.set();  // #216
    ev_mte1_m_ready_1.wait();  // #217
    mmad(l0c, l0a_view_3, l0b_view_3, 16, 64, 16, false);  // #90
    ev_m_mte1_valid_1.set();  // #232
    _l0acnt = 4;  // #92
    _l0bcnt = 4;  // #94
    const Tensor<uint8_t, Position::L0A> l0a_slot_4 = _l0a.get(_l0acnt);  // #95
    const Tensor<bfloat16_t, Position::L0A> l0a_view_4 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #96
    const Tensor<uint8_t, Position::L0B> l0b_slot_4 = _l0b.get(_l0bcnt);  // #97
    const Tensor<bfloat16_t, Position::L0B> l0b_view_4 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #98
    ev_m_mte1_valid_1.wait();  // #233
    l1_to_l0<false>(l0a_view_4, l1a4, 16, 16, 16, 16);  // #99
    l1_to_l0<false>(l0b_view_4, l1b4, 64, 16, 64, 16);  // #100
    ev_mte1_m_ready_1.set();  // #218
    ev_mte1_m_ready_1.wait();  // #219
    mmad(l0c, l0a_view_4, l0b_view_4, 16, 64, 16, false);  // #101
    ev_m_mte1_valid_1.set();  // #234
    _l0acnt = 5;  // #103
    _l0bcnt = 5;  // #105
    const Tensor<uint8_t, Position::L0A> l0a_slot_5 = _l0a.get(_l0acnt);  // #106
    const Tensor<bfloat16_t, Position::L0A> l0a_view_5 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #107
    const Tensor<uint8_t, Position::L0B> l0b_slot_5 = _l0b.get(_l0bcnt);  // #108
    const Tensor<bfloat16_t, Position::L0B> l0b_view_5 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #109
    ev_m_mte1_valid_1.wait();  // #235
    l1_to_l0<false>(l0a_view_5, l1a5, 16, 16, 16, 16);  // #110
    l1_to_l0<false>(l0b_view_5, l1b5, 64, 16, 64, 16);  // #111
    ev_mte1_m_ready_1.set();  // #220
    ev_mte1_m_ready_1.wait();  // #221
    mmad(l0c, l0a_view_5, l0b_view_5, 16, 64, 16, false);  // #112
    ev_m_mte1_valid_1.set();  // #236
    _l0acnt = 6;  // #114
    _l0bcnt = 6;  // #116
    const Tensor<uint8_t, Position::L0A> l0a_slot_6 = _l0a.get(_l0acnt);  // #117
    const Tensor<bfloat16_t, Position::L0A> l0a_view_6 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #118
    const Tensor<uint8_t, Position::L0B> l0b_slot_6 = _l0b.get(_l0bcnt);  // #119
    const Tensor<bfloat16_t, Position::L0B> l0b_view_6 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #120
    ev_m_mte1_valid_1.wait();  // #237
    l1_to_l0<false>(l0a_view_6, l1a6, 16, 16, 16, 16);  // #121
    l1_to_l0<false>(l0b_view_6, l1b6, 64, 16, 64, 16);  // #122
    ev_mte1_m_ready_1.set();  // #222
    ev_mte1_m_ready_1.wait();  // #223
    mmad(l0c, l0a_view_6, l0b_view_6, 16, 64, 16, false);  // #123
    ev_m_mte1_valid_1.set();  // #238
    _l0acnt = 7;  // #125
    _l0bcnt = 7;  // #127
    const Tensor<uint8_t, Position::L0A> l0a_slot_7 = _l0a.get(_l0acnt);  // #128
    const Tensor<bfloat16_t, Position::L0A> l0a_view_7 = _l0a.get(_l0acnt).as<bfloat16_t>();  // #129
    const Tensor<uint8_t, Position::L0B> l0b_slot_7 = _l0b.get(_l0bcnt);  // #130
    const Tensor<bfloat16_t, Position::L0B> l0b_view_7 = _l0b.get(_l0bcnt).as<bfloat16_t>();  // #131
    ev_m_mte1_valid_1.wait();  // #239
    l1_to_l0<false>(l0a_view_7, l1a7, 16, 16, 16, 16);  // #132
    l1_to_l0<false>(l0b_view_7, l1b7, 64, 16, 64, 16);  // #133
    ev_mte1_m_ready_1.set();  // #224
    ev_mte1_m_ready_1.wait();  // #225
    mmad(l0c, l0a_view_7, l0b_view_7, 16, 64, 16, false);  // #134
    ev_m_fix_ready_1.set();  // #240
    _l0acnt = 8;  // #136
    _l0bcnt = 8;  // #138
    ev_m_fix_ready_1.wait();  // #241
    l0c_to_gm_nz2nd(z, l0c, 16, 64, 64, 16, false, 0.0f, 0, false, false);  // #43
    return;  // #44
}
