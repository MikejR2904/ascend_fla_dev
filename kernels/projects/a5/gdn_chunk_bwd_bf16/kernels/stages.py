"""BF16 storage with FP32 checkpoint/replay adjoint; no state inversion or atomics.

One vector worker owns each complete (batch,value-head) recurrence. Local
single-slot autosync protects UB lifetimes. The replay GM tape has an explicit
MTE3->MTE2 publication event and a local drain before its next overwrite.
"""
from ascriptor.a5 import *

D = 128
C = 64
SCALE = 128**-.5


@vf()
def clear_row(row: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    row[0:1,0:D] <<= z
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@vf()
def publish_pair(q: Tensor, k: Tensor, qo: Tensor, ko: Tensor):
    # Narrow only the complete FP32 group sums, once per final output.
    value = RegList(DT.float, 2)
    value <<= q[0:1,0:D]
    qo[0:1,0:D] <<= value
    value <<= k[0:1,0:D]
    ko[0:1,0:D] <<= value
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@vf()
def clear_state(state: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    for i in range(D):
        state[i:i+1, 0:D] <<= z
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def primal(state: Tensor, key: Tensor, value: Tensor, gate: Tensor, beta: Tensor):
    row = RegList(DT.float, 2)
    read = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    residual = RegList(DT.float, 2)
    decay = Reg(DT.float)
    ki = Reg(DT.float)
    weight = Reg(DT.float)
    decay <<= gate[0:1, 0:1].single()
    decay <<= decay.exp()
    weight <<= beta[0:1, 0:1].single()
    read <<= 0.0
    for i in range(D):
        row <<= state[i:i+1, 0:D]
        row <<= row * decay
        state[i:i+1, 0:D] <<= row
        ki <<= key[0:1, i:i+1].single()
        tmp <<= row * ki
        read <<= read + tmp
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    residual <<= value[0:1, 0:D]
    residual <<= residual - read
    residual <<= residual * weight
    for i in range(D):
        row <<= state[i:i+1, 0:D]
        ki <<= key[0:1, i:i+1].single()
        tmp <<= residual * ki
        row <<= row + tmp
        state[i:i+1, 0:D] <<= row
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def gdn_bf16_bwd_checkpoints(
    k: GM[bf16, ("B", "T", "H", 128)], v: GM[bf16, ("B", "T", "HV", 128)],
    g: GM[f32, ("B", "T", "HV")], beta: GM[f32, ("B", "T", "HV")],
    checkpoints: GM[f32, ("B", "N", "HV", 128, 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32, T: i32, H: i32, HV: i32, N: i32,
):
    state = Tensor(DT.float, [D,D], Position.UB)
    ku = Tensor(DT.bfloat16, [1,D], Position.UB)
    vu = Tensor(DT.bfloat16, [1,D], Position.UB)
    gu = Tensor(DT.float, [1,8], Position.UB)
    bu = Tensor(DT.float, [1,8], Position.UB)
    per = CeilDiv(B*HV, GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per, B*HV)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//HV)
            hh = Var(item%HV)
            kh = Var(hh//(HV//H))
            clear_state(state)
            for chunk in range(N):
                checkpoints[bb,chunk,hh,:,:] <<= state[:,:]
                for token in range(C):
                    tt = Var(chunk*C+token)
                    ku[:,:] <<= k[bb,tt,kh:kh+1,:]
                    vu[:,:] <<= v[bb,tt,hh:hh+1,:]
                    gu[:,0:1] <<= g[bb,tt,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt,hh:hh+1]
                    primal(state,ku,vu,gu,bu)
            final_state[bb,hh,:,:] <<= state[:,:]
    return checkpoints, final_state


@vf()
def adjoint(state: Tensor, back: Tensor, query: Tensor, key: Tensor, value: Tensor,
            gate: Tensor, beta: Tensor, dout: Tensor, dq: Tensor, dk: Tensor,
            dv: Tensor, dg: Tensor, db: Tensor):
    # state enters as Sprev and is overwritten with D. back enters as dS.
    row = RegList(DT.float, 2)
    grad = RegList(DT.float, 2)
    read = RegList(DT.float, 2)
    residual = RegList(DT.float, 2)
    z = RegList(DT.float, 2)
    dz = RegList(DT.float, 2)
    dr = RegList(DT.float, 2)
    dot = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    tmp2 = RegList(DT.float, 2)
    gate_grad = RegList(DT.float, 2)
    decay = Reg(DT.float)
    weight = Reg(DT.float)
    ki = Reg(DT.float)
    qi = Reg(DT.float)
    scalar = Reg(DT.float)
    other = Reg(DT.float)
    decay <<= gate[0:1,0:1].single()
    decay <<= decay.exp()
    weight <<= beta[0:1,0:1].single()
    read <<= 0.0
    for i in range(D):
        row <<= state[i:i+1,0:D]
        row <<= row * decay
        state[i:i+1,0:D] <<= row
        ki <<= key[0:1,i:i+1].single()
        tmp <<= row * ki
        read <<= read + tmp
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    residual <<= value[0:1,0:D]
    residual <<= residual - read
    z <<= residual * weight
    dot <<= dout[0:1,0:D]
    dz <<= 0.0
    for i in range(D):
        row <<= state[i:i+1,0:D]
        ki <<= key[0:1,i:i+1].single()
        tmp <<= z * ki
        tmp <<= row + tmp
        tmp <<= tmp * dot
        scalar <<= tmp.cadd()
        scalar <<= scalar * SCALE
        dq[0:1,i:i+1] <<= scalar.single_value()
        grad <<= back[i:i+1,0:D]
        qi <<= query[0:1,i:i+1].single()
        qi <<= qi * SCALE
        tmp <<= dot * qi
        grad <<= grad + tmp
        back[i:i+1,0:D] <<= grad
        tmp <<= grad * ki
        dz <<= dz + tmp
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    dr <<= dz * weight
    dv[0:1,0:D] <<= dr
    tmp <<= dz * residual
    scalar <<= tmp.cadd()
    db[0:1,0:1] <<= scalar.single_value()
    gate_grad <<= 0.0
    for i in range(D):
        grad <<= back[i:i+1,0:D]
        row <<= state[i:i+1,0:D]
        tmp <<= grad * z
        scalar <<= tmp.cadd()
        tmp <<= row * dr
        other <<= tmp.cadd()
        scalar <<= scalar - other
        dk[0:1,i:i+1] <<= scalar.single_value()
        ki <<= key[0:1,i:i+1].single()
        tmp <<= dr * ki
        grad <<= grad - tmp
        tmp2 <<= grad * row
        gate_grad <<= gate_grad + tmp2
        grad <<= grad * decay
        back[i:i+1,0:D] <<= grad
    scalar <<= gate_grad.cadd()
    dg[0:1,0:1] <<= scalar.single_value()
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@kernel()
def gdn_bf16_bwd_reverse(
    q: GM[bf16, ("B","T","H",128)], k: GM[bf16, ("B","T","H",128)],
    v: GM[bf16, ("B","T","HV",128)], g: GM[f32, ("B","T","HV")],
    beta: GM[f32, ("B","T","HV")], dout: GM[bf16, ("B","T","HV",128)],
    dht: GM[f32, ("B","HV",128,128)],
    checkpoints: GM[f32, ("B","N","HV",128,128)],
    tape: GM[f32, ("B","HV",64,128,128)],
    dq_parts: GM[f32, ("B","T","HV",128)], dk_parts: GM[f32, ("B","T","HV",128)],
    dv: GM[bf16, ("B","T","HV",128)], dg: GM[f32, ("B","T","HV")],
    dbeta: GM[f32, ("B","T","HV")],
    B: i32, T: i32, H: i32, HV: i32, N: i32, has_do: i32, has_dht: i32,
):
    state = Tensor(DT.float,[D,D],Position.UB)
    back = Tensor(DT.float,[D,D],Position.UB)
    qu = Tensor(DT.bfloat16,[1,D],Position.UB)
    ku = Tensor(DT.bfloat16,[1,D],Position.UB)
    vu = Tensor(DT.bfloat16,[1,D],Position.UB)
    gu = Tensor(DT.float,[1,8],Position.UB)
    bu = Tensor(DT.float,[1,8],Position.UB)
    ou = Tensor(DT.bfloat16,[1,D],Position.UB)
    dqu = Tensor(DT.float,[1,D],Position.UB)
    dku = Tensor(DT.float,[1,D],Position.UB)
    dvu = Tensor(DT.bfloat16,[1,D],Position.UB)
    dgu = Tensor(DT.float,[1,8],Position.UB)
    dbu = Tensor(DT.float,[1,8],Position.UB)
    ready = SEvent(Pipe.MTE3,Pipe.MTE2)
    per = CeilDiv(B*HV,GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per,B*HV)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//HV)
            hh = Var(item%HV)
            kh = Var(hh//(HV//H))
            if has_dht != 0:
                back[:,:] <<= dht[bb,hh,:,:]
            else:
                clear_state(back)
            if has_do == 0:
                clear_row(ou)
            for reverse_chunk in range(N):
                chunk = Var(N-1-reverse_chunk)
                state[:,:] <<= checkpoints[bb,chunk,hh,:,:]
                for token in range(C):
                    tt = Var(chunk*C+token)
                    tape[bb,hh,token,:,:] <<= state[:,:]
                    ku[:,:] <<= k[bb,tt,kh:kh+1,:]
                    vu[:,:] <<= v[bb,tt,hh:hh+1,:]
                    gu[:,0:1] <<= g[bb,tt,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt,hh:hh+1]
                    primal(state,ku,vu,gu,bu)
                ready.set()
                ready.wait()
                for reverse_token in range(C):
                    token = Var(C-1-reverse_token)
                    tt = Var(chunk*C+token)
                    state[:,:] <<= tape[bb,hh,token,:,:]
                    qu[:,:] <<= q[bb,tt,kh:kh+1,:]
                    ku[:,:] <<= k[bb,tt,kh:kh+1,:]
                    vu[:,:] <<= v[bb,tt,hh:hh+1,:]
                    gu[:,0:1] <<= g[bb,tt,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt,hh:hh+1]
                    if has_do != 0:
                        ou[:,:] <<= dout[bb,tt,hh:hh+1,:]
                    adjoint(state,back,qu,ku,vu,gu,bu,ou,dqu,dku,dvu,dgu,dbu)
                    dq_parts[bb,tt,hh:hh+1,:] <<= dqu[:,:]
                    dk_parts[bb,tt,hh:hh+1,:] <<= dku[:,:]
                    dv[bb,tt,hh:hh+1,:] <<= dvu[:,:]
                    dg[bb,tt,hh:hh+1] <<= dgu[:,0:1]
                    dbeta[bb,tt,hh:hh+1] <<= dbu[:,0:1]
                # All tape readers and gradient publications retire locally.
                # The next replay can safely overwrite this owner's GM slice.
                barrier(Pipe.ALL)
    return tape,dq_parts,dk_parts,dv,dg,dbeta


@vf()
def clear_pair(q: Tensor,k: Tensor):
    z = RegList(DT.float,2)
    z <<= 0.0
    q[0:1,0:D] <<= z
    k[0:1,0:D] <<= z
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@vf()
def add_pair(q: Tensor,k: Tensor,qi: Tensor,ki: Tensor):
    total = RegList(DT.float,2)
    part = RegList(DT.float,2)
    total <<= q[0:1,0:D]
    part <<= qi[0:1,0:D]
    total <<= total + part
    q[0:1,0:D] <<= total
    total <<= k[0:1,0:D]
    part <<= ki[0:1,0:D]
    total <<= total + part
    k[0:1,0:D] <<= total
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@kernel()
def gdn_bf16_bwd_group_reduce(
    dq_parts: GM[f32,("B","T","HV",128)], dk_parts: GM[f32,("B","T","HV",128)],
    dq: GM[bf16,("B","T","H",128)], dk: GM[bf16,("B","T","H",128)],
    B: i32,T: i32,H: i32,HV: i32,
):
    qu = Tensor(DT.float,[1,D],Position.UB)
    ku = Tensor(DT.float,[1,D],Position.UB)
    qi = Tensor(DT.float,[1,D],Position.UB)
    ki = Tensor(DT.float,[1,D],Position.UB)
    qo = Tensor(DT.bfloat16,[1,D],Position.UB)
    ko = Tensor(DT.bfloat16,[1,D],Position.UB)
    per = CeilDiv(B*T*H,GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per,B*T*H)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//(T*H))
            tt = Var((item//H)%T)
            kh = Var(item%H)
            clear_pair(qu,ku)
            for group in range(HV//H):
                hh = Var(kh*(HV//H)+group)
                qi[:,:] <<= dq_parts[bb,tt,hh:hh+1,:]
                ki[:,:] <<= dk_parts[bb,tt,hh:hh+1,:]
                add_pair(qu,ku,qi,ki)
            publish_pair(qu,ku,qo,ko)
            dq[bb,tt,kh:kh+1,:] <<= qo[:,:]
            dk[bb,tt,kh:kh+1,:] <<= ko[:,:]
    return dq,dk
