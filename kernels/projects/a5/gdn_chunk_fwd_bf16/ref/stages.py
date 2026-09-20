"""FP32 mathematical stage references, separate from the recurrent oracle."""
import torch


def reference_stages(inputs):
    q = inputs['q'].float()
    hv = inputs['v'].shape[2]
    key_indices = torch.arange(hv) // (hv // q.shape[2])
    k = inputs['k'].float()
    if hv != q.shape[2]:
        q, k = q.index_select(2, key_indices), k.index_select(2, key_indices)
    batch, time, heads, _ = q.shape
    chunks = (time + 63) // 64
    qn = q * 128**-0.5
    kn = k
    def pack(x):
        padding = torch.zeros(batch, chunks * 64 - time, heads, 128)
        return torch.cat((x, padding), dim=1).reshape(batch, chunks, 64, heads, 128).transpose(2, 3).contiguous()
    qn, kn, gc, bk, wv = (
        pack(qn), pack(kn), pack(inputs['g'].unsqueeze(-1).expand_as(q)).cumsum(-2),
        pack(inputs['beta'].unsqueeze(-1) * kn), pack(inputs['beta'].unsqueeze(-1) * inputs['v'].float()))
    lower = torch.zeros(batch, chunks, heads, 64, 64)
    score = torch.zeros_like(lower)
    for i in range(64):
        decayed = kn[..., :i+1, :] * (gc[..., i:i+1, :] - gc[..., :i+1, :]).exp()
        score[..., i, :i+1] = (qn[..., i:i+1, :] * decayed).sum(-1)
        if i:
            lower[..., i, :i] = (bk[..., i:i+1, :] * decayed[..., :i, :]).sum(-1)
    system = lower + torch.eye(64)
    u = torch.linalg.solve_triangular(system, wv, upper=False, unitriangular=True)
    wy = torch.linalg.solve_triangular(system, bk * gc.exp(), upper=False, unitriangular=True)
    state = torch.zeros(batch, heads, 128, 128)
    states, deltas, outputs = [], [], []
    for c in range(chunks):
        states.append(state)
        delta = u[:, c] - wy[:, c] @ state
        deltas.append(delta)
        outputs.append((qn[:, c] * gc[:, c].exp()) @ state + score[:, c] @ delta)
        tail = kn[:, c] * (gc[:, c, :, -1:, :] - gc[:, c]).exp()
        state = gc[:, c, :, -1, :].exp().unsqueeze(-1) * state + tail.transpose(-1, -2) @ delta
    o = torch.stack(outputs, dim=1).transpose(2, 3).reshape(batch, chunks*64, heads, 128)[:, :time].contiguous()
    return dict(qn=qn, kn=kn, gc=gc, bk=bk, wv=wv, lower=lower, score=score,
                u=u, wy=wy, states=torch.stack(states, dim=1),
                delta=torch.stack(deltas, dim=1), final_state=state, o=o)
