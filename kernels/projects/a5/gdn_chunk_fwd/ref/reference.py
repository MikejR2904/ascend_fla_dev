"""CPU FP32 block-solve reference, independent of the kernel WY implementation."""
import torch

def block_solve(q, k, v, beta, g):
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))
    beta, g = (x.transpose(1, 2).float() for x in (beta, g))
    state = torch.zeros(*q.shape[:2], 128, 128)
    output = torch.empty_like(v)
    eye = torch.eye(64)
    causal = torch.ones(64, 64, dtype=torch.bool).tril()
    for start in range(0, q.shape[2], 64):
        sl = slice(start, start + 64)
        kc, qc, vc, bc = k[:, :, sl], q[:, :, sl], v[:, :, sl], beta[:, :, sl]
        prefix = g[:, :, sl].cumsum(-1)
        delta = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        # Never evaluate positive acausal exponents, even when gate spans are large.
        decay = delta.masked_fill(~causal, -torch.inf).exp()
        lower = ((kc @ kc.transpose(-1, -2)) * decay * bc.unsqueeze(-1)).tril(-1)
        rhs = bc.unsqueeze(-1) * (vc - prefix.exp().unsqueeze(-1) * (kc @ state))
        updates = torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)
        output[:, :, sl] = ((qc @ state) * prefix.exp().unsqueeze(-1)
                           + ((qc @ kc.transpose(-1, -2)) * decay) @ updates) / (128 ** 0.5)
        weights = (prefix[..., -1:] - prefix).exp()
        state = state * prefix[..., -1:].exp().unsqueeze(-1) + (kc * weights.unsqueeze(-1)).transpose(-1, -2) @ updates
    return output.transpose(1, 2).contiguous(), state



def make_inputs(case):
    p = case['parameters']
    b, t, h = (p[n] for n in ('B', 'T', 'H'))
    rng = torch.Generator().manual_seed(case['seed'])
    data = {n: torch.randn(b, t, h, 128, generator=rng) * 0.05 for n in ('q','k','v')}
    if p.get('normalized_keys', False):
        data['k'] = torch.nn.functional.normalize(data['k'], dim=-1)
    if p.get('bf16_inputs', False):
        data = {n: x.bfloat16().float() for n, x in data.items()}
    data['beta'] = torch.rand(b,t,h,generator=rng)
    data['g'] = -torch.rand(b,t,h,generator=rng) * p.get('gate_scale',1.0)
    data['initial_state'] = torch.zeros(b,h,128,128)
    return data


def validate_inputs(inputs, case=None):
    if set(inputs) != {'q','k','v','beta','g','initial_state'}:
        raise ValueError('expected q/k/v/beta/g/initial_state')
    q=inputs['q']
    if q.ndim != 4 or q.shape[-1] != 128 or min(q.shape[:3]) < 1 or q.shape[1]%64 or q.shape[1]>4096:
        raise ValueError('expected positive B/H, T multiple of 64 up to 4096, D=128')
    b,t,h,_=q.shape
    for n,x in inputs.items():
        shape = (b,h,128,128) if n=='initial_state' else ((b,t,h) if n in ('beta','g') else q.shape)
        if x.shape != shape or x.dtype != torch.float32 or x.device.type != 'cpu' or not x.is_contiguous():
            raise ValueError(f'{n} requires contiguous CPU float32 {tuple(shape)}; no GQA')
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f'{n} must be finite')
    if bool((inputs['g']>0).any()) or bool(((inputs['beta']<0)|(inputs['beta']>1)).any()):
        raise ValueError('g<=0 and beta in [0,1] required')
    if bool(inputs['initial_state'].any()):
        raise ValueError('only zero initial state supported')
    if case and case.get('block_dim',1) not in (1,2):
        raise ValueError('block_dim must be 1 or 2')


def reference(inputs):
    validate_inputs(inputs)
    o,s=block_solve(*(inputs[n] for n in ('q','k','v','beta','g')))
    return {'o':o,'final_state':s}


def validate_reference(inputs, outputs, case=None):
    b,t,h,_=inputs['q'].shape
    for n,shape in (('o',(b,t,h,128)),('final_state',(b,h,128,128))):
        x=outputs[n]
        if x.shape!=shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():
            raise ValueError(f'invalid reference {n}')
        if x.norm()>0:
            for wrong in (torch.zeros_like(x), x*1.25):
                if torch.allclose(wrong,x,atol=2e-5,rtol=2e-4):
                    raise AssertionError('reference comparison cannot reject wrong output')
