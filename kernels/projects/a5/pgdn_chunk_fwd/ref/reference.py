"""Independent CPU block solves for ATK and the asymmetric PGDN delta rule."""
import torch

OUTPUTS = ('o', 'final_state', 'final_A_state')


def make_inputs(case):
    p = case['parameters']
    b, t, h, hv = (p[n] for n in ('B', 'T', 'H', 'HV'))
    rng = torch.Generator().manual_seed(case['seed'])
    data = {n: torch.randn(b, t, hv if n == 'v' else h, 128, generator=rng) * 0.05
            for n in ('q', 'k', 'v')}
    if p.get('bf16_inputs', False):
        data = {n: x.bfloat16().float() for n, x in data.items()}
    if 'norm' in p:
        for n in ('q', 'k'):
            data[n].zero_()
            data[n][..., 0] = p['norm']
    for n, heads in (('beta', hv), ('beta_atk', h)):
        data[n] = torch.rand(b, t, heads, generator=rng)
        if n in p:
            data[n].fill_(p[n])
    data['g'] = -torch.rand(b, t, hv, generator=rng) * p.get('gate_scale', 1.0)
    data['g_atk'] = -torch.rand(b, t, h, generator=rng) * p.get('atk_gate_scale', 0.03)
    data['initial_state'] = torch.zeros(b, hv, 128, 128)
    return data


def validate_inputs(inputs, case=None):
    names = {'q', 'k', 'v', 'g', 'beta', 'g_atk', 'beta_atk', 'initial_state'}
    if set(inputs) != names:
        raise ValueError(f'expected inputs {sorted(names)}')
    q, v = inputs['q'], inputs['v']
    if q.ndim != 4 or min(q.shape[:3]) < 1 or q.shape[-1] != 128 or q.shape[1] % 64 or q.shape[1] > 4096:
        raise ValueError(f'requires positive B/H, T multiple of64 up to4096, K=128; got {tuple(q.shape)}')
    b, t, h, _ = q.shape
    if v.ndim != 4 or v.shape[:2] != (b, t) or v.shape[-1] != 128 or v.shape[2] < 1 or v.shape[2] % h:
        raise ValueError(f'v requires [B,T,HV,128], positive HV divisible by H={h}; got {tuple(v.shape)}')
    hv = v.shape[2]
    shapes = dict(q=q.shape, k=q.shape, v=v.shape, g=(b,t,hv), beta=(b,t,hv),
                  g_atk=(b,t,h), beta_atk=(b,t,h), initial_state=(b,hv,128,128))
    for name, value in inputs.items():
        if value.shape != shapes[name] or value.dtype != torch.float32 or value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError(f'{name} requires contiguous CPU FP32 {tuple(shapes[name])}; got {tuple(value.shape)}/{value.dtype}')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite')
    for name in ('g', 'g_atk'):
        if bool((inputs[name] > 0).any()):
            raise ValueError(f'{name} must be nonpositive')
        if not bool(torch.isfinite(inputs[name].reshape(b,t//64,64,-1).sum(2)).all()):
            raise ValueError(f'{name} chunk prefix must remain finite in FP32')
    for name in ('beta', 'beta_atk'):
        if bool(((inputs[name] < 0) | (inputs[name] > 1)).any()):
            raise ValueError(f'{name} must be in [0,1]')
    if bool(inputs['initial_state'].any()):
        raise ValueError('only zero initial state is supported')
    if case and case.get('block_dim', 1) not in (1, 2):
        raise ValueError('block_dim must be1 or2')


def block_solve(inputs, *, retain_stages=False):
    """ATK as a causal weighted sum; main update as a triangular solve.

    This does not import the pinned recurrent oracle, production graph, kernels,
    or simulator. ATK is computed at H before independent value-head indexing.
    """
    q, k, v = (inputs[n].float() for n in ('q', 'k', 'v'))
    b, t, h, _ = q.shape
    hv = v.shape[2]
    q_norm = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    k_read = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.zeros(b, h, 128)
    k_writes = []
    causal = torch.ones(64, 64, dtype=torch.bool).tril()
    for start in range(0, t, 64):
        sl = slice(start, start + 64)
        prefix = inputs['g_atk'][:, sl].transpose(1, 2).cumsum(-1)
        diff = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        weights = diff.masked_fill(~causal, -torch.inf).exp()
        key = k_read[:, sl].transpose(1, 2)
        updates = key.square() * inputs['beta_atk'][:, sl].transpose(1, 2).unsqueeze(-1)
        diagonal = weights @ updates + prefix.exp().unsqueeze(-1) * a.unsqueeze(-2)
        centered = (diagonal + 1e-6).log() + 0.2
        multiplier = (-torch.tensor(1.5).log() * centered / (1 + centered.abs())).exp()
        k_writes.append((key * multiplier).transpose(1, 2))
        a = diagonal[..., -1, :].clone()
    k_write = torch.cat(k_writes, dim=1).contiguous()
    indices = torch.arange(hv) // (hv // h)
    def grouped(x):
        return x.index_select(2, indices).transpose(1, 2)
    queries, reads, writes = grouped(q_norm) * 128**-0.5, grouped(k_read), grouped(k_write)
    values = v.transpose(1, 2)
    beta = inputs['beta'].transpose(1, 2)
    g = inputs['g'].transpose(1, 2)
    state = inputs['initial_state'].clone()
    output = torch.empty_like(values)
    collected = {n: [] for n in ('qn','kw','gc','bk','wv','lower','score','u','wy','states','delta')}
    for start in range(0, t, 64):
        sl = slice(start, start + 64)
        qr, kr, kw, vc, bc = queries[:,:,sl], reads[:,:,sl], writes[:,:,sl], values[:,:,sl], beta[:,:,sl]
        prefix = g[:,:,sl].cumsum(-1)
        decay = (prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp()
        bk = bc.unsqueeze(-1) * kr
        wv = bc.unsqueeze(-1) * vc
        lower = ((bk @ kw.transpose(-1,-2)) * decay).tril(-1)
        system = torch.eye(64) + lower
        # One direct RHS solve is independent of the kernel's split WY/U solve.
        rhs = bc.unsqueeze(-1) * (vc - prefix.exp().unsqueeze(-1) * (kr @ state))
        delta = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        score = (qr @ kw.transpose(-1,-2)) * decay
        output[:,:,sl] = prefix.exp().unsqueeze(-1) * (qr @ state) + score @ delta
        if retain_stages:
            stage = dict(qn=qr, kw=kw, gc=prefix.unsqueeze(-1).expand_as(kr), bk=bk, wv=wv,
                         lower=lower, score=score, delta=delta, states=state)
            stage['u'] = torch.linalg.solve_triangular(system,wv,upper=False,unitriangular=True)
            stage['wy'] = torch.linalg.solve_triangular(system,bk*prefix.exp().unsqueeze(-1),upper=False,unitriangular=True)
            for name, value in stage.items():
                collected[name].append(value.clone())
        tail = kw * (prefix[..., -1:] - prefix).exp().unsqueeze(-1)
        state = state * prefix[..., -1:].exp().unsqueeze(-1) + tail.transpose(-1,-2) @ delta
    result = dict(o=output.transpose(1,2).contiguous(), final_state=state.contiguous(), final_A_state=a.contiguous())
    if retain_stages:
        result.update(q_norm=q_norm.contiguous(), k_read=k_read.contiguous(), k_write=k_write)
        result.update({name: torch.stack(parts,dim=1).contiguous() for name,parts in collected.items()})
    return result


def reference(inputs):
    validate_inputs(inputs)
    return block_solve(inputs)


def reference_stages(inputs):
    validate_inputs(inputs)
    return block_solve(inputs, retain_stages=True)


def validate_reference(inputs, outputs, case=None):
    b,t,hv,_ = inputs['v'].shape
    h = inputs['q'].shape[2]
    if set(outputs) != set(OUTPUTS):
        raise ValueError('all three named outputs are required')
    for name, shape in (('o',(b,t,hv,128)),('final_state',(b,hv,128,128)),('final_A_state',(b,h,128))):
        x = outputs[name]
        if x.shape != shape or x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError(f'invalid reference {name}: {tuple(x.shape)}/{x.dtype}')
        if x.norm() > 0:
            for wrong in (torch.zeros_like(x), -x, x*1.25):
                if ((wrong-x).norm()/x.norm()).item() <= 1e-4:
                    raise AssertionError(f'{name}: negative control was not rejected')
