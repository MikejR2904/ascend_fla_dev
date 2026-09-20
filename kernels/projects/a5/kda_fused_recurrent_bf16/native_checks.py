"""Additional actual-public-call checks, executed after the complete workload."""
import itertools

import torch


def per_head(got, refs, dtype, research):
    rows = []
    b, _, hv, _ = got[0].shape
    for bi, hi in itertools.product(range(b), range(hv)):
        current = got[0][bi,:,hi], got[1][bi,hi]
        for name, expected in refs.items():
            oref, sref = expected[0][bi,:,hi], expected[1][bi,hi]
            om, sm = research.metrics(current[0],oref), research.metrics(current[1],sref)
            floor = research.metrics(oref.bfloat16(),oref)['relative_l2']
            budget = min(.01,3*floor) if dtype==torch.bfloat16 else 1e-5
            passed = om['finite'] and sm['finite'] and om['relative_l2']<=budget and sm['relative_l2']<=1e-5
            rows.append(dict(batch=bi,head=hi,reference=name,o=om,final_state=sm,
                             output_floor=floor,output_budget=budget,passed=passed))
    return rows


def compare(got, refs, dtype, research):
    if dtype == torch.bfloat16:
        result = research.compare(got, refs)
    else:
        result = {}
        for name, expected in refs.items():
            om, sm = (research.metrics(a, b) for a, b in zip(got, expected))
            result[name] = dict(o=om, final_state=sm,
                                passed=all(m['finite'] and m['relative_l2'] <= 1e-5 for m in (om, sm)))
    assert all(r['passed'] for r in result.values()), result
    return result


def boundaries(api, research, bd, public, original_fp32, to_device, fla):
    def cpu(x):
        return x.detach().cpu().contiguous()

    def hashes(xs):
        return list(map(research.digest, xs))

    rows = []
    # Original decode acceptance dimensions, with unrounded FP32 inputs.
    for b, t, h, hv in ((1,1,1,1), (1,1,1,2), (1,4,1,1), (1,1,2,4),
                        (1,16,1,1), (1,1,16,32), (1,1,32,32), (1,8,32,32)):
        gen = torch.Generator().manual_seed(2026)
        def randn(*shape):
            return torch.randn(*shape, generator=gen)
        data = dict(q=torch.nn.functional.normalize(randn(b,t,h,128), dim=-1),
                    k=torch.nn.functional.normalize(randn(b,t,h,128), dim=-1),
                    v=randn(b,t,hv,128)*.04,
                    g=-torch.rand(b,t,hv,128,generator=gen)*.03,
                    beta=torch.rand(b,t,hv,generator=gen)*.45+.05,
                    initial_state=randn(b,hv,128,128)*.01,
                    scale=128**-.5, output_final_state=True)
        dev = to_device(data)
        got, ops = public(dev)
        result = compare(got, research.references(data, fla), torch.float32, research)
        exact = hashes(got) == hashes(original_fp32(dev))
        assert exact
        rows.append(dict(kind='original_unrounded_fp32', B=b,T=t,H=h,HV=hv,
                         comparison=result, original_bitwise=exact, host_operations=ops,
                         output_sha256=hashes(got)))

    for dtype, state in itertools.product((torch.bfloat16, torch.float32), (False, True)):
        p = dict(B=2,T=16,H=3,G=4,state=state)
        data = research.make_inputs(p, seed=6102)
        for n in ('q','k','v'):
            data[n] = data[n].to(dtype)
        dev = to_device(data)
        full, _ = public(dev, return_cpu=False)
        without_state, _ = public(dict(dev, output_final_state=False), return_cpu=False)
        assert without_state[1] is None
        assert hashes([cpu(without_state[0])]) == hashes([cpu(full[0])])
        rows.append(dict(kind='output_final_state_false', dtype=str(dtype), initial_state=state,
                         exact_output=True, state_returned=False))
        expected = hashes(tuple(cpu(x) for x in full))
        for width in (1,2,4,8):
            current = dev['initial_state']
            outputs = []
            for begin in range(0, 16, width):
                step = {n: dev[n][:,begin:begin+width].contiguous() for n in ('q','k','v','g','beta')}
                step.update(scale=dev['scale'], initial_state=current, output_final_state=True)
                got, _ = public(step, return_cpu=False)
                outputs.append(got[0]); current = got[1]
            combined = torch.cat(outputs, dim=1), current
            actual = hashes(tuple(cpu(x) for x in combined))
            assert actual == expected, dict(dtype=str(dtype),state=state,width=width,expected=expected,actual=actual)
            rows.append(dict(kind='decode_chaining', dtype=str(dtype), initial_state=state,
                             tokens_per_call=width, bitwise=True, output_sha256=actual))
        unchanged = {n:research.digest(cpu(dev[n]))==research.digest(x) for n,x in data.items()
                     if isinstance(x,torch.Tensor)}
        assert all(unchanged.values())

    # Actual NPU tensors exercise metadata rejection without executing invalid kernels.
    data = to_device(research.make_inputs(dict(B=2,T=2,H=2,G=4,state=True)))
    bad = []
    for name,dtype in (('q',torch.float32),('k',torch.float32),('v',torch.float16),
                        ('v',torch.float64),('g',torch.bfloat16),('beta',torch.bfloat16),
                        ('initial_state',torch.bfloat16)):
        # Unsupported-dtype tensors are constructed in the verifier, not the operator.
        value = data[name].cpu().to(dtype)
        # The NPU cannot represent FP64; reject that CPU tensor at the same ABI
        # boundary instead of relying on a device transfer's implicit cast.
        if dtype != torch.float64:
            value = value.npu()
        bad.append((f'{name}_{dtype}',dict(data,**{name:value})))
    for name in ('q','k','v','g','beta','initial_state'):
        value = torch.stack([data[name],data[name]],dim=-1)[...,0]
        bad.append((name+'_noncontiguous',dict(data,**{name:value})))
    for name,invalid in bad:
        original = api._compiled
        def fail_compile(*args):
            raise AssertionError('invalid inputs reached custom compilation/launch')
        api._compiled = fail_compile
        try:
            try:
                api.fused_recurrent_kda(**invalid, block_dim=bd)
            except ValueError as error:
                rows.append(dict(kind='metadata_rejection',case=name,error=str(error),no_launch=True))
            else:
                raise AssertionError('invalid case accepted: '+name)
        finally:
            api._compiled = original

    # Independent raw-input formulas; no production preprocessing as CPU oracle.
    for dtype, flags in itertools.product((torch.bfloat16, torch.float32), itertools.product((False,True), repeat=3)):
        gen = torch.Generator().manual_seed(6244)
        q, k = [torch.randn(2,2,2,128,generator=gen) for _ in range(2)]
        g = torch.randn(2,2,8,128,generator=gen)*.1
        beta = torch.randn(2,2,8,generator=gen)
        a, bias = torch.linspace(-.2,.1,8), torch.linspace(-3.,-2.,8*128)
        qp = (q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()).to(dtype)
        kp = (k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()).to(dtype)
        gp = -a.exp()[None,None,:,None]*torch.nn.functional.softplus(g+bias.view(8,128))
        bp = beta.sigmoid()
        v = (torch.randn(2,2,8,128,generator=gen)*.2).to(dtype)
        state = torch.randn(2,8,128,128,generator=gen)*.05
        expected = dict(q=qp,k=kp,v=v,g=gp,beta=bp,initial_state=state,scale=128**-.5,output_final_state=True)
        raw = dict(expected, q=q if flags[0] else qp, k=k if flags[0] else kp,
                   g=g if flags[1] else gp, beta=beta if flags[2] else bp,
                   A_log=a, dt_bias=bias, use_qk_l2norm_in_kernel=flags[0],
                   use_gate_in_kernel=flags[1], use_beta_sigmoid_in_kernel=flags[2])
        dev = to_device(raw)
        got, ops = public(dev, legacy=any(flags))
        result = compare(got, research.references(expected, fla), dtype, research)
        unchanged = {n:research.digest(cpu(dev[n]))==research.digest(x) for n,x in raw.items()
                     if isinstance(x,torch.Tensor)}
        assert all(unchanged.values())
        exact = None
        if dtype == torch.float32:
            exact = hashes(got)==hashes(original_fp32(dev))
            assert exact
        rows.append(dict(kind='raw_flags_legacy' if any(flags) else 'prepared_default',
                         dtype=str(dtype), flags=flags, comparison=result, original_fp32_bitwise=exact,
                         input_unchanged=unchanged,host_operations=ops,host_categories=public.last_categories,
                         output_sha256=hashes(got)))
    return dict(passed=True, block_dim=bd, cases=rows)
