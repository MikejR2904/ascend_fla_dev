"""Leaf, gradient-view and public-gate checks after the full hardware workload."""
from __future__ import annotations

import json
import math
import sys

import torch

import native_support as check

# The readonly reference modules add their own unit directory to sys.path.
# Load this unit by identity instead of accidentally importing their unit.py.
unit = check.load_file('_fmt02_layout_leaf_unit', check.ROOT / 'unit.py')


def original_backward_inputs(backward):
    """Exact five original contract distributions, with no kernel or model import."""
    sys.path.insert(0, str(backward._bwd_kernels_root()))
    from ref.inputs import make_inputs
    cases = [('single_chunk', 1, 1, 1, 1, 1.), ('multi_chunk', 1, 1, 1, 2, 1.),
             ('grouped_heads', 1, 1, 2, 2, 1.), ('gentle_decay', 1, 1, 1, 2, .03),
             ('grouped_idle_cores', 1, 1, 2, 1, 1.)]
    for name, b, h, hv, c, factor in cases:
        src = vars(make_inputs(B=b, H=h, HV=hv, C=c, K=128, V=128, chunk_size=64, seed=2026))
        x = {n: src[n] for n in ('q', 'k', 'v', 'do', 'dht')}
        x['g'] = (src['g'].float()*factor).bfloat16().float() if factor != 1 else src['g'].float()
        x['beta'] = src['beta'].float()
        x['h0'] = src['initial_state'].float()
        yield dict(id='original_'+name, B=b, H=h, HV=hv, C=c, state=True), x


def stable_contracts(chunk, before, real, reference, bd):
    """Run declared source-generated inputs; public gates stay enabled."""
    root = check.ROOT.parent / 'kda_fwd_stable'
    sys.path.insert(0, str(root))
    source_unit = check.load_file('_fmt02_stable_source_unit', root/'unit.py')
    rows = []
    for family in ('kda_fwd_stable', 'kda_bwd_stable'):
        contract = json.loads((check.ROOT.parent / family / 'contract.json').read_text())
        for case in contract['cases']:
            inputs = source_unit.make_inputs(case)
            x = {**inputs, 'g': inputs['g_raw'], 'h0': inputs['initial_state']}
            span = chunk._gate_span(x['g'], case['parameters']['C'], on_cpu=True)
            for path, limit in (('forward',155), ('backward',105)):
                data = {n: x[n].npu() for n in ('q','k','v','g','beta')}
                data['initial_state'] = x['h0'].npu()
                outputs = []
                for module in (chunk,before):
                    function = module.chunk_kda_fwd if path=='forward' else module.chunk_kda_fwd_with_caches
                    try:
                        with check.instrument(audit=module is chunk) as (audit, launches):
                            got = function(**data, block_dim=bd, layout_device='npu',
                                           **({'output_final_state':True} if path=='forward' else {}))
                        torch.npu.synchronize()
                        assert span <= limit
                        if module is chunk: assert not audit.unexpected(), audit.unexpected()
                        outputs.append(check.cpu(got))
                    except ValueError as error:
                        assert span > limit and '门控跨度' in str(error) and not launches
                        outputs.append(str(error))
                row = dict(family=family, id=case['id'], path=path, span=span, limit=limit,
                           source_case=case, rejected=span>limit)
                if span>limit:
                    row['passed'] = outputs[0]==outputs[1]
                else:
                    left,right=[dict(o=y[0],final_state=y[1],**(y[2] if len(y)==3 else {})) for y in outputs]
                    row['comparison']=check.exact(left,right)
                    oracle={n:x[n] for n in ('q','k','v','g','beta','h0')}
                    refs=dict(independent=reference.independent_reference(oracle),fla=reference.fla_reference(oracle))
                    row['cpu_fp32']={label:{n:reference.metrics(left[n],ref[n]) for n in ('o','final_state')}
                                     for label,ref in refs.items()}
                    row['passed']=(all(v['passed'] for v in row['comparison'].values())
                        and all(m['passed'] for ref in row['cpu_fp32'].values() for m in ref.values()))
                rows.append(row)
                assert row['passed'],row
    return dict(passed=True,cases=rows)


def leaf(runtime, bd):
    rows = []
    cases = json.loads((check.ROOT / 'contract.json').read_text())['cases']
    for case in cases:
        x = unit.make_inputs(case)
        expected = unit.reference(x)['destination']
        key, shape, scalars = unit.launch_description(x)
        source = x['source']
        if source.storage_offset():
            # Transfer the complete storage first, retaining a real offset on NPU.
            base = source._base.npu()
            dev = base.as_strided(source.shape, source.stride(), source.storage_offset())
        else:
            dev = source.npu()
        with check.instrument() as (audit, launches):
            if key.startswith('zero'):
                got = runtime.zeros(shape, unit.DTYPES[case['parameters']['destination']], 'npu', block_dim=bd)
            else:
                got = runtime.move(dev, shape, tuple(scalars[f'S{i}'] for i in range(5)),
                                   dtype=expected.dtype, block_dim=bd,
                                   multiply=bool(scalars.get('multiply', False)), factor=scalars.get('factor', 1))
        torch.npu.synchronize()
        comparison = check.exact({'destination': got.view_as(expected)}, {'destination': expected})
        unchanged = check.digest(dev) == check.digest(source)
        row = dict(id=case['id'], comparison=comparison, input_unchanged=unchanged,
                   host_operations=audit.report(), launches=launches,
                   passed=comparison['destination']['passed'] and unchanged and not audit.unexpected())
        rows.append(row)
        assert row['passed'], row
    # Direct leaf launches retain canaries on both sides of the actual output.
    # FP32 values include signed zeros, RNE ties, and finite BF16 subnormals.
    values = torch.tensor([0., -0., 1+2**-8, 1+3*2**-8, -(1+2**-8),
                           2**-133, -2**-133, 2**-126], dtype=torch.float32).repeat(16)
    for src in (torch.bfloat16, torch.float32):
        for dst in (torch.bfloat16, torch.float32):
            source = values.to(src).npu().view(1, -1)
            expected = values.to(src).to(dst).view(1, -1)
            backing = torch.full((1, 256), 7.5, dtype=dst).npu()
            actual = backing[:, 64:192]
            key = ('bf16' if src == torch.bfloat16 else 'f32') + '_' + ('bf16' if dst == torch.bfloat16 else 'f32')
            scalars = dict(Storage=128, N=128, D1=1, D2=1, D3=1, D4=128,
                           S0=0, S1=0, S2=0, S3=0, S4=1, **runtime.tile_scalars((1,1,1,1,128)))
            if key == 'f32_bf16': scalars.update(multiply=0, factor=1.)
            with check.instrument(audit=False):
                runtime.prepare('a5', bd)[key]({'source': source}, scalars, {'destination': actual})
            torch.npu.synchronize()
            host = check.cpu(backing)
            comparison = check.exact({'destination': host[:, 64:192]}, {'destination': expected})
            guarded = bool((host[:, :64] == 7.5).all() and (host[:, 192:] == 7.5).all())
            row = dict(id='canary_special_' + key, comparison=comparison, guards_intact=guarded,
                       passed=comparison['destination']['passed'] and guarded)
            rows.append(row)
            assert row['passed'], row
    return dict(passed=True, cases=rows)


def autograd_views(candidate, before, real, bd):
    rows = []
    x = real.make_inputs(B=2, H=2, HV=4, C=3, span=46, want_grads=True)
    names = ('q', 'k', 'v', 'g', 'beta', 'h0')
    for mode in ('contiguous', 'transposed', 'broadcast', 'missing_dht', 'absent_h0'):
        def run(module, audit_enabled):
            leaves = {n: x[n].npu().requires_grad_(True) for n in names}
            fwd_audit, bwd_audit = check.Audit(), check.Audit()
            original_backward = module._ChunkKDA.backward
            def audited_backward(ctx, do, dht):
                with bwd_audit:
                    return original_backward(ctx, do, dht)
            if audit_enabled:
                module._ChunkKDA.backward = staticmethod(audited_backward)
            try:
                with check.instrument(audit=False) as (_, launches):
                    with fwd_audit:
                        o, ht = module.chunk_kda(*(leaves[n] for n in ('q', 'k', 'v', 'g', 'beta')),
                            initial_state=None if mode == 'absent_h0' else leaves['h0'],
                            output_final_state=True, block_dim=bd, layout_device='npu')
                    if mode == 'broadcast':
                        do = torch.tensor(.25, dtype=o.dtype).npu().expand_as(o)
                        dht = torch.tensor(-.125, dtype=ht.dtype).npu().expand_as(ht)
                    elif mode == 'transposed':
                        do = x['do'].transpose(1, 2).contiguous().npu().transpose(1, 2)
                        dht = x['dht'].float().transpose(-1, -2).contiguous().npu().transpose(-1, -2)
                    else:
                        do, dht = x['do'].npu(), x['dht'].float().npu()
                    wanted = names if mode != 'absent_h0' else names[:-1]
                    if mode == 'missing_dht':
                        gradients = torch.autograd.grad(o, [leaves[n] for n in wanted], grad_outputs=do)
                    else:
                        gradients = torch.autograd.grad((o, ht), [leaves[n] for n in wanted], grad_outputs=(do, dht))
                torch.npu.synchronize()
            finally:
                module._ChunkKDA.backward = staticmethod(original_backward)
            got = check.cpu(dict(o=o, final_state=ht, **dict(zip(wanted, gradients))))
            unchanged = all(check.digest(leaves[n]) == check.digest(x[n]) for n in names)
            return got, dict(forward=fwd_audit.report(), backward=bwd_audit.report(),
                             unexpected=fwd_audit.unexpected()+bwd_audit.unexpected(),
                             launches=launches, input_unchanged=unchanged)
        got, audit = run(candidate, True)
        old, _ = run(before, False)
        result = check.exact(got, old)
        row = dict(id=mode, comparison=result, audit=audit,
                   passed=all(r['passed'] for r in result.values()) and not audit['unexpected'] and audit['input_unchanged'])
        rows.append(row)
        assert row['passed'], row
    return dict(passed=True, cases=rows)


def gates(chunk, before, real, bd):
    rows = []
    x = real.make_inputs(B=1, H=1, HV=2, C=2, span=46, want_grads=True)
    for path, limit in (('forward', 155), ('backward', 105)):
        for span in (0, limit, limit+1):
            g = torch.empty_like(x['g']).view(1, 2, 64, 2, 128)
            # Exactly representable increments: 0 + 62*(-1) + (62-span).
            g.fill_(-1 if span else 0)
            g[:, :, 0] = 0
            g[:, :, -1] = 62-span if span else 0
            x['g'] = g.view_as(x['g'])
            assert chunk._gate_span(x['g'], 2, on_cpu=True) == span
            data = {n: x[n].npu() for n in ('q', 'k', 'v', 'g', 'beta')}
            data['initial_state'] = x['h0'].npu()
            results = []
            for module in (chunk, before):
                function = module.chunk_kda_fwd if path == 'forward' else module.chunk_kda_fwd_with_caches
                try:
                    with check.instrument(audit=module is chunk) as (audit, launches):
                        result = function(**data, block_dim=bd, layout_device='npu',
                                          **({'output_final_state': True} if path == 'forward' else {}))
                    torch.npu.synchronize()
                    assert span <= limit
                    if module is chunk: assert not audit.unexpected(), audit.unexpected()
                    results.append(check.cpu(result))
                except ValueError as error:
                    assert span > limit and '门控跨度' in str(error), str(error)
                    assert not launches, 'gate must reject before first kernel'
                    results.append(str(error))
            if span > limit:
                same = results[0] == results[1]
                row = dict(path=path, span=span, limit=limit, rejected=True, same_error=same, passed=same)
            else:
                left = dict(o=results[0][0], final_state=results[0][1])
                right = dict(o=results[1][0], final_state=results[1][1])
                if path == 'backward':
                    left.update(results[0][2]); right.update(results[1][2])
                comparison = check.exact(left, right)
                row = dict(path=path, span=span, limit=limit, comparison=comparison,
                           passed=all(r['passed'] for r in comparison.values())
                           and all(bool(t.isfinite().all()) for t in left.values()))
            rows.append(row)
            assert row['passed'], row
    return dict(passed=True, cases=rows)


def decode_audit(api, real, bd):
    rows=[]
    for dtype in (torch.bfloat16,torch.float32):
        x=real.make_inputs(B=1,H=2,HV=4,C=1,span=46)
        x={n:(t[:,:16].contiguous() if n!='h0' else t) for n,t in x.items()}
        for n in ('q','k','v'):x[n]=x[n].to(dtype)
        reference=real.ref_fwd(x)
        data={n:x[n].npu() for n in ('q','k','v','g','beta')}
        data['initial_state']=x['h0'].npu()
        if bd not in api.SUPPORTED_BLOCK_DIM:
            # Chunk admits bd3; the unchanged decode contract deliberately does not.
            with check.instrument() as (audit,launches):
                try:
                    api.fused_recurrent_kda(**data,output_final_state=True,block_dim=bd)
                except ValueError as error:
                    assert 'block_dim' in str(error) and not launches
                else:
                    raise AssertionError('decode accepted an undeclared block dimension')
            assert not audit.unexpected(),audit.unexpected()
            rows.append(dict(dtype=str(dtype),block_dim=bd,rejected=True,
                             reason='unchanged decode block-dimension domain',
                             host_operations=audit.report(),launches=launches,passed=True))
            continue
        with check.instrument() as (audit,launches):
            got=api.fused_recurrent_kda(**data,output_final_state=True,block_dim=bd)
        torch.npu.synchronize()
        measures={n:check.metrics(a,e,.005 if dtype==torch.bfloat16 and n=='o' else 1e-5)
                  for n,a,e in zip(('o','final_state'),got,reference)}
        unchanged=all(check.digest(data[n])==check.digest(x[n]) for n in ('q','k','v','g','beta'))
        row=dict(dtype=str(dtype),measurements=measures,host_operations=audit.report(),
                 launches=launches,input_unchanged=unchanged,
                 passed=all(m['passed'] for m in measures.values()) and not audit.unexpected() and unchanged)
        rows.append(row)
        assert row['passed'],row
    return dict(passed=True,cases=rows)
