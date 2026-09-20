"""Three-layer prefill/decode acceptance, fixed by PM comment 5746672559."""
import itertools

import torch


def verify(chunk, research, bd, public, to_device, fla, write, output_dir):
    def cpu(x):
        return x.detach().cpu().contiguous()

    def digest_pair(pair):
        return [research.digest(cpu(x)) for x in pair]

    chunk_bd = min(bd, 4)
    original_chain = chunk._compiled_chain
    stages = original_chain('a5', chunk_bd, 'stable')

    def poisoned(op):
        def launch(inputs, scalars, outputs):
            for x in outputs.values():
                x.fill_(float('nan'))
            return op(inputs, scalars, outputs)
        return launch

    wrapped = {name: poisoned(op) for name, op in stages.items()}

    def prefill(data):
        chunk._compiled_chain = lambda *args: wrapped
        try:
            got = chunk.chunk_kda_fwd(**data, block_dim=chunk_bd, layout_device='npu', impl='stable')
        finally:
            chunk._compiled_chain = original_chain
        torch.npu.synchronize()
        assert all(torch.isfinite(cpu(x)).all() for x in got)
        return got

    def slice_inputs(data, start, stop):
        return {n:data[n][:,start:stop].contiguous() for n in ('q','k','v','g','beta')}

    cases = [dict(B=1,H=2,G=g,prefix=prefix,state=state)
             for prefix,g,state in itertools.product((64,128),(1,2,4,8),(False,True))]
    cases += [dict(B=2,H=4,G=8,prefix=128,state=True),
              dict(B=1,H=32,G=1,prefix=128,state=True)]
    cases += [dict(B=1,H=2,G=4,prefix=128,state=True,span=span) for span in (0.,105.,155.)]
    records = []
    for index,p in enumerate(cases):
        name = f'prefill_{index:02d}'
        print('PREFILL_START', name, p, flush=True)
        data = research.make_inputs(dict(p,T=p['prefix']+64),seed=6506)
        if 'span' in p:
            data['g'].fill_(-p['span']/64)
        refs = research.references(data,fla)
        prefix_cpu = dict(slice_inputs(data,0,p['prefix']),initial_state=data['initial_state'],
                          scale=data['scale'],output_final_state=True)
        prefix_refs = research.references(prefix_cpu,fla)
        dev = to_device(data)
        one = prefill(dev)
        one_cpu = tuple(cpu(x) for x in one)
        prefix_dev = dict(slice_inputs(dev,0,p['prefix']),initial_state=dev['initial_state'],
                          scale=dev['scale'],output_final_state=True)
        prefix = prefill(prefix_dev)
        prefix_state_hash = research.digest(cpu(prefix[1]))
        first_layer = []
        segment_hashes = {}

        def decode_suffix(width):
            state = prefix[1]
            outs = []
            segments = []
            for begin in range(p['prefix'], p['prefix']+64, width):
                step = dict(slice_inputs(dev,begin,begin+width),initial_state=state,
                            scale=data['scale'],output_final_state=True)
                cpu_step = dict(slice_inputs(data,begin,begin+width),initial_state=cpu(state),
                                scale=data['scale'],output_final_state=True)
                expected = research.references(cpu_step,fla)
                state_before = research.digest(cpu_step['initial_state'])
                got, ops = public(step,return_cpu=False)
                result = research.compare(tuple(cpu(x) for x in got),expected)
                unchanged = research.digest(cpu(state)) == state_before
                row = dict(begin=begin,tokens=width,comparison=result,input_state_unchanged=unchanged,
                           host_operations=ops,passed=unchanged and all(r['passed'] for r in result.values()))
                first_layer.append(row)
                write(name+'-decode-steps',dict(complete=False,cases=first_layer))
                if not row['passed']:
                    torch.save(dict(input=cpu_step,actual=tuple(cpu(x) for x in got),reference=expected),
                               output_dir/(name+f'-decode-{begin}-{width}-failure.pt'))
                    raise AssertionError(row)
                outs.append(got[0]);state=got[1]
                if (begin+width-p['prefix']) % 16 == 0:
                    group_output = got[0] if width == 16 else torch.cat(outs[-16:],dim=1)
                    segments.append(digest_pair((group_output,state)))
            segment_hashes[width] = segments
            return torch.cat(outs,dim=1),state

        sixteen = decode_suffix(16)
        single = decode_suffix(1)
        exact = digest_pair(sixteen) == digest_pair(single) and segment_hashes[16] == segment_hashes[1]
        assert research.digest(cpu(prefix[1])) == prefix_state_hash
        chain = (torch.cat((prefix[0],sixteen[0]),dim=1),sixteen[1])
        chain_cpu = tuple(cpu(x) for x in chain)
        prefix_actual = tuple(cpu(x) for x in prefix)
        comparison = {}
        for ref_name,expected in refs.items():
            eo = research.metrics(one_cpu[0],expected[0])
            es = research.metrics(one_cpu[1],expected[1])
            obudget,sbudget = min(.01,3*eo['relative_l2']),min(.01,3*es['relative_l2'])
            metrics = dict(o=research.metrics(chain_cpu[0],expected[0]),
                           final_state=research.metrics(chain_cpu[1],expected[1]),
                           prefix_o=research.metrics(chain_cpu[0][:,:p['prefix']],expected[0][:,:p['prefix']]),
                           suffix_o=research.metrics(chain_cpu[0][:,p['prefix']:],expected[0][:,p['prefix']:]),
                           prefix_state=research.metrics(prefix_actual[1],prefix_refs[ref_name][1]))
            passed = all(m['finite'] and m['relative_l2'] <= (
                sbudget if name in ('final_state','prefix_state') else obudget) for name,m in metrics.items())
            comparison[ref_name] = dict(oneshot_o=eo,oneshot_state=es,o_budget=obudget,state_budget=sbudget,
                                        metrics=metrics,passed=passed)
        unchanged = {n:research.digest(cpu(dev[n]))==research.digest(x) for n,x in data.items()
                     if isinstance(x,torch.Tensor)}
        row = dict(case=p,block_dim=bd,chunk_block_dim=chunk_bd,decode_steps=first_layer,
                   sixteen_vs_single_bitwise=exact,comparison=comparison,input_unchanged=unchanged,
                   segment_hashes=segment_hashes,
                   chain_vs_oneshot=dict(o=research.metrics(chain_cpu[0],one_cpu[0]),
                                         final_state=research.metrics(chain_cpu[1],one_cpu[1])),
                   output_sha256=dict(zip(('o','final_state'),map(research.digest,chain_cpu))),
                   oneshot_sha256=dict(zip(('o','final_state'),map(research.digest,one_cpu))))
        row['passed'] = exact and all(unchanged.values()) and all(r['passed'] for r in comparison.values())
        write(name,row)
        records.append(row)
        if not row['passed']:
            torch.save(dict(inputs=data,actual=chain_cpu,oneshot=one_cpu,reference=refs),
                       output_dir/(name+'-failure.pt'))
            raise AssertionError(row)
        print('PREFILL_PASS',name,json_summary(comparison),flush=True)
    return dict(passed=True,block_dim=bd,cases=records,
                ruling='PM5746672559: strict local decode, bitwise segmentation, global min(.01,3E_oneshot)',
                prefill_scope='Existing stable chunk public path; its host layout work is outside BF-06 decode audit')


def json_summary(comparison):
    import json
    return json.dumps({n:dict(o=r['metrics']['o']['relative_l2'],state=r['metrics']['final_state']['relative_l2'],
                             o_budget=r['o_budget'],state_budget=r['state_budget']) for n,r in comparison.items()})
