"""Native public grid; each process owns one chunk and one decode block_dim."""
import argparse
import hashlib
import traceback
from pathlib import Path
import torch
from torch.utils._python_dispatch import _disable_current_modes
from ascend_fla.ops.kda import chunk,autograd,fused_recurrent
import native_context
import native_grid
import unit


def decode_sequence(api,data,block_dim):
    # Public decode accepts at most16 tokens. This harness streams a complete
    # C*64-token population through that supported ABI and carries FP32 state.
    state=data['initial_state'];outputs=[]
    for start in range(0,data['q'].shape[1],16):
        part=dict(data,initial_state=state)
        for name in ('q','k','v','g','beta'):part[name]=data[name][:,start:start+16]
        o,state=api.fused_recurrent_kda(**part,block_dim=block_dim)
        outputs.append(o)
    # Concatenation belongs only to verification, outside the public op audit.
    with _disable_current_modes():
        return torch.cat(outputs,dim=1),state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',choices=('ordinary','boundary'),default='ordinary')
    parser.add_argument('--route',choices=('chunk','decode'),required=True)
    parser.add_argument('--block-dim',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id')
    parser.add_argument('--chunk-count',type=int,choices=(1,2,3))
    args=parser.parse_args()
    bd=args.block_dim if args.route=='chunk' else min(args.block_dim,4)
    dbd=(4 if bd==3 else bd) if args.route=='chunk' else args.block_dim
    context=native_context.Context(args.output,bd,dbd,__file__)
    check=context.check;old=context.old
    decode_checks=unit.module('verify_native').load('_bf07_decode_checks',unit.ROOT.parent/'kda_fused_recurrent_bf16/research.py')
    generator=native_grid.cases if args.population=='ordinary' else native_grid.boundary_cases
    cases=[p for p in generator(args.route) if not args.chunk_count or p['C']==args.chunk_count]
    if args.case_id:cases=[p for p in cases if p['id']==args.case_id]
    assert cases, 'selected population is empty'
    rows=[]
    original_forward_metric=context.reference.metrics
    def forward_metric(actual,expected):
        result=original_forward_metric(actual,expected)
        a,e=actual.detach().cpu().double(),expected.detach().cpu().double()
        norm=float(e.norm());residual=float((a-e).norm())
        result['relative_l2_fp64_accumulation']=residual/norm if norm else (0. if residual==0 else float('inf'))
        result['passed']=result['passed'] and result['relative_l2_fp64_accumulation']<=.05
        return result
    if args.population=='boundary':
        context.reference.metrics=forward_metric
        original_decode_metric=decode_checks.metrics
        def decode_metric(actual,expected):
            result=original_decode_metric(actual,expected)
            a,e=actual.detach().cpu().double(),expected.detach().cpu().double()
            norm=float(e.norm());residual=float((a-e).norm())
            result['relative_l2']=residual/norm if norm else (0. if residual==0 else float('inf'))
            return result
        decode_checks.metrics=decode_metric
    for case in cases:
        label=case['id'];print('CASE_START',args.route,label,flush=True)
        x=(native_grid.inputs if args.population=='ordinary' else native_grid.boundary_inputs)(case);flags=case['flags'];dtype=x['v'].dtype
        dev={n:t.npu() for n,t in x.items()}
        prep_kwargs=dict(flags,A_log=dev['A_log'],dt_bias=dev['dt_bias'],qk_dtype=dtype)
        data=dict(q=dev['q'],k=dev['k'],v=dev['v'],g=dev['g'],beta=dev['beta'],
                  initial_state=dev['h0'],A_log=dev['A_log'],dt_bias=dev['dt_bias'],output_final_state=True,**flags)
        try:
            with torch.no_grad(),check.instrument() as (audit,launches):
                prepared=chunk._prepare_kernel_inputs(*(dev[n] for n in ('q','k','g','beta')),
                    **prep_kwargs,namespace=args.route,block_dim=bd if args.route=='chunk' else dbd)
                if args.route=='chunk':
                    public=autograd.chunk_kda(**data,block_dim=bd,layout_device='npu')
                    q,k,g,beta=prepared
                    o,state,caches=chunk.chunk_kda_fwd_with_caches(q,k,dev['v'],g,beta,initial_state=dev['h0'],block_dim=bd,layout_device='npu')
                    assert set(caches)==set(chunk.BWD_CACHE_NAMES) and len(caches)==9
                    candidate=dict(o=o,final_state=state,**{'cache_'+n:t for n,t in caches.items()})
                    plain_cached_exact=None
                else:
                    o,state=decode_sequence(fused_recurrent,data,dbd)
                    candidate=dict(o=o,final_state=state);plain_cached_exact=None
            torch.npu.synchronize()
            if args.route=='chunk':plain_cached_exact=[check.digest(a)==check.digest(b) for a,b in zip(public,(o,state))]
            context.write('audits/'+label,dict(operations=audit.report(),unexpected=audit.unexpected(),launches=launches))
            got=check.cpu(candidate);prep_cpu=check.cpu(prepared)
            disabled={n:prepared[i] is dev[n] for i,n in enumerate(('q','k','g','beta')) if not flags[native_grid.FLAG_NAMES[0 if i<2 else i-1]]}
            old_cpu=old._prepare_inputs(*(x[n] for n in ('q','k','g','beta')),
                **flags,A_log=x['A_log'],dt_bias=x['dt_bias'],qk_dtype=dtype)
            prec={}
            for i,name in enumerate(('q','k','g','beta')):
                enabled=flags[native_grid.FLAG_NAMES[0 if i<2 else i-1]]
                if not enabled:continue
                kind='norm' if i<2 else name if name=='beta' else 'gate'
                values={'source':x[name]}
                types=case['types'][name]
                if kind=='norm':types+='_'+case['types']['v']
                if kind=='gate':
                    values.update(alog=x['A_log'],bias=x['dt_bias'])
                    types+='_'+case['types']['A_log']+'_'+case['types']['dt_bias']
                inp=dict(values=values,parameters=dict(kind=kind,types=types))
                if case.get('boundary')=='beta_saturation' and kind=='beta':
                    # Frozen calibration excludes saturation endpoints from the
                    # ordinary floor. Retain their full FP64 discrepancy, require
                    # actual CPU FP32 endpoint bytes, and keep ordinary limits on
                    # the remaining nonsaturated members of this same population.
                    actual=prep_cpu[i].reshape(1,-1);high=unit.reference(inp)['destination']
                    expected_cpu=unit.host_reference(inp)['destination']
                    endpoint=values['source'].reshape(1,-1).abs()>20
                    ordinary_input=dict(values={'source':values['source'].reshape(1,-1)[~endpoint].reshape(1,-1)},parameters=inp['parameters'])
                    ordinary=unit.compare(ordinary_input,{'destination':actual[~endpoint].reshape(1,-1)},unit.reference(ordinary_input),context.budgets)
                    exact=check.digest(actual[endpoint])==check.digest(expected_cpu[endpoint])
                    prec[name]=dict(ordinary=ordinary,endpoint_count=int(endpoint.sum()),endpoint_cpu_exact=exact,
                        full_to_fp64=context.precision.metrics(actual,high),passed=ordinary['passed'] and exact)
                else:
                    prec[name]=unit.compare(inp,{'destination':prep_cpu[i].reshape(1,-1)},unit.reference(inp),context.budgets)
            with torch.no_grad():
                old_prep=old._prepare_inputs(*(dev[n] for n in ('q','k','g','beta')),**prep_kwargs)
                if args.route=='chunk':
                    old_o,old_state,old_caches=old.chunk_kda_fwd_with_caches(old_prep[0],old_prep[1],dev['v'],old_prep[2],old_prep[3],initial_state=dev['h0'],block_dim=bd,layout_device='npu')
                    before=check.cpu(dict(o=old_o,final_state=old_state,**{'cache_'+n:t for n,t in old_caches.items()}))
                else:
                    before=dict(zip(('o','final_state'),check.cpu(decode_sequence(context.old_decode,data,dbd))))
            all_flags_disabled=not any(flags.values())
            exact_disabled=check.exact(got,before) if all_flags_disabled else None
            oracle_x=dict(zip(('q','k','g','beta'),old_cpu));oracle_x.update(v=x['v'],h0=x['h0'])
            refs=dict(independent=context.reference.independent_reference(oracle_x),fla=context.reference.fla_reference(oracle_x))
            comparisons={}
            for name,ref in refs.items():
                if args.route=='chunk':
                    measures={n:context.reference.metrics(got[n],ref[n]) for n in ('o','final_state')}
                    measures['per_head_chunk']=[dict(chunk=c,head=h,**context.reference.metrics(got['o'][:,c*64:(c+1)*64,h],ref['o'][:,c*64:(c+1)*64,h])) for c in range(case['C']) for h in range(case['HV'])]
                    prior=torch.cat((x['h0'][:,None],ref['chunk_states'][:,:-1]),dim=1)
                    measures['cached_states']=[dict(chunk=c,head=h,**check.metrics(got['cache_h'][:,c,h],prior[:,c,h],.05)) for c in range(case['C']) for h in range(case['HV'])]
                else:
                    measures={}
                    for n in ('o','final_state'):
                        metric=decode_checks.metrics(got[n],ref[n])
                        floor=decode_checks.metrics(ref[n].bfloat16().float(),ref[n])['relative_l2']
                        limit=min(.01,3*floor) if dtype==torch.bfloat16 and n=='o' else 1e-5
                        metric.update(budget=limit,bf16_floor=floor,passed=metric['finite'] and metric['relative_l2']<=limit)
                        measures[n]=metric
                    measures['per_head_chunk']=[]
                    for c in range(case['C']):
                        for h in range(case['HV']):
                            a,e=got['o'][:,c*64:(c+1)*64,h],ref['o'][:,c*64:(c+1)*64,h]
                            m=decode_checks.metrics(a,e);floor=decode_checks.metrics(e.bfloat16().float(),e)['relative_l2']
                            limit=min(.01,3*floor) if dtype==torch.bfloat16 else 1e-5
                            measures['per_head_chunk'].append(dict(chunk=c,head=h,**m,budget=limit,passed=m['finite'] and m['relative_l2']<=limit))
                comparisons[name]=measures
            row=dict(case=case,route=args.route,population=args.population,block_dim=args.block_dim,
                     decode_test_schedule='16-token public calls with carried FP32 state' if args.route=='decode' else None,
                     input_sha256={n:check.digest(t) for n,t in x.items()},
                     input_unchanged={n:check.digest(dev[n])==check.digest(t) for n,t in x.items()},
                     output_sha256={n:check.digest(t) for n,t in got.items()},
                     prep_sha256={n:check.digest(t) for n,t in zip(('q','k','g','beta'),prep_cpu)},
                     finite={n:bool(t.isfinite().all()) for n,t in got.items()},
                     disabled_same_object=disabled,plain_cached_exact=plain_cached_exact,
                     all_flags_disabled_exact=exact_disabled,preparation=prec,
                     predecessor_npu_differences={n:check.metrics(t,before[n],.05) for n,t in got.items()},
                     cpu_fp32_references=comparisons,passed=False)
            row['passed']=(all(row['input_unchanged'].values()) and all(row['finite'].values()) and all(disabled.values())
                and (plain_cached_exact is None or all(plain_cached_exact)) and not audit.unexpected()
                and all(p['passed'] for p in prec.values())
                and (exact_disabled is None or all(p['passed'] for p in exact_disabled.values()))
                and all(m[n]['passed'] for m in comparisons.values() for n in ('o','final_state'))
                and all(p['passed'] for m in comparisons.values() for p in m['per_head_chunk'])
                and all(p['passed'] for m in comparisons.values() for p in m.get('cached_states',[])))
            if not row['passed'] and args.route=='decode':
                oracle_native=dict(zip(('q','k','g','beta'),prep_cpu));oracle_native.update(v=x['v'],h0=x['h0'])
                isolated_refs=dict(independent=context.reference.independent_reference(oracle_native),fla=context.reference.fla_reference(oracle_native))
                row['located_actual_preparation_references']={label:{n:decode_checks.metrics(got[n],ref[n]) for n in ('o','final_state')} for label,ref in isolated_refs.items()}
                direct_data={n:prepared[i] for i,n in enumerate(('q','k','g','beta'))}
                direct_data.update(v=dev['v'],initial_state=dev['h0'],output_final_state=True)
                direct=check.cpu(decode_sequence(fused_recurrent,direct_data,dbd))
                row['raw_vs_actual_prepared_public_bitwise']={n:check.digest(got[n])==check.digest(direct[i]) for i,n in enumerate(('o','final_state'))}
            context.write('cases/'+label,row)
            rows.append(dict(id=label,passed=row['passed'],output_sha256=row['output_sha256'],prep_sha256=row['prep_sha256']))
            context.write('summary',dict(complete=False,passed=False,expected=len(cases),cases=rows))
            if not row['passed']:
                torch.save(dict(inputs=x,actual=got,before=before,reference=refs,actual_preparation=prep_cpu,predecessor_npu_preparation=check.cpu(old_prep)),args.output/(label+'.private.pt'))
                raise AssertionError('public grid case failed: '+label)
            print('CASE_PASS',label,flush=True)
        except BaseException as exc:
            context.write('failure',dict(case=case,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed=len(rows)))
            raise
    context.write('summary',dict(complete=True,passed=True,expected=len(cases),cases=rows))
    print('GRID_DONE',args.route,args.block_dim,len(rows),flush=True)


if __name__=='__main__':main()
