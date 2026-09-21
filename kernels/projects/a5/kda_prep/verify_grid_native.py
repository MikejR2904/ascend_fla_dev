"""Native public grid; each process owns one chunk and one decode block_dim."""
import argparse
import hashlib
import math
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
    parser.add_argument('--boundary-kind',nargs='+',choices=('zero','nearzero','threshold','beta_saturation'),
                        help='Run a named diagnostic partition; other required populations remain pending.')
    parser.add_argument('--observe-all-cases',action='store_true',
                        help='Diagnostic collection: retain every numerical failure; exit zero means collection completed, not acceptance passed.')
    parser.add_argument('--dpm51-nearzero-decode',action='store_true',
                        help='User-approved nearzero decode comparison; retains old CPU-prep metrics and all original limits.')
    parser.add_argument('--chunk-count',type=int,choices=(1,2,3))
    args=parser.parse_args()
    bd=args.block_dim if args.route=='chunk' else min(args.block_dim,4)
    dbd=(4 if bd==3 else bd) if args.route=='chunk' else args.block_dim
    context=native_context.Context(args.output,bd,dbd,__file__)
    check=context.check;old=context.old
    decode_checks=unit.module('verify_native').load('_bf07_decode_checks',unit.ROOT.parent/'kda_fused_recurrent_bf16/research.py')
    generator=native_grid.cases if args.population=='ordinary' else native_grid.boundary_cases
    cases=[p for p in generator(args.route) if not args.chunk_count or p['C']==args.chunk_count]
    if args.boundary_kind:
        assert args.population=='boundary'
        cases=[p for p in cases if p['boundary'] in args.boundary_kind]
    if args.case_id:cases=[p for p in cases if p['id']==args.case_id]
    assert cases, 'selected population is empty'
    if args.dpm51_nearzero_decode:
        assert args.route=='decode' and args.population=='boundary'
        assert all(p['boundary']=='nearzero' and p['id'].startswith('nearzero_') for p in cases)
    rows=[]
    original_forward_metric=context.reference.metrics
    original_decode_metric=decode_checks.metrics
    metric_comparisons=[]
    metric_stats=dict(comparisons=0,same_at_six_significant_digits=0,max_absolute_difference=0.,nonfinite_comparisons=0)

    def relative_l2(actual,expected):
        a,e=actual.detach().cpu().double(),expected.detach().cpu().double()
        norm=float(e.norm());residual=float((a-e).norm())
        return residual/norm if norm else (0. if residual==0 else float('inf'))

    def zero_representation_endpoint(actual,expected):
        # PM5750678047: only BF16 output slices whose correctly rounded
        # CPU FP32 golden is entirely zero. The original relative error stays.
        if args.population!='boundary' or actual.dtype!=torch.bfloat16:
            return None
        rounded=expected.to(torch.bfloat16)
        if not bool(expected.isfinite().all()) or not bool((rounded==0).all()):
            return None
        # Distance from numeric zero in BF16 representable steps. Signed-zero
        # encodings are recorded separately; +0 and -0 have zero ULP distance.
        distances=(actual.contiguous().view(torch.int16).long() & 0x7fff)
        finite=bool(actual.isfinite().all())
        return dict(owner_comment='https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5750678047',
                    definition='CPU FP32 golden correctly rounded to BF16 is zero throughout this slice',
                    elements=actual.numel(),max_ulp_from_rounded_zero=int(distances.max()),
                    signed_zero_differences=int((torch.signbit(actual)!=torch.signbit(rounded)).sum()),
                    bitwise_equal_to_rounded_reference=check.digest(actual)==check.digest(rounded),
                    passed=finite and bool((distances<=1).all()))

    def record_metrics(old,new):
        same=format(old,'.6g')==format(new,'.6g')
        difference=abs(old-new)
        metric_comparisons.append(dict(legacy_relative_l2=old,corrected_fp64_relative_l2=new,
            legacy_implementation='FP32 norm' if args.route=='chunk' else 'FP64 norm with denominator clamped at 1e-30',
            absolute_difference=difference,same_at_six_significant_digits=same,
            same_at_six_decimal_places=format(old,'.6f')==format(new,'.6f')))
        metric_stats['comparisons']+=1
        metric_stats['same_at_six_significant_digits']+=int(same)
        if math.isfinite(difference):metric_stats['max_absolute_difference']=max(metric_stats['max_absolute_difference'],difference)
        else:metric_stats['nonfinite_comparisons']+=1

    def forward_metric(actual,expected):
        result=original_forward_metric(actual,expected)
        precise=relative_l2(actual,expected)
        record_metrics(result['relative_l2'],precise)
        result.update(relative_l2_fp64_accumulation=precise,legacy_passed=result['passed'])
        result['passed']=result['legacy_passed'] and precise<=.05
        endpoint=zero_representation_endpoint(actual,expected)
        if endpoint is not None:
            result['zero_representation_endpoint']=endpoint
            result['passed']=endpoint['passed']
        return result

    context.reference.metrics=forward_metric

    def decode_measure(actual,expected):
        result=original_decode_metric(actual,expected)
        legacy=result['relative_l2'];precise=relative_l2(actual,expected)
        old_floor=original_decode_metric(expected.bfloat16().float(),expected)['relative_l2']
        floor=relative_l2(expected.bfloat16().float(),expected)
        old_limit=min(.01,3*old_floor) if actual.dtype==torch.bfloat16 else 1e-5
        limit=min(.01,3*floor) if actual.dtype==torch.bfloat16 else 1e-5
        old_pass=result['finite'] and legacy<=old_limit
        record_metrics(legacy,precise)
        result.update(relative_l2=precise,legacy_relative_l2=legacy,
            legacy_budget=old_limit,legacy_passed=old_pass,budget=limit,bf16_floor=floor,
            passed=old_pass and precise<=limit)
        endpoint=zero_representation_endpoint(actual,expected)
        if endpoint is not None:
            result['zero_representation_endpoint']=endpoint
            result['passed']=endpoint['passed']
        return result

    for case in cases:
        metric_comparisons.clear()
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
                    measures={n:decode_measure(got[n],ref[n]) for n in ('o','final_state')}
                    measures['per_head_chunk']=[dict(chunk=c,head=h,**decode_measure(
                        got['o'][:,c*64:(c+1)*64,h],ref['o'][:,c*64:(c+1)*64,h]))
                        for c in range(case['C']) for h in range(case['HV'])]
                comparisons[name]=measures
            zero_slices=[]
            for oracle,measures in comparisons.items():
                for output in ('o','final_state'):
                    if 'zero_representation_endpoint' in measures[output]:
                        zero_slices.append(dict(oracle=oracle,output=output,scope='whole',**measures[output]['zero_representation_endpoint']))
                for part in measures['per_head_chunk']:
                    if 'zero_representation_endpoint' in part:
                        zero_slices.append(dict(oracle=oracle,output='o',scope='head_chunk',chunk=part['chunk'],head=part['head'],**part['zero_representation_endpoint']))
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
                     cpu_fp32_references=comparisons,zero_representation_slices=zero_slices,passed=False)
            non_reference_checks=(all(row['input_unchanged'].values()) and all(row['finite'].values()) and all(disabled.values())
                and (plain_cached_exact is None or all(plain_cached_exact)) and not audit.unexpected()
                and all(p['passed'] for p in prec.values())
                and (exact_disabled is None or all(p['passed'] for p in exact_disabled.values())))
            row['passed']=(non_reference_checks
                and all(m[n]['passed'] for m in comparisons.values() for n in ('o','final_state'))
                and all(p['passed'] for m in comparisons.values() for p in m['per_head_chunk'])
                and all(p['passed'] for m in comparisons.values() for p in m.get('cached_states',[])))
            if args.route=='decode' and (not row['passed'] or args.dpm51_nearzero_decode):
                oracle_native=dict(zip(('q','k','g','beta'),prep_cpu));oracle_native.update(v=x['v'],h0=x['h0'])
                isolated_refs=dict(independent=context.reference.independent_reference(oracle_native),fla=context.reference.fla_reference(oracle_native))
                row['located_actual_preparation_references']={label:{n:dict(original_decode_metric(got[n],ref[n]),relative_l2=relative_l2(got[n],ref[n])) for n in ('o','final_state')} for label,ref in isolated_refs.items()}
                direct_data={n:prepared[i] for i,n in enumerate(('q','k','g','beta'))}
                direct_data.update(v=dev['v'],initial_state=dev['h0'],output_final_state=True)
                direct=check.cpu(decode_sequence(fused_recurrent,direct_data,dbd))
                row['raw_vs_actual_prepared_public_bitwise']={n:check.digest(got[n])==check.digest(direct[i]) for i,n in enumerate(('o','final_state'))}
                if args.dpm51_nearzero_decode:
                    # This bounds the named diagnostic population; it does not
                    # add a public input gate. Goldens remain CPU FP32.
                    eps_ratios={n:float(x[n].double().square().sum(-1).max()/1e-6) for n in ('q','k')}
                    assert all(value<1e-12 for value in eps_ratios.values()),eps_ratios
                    native_comparisons={}
                    for name,ref in isolated_refs.items():
                        measures={n:decode_measure(got[n],ref[n]) for n in ('o','final_state')}
                        measures['per_head_chunk']=[dict(chunk=c,head=h,**decode_measure(
                            got['o'][:,c*64:(c+1)*64,h],ref['o'][:,c*64:(c+1)*64,h]))
                            for c in range(case['C']) for h in range(case['HV'])]
                        native_comparisons[name]=measures
                    row['old_cpu_preparation_criteria_passed']=row['passed']
                    row['native_preparation_cpu_fp32_references']=native_comparisons
                    row['comparison_contract']=dict(decision='D-PM-51',
                        owner_comment='https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5750986507',
                        scope='Named nearzero decode boundary population only',sum_squared_over_epsilon=eps_ratios,
                        limits_changed=False,old_cpu_preparation_metrics_retained=True)
                    row['passed']=(non_reference_checks and all(row['raw_vs_actual_prepared_public_bitwise'].values())
                        and all(m[n]['passed'] for m in native_comparisons.values() for n in ('o','final_state'))
                        and all(p['passed'] for m in native_comparisons.values() for p in m['per_head_chunk']))
            context.write('cases/'+label,row)
            context.write('metric-comparisons/'+label,dict(ordinary_decisions_require_legacy_and_corrected_pass=True,population=args.population,comparisons=metric_comparisons))
            rows.append(dict(id=label,passed=row['passed'],output_sha256=row['output_sha256'],prep_sha256=row['prep_sha256']))
            context.write('summary',dict(complete=False,passed=False,expected=len(cases),cases=rows,
                          metric_implementation_summary=metric_stats,selected_boundary_kinds=args.boundary_kind,full_population=args.boundary_kind is None and args.case_id is None and args.chunk_count is None))
            if not row['passed']:
                torch.save(dict(inputs=x,actual=got,before=before,reference=refs,
                    native_preparation_reference=isolated_refs if args.dpm51_nearzero_decode else None,
                    actual_preparation=prep_cpu,predecessor_npu_preparation=check.cpu(old_prep)),args.output/(label+'.private.pt'))
                if not args.observe_all_cases:
                    raise AssertionError('public grid case failed: '+label)
            print('CASE_PASS' if row['passed'] else 'CASE_FAIL_RETAINED',label,flush=True)
        except BaseException as exc:
            context.write('failure',dict(case=case,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed=len(rows)))
            raise
    context.write('summary',dict(complete=True,passed=all(row['passed'] for row in rows),expected=len(cases),cases=rows,
                  diagnostic_collection_only=args.observe_all_cases,
                  metric_implementation_summary=metric_stats,selected_boundary_kinds=args.boundary_kind,full_population=args.boundary_kind is None and args.case_id is None and args.chunk_count is None))
    print('GRID_COLLECTED' if args.observe_all_cases else 'GRID_DONE',args.route,args.block_dim,len(rows),
          'passing',sum(row['passed'] for row in rows),flush=True)


if __name__=='__main__':main()
