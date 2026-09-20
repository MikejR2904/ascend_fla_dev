"""Fresh original backward, real-shape, gate-boundary and retained-training checks."""
import argparse
import itertools
import sys
import traceback
from pathlib import Path
import torch
from ascend_fla.ops.kda import chunk,chunk_bwd,autograd
from native_context import Context
import unit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bd=args.block_dim
    ctx=Context(args.output,bd,4 if bd==3 else bd,__file__);check=ctx.check
    sys.modules['native_support']=check
    extra=unit.module('verify_native').load('_bf07_existing_distributions',unit.ROOT.parent/'kda_layout/native_checks.py')
    options=dict(block_dim=bd,layout_device='npu');rows=[]

    def execute(case,x):
        label=case['id'];print('BACKWARD_CASE_START',label,flush=True)
        dev={n:t.npu() for n,t in x.items()}
        data={n:dev[n] for n in ('q','k','v','g','beta')};data['initial_state']=dev['h0']
        with torch.no_grad(),check.instrument() as (audit,launches):
            plain=autograd.chunk_kda(**data,output_final_state=True,**options)
            o,state,caches=chunk.chunk_kda_fwd_with_caches(**data,**options)
            beta=chunk._layout_runtime().cast(dev['beta'],torch.bfloat16,block_dim=bd)
            grads=chunk_bwd.chunk_kda_bwd(**{n:dev[n] for n in ('q','k','v','do','dht')},beta=beta,caches=caches,**options)
        torch.npu.synchronize()
        got=check.cpu(dict(o=o,final_state=state,**{'cache_'+n:t for n,t in caches.items()},**grads))
        old_o,old_state,old_cache=ctx.old.chunk_kda_fwd_with_caches(**data,**options)
        old_grads=chunk_bwd.chunk_kda_bwd(**{n:dev[n] for n in ('q','k','v','do','dht')},beta=dev['beta'].bfloat16(),caches=old_cache,**options)
        torch.npu.synchronize();before=check.cpu(dict(o=old_o,final_state=old_state,**{'cache_'+n:t for n,t in old_cache.items()},**old_grads))
        exact=check.exact(got,before);refs={}
        for name,fwd,fn in (('independent',ctx.reference.independent_reference(x),ctx.real.kda_recurrent_ref),('fla',ctx.reference.fla_reference(x),ctx.fla.naive_recurrent_kda)):
            grad_ref=check.gradients(x,fn)
            refs[name]=dict(**{n:ctx.reference.metrics(got[n],fwd[n]) for n in ('o','final_state')},
                gradients={n:check.metrics(got[n],t,ctx.real.BUDGET[n]) for n,t in grad_ref.items()},
                per_head_chunk=[dict(chunk=c,head=h,**ctx.reference.metrics(got['o'][:,c*64:(c+1)*64,h],fwd['o'][:,c*64:(c+1)*64,h])) for c in range(case['C']) for h in range(case['HV'])])
        row=dict(case=case,block_dim=bd,bitwise=exact,input_sha256={n:check.digest(t) for n,t in x.items()},
                 input_unchanged={n:check.digest(t)==check.digest(dev[n]) for n,t in x.items()},
                 output_sha256={n:check.digest(t) for n,t in got.items()},finite={n:bool(t.isfinite().all()) for n,t in got.items()},
                 plain_cached_exact=[check.digest(a)==check.digest(b) for a,b in zip(plain,(o,state))],references=refs,
                 operations=audit.report(),unexpected=audit.unexpected(),launches=launches)
        row['passed']=(all(t['passed'] for t in exact.values()) and all(row['input_unchanged'].values()) and all(row['finite'].values())
            and all(row['plain_cached_exact']) and not audit.unexpected() and all(r[n]['passed'] for r in refs.values() for n in ('o','final_state'))
            and all(t['passed'] for r in refs.values() for t in r['gradients'].values()) and all(t['passed'] for r in refs.values() for t in r['per_head_chunk']))
        ctx.write('backward/'+label,row);rows.append(dict(id=label,passed=row['passed']))
        if not row['passed']:
            torch.save(dict(inputs=x,actual=got,before=before),args.output/(label+'.private.pt'))
            raise AssertionError(label)
        print('BACKWARD_CASE_PASS',label,flush=True)

    try:
        for case,x in extra.original_backward_inputs(chunk_bwd):execute(case,x)
        for h,hv in ((32,32),(16,32)):
            case=dict(id=f'real_t1024_h{h}_hv{hv}',B=1,H=h,HV=hv,C=16)
            execute(case,ctx.real.make_inputs(B=1,H=h,HV=hv,C=16,span=46,want_grads=True))
        for span in (0.,100.8,105.,155.,105.001,155.001):
            x=ctx.real.make_inputs(B=1,H=1,HV=1,C=2,span=46,want_grads=True)
            raw=x['g'].clone().view(1,2,64,1,128);raw.fill_(-120.)
            count=4 if span==100.8 else 5
            if span:raw[:,:,64-count:].fill_(span/count)
            a=torch.zeros(1);bias=torch.zeros(128)
            cpu_prep=ctx.old._prepare_inputs(x['q'],x['k'],raw.view_as(x['g']),x['beta'],A_log=a,dt_bias=bias,use_gate_in_kernel=True)
            native=chunk._prepare_kernel_inputs(x['q'].npu(),x['k'].npu(),raw.view_as(x['g']).npu(),x['beta'].npu(),A_log=a.npu(),dt_bias=bias.npu(),use_gate_in_kernel=True,block_dim=bd)
            native_cpu=check.cpu(native);before_span=chunk._gate_span(cpu_prep[2],2,on_cpu=True);after_span=chunk._gate_span(native_cpu[2],2,on_cpu=True)
            row=dict(requested_span=span,predecessor_cpu_span=before_span,candidate_span=after_span,paths={},prep_difference=ctx.precision.metrics(native_cpu[2],cpu_prep[2]))
            for path,limit in (('forward',155.),('cached',105.)):
                pair=[]
                for module,prep in ((ctx.old,cpu_prep),(chunk,native_cpu)):
                    data={n:t.npu() for n,t in zip(('q','k','g','beta'),prep)};data.update(v=x['v'].npu(),initial_state=x['h0'].npu())
                    function=module.chunk_kda_fwd if path=='forward' else module.chunk_kda_fwd_with_caches
                    try:
                        result=function(**data,**options,**({'output_final_state':True} if path=='forward' else {}))
                        torch.npu.synchronize();pair.append(dict(accepted=True,outputs=check.cpu(result)))
                    except ValueError as exc:pair.append(dict(accepted=False,error=str(exc)))
                expected=(before_span<=limit,after_span<=limit)
                report=dict(expected_acceptance=list(expected),actual_acceptance=[p['accepted'] for p in pair],same_decision=pair[0]['accepted']==pair[1]['accepted'])
                assert report['actual_acceptance']==list(expected)
                assert report['same_decision']
                if pair[1]['accepted']:
                    expected_input=dict(zip(('q','k','g','beta'),cpu_prep));expected_input.update(v=x['v'],h0=x['h0'])
                    refs=dict(independent=ctx.reference.independent_reference(expected_input),fla=ctx.reference.fla_reference(expected_input))
                    report['references']={label:{n:ctx.reference.metrics(pair[1]['outputs'][i],ref[n]) for i,n in enumerate(('o','final_state'))} for label,ref in refs.items()}
                    assert all(m['passed'] for ref in report['references'].values() for m in ref.values())
                row['paths'][path]=report
            row['passed']=True;ctx.write(f'gates/span{span:g}',row)
            if span<=105.:
                prepared=dict(x,**dict(zip(('q','k','g','beta'),native_cpu)))
                execute(dict(id=f'gate_span{span:g}_backward',B=1,H=1,HV=1,C=2),prepared)
            print('GATE_PASS',span,before_span,after_span,flush=True)
        training=[]
        names=('q','k','v','g','beta','h0','A_log','dt_bias')
        flags_list=list(itertools.product((False,True),repeat=3))
        populations=[(dtype,flags,names) for dtype,flags in itertools.product((torch.bfloat16,torch.float32),flags_list)]
        populations += [(dtype,(True,True,True),selected) for dtype,selected in itertools.product((torch.bfloat16,torch.float32),(('A_log',),('dt_bias',),('A_log','dt_bias'),('q',),('beta',),('v',),('h0',)))]
        for index,(dtype,flags,selected) in enumerate(populations):
            flags=dict(zip(('use_qk_l2norm_in_kernel','use_gate_in_kernel','use_beta_sigmoid_in_kernel'),flags))
            x=ctx.real.make_inputs(B=2,H=2,HV=4,C=3,span=46,want_grads=True)
            gen=torch.Generator().manual_seed(7012)
            if flags['use_qk_l2norm_in_kernel']:
                for n in ('q','k'):x[n]=torch.randn(x[n].shape,generator=gen).to(dtype)
            if flags['use_gate_in_kernel']:x['g']=(torch.randn(x['g'].shape,generator=gen)*.1).to(dtype)
            if flags['use_beta_sigmoid_in_kernel']:x['beta']=torch.randn(x['beta'].shape,generator=gen).to(dtype)
            x.update(A_log=torch.linspace(-.2,.2,4).to(dtype),dt_bias=torch.linspace(-.3,-.1,4*128).to(dtype))
            # Disabled parameters remain leaves but are intentionally unused.
            active=tuple(n for n in selected if n not in ('A_log','dt_bias') or flags['use_gate_in_kernel'])
            results={};audits={};unchanged={}
            for label,module in (('predecessor',ctx.old_auto),('candidate',autograd)):
                leaves={n:x[n].npu().requires_grad_(n in selected) for n in names}
                with check.instrument(audit=False):
                    with check.Audit() as audit:
                        o,state=module.chunk_kda(*(leaves[n] for n in ('q','k','v','g','beta')),initial_state=leaves['h0'],A_log=leaves['A_log'],dt_bias=leaves['dt_bias'],output_final_state=True,**flags,**options)
                    grads=torch.autograd.grad((o,state),[leaves[n] for n in active],grad_outputs=(x['do'].npu(),x['dht'].float().npu()))
                torch.npu.synchronize();results[label]=check.cpu(dict(o=o,final_state=state,**dict(zip(active,grads))))
                audits[label]=audit.report();unchanged[label]={n:check.digest(t)==check.digest(x[n]) for n,t in leaves.items()}
            exact=check.exact(results['candidate'],results['predecessor'])
            row=dict(index=index,dtype=str(dtype),flags=flags,requires_grad=list(selected),actual_gradient_names=list(active),
                exact=exact,input_unchanged=unchanged,forward_operations=audits,
                finite={n:bool(t.isfinite().all()) for n,t in results['candidate'].items()},
                note='Retained differentiable host graph; explicit noncompliant BF-08 exception, not custom prep backward acceptance.')
            row['passed']=all(v['passed'] for v in exact.values()) and all(row['finite'].values()) and all(v for p in unchanged.values() for v in p.values())
            ctx.write(f'training/case{index:02d}',row);training.append(dict(index=index,passed=row['passed']))
            if not row['passed']:
                torch.save(dict(inputs=x,outputs=results),args.output/f'training{index:02d}.private.pt');raise AssertionError('retained training differs')
            print('TRAINING_PASS',index,flush=True)
        ctx.write('summary',dict(complete=True,passed=True,block_dim=bd,backward_cases=rows,training_cases=training,gate_populations=6))
        print('REGRESSIONS_DONE',bd,len(rows),len(training),flush=True)
    except BaseException as exc:
        ctx.write('failure',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed=rows));raise

if __name__=='__main__':main()
