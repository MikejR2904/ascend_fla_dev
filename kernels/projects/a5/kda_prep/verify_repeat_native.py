"""Fresh D-PM-44 predecessor/candidate distributions and separately located scan replay."""
import argparse
import collections
from pathlib import Path
import torch
from ascend_fla.ops.kda import autograd
from ascend_fla.runtime.compile import CompiledKernel
from native_context import Context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ctx=Context(args.output,4,4,__file__);check=ctx.check
    x=ctx.real.make_inputs(B=2,H=2,HV=4,C=3,span=46,want_grads=True)
    names=('q','k','v','g','beta','h0');gradient_names=('dq','dk','dv','dg','dbeta','dh0')
    references={label:dict(forward=fwd,gradients=check.gradients(x,fn)) for label,fwd,fn in (
        ('independent',ctx.reference.independent_reference(x),ctx.real.kda_recurrent_ref),
        ('fla',ctx.reference.fla_reference(x),ctx.fla.naive_recurrent_kda))}
    hashes={n:check.digest(t) for n,t in x.items()};rows=[];saved={}
    def run(module):
        leaves={n:x[n].npu().requires_grad_(True) for n in names}
        do=x['do'].npu();dht=x['dht'].float().npu()
        with check.instrument(audit=False):
            o,ht=module.chunk_kda(*(leaves[n] for n in ('q','k','v','g','beta')),initial_state=leaves['h0'],
                output_final_state=True,block_dim=4,layout_device='npu')
            grads=torch.autograd.grad((o,ht),[leaves[n] for n in names],grad_outputs=(do,dht))
        torch.npu.synchronize()
        got=check.cpu(dict(o=o,final_state=ht,**dict(zip(gradient_names,grads))))
        return got,{n:check.digest(leaves[n])==hashes[n] for n in names}
    for label,module in (('predecessor',ctx.old_auto),('candidate',autograd)):
        for repeat in range(12):
            got,unchanged=run(module)
            comparisons={name:dict(**{n:ctx.reference.metrics(got[n],ref['forward'][n]) for n in ('o','final_state')},
                **{n:check.metrics(got[n],t,ctx.real.BUDGET[n]) for n,t in ref['gradients'].items()}) for name,ref in references.items()}
            row=dict(path=label,repeat=repeat,input_sha256=hashes,output_sha256={n:check.digest(t) for n,t in got.items()},
                     unchanged=unchanged,references=comparisons,passed=all(unchanged.values()) and all(m['passed'] for ref in comparisons.values() for m in ref.values()))
            rows.append(row);saved[f'{label}-{repeat}']=got
            ctx.write('public-repeats',dict(complete=False,rows=rows))
            print('PUBLIC_REPEAT',label,repeat,row['passed'],row['output_sha256']['dh0'],flush=True)
    distributions={label:{n:dict(collections.Counter(row['output_sha256'][n] for row in rows if row['path']==label)) for n in rows[0]['output_sha256']} for label in ('predecessor','candidate')}
    public_pass=all(row['passed'] for row in rows) and all(len(counts)==1 for outputs in distributions.values() for counts in outputs.values())
    same={n:set(distributions['predecessor'][n])==set(distributions['candidate'][n]) for n in rows[0]['output_sha256']}
    ctx.write('public-repeats',dict(complete=True,passed=public_pass and all(same.values()),
        scope='12+12 original public autograd calls; existing poison pattern; no added internal barriers or scan captures in these loops.',
        rows=rows,distributions=distributions,predecessor_candidate_bitwise=same))
    # Locate the scan boundary in a separate diagnostic after unbiased repeats.
    captures={}
    for label,module in (('predecessor',ctx.old_auto),('candidate',autograd)):
        original=CompiledKernel.__call__;capture={}
        def intercept(op,inputs,scalars,outputs):
            if 'dh0' in outputs:
                capture['inputs']=check.cpu(inputs);capture['op']=op
                capture['scalars']=dict(scalars);capture['templates']={n:torch.empty_like(t) for n,t in outputs.items()}
            return original(op,inputs,scalars,outputs)
        CompiledKernel.__call__=intercept
        try:run(module)
        finally:CompiledKernel.__call__=original
        assert len(capture['inputs'])==8
        captures[label]=capture
    scan_equal=check.exact(captures['candidate']['inputs'],captures['predecessor']['inputs'])
    capture=captures['candidate'];inputs={n:t.npu() for n,t in capture['inputs'].items()}
    input_hash={n:check.digest(t) for n,t in inputs.items()};replays=[];replay_tensors={}
    for repeat in range(12):
        outputs={n:torch.empty_like(t) for n,t in capture['templates'].items()}
        with check.instrument(audit=False):capture['op'](inputs,capture['scalars'],outputs)
        torch.npu.synchronize();got=check.cpu(outputs);replay_tensors[str(repeat)]=got
        row=dict(repeat=repeat,input_sha256=input_hash,input_unchanged={n:check.digest(t)==input_hash[n] for n,t in inputs.items()},
                 output_sha256={n:check.digest(t) for n,t in got.items()},finite={n:bool(t.isfinite().all()) for n,t in got.items()})
        replays.append(row);print('SCAN_REPEAT',repeat,row['output_sha256'],flush=True)
    replay_pass=(all(v['passed'] for v in scan_equal.values()) and all(all(r['input_unchanged'].values()) and all(r['finite'].values()) for r in replays)
        and all(len({r['output_sha256'][n] for r in replays})==1 for n in replays[0]['output_sha256']))
    ctx.write('scan-replays',dict(complete=True,passed=replay_pass,predecessor_candidate_inputs=scan_equal,rows=replays,
        scope='Separate diagnostic: all eight identical scan inputs,12 direct native scan calls, no new layout within loop.'))
    torch.save(dict(inputs=x,public=saved,references=references,scan_inputs=capture['inputs'],scan_replays=replay_tensors),args.output/'all-samples.private.pt')
    passed=public_pass and all(same.values()) and replay_pass
    ctx.write('summary',dict(complete=True,passed=passed,public_repeats=24,direct_scan_repeats=12,
        limitation='Qualification is specific to this measured environment/device/input population; inherited scan defect is not repaired.'))
    print('REPEAT_DONE',passed,flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
