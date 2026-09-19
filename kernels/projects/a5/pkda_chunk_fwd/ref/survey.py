"""CPU FP32 dual-oracle, ATK precision and near-zero normalization report."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ref.reference import make_inputs, reference, validate_reference
from ref.oracles import oracle, metrics
from ref.stages import reference_stages


def atk_precision(data):
    k=data['k'];a=data['initial_A_state'].clone();ad=a.double()
    center=data['log_atk_scale'][None,:,None]
    minimum=float('inf');maximum=-float('inf');logerr=0.;merr=0.;mlo=10.;mhi=0.
    for t in range(k.shape[1]):
        ga=data['g_atk'][:,t,None].transpose(1,2)
        ba=data['beta_atk'][:,t,None].transpose(1,2)
        a=ga.exp()*a+ba*k[:,t].square()
        ad=ga.double().exp()*ad+ba.double()*k[:,t].double().square()
        ell=(a+1e-6).log();r=ell-center
        m=(-torch.log(torch.tensor(1.5))*r/(1+r.abs())).exp()
        rd=(ad+1e-6).log()-center.double()
        md=(-torch.log(torch.tensor(1.5,dtype=torch.float64))*rd/(1+rd.abs())).exp()
        minimum=min(minimum,float(ell.min()));maximum=max(maximum,float(ell.max()))
        logerr=max(logerr,float((ell.double()-(ad+1e-6).log()).abs().max()))
        merr=max(merr,float((m.double()-md).abs().max()));mlo=min(mlo,float(m.min()));mhi=max(mhi,float(m.max()))
    return dict(log_range=[minimum,maximum],M_range=[mlo,mhi],log_fp32_vs_fp64_max_abs=logerr,M_fp32_vs_fp64_max_abs=merr)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    contract=json.loads((root/'contract.json').read_text())
    rows=[]
    for case in contract['cases']:
        data=make_inputs(case);local=reference(data);fla=oracle(data)
        validate_reference(data,local);validate_reference(data,fla)
        error=metrics(local,fla)
        assert all(m['relative_l2']<=1e-4 for m in error.values()),(case['id'],error)
        rows.append(dict(case=case['id'],parameters=case['parameters'],chunk_vs_fla_naive=error,atk=atk_precision(data),passed=True))
        print(json.dumps(rows[-1]),flush=True)
    near=[]
    for norm in (1e-2,1e-4,1e-6,0.):
        data=make_inputs(dict(seed=713,parameters=dict(B=1,T=2,H=1,row_norm=norm)))
        raw=oracle(data)
        chunk_input=dict(data)
        unit_input=dict(data)
        for n in ('q','k'):
            chunk_input[n]=data[n]*torch.rsqrt(data[n].square().sum(-1,keepdim=True)+1e-6)
            unit_input[n]=torch.nn.functional.normalize(data[n],dim=-1)
        chunk=oracle(chunk_input);normalized=oracle(unit_input)
        near.append(dict(row_norm=norm,raw_naive_output_norm=float(raw['o'].norm()),
                         chunk_normalized_output_norm=float(chunk['o'].norm()),
                         caller_F_normalized_output_norm=float(normalized['o'].norm()),
                         chunk_formula_vs_raw_naive=metrics(chunk,raw),
                         caller_F_normalize_vs_raw_naive=metrics(normalized,raw)))
    gate_cases=[]
    for span in (105.0,155.0,4096.0):
        for seed in (71,72,73):
            data=make_inputs(dict(seed=seed,parameters=dict(B=1,T=192,H=8)))
            gate=data['g'].reshape(1,3,64,8,128)
            gate.fill_(-1e-5);gate[:,:,0].fill_(-span+63e-5)
            # span4096 is an explicitly unsupported CPU experiment. Do not
            # call the public validator or use this result as qualification.
            candidate=reference_stages(data)
            candidate={n:candidate[n] for n in ('o','final_state','final_A_state')}
            error=metrics(candidate,oracle(data))
            ok=all(m['relative_l2']<=1e-4 for m in error.values())
            if span<=155: assert ok,error
            gate_cases.append(dict(span=span,seed=seed,within_public_domain=span<=155,
                                   comparison=error,meets_1e4=ok))
    # A faulty symmetric-key prediction keeps A correct while corrupting S/o.
    data=make_inputs(dict(seed=987,parameters=dict(B=1,T=64,H=1)))
    expected=oracle(data);state=data['initial_state'].clone();a=data['initial_A_state'].clone();bad=[]
    for t in range(64):
        k=data['k'][:,t]
        a=data['g_atk'][:,t].exp().unsqueeze(-1)*a+data['beta_atk'][:,t].unsqueeze(-1)*k.square()
        r=(a+1e-6).log()-data['log_atk_scale'][None,:,None]
        write=k*(-torch.log(torch.tensor(1.5))*r/(1+r.abs())).exp()
        state=state*data['g'][:,t].exp().unsqueeze(-1)
        residual=data['v'][:,t]-(write.unsqueeze(-1)*state).sum(-2)
        state=state+(data['beta'][:,t].unsqueeze(-1)*write).unsqueeze(-1)*residual.unsqueeze(-2)
        bad.append((data['q'][:,t].unsqueeze(-1)*state).sum(-2)*128**-0.5)
    negative=metrics(dict(o=torch.stack(bad,1),final_state=state,final_A_state=a),expected)
    assert negative['o']['relative_l2']>1e-4 and negative['final_state']['relative_l2']>1e-4
    out=dict(schema='pkda-host-survey/1',passed=True,torch=torch.__version__,
             fla_revision='e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2',
             fla_naive_sha256=hashlib.sha256(Path(os.environ['FLA_PKDA_NAIVE']).read_bytes()).hexdigest(),
             cases=rows,near_zero=near,adversarial_gate_cases=gate_cases,
             rejected_symmetric_key_control=negative,
             scope='CPU mathematical evidence only. FLA Triton not executed; native untested.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
