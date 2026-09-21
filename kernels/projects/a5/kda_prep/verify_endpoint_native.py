"""Located native primitive traces for retained endpoint discrepancies."""
import argparse
from pathlib import Path
import torch
from ascend_fla.ops.kda import chunk
from native_context import Context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ctx=Context(args.output,4,4,__file__)
    values=[-120.,-104.,-100.,-90.,-88.,-87.,-86.,-40.,-20.,-16.,-1.,0.,1.,19.999998092651367,20.,20.000001907348633,40.,80.]
    x=torch.tensor(values,dtype=torch.float32);a=torch.full_like(x,-.2)
    def stages(x,a):
        exp=x.exp();sp=torch.nn.functional.softplus(x);ea=a.exp()
        return dict(exp=exp,one_plus_exp=1+exp,log1p_exp=torch.log1p(exp),softplus=sp,exp_alog=ea,gate=-ea*sp,sigmoid=x.sigmoid())
    cpu=stages(x,a);device=stages(x.npu(),a.npu());torch.npu.synchronize()
    device={n:t.cpu() for n,t in device.items()}
    g=x.repeat((128+len(values)-1)//len(values))[:128].view(1,1,1,128)
    kernel=chunk._prep_runtime().gate(g.npu(),torch.tensor([-.2]).npu(),torch.zeros(128).npu(),block_dim=4).cpu().reshape(-1)
    rows=[]
    for i,value in enumerate(values):
        rows.append(dict(input=value,candidate_gate=float(kernel[i]),cpu_fp32={n:float(t[i]) for n,t in cpu.items()},
                         predecessor_npu={n:float(t[i]) for n,t in device.items()},
                         cpu_bits={n:int(t.view(torch.int32)[i]) for n,t in cpu.items()},
                         predecessor_bits={n:int(t.view(torch.int32)[i]) for n,t in device.items()}))
    downstream=[]
    us=[-104.,-100.,-90.,-88.,-87.,-40.,-20.,-16.]
    for alog in (-3.,-.2,2.7,80.,89.,100.):
        g=torch.tensor(us).repeat(16).view(1,1,1,128)
        a=torch.tensor([alog]);bias=torch.zeros(128)
        dg,da,db=g.npu(),a.npu(),bias.npu()
        candidate=chunk._prep_runtime().gate(dg,da,db,block_dim=4)
        old=ctx.old._prepare_inputs(None,None,dg,None,A_log=da,dt_bias=db,use_gate_in_kernel=True)[2]
        cpu=ctx.old._prepare_inputs(None,None,g,None,A_log=a,dt_bias=bias,use_gate_in_kernel=True)[2]
        high=ctx.precision.high_precision_gate(g,a,bias)
        candidate_decay=candidate.exp().cpu().reshape(-1);old_decay=old.exp().cpu().reshape(-1)
        candidate=candidate.cpu().reshape(-1);old=old.cpu().reshape(-1);cpu=cpu.reshape(-1);high=high.reshape(-1)
        for i,u in enumerate(us):
            downstream.append(dict(u=u,A_log=float(a[0]),candidate_gate=float(candidate[i]),predecessor_npu_gate=float(old[i]),
                cpu_fp32_gate=float(cpu[i]),fp64_gate=float(high[i]),candidate_decay_npu=float(candidate_decay[i]),predecessor_decay_npu=float(old_decay[i]),
                candidate_gate_bits=int(candidate.view(torch.int32)[i]),predecessor_gate_bits=int(old.view(torch.int32)[i]),cpu_gate_bits=int(cpu.view(torch.int32)[i]),
                candidate_decay_bits=int(candidate_decay.view(torch.int32)[i]),predecessor_decay_bits=int(old_decay.view(torch.int32)[i]),
                decay_ulp_difference=int(abs(candidate_decay.view(torch.int32)[i].long()-old_decay.view(torch.int32)[i].long()))))
    ctx.write('downstream',dict(cases=downstream,scope='Measured endpoint cross-products and their NPU exp(g) effect; NaN/Inf encoded by explicit IEEE bits when JSON numeric value is null. No automatic acceptance.'))
    ctx.write('stages',dict(cases=rows,conclusion='Located diagnostics only; no automatic endpoint acceptance or budget modification.'))
    print('ENDPOINT_TRACE_DONE',len(rows),flush=True)

if __name__=='__main__':main()
