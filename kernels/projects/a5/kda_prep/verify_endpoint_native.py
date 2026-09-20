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
    ctx.write('stages',dict(cases=rows,conclusion='Located diagnostics only; no automatic endpoint acceptance or budget modification.'))
    print('ENDPOINT_TRACE_DONE',len(rows),flush=True)

if __name__=='__main__':main()
