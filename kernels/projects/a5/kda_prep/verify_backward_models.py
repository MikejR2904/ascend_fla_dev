"""Bounded backward sim/pipesim diagnostics, after full native training."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,platform,traceback
from pathlib import Path
import torch
import ascriptor
ROOT=Path(__file__).resolve().parent

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',choices=('sim','pipesim'),required=True)
    parser.add_argument('--kind',choices=('norm','gate','beta'),required=True)
    parser.add_argument('--dtype',choices=('bf16','f32'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    assert Path(ascriptor.__file__).resolve().is_relative_to(args.runtime_root.resolve()/'library')
    manifest=json.loads(args.runtime_manifest.read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((args.runtime_root/name).read_bytes()).hexdigest()==digest,name
    runtime=load('model_runtime','runtime.py');runner=load('model_runner','_unit_runner.py')
    precision=load('model_precision','ref/calibrate.py');cal=load('model_cal','ref/backward_calibrate.py')
    support=load('model_support','backward_native_support.py');old,_=cal.predecessors()
    budget=json.loads((ROOT/'backward_budgets.json').read_text())['groups'];dt=cal.DTYPES[args.dtype]
    environment=dict(python=platform.python_version(),torch=torch.__version__,ascriptor=ascriptor.__version__,
        stage=args.launcher,device_profile='950',block_dim=1,vector_participants=2,
        library_commit='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5',
        kernels_commit='b3b3f9c16df7c4626ed3c081032a1be5a753d0b1',
        accepted_runtime_manifest_sha256=hashlib.sha256(args.runtime_manifest.read_bytes()).hexdigest(),
        verified_runtime_files=len(manifest),
        kernel_source_sha256=hashlib.sha256((ROOT/'kernels/backward.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budget_sha256=hashlib.sha256((ROOT/'backward_budgets.json').read_bytes()).hexdigest(),
        scope='Reduced diagnostic; neither full workload nor board performance qualification')
    rng=torch.Generator().manual_seed(8088);rows=[]
    options=dict(device='a5',backend='cce',block_dim=1,launcher=args.launcher,
                 out_dir=args.output,timeout=45.,board=None)
    def launch(key,values):return runner.launch_kernel(runtime.backward_kernels()[key],values,options)
    def compare(name,key,got,high,before,condition=None,endpoint_mask=None):
        row=support.gradient_record(key,got,high,before,precision,budget[key],args.output/'ulp-details'/name,
            condition=condition if condition is not None else torch.ones_like(high),environment=environment,
            endpoint_zero_mask=endpoint_mask,endpoint_authority='D-PM-55, issue117 comment5755012001' if endpoint_mask is not None else None)
        rows.append(dict(id=name,**row));return row
    try:
        if args.kind=='norm':
            x=torch.randn((1,65,1,128),generator=rng).to(dt);gy=torch.randn(x.shape,generator=rng).bfloat16()
            x[:,0]*=1.e20;x[:,1]*=1.e18;gy[:,2]=x[:,2].bfloat16()
            before=x.clone();before_gy=gy.clone();out=torch.full((1,x.numel()),float('nan'),dtype=dt)
            got=launch('norm_'+args.dtype,(x.reshape(1,-1),gy.reshape(1,-1),out,x.numel())).reshape(x.shape)
            leaf=x.clone().requires_grad_();y=old._prepare_inputs(leaf,leaf,None,None,use_qk_l2norm_in_kernel=True)[0]
            previous,=torch.autograd.grad(y,leaf,gy);high=cal.norm_reference(x,gy)
            z=x.double();d=gy.double();s=(z*z).sum(-1,keepdim=True)+1e-6
            cond=((d/s.sqrt()).abs()+(z*((z*d).sum(-1,keepdim=True)/(s*s.sqrt()))).abs())/high.abs()
            overflow=torch.isinf(x.float().square().sum(-1,keepdim=True)+1.e-6).expand_as(x)
            compare('norm65rows','norm:'+args.dtype+':dx',got,high,previous,cond,overflow)
            assert torch.equal(x,before) and torch.equal(gy,before_gy)
        elif args.kind=='beta':
            x=torch.randn((1,8193,1),generator=rng).to(dt);gy=torch.randn(x.shape,generator=rng)
            probability=x.float().sigmoid();out=torch.full((1,x.numel()),float('nan'),dtype=dt)
            got=launch('beta_'+args.dtype,(x.reshape(1,-1),probability.reshape(1,-1),gy.reshape(1,-1),out,x.numel())).reshape(x.shape)
            leaf=x.clone().requires_grad_();y=leaf.float().sigmoid();previous,=torch.autograd.grad(y,leaf,gy)
            compare('beta8193','beta:'+args.dtype+':dbeta',got,cal.beta_reference(x,gy),previous)
        else:
            for bt,hv in ((65,2),(1,17)):
                x=torch.randn((1,bt,hv,128),generator=rng).to(dt)
                gy=torch.randn(x.shape,generator=rng);a=torch.linspace(-.2,.2,hv).to(dt);bias=torch.linspace(-.3,-.1,hv*128).to(dt)
                n=x.numel();chunks=(bt+31)//32;p=hv*chunks*512
                dg=torch.full((1,n),float('nan'),dtype=dt);partials=torch.full((1,p),float('nan'))
                labels='_'.join([args.dtype]*3)
                gd,part=launch('gate_'+labels,(x.reshape(1,-1),gy.reshape(1,-1),a.reshape(1,-1),bias.reshape(1,-1),dg,partials,n,hv,hv*128,bt,p))
                assert part.isfinite().all(),'unwritten partial workspace'
                da=torch.full((1,hv),float('nan'),dtype=dt);db=torch.full((1,hv*128),float('nan'),dtype=dt)
                ga,gb=launch('gate_reduce_'+args.dtype+'_'+args.dtype,(part,a.reshape(1,-1),da,db,p,hv,hv*128,chunks))
                leaves=[v.clone().requires_grad_() for v in (x,a,bias)]
                y=old._prepare_inputs(None,None,leaves[0],None,A_log=leaves[1],dt_bias=leaves[2],use_gate_in_kernel=True)[2]
                previous=torch.autograd.grad(y,leaves,gy);high=cal.gate_reference(x,a,bias,gy)
                u=x.double()+bias.double().view(hv,128)
                sp=torch.where(u>20,u,torch.logaddexp(u,torch.zeros_like(u)))
                contribution=gy.double()*sp*(-a.double().exp().view(-1,1))
                conditions=dict(dg=torch.ones_like(high['dg']),
                    dA_log=contribution.abs().sum((0,1,3))/high['dA_log'].abs(),
                    ddt_bias=high['dg'].abs().sum((0,1)).reshape(-1)/high['ddt_bias'].abs())
                for name,got,prior in zip(('dg','dA_log','ddt_bias'),(gd.reshape(x.shape),ga.reshape(a.shape),gb.reshape(bias.shape)),previous):
                    compare(f'gate_bt{bt}_hv{hv}_{name}','gate:'+labels+':'+name,got,high[name],prior,conditions[name])
        result=dict(environment=environment,complete=True,passed=all(r['passed'] for r in rows),cases=rows,
            execution_evidence=options.get('_execution_evidence',[]),
            retained_boundaries='two vector participants, repeated4096 tiles, one-row/one-element tail; gate two-stage GM handoff and grouped A-output owners')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('MODEL_RESULT',args.launcher,args.kind,args.dtype,result['passed'],flush=True)
        if not result['passed']:raise AssertionError('Reduced model numerical comparison failed')
    except BaseException as error:
        (args.output/'failure.private.json').write_text(json.dumps(dict(environment=environment,error=str(error),traceback=traceback.format_exc()),indent=2)+'\n')
        raise

if __name__=='__main__':main()
