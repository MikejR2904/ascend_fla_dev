"""Supplemental norm conditioning observations; never alters frozen acceptance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import torch
ROOT=Path(os.environ.get('BF08_UNIT_ROOT',Path(__file__).resolve().parent))

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('saved','calibration'),required=True)
    p.add_argument('--actual',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cal=load('_cond_calibration',ROOT/'ref/backward_calibrate.py')
    precision=load('_cond_precision',ROOT/'ref/calibrate.py')
    support=load('_cond_support',ROOT/'backward_native_support.py')
    old,receipt=cal.predecessors()
    environment=dict(python=platform.python_version(),torch=torch.__version__,execution='CPU-only arithmetic on saved/generated tensors',
        predecessor=receipt,analysis_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_driver_sha256=hashlib.sha256((ROOT/'ref/backward_calibrate.py').read_bytes()).hexdigest(),
        metric_driver_sha256=hashlib.sha256((ROOT/'ref/calibrate.py').read_bytes()).hexdigest(),
        frozen_budget_sha256=hashlib.sha256((ROOT/'backward_budgets.json').read_bytes()).hexdigest())
    rows=[]
    def write():
        dest=args.output/'conditioning.json';tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(environment=environment,rows=rows,scope='observations only; no new criterion or budget'),indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    def observe(label,x,gy,candidate=None,extra=None):
        z,d=x.double(),gy.double();s=(z*z).sum(-1,keepdim=True)+1.e-6
        first=d/s.sqrt();second=z*((z*d).sum(-1,keepdim=True)/(s*s.sqrt()))
        high=first-second;scale=first.abs()+second.abs()
        leaf=x.clone().requires_grad_();y=old._prepare_inputs(leaf,leaf,None,None,use_qk_l2norm_in_kernel=True)[0]
        previous,=torch.autograd.grad(y,leaf,gy)
        row=dict(id=label,shape=list(x.shape),dtype=str(x.dtype),input_sha256=precision.digest(x),
            sensitivity_sha256=precision.digest(gy),fp64_sha256=precision.digest(high),**(extra or {}))
        values={'old_cpu':previous}
        if candidate is not None:values['candidate']=candidate
        for name,value in values.items():
            error=(value.double()-high).abs();valid=torch.isfinite(error)&torch.isfinite(scale)
            nonzero=valid&(scale!=0);normalized=error[nonzero]/scale[nonzero]
            _,_,dist=support.numeric_ulps(value,high)
            metrics=precision.metrics(value,high)
            metrics['ulp_distribution_to_fp64']=dist
            metrics['over_one_ulp_elements']=sum(v['count'] for v in dist if v['ulp']>1)
            metrics['term_normalized']=dict(elements=int(normalized.numel()),nonfinite_pairs=int((~valid).sum()),
                zero_scale_nonzero_error=int((valid&(scale==0)&(error!=0)).sum()),
                quantile_probabilities=[.99,.999,.9999,1.],
                quantiles=torch.quantile(normalized,torch.tensor([.99,.999,.9999,1.],dtype=torch.float64)).tolist() if normalized.numel() else [0.]*4,
                max=float(normalized.max()) if normalized.numel() else 0.)
            row[name]=metrics
        rows.append(row);write();print('CONDITION_ROW',label,json.dumps({n:dict(rel=row[n]['max_relative_nonzero'],l2=row[n]['relative_l2'],over_one=row[n]['over_one_ulp_elements'],normalized_max=row[n]['term_normalized']['max']) for n in values}),flush=True)
        return previous,row
    if args.mode=='saved':
        assert args.actual is not None
        source=json.loads((args.actual.parent/'full.json').read_text())
        environment['source_execution_environment']=source['environment']
        environment['saved_actual_sha256']=hashlib.file_digest(args.actual.open('rb'),'sha256').hexdigest()
        data=torch.load(args.actual,map_location='cpu',weights_only=True)
        for name in ('q','k'):
            previous,row=observe('full_v1_'+name,data['raw'][name],data['sensitivities'][name],data['actual']['d'+name])
            assert row['candidate']['actual_sha256']==source['prep_gradients'][name]['to_fp64']['actual_sha256']
            assert row['fp64_sha256']==source['prep_gradients'][name]['to_fp64']['reference_fp64_sha256']
            assert row['old_cpu']['actual_sha256']==source['prep_gradients'][name]['to_old_cpu']['reference_fp64_sha256'] or precision.digest(previous.double())==source['prep_gradients'][name]['to_old_cpu']['reference_fp64_sha256']
            details=[]
            for p in sorted((args.actual.parent/'ulp-details'/name).glob('*.json')):details+=json.loads(p.read_text())['locations']
            count=0
            for item in details:
                ix=tuple(item['index']);a=int(data['actual']['d'+name].contiguous().view(torch.int16)[ix])&65535;b=int(previous.contiguous().view(torch.int16)[ix])&65535
                assert a==item['candidate_bits'] and b==item['old_bits'];count+=a==b
            row['located_candidate_equal_old']=dict(total=len(details),equal=count);write()
    else:
        frozen=json.loads((ROOT/'evidence/backward/calibration/norm.json').read_text())
        assert platform.python_version()==frozen['environment']['python'] and torch.__version__==frozen['environment']['torch']
        for seed in (8008,8009,8010):
            rng=torch.Generator().manual_seed(seed)
            for dtype,dt in cal.DTYPES.items():
                cases=[]
                for kind,shape in (('gaussian',(2,192,2,128)),('full_kimi',(1,4096,32,128))):
                    x=torch.randn(shape,generator=rng).to(dt);gy=torch.randn(shape,generator=rng).bfloat16()
                    _,row=observe(f'seed{seed}_{dtype}_{kind}',x,gy,extra=dict(seed=seed,population='original norm calibration' if seed==8008 else 'additional norm holdout; original seeds8009/8010 calibrated gate/beta',kind=kind))
                    if seed==8008:
                        baseline=next(v for v in frozen['rows'] if v['types']==dtype and v['kind']==kind)
                        assert row['input_sha256']==baseline['input_sha256'][0] and row['sensitivity_sha256']==baseline['sensitivity_sha256']
                        assert row['old_cpu']['actual_sha256']==baseline['metrics']['actual_sha256']
                        row['original_calibration_bytes_verified']=True;write()
                    del x,gy
                for scale in (0.,1e-20,1e-8,1e-4,30.,1e10,1e18,1e20):
                    x=(torch.randn((1,64,2,128),generator=rng)*scale).to(dt);gy=torch.randn(x.shape,generator=rng).bfloat16()
                    if seed==8008:
                        _,row=observe(f'seed{seed}_{dtype}_scale{scale:g}',x,gy,extra=dict(seed=seed,kind='scale_'+str(scale),ordinary=scale<=30.))
                        baseline=next(v for v in frozen['rows'] if v['types']==dtype and v['kind']==f'scale_{scale:g}')
                        assert row['input_sha256']==baseline['input_sha256'][0] and row['sensitivity_sha256']==baseline['sensitivity_sha256']
                        assert row['old_cpu']['actual_sha256']==baseline['metrics']['actual_sha256']
                        row['original_calibration_bytes_verified']=True;write()
                x=torch.randn((1,64,2,128),generator=rng).to(dt)
                if seed==8008:observe(f'seed{seed}_{dtype}_parallel',x,x.bfloat16(),extra=dict(seed=seed,ordinary=False))
    write();print('CONDITIONING_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
