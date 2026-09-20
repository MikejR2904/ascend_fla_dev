"""Classify retained native failures without modifying their acceptance decisions.

CPU-only analysis of trusted, locally generated tensor receipts. The original
CPU FP32 reference tensors remain unchanged; FP64 is used for metric accumulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def rel(a, e):
    a, e = a.double(), e.double()
    residual, norm = float((a-e).norm()), float(e.norm())
    return residual/norm if norm else (0. if residual == 0 else None)


def ordered(t):
    bits = t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32).long()
    offset = 32768 if t.dtype == torch.bfloat16 else 2147483648
    return torch.where(bits < 0, -bits-offset, bits)


def classify(a, e, old, route):
    assert e.dtype == torch.float32
    aa, ee = a.float(), e
    finite = torch.isfinite(aa) & torch.isfinite(ee)
    normal = finite & (ee.abs() >= torch.finfo(torch.float32).tiny)
    subnormal = finite & (ee != 0) & (ee.abs() < torch.finfo(torch.float32).tiny)
    flush = subnormal & (aa == 0)
    retained = subnormal & (aa != 0)
    exact_zero = finite & (ee == 0) & (aa == 0)
    zero_mismatch = finite & (ee == 0) & (aa != 0)
    n = int(normal.sum())
    normal_error = rel(aa[normal], ee[normal]) if n else None
    normal_floor = rel(ee[normal].bfloat16().float(), ee[normal]) if n else None
    limit = .05 if route == 'chunk' else (min(.01, 3*normal_floor) if a.dtype == torch.bfloat16 and n else 1e-5)
    other = retained | zero_mismatch | ~finite
    rounded = e.to(a.dtype)
    locations = other.nonzero()[:8].tolist()
    normal_locations = (normal & (aa != ee)).nonzero()[:8].tolist()
    different = (a != old).nonzero()[:8].tolist()
    return dict(elements=a.numel(), actual_dtype=str(a.dtype), golden_dtype=str(e.dtype),
        raw_relative_l2=rel(a,e), actual_max_abs=float(aa.abs().max()),golden_max_abs=float(ee.abs().max()),
        counts=dict(cpu_normal=n,cpu_subnormal_flushed_to_signed_zero=int(flush.sum()),
            cpu_subnormal_actual_nonzero=int(retained.sum()),
            cpu_subnormal_actual_nonzero_numerically_exact=int((retained & (aa==ee)).sum()),
            both_numeric_zero=int(exact_zero.sum()),cpu_zero_actual_nonzero=int(zero_mismatch.sum()),
            nonfinite=int((~finite).sum())),
        normal_subset=dict(relative_l2=normal_error,old_npu_relative_l2=rel(old[normal],ee[normal]) if n else None,
            budget=limit,passed=(normal_error<=limit) if n else None),
        candidate_equals_old_npu_bitwise=digest(a)==digest(old),
        candidate_to_old_npu=dict(relative_l2=rel(a,old),max_abs=float((aa-old.float()).abs().max()),
            numeric_differences=int((a!=old).sum()),max_ulp=int((ordered(a)-ordered(old)).abs().max()),
            samples=[dict(index=i,actual=float(aa[tuple(i)]),old_npu=float(old[tuple(i)]),cpu_fp32=float(ee[tuple(i)])) for i in different]),
        candidate_equals_rounded_cpu_bitwise=digest(a)==digest(rounded),
        rounded_cpu_nonzero=int((rounded!=0).sum()),
        signed_zero_difference_count=int(((aa==0)&(rounded==0)&(torch.signbit(a)!=torch.signbit(rounded))).sum()),
        unqualified_other_count=int(other.sum()),
        normal_samples=[dict(index=i,actual=float(aa[tuple(i)]),cpu_fp32=float(ee[tuple(i)]),old_npu=float(old[tuple(i)])) for i in normal_locations],
        other_samples=[dict(index=i,actual=float(aa[tuple(i)]),cpu_fp32=float(ee[tuple(i)]),old_npu=float(old[tuple(i)])) for i in locations],
        scope='Classification only; original numerical pass/fail is retained. Exact numeric zeros are separate from flush. Nonzero subnormal results are not covered by flush qualification.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    for run in args.runs:
        summary=json.loads((run/'summary.json').read_text());result=[]
        for entry in summary['cases']:
            if entry['passed']:continue
            label=entry['id'];receipt=json.loads((run/'cases'/(label+'.json')).read_text())
            path=run/(label+'.private.pt');raw=path.read_bytes()
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert all(digest(t)==receipt['output_sha256'][n] for n,t in saved['actual'].items())
            assert all(digest(t)==receipt['input_sha256'][n] for n,t in saved['inputs'].items())
            slices=[]
            for oracle, measures in receipt['cpu_fp32_references'].items():
                for name in ('o','final_state'):
                    if not measures[name]['passed']:
                        slices.append(dict(oracle=oracle,output=name,scope='whole',original_metric=measures[name],classification=classify(saved['actual'][name],saved['reference'][oracle][name],saved['before'][name],receipt['route'])))
                for m in measures['per_head_chunk']:
                    if m['passed']:continue
                    c,h=m['chunk'],m['head'];idx=(slice(None),slice(c*64,(c+1)*64),h)
                    slices.append(dict(oracle=oracle,output='o',scope='head_chunk',chunk=c,head=h,original_metric=m,classification=classify(saved['actual']['o'][idx],saved['reference'][oracle]['o'][idx],saved['before']['o'][idx],receipt['route'])))
            result.append(dict(id=label,raw_tensor_archive_sha256=hashlib.sha256(raw).hexdigest(),actual_and_input_hashes_verified=True,slices=slices))
        report=dict(environment=summary['environment'],run=run.name,stage='CPU classification of saved actual native outputs',
            classifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),owner_comment='https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5750896563',
            changes_numerical_acceptance=False,original_complete=summary['complete'],original_passed=summary['passed'],cases=result)
        (args.output/(run.name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(run.name,len(result),'classified',flush=True)


if __name__=='__main__':main()
