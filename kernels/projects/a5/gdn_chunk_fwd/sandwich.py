"""Compare two immutable checkouts in fresh processes on one reserved device.

Both revisions must provide this unit's benchmark.py. The caller owns the
shared-device lock and environment; no SSH, credentials or device assignment
are embedded here. Each sample includes the public operator's allocations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--fla-naive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    args=p.parse_args()
    if args.rounds<1: raise ValueError('rounds must be positive')
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'scope':'fresh-process same-device synchronized baseline/candidate/baseline',
            'warmup':10,'repeat':50,'rounds':[]}
    roots={'baseline':args.baseline.resolve(),'candidate':args.candidate.resolve()}
    relative=Path('kernels/projects/a5/gdn_chunk_fwd')
    report['kernel_sha256']={n:hashlib.sha256((r/relative/'kernels/stages.py').read_bytes()).hexdigest() for n,r in roots.items()}
    for index in range(args.rounds):
        for case in ('b1_t1024_h16_g0.03','b1_t4096_h16_g1'):
            row={'round':index+1,'case':case,'samples':{}}
            for label in ('baseline_before','candidate','baseline_after'):
                root=roots['candidate' if label=='candidate' else 'baseline']
                output=args.output.parent/f'{args.output.stem}-r{index+1}-{case}-{label}.json'
                command=[sys.executable,str(root/relative/'benchmark.py'),'--block-dim','2','--case',case,
                         '--profile','--warmup','10','--repeat','50','--fla-naive',str(args.fla_naive.resolve()),'--output',str(output)]
                subprocess.run(command,cwd=root,env=os.environ.copy(),check=True)
                child=json.loads(output.read_text())
                if child['kernel_sha256']!=report['kernel_sha256']['candidate' if label=='candidate' else 'baseline']:
                    raise RuntimeError('source changed during the sandwich')
                result=child['cases'][0]
                row['samples'][label]={'median_us':result['public_timing']['median_us'],
                                       'source_sha256':child['kernel_sha256'],'device':child['device'],
                                       'retained_stage_workspace_bytes':result['retained_stage_workspace_bytes'],
                                       'public_peak_allocated_bytes':result['public_peak_allocated_bytes'],
                                       'sha256':result['sha256']}
            samples=row['samples']
            assert samples['baseline_before']['sha256']==samples['candidate']['sha256']==samples['baseline_after']['sha256'],'stage bytes changed'
            assert samples['baseline_before']['device']==samples['candidate']['device']==samples['baseline_after']['device'],'device/toolchain changed'
            row['speedup']=min(samples['baseline_before']['median_us'],samples['baseline_after']['median_us'])/samples['candidate']['median_us']
            report['rounds'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'round':index+1,'case':case,'speedup':row['speedup']}),flush=True)
    report['consistently_faster']=all(r['speedup']>1 for r in report['rounds'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
