"""Bounded actual-stage dual-oracle verification; one block_dim per process."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import unit
from ref.oracles import metrics, oracle
from _unit_runner import compare_outputs, validate_contract


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launcher',choices=('sim','pipesim'),required=True)
    p.add_argument('--case',action='append',required=True)
    p.add_argument('--block-dim',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    contract=validate_contract(json.loads((Path(__file__).parent/'contract.json').read_text()))
    rows=[]
    for case_id in args.case:
        case=next(c for c in contract['cases'] if c['id']==case_id)
        data=unit.make_inputs(case)
        expected=unit.reference(data)
        independent=oracle(data)
        options=dict(device='a5',backend='cce',launcher=args.launcher,board=None,
                     block_dim=args.block_dim,timeout=240,sim_processes='threads',
                     out_dir=str(args.output/case_id))
        stages=unit.execute_stages(data,options)
        stage_metrics=compare_outputs(stages,unit.reference_stages(data),contract,stage=True)
        got={n:stages[n] for n in ('o','final_state','final_A_state')}
        compare_outputs(got,expected,contract)
        compare_outputs(got,independent,contract)
        hashes={n:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for n,t in stages.items()}
        row=dict(case=case_id,block_dim=args.block_dim,launcher=args.launcher,
                 vs_chunk=metrics(got,expected),vs_fla_naive=metrics(got,independent),
                 stage_comparison=stage_metrics,stage_sha256=hashes,
                 execution_evidence=options.get('_execution_evidence', []),passed=True)
        rows.append(row)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'summary.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('stage_comparison','stage_sha256','execution_evidence')}),flush=True)


if __name__=='__main__':main()
