"""Bounded unit diagnostics; run after the complete native workload."""
import argparse
import hashlib
import json
from pathlib import Path

import unit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('reference','check'))
    parser.add_argument('--launcher',choices=('sim','pipesim'),default='sim')
    parser.add_argument('--case',action='append')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    from _unit_runner import validate_contract
    contract=validate_contract(json.loads((root/'contract.json').read_text()))
    budgets=json.loads((root/'budgets.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in contract['cases']:
        if args.case and case['id'] not in args.case:continue
        inputs=unit.make_inputs(case);expected=unit.reference(inputs)
        row=dict(case=case,stage='reference' if args.command=='reference' else args.launcher)
        if args.command=='check':
            options=dict(device='a5',backend='cce',block_dim=case['block_dim'],
                         launcher=args.launcher,out_dir=args.output/case['id'],timeout=60.,board=None)
            actual=unit.execute(inputs,options)
            row.update(unit.compare(inputs,actual,expected,budgets))
            row['execution_evidence']=options.get('_execution_evidence',[])
        else:
            row.update(unit.compare(inputs,unit.host_reference(inputs),expected,budgets))
        (args.output/(case['id']+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        rows.append(row)
        print(case['id'],row['passed'],flush=True)
    assert rows,'no selected cases'
    result=dict(complete=True,passed=all(r['passed'] for r in rows),cases=rows,
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in root.rglob('*.py')})
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
