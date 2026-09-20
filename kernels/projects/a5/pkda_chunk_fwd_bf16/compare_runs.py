"""Recompute fixed budgets and cross-bd bytes from native verifier receipts."""
import argparse
import json
from pathlib import Path


def compare(roots):
    if len(roots)!=4:raise ValueError('provide all four independent bd result directories')
    runs={};rows=[]
    for root in roots:
        summary=json.loads((root/'summary.json').read_text())
        build=json.loads((root/'compile.json').read_text())
        boundaries=json.loads((root/'boundaries.json').read_text())
        fp32=json.loads((root/'fp32-full-before-after.json').read_text())
        assert summary['complete'] and summary['passed'] and build['complete']
        assert len(build['entries'])==11 and sum(e['family']=='BF16' for e in build['entries'])==6
        bd=build['block_dim'];assert bd in (1,2,3,4) and bd not in runs
        assert len(summary['cases'])==89 and boundaries['passed'] and len(boundaries['cases'])==43
        assert all(c['passed'] for c in boundaries['cases']) and fp32['passed'] and all(fp32['bytes_equal'].values())
        cases={c['case']['id']:c for c in summary['cases']};assert len(cases)==89
        for case in cases.values():
            assert case['passed'] and all(case['input_unchanged'].values()) and all(case['poison_all_written'].values())
            assert case['public_equals_staged']
            assert set(case['host_operations'])<={'aten.empty.memory_format','aten.view.default'}
            assert case['control_readbacks'] and all(r['kind']=='control_metadata' for r in case['control_readbacks'])
            for name in ('o','final_state','final_A_state'):
                limit=1e-4 if name=='final_A_state' else min(.01,3*min(case['floors'][name].values()))
                assert case['budgets'][name]==limit
                for oracle in ('A','B'):assert case['errors'][oracle][name]['relative_l2']<=limit
        runs[bd]=cases
        rows.append(dict(block_dim=bd,cases=len(cases),boundary_checks=len(boundaries['cases']),vendors=len(build['entries']),
                         max_relative_l2={oracle:{name:max(c['errors'][oracle][name]['relative_l2'] for c in cases.values()) for name in ('o','final_state','final_A_state')} for oracle in ('A','B')}))
    cross=[]
    for bd,cases in sorted(runs.items()):
        assert set(cases)==set(runs[1])
        for name,case in cases.items():
            base=runs[1][name]
            for field in ('input_sha256','stage_sha256','output_sha256'):
                assert case[field]==base[field],(bd,name,field)
        cross.append(dict(block_dim=bd,cases=89,input_bytes_equal=True,all19_stage_bytes_equal=True,public_output_bytes_equal=True))
    return dict(passed=True,cases=356,boundary_checks=172,per_block_dim=sorted(rows,key=lambda r:r['block_dim']),cross_block_dim=cross,
                criteria='Both CPU FP32 references; o/state<=1e-2 AND<=3F for each case; ATK<=1e-4. No model or historical qualification used.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('roots',type=Path,nargs=4);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.roots);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
