"""Recompute strict grid maxima, output equality and the audited host-op union.

Input consists of restored original receipt directories for every block_dim.
This reads every case. Selected full receipts retain each numerical worst case;
selection is for bounded publication and never substitutes for full validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def summarize(root, prefix, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries, files, groups, audits, picks = [], {}, {}, {}, set()
    raw_groups, raw_audits = [], []
    environment = None
    audit_sources = {}
    for bd in (1, 2, 3, 4):
        folder = root/f'{prefix}-bd{bd}'
        summary_path = folder/'summary.json'
        summary = json.loads(summary_path.read_text())
        assert summary['complete'] and summary['passed'] and summary['expected_cases'] == 1620
        assert summary['gradient_selections'] == 9 and len(summary['cases']) == 1620
        summaries.append(summary)
        environment = summary['environment']
        files[str(bd)] = dict(summary_sha256=digest(summary_path.read_bytes()), cases=1620)
        compile = json.loads((folder/'compile.json').read_text())
        assert compile['complete'] and len(compile['entries']) == 50
        assert sum(e['family'] == 'backward' for e in compile['entries']) == 9
        assert {e['block_dim'] for e in compile['entries']} == {bd}
        for item in summary['cases']:
            case = item['id']
            path = folder/'cases'/(case+'.json')
            raw = path.read_bytes()
            row = json.loads(raw)
            assert row['passed']
            assert row['all_gradients']['cache_count'] == 9
            assert row['all_gradients']['outputs'] == item['all_outputs']
            assert {k:v['outputs'] for k,v in row['selections'].items()} == item['selection_outputs']
            for grad, measurement in row['prep_gradients'].items():
                key = measurement['key']
                values = dict(relative_l2=measurement['to_fp64']['relative_l2'],
                    max_relative=measurement['to_fp64']['max_relative_nonzero'],
                    relative_l2_ratio=measurement['to_fp64']['relative_l2']/measurement['budget']['relative_l2_limit'],
                    max_relative_ratio=measurement['to_fp64']['max_relative_nonzero']/measurement['budget']['elementwise_relative_limit'])
                assert measurement['passed'] and values['relative_l2_ratio'] <= 1 and values['max_relative_ratio'] <= 1
                state = groups.setdefault(key, {})
                for metric, value in values.items():
                    if metric not in state or value > state[metric]['value']:
                        state[metric] = dict(value=value, block_dim=bd, case=case, gradient=grad,
                            raw_receipt_sha256=digest(raw))
                # Original raw metric objects can be recomputed from Git without
                # publishing each large full training receipt.
                if bd == 1:
                    fields=('elements','finite_pairs','relative_l2','max_relative_nonzero',
                        'zero_reference_nonzero_actual','rounded_reference_max_ulp','rounded_reference_over_one_ulp',
                        'rounded_reference_sign_differences')
                    raw_groups.append(dict(case=case,gradient=grad,key=key,raw_receipt_sha256=digest(raw),
                        to_fp64={k:measurement['to_fp64'][k] for k in fields},
                        to_old_cpu={k:measurement['to_old_cpu'][k] for k in fields},
                        ulp_to_fp64_distribution=measurement['ulp_to_fp64_distribution'],
                        ulp_to_old_distribution=measurement['ulp_to_old_distribution'],
                        bf16_over_one_locations=measurement['bf16_over_one_locations'],passed=measurement['passed']))
            for selection, record in [('all',row['all_gradients']), *row['selections'].items()]:
                assert not record['unexpected'] and not record['old_host_preparation']
                assert all(record['unchanged'].values()) and all(record['finite'].values())
                key = digest(json.dumps(record['audit'],sort_keys=True,separators=(',',':')).encode())
                table = audits.setdefault(key,dict(operations=record['audit'],uses_per_bd={str(b):0 for b in (1,2,3,4)}))
                table['uses_per_bd'][str(bd)] += 1
                audit_sources.setdefault(key,dict(block_dim=bd,case=case,selection=selection))
            if bd == 1:
                raw_audits.append(dict(case=case,table_by_selection={n:digest(json.dumps(v['audit'],sort_keys=True,separators=(',',':')).encode())
                    for n,v in [('all',row['all_gradients']),*row['selections'].items()]}))
    assert all(s['cases'] == summaries[0]['cases'] for s in summaries)
    assert len({v['summary_sha256'] for v in files.values()}) == 1
    # Publish six complete worst-case receipts, one L2 and one elementwise
    # maximum for each leaf family. Every group's exact maxima and all raw
    # numerical fields remain independently recomputable from the JSONL rows.
    for family in ('norm','gate','beta'):
        for metric in ('relative_l2_ratio','max_relative_ratio'):
            row=max((v[metric] for k,v in groups.items() if k.startswith(family+':')),key=lambda x:x['value'])
            picks.add((row['block_dim'],row['case']))
    union = {}
    for key,table in audits.items():
        assert len(set(table['uses_per_bd'].values())) == 1
        for op in table['operations']:
            identity=tuple(op.get(k) for k in ('category','file','line','operator'))
            target=union.setdefault(identity,dict(**{k:op.get(k) for k in ('category','file','line','operator')},count=0))
            target['count'] += op['count'] * sum(table['uses_per_bd'].values())
    report=dict(environment=environment,passed=True,task_complete=False,cases_per_bd=1620,gradient_selections=9,
        candidate_calls=58320,cross_bd_all_returned_hashes_equal=True,nine_caches_checked=True,
        original_summaries=files,groups=groups,audit_operation_union=list(union.values()),
        audit_tables=[dict(sha256=key,**value,example=audit_sources[key]) for key,value in audits.items()],
        selected_receipts=[dict(block_dim=b,case=c) for b,c in sorted(picks)],
        script_sha256=digest(Path(__file__).read_bytes()),
        frozen_budget_sha256=digest(Path(__file__).with_name('backward_budgets.json').read_bytes()))
    def save(name,value):
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    save('summary.json',report)
    # Original summary metadata and all case rows reconstruct the exact source
    # summaries. They are split solely to keep each committed text file bounded.
    metadata=dict(summaries[0]);metadata.pop('cases')
    save('original-summary-header.json',dict(environment=environment,original_metadata=metadata,
        original_summary_sha256=files['1']['summary_sha256'], original_key_order=list(summaries[0])))
    for i in range(0,1620,100):
        dictionary=[];positions={}
        def encode(value):
            if isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value):
                if value not in positions:positions[value]=len(dictionary);dictionary.append(value)
                return {'$sha256':positions[value]}
            if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
            if isinstance(value,list):return [encode(v) for v in value]
            return value
        encoded=encode(summaries[0]['cases'][i:i+100])
        save(f'output-hashes/part-{i//100:03d}.json',dict(environment=environment,sha256_dictionary=dictionary,cases=encoded))
    for i in range(0,len(raw_groups),1000):
        columns=list(raw_groups[0]);metric_fields=list(raw_groups[0]['to_fp64'])
        rows=[]
        for record in raw_groups[i:i+1000]:
            rows.append([list(record[k].values()) if k in ('to_fp64','to_old_cpu') else record[k] for k in columns])
        save(f'raw-metrics/part-{i//1000:03d}.json',dict(environment=environment,
            columns=columns,metric_fields=metric_fields,rows=rows))
    audit_dictionary=sorted(audits);audit_index={v:i for i,v in enumerate(audit_dictionary)}
    selections=list(raw_audits[0]['table_by_selection'])
    save('audit-uses.json',dict(environment=environment,table_dictionary=audit_dictionary,selections=selections,
        cases=[dict(case=row['case'],tables=[audit_index[row['table_by_selection'][n]] for n in selections]) for row in raw_audits]))
    samples=output/'selected-original';samples.mkdir()
    for b,c in sorted(picks):
        data=(root/f'{prefix}-bd{b}'/'cases'/(c+'.json')).read_bytes()
        # Preserve the complete original object and exact-byte digest in bounded
        # compact text. Verification reconstructs the original indent=2 bytes.
        save('selected-original/'+f'bd{b}-{c}.json',dict(environment=environment,
            original_bytes=len(data),original_sha256=digest(data),receipt=json.loads(data)))
    manifest={str(p.relative_to(output)):digest(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    save('manifest.json',dict(environment=environment,sha256=manifest))
    return dict(passed=True,original_cases_read=6480,selected_receipts=len(picks),
        metric_rows=len(raw_groups),audit_tables=len(audits),bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root,args.prefix,args.output)),flush=True)


if __name__ == '__main__':main()
