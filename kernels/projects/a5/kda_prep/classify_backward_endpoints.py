"""Classify measured four-column endpoints without creating an accuracy exemption.

This consumes actual native receipts. It neither runs kernels nor changes any
frozen budget. Native flushes and range observations are never CPU PASS claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def classify(root):
    root = Path(root)
    result = []; total_flush = total_unclassified = total_new_nan = 0
    for path in sorted((root/'four-columns').glob('*.json')):
        j = json.loads(path.read_text())
        gradients = j['gradients'] if isinstance(j['gradients'],dict) else {'dbeta':j['gradients']}
        for name, rows in gradients.items():
            counts = dict(elements=len(rows),native_flush=0,cpu_zero_candidate_zero=0,
                ordinary_cpu_normal=0,subnormal_candidate_nonzero=0,cpu_zero_candidate_nonzero=0,
                candidate_only_nan=0,nonfinite_observations=0,outside_literal_three_equalities=0,
                candidate_equals_cpu=0,candidate_equals_old_npu_differs_cpu=0,
                old_npu_differs_candidate_equals_cpu=0,zero_sign_differences_from_old_npu=0)
            located = []
            for row in rows:
                candidate, old, cpu = (row[k] for k in ('candidate','old_npu','cpu_fp32'))
                a,b = candidate['ieee_class'],cpu['ieee_class']
                nonfinite = any(row[k]['ieee_class'] in ('nan','positive_inf','negative_inf')
                                for k in ('candidate','old_npu','cpu_fp32'))
                classification = None
                if b == 'subnormal':
                    classification='native_flush' if a == 'zero' else 'subnormal_candidate_nonzero'
                elif b == 'zero':
                    classification='cpu_zero_candidate_zero' if a == 'zero' else 'cpu_zero_candidate_nonzero'
                elif b == 'normal':
                    classification='ordinary_cpu_normal'
                if classification:counts[classification] += 1
                for k in ('candidate_only_nan','candidate_equals_cpu','candidate_equals_old_npu_differs_cpu',
                          'old_npu_differs_candidate_equals_cpu'):
                    counts[k] += row[k]
                counts['nonfinite_observations'] += nonfinite
                unclassified = nonfinite and not row['candidate_equals_cpu'] and not row['candidate_equals_old_npu']
                counts['outside_literal_three_equalities'] += unclassified
                signed_zero = a == 'zero' and old['ieee_class'] == 'zero' and candidate['bits'] != old['bits']
                counts['zero_sign_differences_from_old_npu'] += signed_zero
                if classification in ('native_flush','subnormal_candidate_nonzero','cpu_zero_candidate_nonzero') or unclassified or row['candidate_only_nan']:
                    located.append(dict(index=row['index'],classification=classification,
                        outside_literal_three_equalities=unclassified,**{k:row[k] for k in ('candidate','old_npu','cpu_fp32','fp64')}))
            total_flush += counts['native_flush']; total_unclassified += counts['outside_literal_three_equalities']
            total_new_nan += counts['candidate_only_nan']
            result.append(dict(case=path.stem,output=name,counts=counts,locations=located,
                original_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert result
    return dict(environment=j['environment'],source_stage='postprocess actual native four-column receipts',
        authority='D-PM-56(2,3) and D-PM-48/50; unresolved equality membership remains unresolved',
        task_complete=False,cpu_correctness_pass=False,native_flush_elements=total_flush,
        outside_literal_three_equalities=total_unclassified,candidate_only_nan=total_new_nan,rows=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=classify(args.root)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('native_flush_elements','outside_literal_three_equalities','candidate_only_nan','cpu_correctness_pass')}))


if __name__=='__main__':main()
