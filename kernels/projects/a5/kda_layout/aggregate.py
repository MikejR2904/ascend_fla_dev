"""Independently recompute FMT-02 byte/accuracy predicates and cross-bd identity."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    roots = [root / f'suite-bd{bd}' for bd in (1,2,3,4)]
    common = None
    totals = []
    for bd, path in zip((1,2,3,4),roots):
        summary=json.loads((path/'summary.json').read_text())
        assert summary['complete'] and summary['passed'] and summary['scope']=='suite'
        compile=json.loads((path/'compile.json').read_text())
        assert compile['complete'] and len(compile['entries'])==22
        assert sum(e['family']=='backward' for e in compile['entries'])==9
        assert all(e['block_dim']==bd and e['vendor_files'] for e in compile['entries'])
        assert len(summary['cases'])==116
        hashes={}
        for item in summary['cases']:
            receipt=path/item['receipt']
            assert hashlib.sha256(receipt.read_bytes()).hexdigest()==item['receipt_sha256']
            row=json.loads(receipt.read_text())
            assert row['passed'] and row['block_dim']==bd
            assert all(row['input_unchanged'].values()) and all(row['plain_cached_exact']) and row['all_outputs_finite']
            for name,sha in row['input_sha256'].items():hashes[(row['case']['id'],'input',name)]=sha
            for name,value in row['bitwise'].items():
                assert value['sha256']==value['before_sha256'] and value['passed']
                hashes[(row['case']['id'],name)]=value['sha256']
            assert len(row['bitwise'])==19
            assert set(row['cpu_references'])=={'independent','fla'}
            for oracle in row['cpu_references'].values():
                for name in ('o','final_state'):
                    m=oracle[name]
                    assert m['finite'] and m['allclose'] and m['relative_l2']<=.05
                for m in oracle['per_head_chunk']:
                    assert m['finite'] and m['allclose'] and m['relative_l2']<=.05
                assert len(oracle['gradients'])==6
                for name,m in oracle['gradients'].items():
                    budget={'dk':.15,'dg':.25}.get(name,.05)
                    assert m['budget']==budget and m['finite'] and m['relative_l2']<=budget
            audit=json.loads((path/(row['case']['id']+'-audit.json')).read_text())
            assert not audit['unexpected'] and all(l['poisoned'] for l in audit['launches'])
        for name,count in (('leaf',31),('autograd_views',5),('gates',6),('decode_audit',2),('stable_contracts',24)):
            extra=json.loads((path/(name+'.json')).read_text())
            assert extra['passed'] and len(extra['cases'])==count and all(r['passed'] for r in extra['cases'])
            for row in extra['cases']:
                for output,value in row.get('comparison',{}).items():
                    assert value['sha256']==value['before_sha256'] and value['passed']
                    hashes[(name,row.get('family',''),row.get('id',''),row.get('path',''),row.get('span',''),output)]=value['sha256']
        if common is None:common=hashes
        else:assert hashes==common, f'cross-bd mismatch at bd{bd}'
        totals.append(dict(block_dim=bd,full_chain_cases=len(summary['cases']),hashed_outputs=len(hashes)))
    return dict(passed=True,block_dims=totals,cross_block_dim_bitwise=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipts',type=Path)
    print(json.dumps(verify(p.parse_args().receipts),indent=2))
