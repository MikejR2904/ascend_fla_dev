"""Replay native receipt budgets and cross-block-dimension byte comparisons."""
import argparse
import json
import math
from pathlib import Path


def check_metric(metric, limit):
    assert metric['finite'] is True
    for name in ('relative_l2','max_abs'):
        assert math.isfinite(metric[name]) and metric[name] >= 0
    assert metric['relative_l2'] <= limit, (metric,limit)


def replay(root, block_dims):
    reference = {}
    boundary_reference = {}
    reports = []
    for bd in block_dims:
        mode = 'suite' if bd in (1,2,4) else 'boundaries'
        folder = root/f'{mode}-bd{bd}'
        build = json.loads((folder/'compile.json').read_text())
        assert build['complete'] and len(build['entries']) == 17
        assert len({x['entry'] for x in build['entries']}) == 17
        assert {x['entry'] for x in build['entries'] if x['family']=='backward'} == {
            'scan_fused_kernel','inverse_mm_bounded_kernel','inverse_epilogue_kernel',
            'inverse_dainv_kernel','inverse_dakk_fused_kernel','finalize_pre_stable_kernel',
            'finalize_pair_kernel','finalize_post_stable_kernel','finalize_reduce_kernel'}
        for entry in build['entries']:
            assert entry['block_dim'] == (bd if entry['family'] in ('decode','original_decode') else min(bd,4))
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['passed']
        assert len(summary['cases']) == (136 if bd in (1,2,4) else 2)
        observed = set()
        worst = dict(bf16_o=0.,fp32_o=0.,state=0.,per_head_state=0.)
        for row in summary['cases']:
            key = (row['case']['id'],row['dtype'])
            assert key not in observed
            observed.add(key)
            assert row['passed'] and all(row['input_unchanged'].values()) and all(row['poison_all_written'])
            assert set(row['host_operations']) <= {'aten.empty.memory_format','aten.view.default'}
            assert set(row['comparison']) == {'A','B'}
            for comp in row['comparison'].values():
                limit = min(.01,3*comp['output_floor']) if row['dtype']=='bfloat16' else 1e-5
                check_metric(comp['o'],limit)
                check_metric(comp['final_state'],1e-5)
                field = 'bf16_o' if row['dtype']=='bfloat16' else 'fp32_o'
                worst[field] = max(worst[field],comp['o']['relative_l2'])
                worst['state'] = max(worst['state'],comp['final_state']['relative_l2'])
            if row['dtype']=='float32':
                assert row['original_fp32_bitwise'] == [True,True]
            assert {(x['batch'],x['head'],x['reference']) for x in row['per_head']} == {
                (bi,hi,ref) for bi in range(row['case']['B'])
                for hi in range(row['case']['H']*row['case']['G']) for ref in ('A','B')}
            for comp in row['per_head']:
                limit = min(.01,3*comp['output_floor']) if row['dtype']=='bfloat16' else 1e-5
                check_metric(comp['o'],limit);check_metric(comp['final_state'],1e-5)
                worst['per_head_state'] = max(worst['per_head_state'],comp['final_state']['relative_l2'])
            artifact = dict(inputs=row['input_sha256'],outputs=row['output_sha256'])
            if key in reference:
                assert artifact==reference[key],dict(block_dim=bd,case=key,cross_bd_mismatch=True)
            else:
                reference[key] = artifact
        boundary = json.loads((folder/'boundaries.json').read_text())
        assert boundary['passed'] and len(boundary['cases']) == 57
        numerical = 0
        for row in boundary['cases']:
            if 'comparison' in row:
                for comp in row['comparison'].values():
                    limit = min(.01,3*comp['output_floor']) if 'output_floor' in comp else 1e-5
                    check_metric(comp['o'],limit);check_metric(comp['final_state'],1e-5)
            if 'output_sha256' in row:
                key = json.dumps({k:row[k] for k in ('kind','B','T','H','HV','dtype','initial_state',
                                                    'tokens_per_call','flags') if k in row},sort_keys=True)
                if key in boundary_reference:
                    assert boundary_reference[key]==row['output_sha256'],dict(block_dim=bd,boundary=key)
                else:
                    boundary_reference[key] = row['output_sha256']
                numerical += 1
            if 'input_unchanged' in row:
                assert all(row['input_unchanged'].values())
            if row.get('original_fp32_bitwise') is not None:
                assert row['original_fp32_bitwise']
            if 'host_categories' in row:
                for item in row['host_categories']:
                    assert item['category'] in ('operator','registered_A2-44_legacy_exception_to_BF-07')
                    if item['category']=='operator':
                        assert item['operator'] in ('aten.empty.memory_format','aten.view.default')
            if row['kind']=='decode_chaining':
                assert row['bitwise']
            if row['kind']=='metadata_rejection':
                assert row['no_launch']
        reports.append(dict(block_dim=bd,numerical_cases=len(observed),additional_checks=57,
                            additional_hashed_cases=numerical,compile_entries=17,backward_entries=9,worst=worst))
    return dict(passed=True,block_dims=block_dims,
                complete_required_grid=set(block_dims)=={1,2,4,8,16,28},
                cross_bd_byte_identical=True,unique_numerical_cases=len(reference),
                unique_additional_hashed_cases=len(boundary_reference),runs=reports,
                scope='Grid and boundary receipts only; prefill, timing and models have separate acceptance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--block-dims',type=int,nargs='+',default=[1,2,4,8,16,28])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = replay(args.root,args.block_dims)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
