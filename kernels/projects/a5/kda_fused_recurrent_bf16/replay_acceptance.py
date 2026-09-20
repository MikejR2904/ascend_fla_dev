"""Independent replay of the fixed prefill budgets and measured timings."""
import argparse
import json
import math
from pathlib import Path
import statistics

from aggregate import check_metric


def read(path):
    return json.loads(path.read_text())


def build_and_baseline(folder, bd):
    build = read(folder/'compile.json')
    assert build['complete'] and len(build['entries']) == 17
    assert len({x['entry'] for x in build['entries']}) == 17
    assert {x['entry'] for x in build['entries'] if x['family']=='backward'} == {
        'scan_fused_kernel', 'inverse_mm_bounded_kernel', 'inverse_epilogue_kernel',
        'inverse_dainv_kernel', 'inverse_dakk_fused_kernel', 'finalize_pre_stable_kernel',
        'finalize_pair_kernel', 'finalize_post_stable_kernel', 'finalize_reduce_kernel'}
    for row in build['entries']:
        assert row['block_dim'] == (bd if row['family'] in ('decode','original_decode') else min(bd,4))
    assert read(folder/'baseline.json')['public_sha256'] == '3435560723662531a409ad6a208283dc3acbf01a7adc47c361ef22f2645caf2e'
    env = read(folder/'environment.json')
    assert env['source_sha256']['kernels/step.py'] == 'c4e341a99c005259e0c9bfb59541ba192ac177d4da648862c1a68f7c8b3cea14'


def replay_prefill(root, block_dims):
    outputs, runs = {}, []
    for bd in block_dims:
        folder = root/f'prefill-bd{bd}'
        build_and_baseline(folder, bd)
        summary = read(folder/'summary.json')
        assert summary['complete'] and summary['passed'] and len(summary['cases']) == 6
        for row in summary['cases']:
            assert row['passed'] and all(row['input_unchanged'].values())
            assert all(row['poison_all_written'])
            assert set(row['host_operations']) <= {'aten.empty.memory_format','aten.view.default'}
            for comp in list(row['comparison'].values())+row['per_head']:
                limit=min(.01,3*comp['output_floor']) if row['dtype']=='bfloat16' else 1e-5
                check_metric(comp['o'],limit)
                check_metric(comp['final_state'],1e-5)
            if row['dtype']=='float32':
                assert row['original_fp32_bitwise'] == [True, True]
        boundary = read(folder/'boundaries.json')
        assert boundary['passed'] and len(boundary['cases']) == 57
        for row in boundary['cases']:
            for comp in row.get('comparison',{}).values():
                check_metric(comp['o'],min(.01,3*comp['output_floor']) if 'output_floor' in comp else 1e-5)
                check_metric(comp['final_state'],1e-5)
            if row['kind']=='original_unrounded_fp32':
                assert row['original_bitwise']
            if row.get('original_fp32_bitwise') is not None:
                assert row['original_fp32_bitwise']
        result = read(folder/'prefill.json')
        assert result['passed'] and len(result['cases']) == 21
        worst = dict(local_o=0., local_state=0., global_o=0., global_state=0., prefix_o=0., suffix_o=0.)
        for index, row in enumerate(result['cases']):
            assert row['passed'] and all(row['input_unchanged'].values())
            assert row['sixteen_vs_single_bitwise']
            assert row['segment_hashes']['16'] == row['segment_hashes']['1']
            assert len(row['segment_hashes']['16']) == 4
            if 'span' in row['case']:
                assert row['gate_span']['cpu'] == row['gate_span']['npu'] == row['case']['span']
            start = row['case']['prefix']
            steps = row['decode_steps']
            assert len(steps) == 68
            assert {(x['begin'],x['tokens']) for x in steps} == {
                (t,w) for w in (1,16) for t in range(start,start+64,w)}
            for step in steps:
                assert step['input_state_unchanged'] and step['passed']
                assert set(step['host_operations']) <= {'aten.empty.memory_format','aten.view.default'}
                assert set(step['comparison']) == {'A','B'}
                for comp in step['comparison'].values():
                    check_metric(comp['o'],min(.01,3*comp['output_floor']))
                    check_metric(comp['final_state'],1e-5)
                    worst['local_o'] = max(worst['local_o'],comp['o']['relative_l2'])
                    worst['local_state'] = max(worst['local_state'],comp['final_state']['relative_l2'])
            assert set(row['comparison']) == {'A','B'}
            for comp in row['comparison'].values():
                check_metric(comp['oneshot_o'],math.inf)
                check_metric(comp['oneshot_state'],math.inf)
                ob = min(.01,3*comp['oneshot_o']['relative_l2'])
                sb = min(.01,3*comp['oneshot_state']['relative_l2'])
                assert comp['o_budget']==ob and comp['state_budget']==sb
                assert set(comp['metrics']) == {'o','final_state','prefix_o','suffix_o','prefix_state'}
                for key, metric in comp['metrics'].items():
                    check_metric(metric,sb if key in ('final_state','prefix_state') else ob)
                    field = {'o':'global_o','final_state':'global_state'}.get(key,key)
                    if field in worst:
                        worst[field] = max(worst[field],metric['relative_l2'])
            pair = (row['output_sha256'],row['oneshot_sha256'],row['segment_hashes'])
            if index in outputs:
                assert outputs[index]==pair, ('cross-bd prefill mismatch',bd,index)
            else:
                outputs[index]=pair
        runs.append(dict(block_dim=bd,chains=21,local_decode_calls=21*68,worst=worst))
    return dict(passed=True,block_dims=block_dims,runs=runs,cross_bd_bitwise=True,
                complete=set(block_dims)=={1,2,4,8,16,28})


def replay_performance(root):
    folder = root/'perf-bd4'
    build_and_baseline(folder,4)
    summary=read(folder/'summary.json')
    assert summary['complete'] and summary['passed']
    result = read(folder/'performance.json')
    assert result['passed'] and result['warmup']==5 and result['repeat']==20 and result['rounds']==3
    assert len(result['cases'])==4
    rows=[]
    for case in result['cases']:
        assert case['warm_signature_cache_check']
        for comp in case['correctness'].values():
            for oracle in comp.values():
                check_metric(oracle['o'],min(.01,3*oracle['output_floor']))
                check_metric(oracle['final_state'],1e-5)
        comparisons={}
        for field,label,prefix in (('original_fp32_rounds','original_fp32','original_fp32'),
                                   ('torch_npu_rounds','torch_npu','torch_npu')):
            rounds=case[field];assert len(rounds)==3
            rates=[];times=[]
            for row in rounds:
                for timing in (value for value in row.values() if isinstance(value,dict)):
                    for key in ('synchronized_us','enqueue_wall_us'):
                        assert len(timing[key])==20 and all(math.isfinite(x) and x>0 for x in timing[key])
                    assert statistics.median(timing['synchronized_us'])==timing['median_us']
                baseline=(row[prefix+'_before']['median_us']+row[prefix+'_after']['median_us'])/2
                candidate=row['native_bf16']['median_us']
                rate=baseline/candidate
                assert rate==row['bf16_speedup_vs_'+label]
                rates.append(rate);times.append(dict(baseline_us=baseline,candidate_us=candidate))
            comparisons[label]=dict(round_speedups=rates,round_medians=times,
                                    median_speedup=statistics.median(rates))
        rows.append(dict(shape=case['shape'],comparisons=comparisons))
    return dict(passed=True,cases=rows,fixed_cost=result['fixed_cost'],
                speed_gate='None; report all measured results, including slowdowns.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--block-dims',type=int,nargs='+',default=[1,2,4,8,16,28])
    parser.add_argument('--without-performance',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(prefill=replay_prefill(args.root,args.block_dims))
    if not args.without_performance:
        result['performance']=replay_performance(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
