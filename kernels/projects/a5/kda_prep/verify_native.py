"""BF-07 native acceptance: the complete Kimi workload is the first launch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(os.environ.get('BF07_UNIT_ROOT', Path(__file__).resolve().parent)).resolve()
REPO = ROOT.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predecessors(chunk, auto, decode):
    receipt = json.loads((ROOT / 'baseline/source.json').read_text())
    result = []
    for name, actual in (('chunk', chunk), ('autograd', auto), ('fused_recurrent', decode)):
        path = ROOT / 'baseline' / (name + '.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['files'][path.name]['sha256']
        module = types.ModuleType('ascend_fla.ops.kda._bf07_before_' + name)
        module.__package__ = 'ascend_fla.ops.kda'
        module.__file__ = actual.__file__
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        result.append(module)
    old, old_auto, old_decode = result
    old._compiled_chain = chunk._compiled_chain
    old_auto._prepare_inputs = old._prepare_inputs
    old_auto.chunk_kda_fwd = old.chunk_kda_fwd
    old_auto.chunk_kda_fwd_with_caches = old.chunk_kda_fwd_with_caches
    old_decode._prepare_inputs = old._prepare_inputs
    old_decode._compiled = decode._compiled
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--mode', choices=('full',), default='full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('BF07_EXTERNAL_DEVICE_LOCK') == '1'
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu
    import ascriptor
    from ascend_fla.ops.kda import chunk, chunk_bwd, fused_recurrent, prepare
    from ascend_fla.ops.kda import autograd as auto
    from ascend_fla.runtime.compile import compile_kernel
    import native_environment

    assert platform.python_version() == '3.12.14'
    assert torch.__version__ == '2.12.0+cu130' and torch_npu.__version__ == '2.12.0'
    assert ascriptor.__version__ == '0.1.0'
    assert Path(ascriptor.__file__).is_relative_to(Path(os.environ['BF07_NATIVE_ROOT']) / 'library')
    torch.set_num_threads(1)
    environment = native_environment.collect()
    environment['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob('*.py'))}
    environment['wrappers_sha256'] = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (chunk, chunk_bwd, auto, fused_recurrent)}

    def write(name, value):
        def finite_json(item):
            if isinstance(item,float) and not math.isfinite(item):
                return None
            if isinstance(item,dict):
                return {k:finite_json(v) for k,v in item.items()}
            if isinstance(item,(list,tuple)):
                return [finite_json(v) for v in item]
            return item
        target = args.output / (name + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(finite_json(dict(environment=environment, **value)), indent=2, allow_nan=False) + '\n')
        temporary.replace(target)

    write('environment', {})
    bd = args.block_dim
    decode_bd = int(os.environ.get('BF07_DECODE_BD', str(bd)))
    plan = [('prep_chunk', e, bd) for e in chunk._prep_runtime().kernels('chunk').values()]
    plan += [('prep_decode', e, decode_bd) for e in chunk._prep_runtime().kernels('decode').values()]
    plan += [('layout', e, bd) for e in chunk._layout_runtime().kernels().values()]
    plan += [('forward', e, bd) for e in chunk.kda_fwd_kernels().values()]
    plan += [('backward', e, bd) for e in chunk_bwd.kda_bwd_kernels().values()]
    plan += [('decode', fused_recurrent._native_kernel(d), decode_bd) for d in (torch.bfloat16, torch.float32)]
    assert len(plan) == 50 and sum(f == 'backward' for f, _, _ in plan) == 9
    builds = []
    for family, entry, entry_bd in plan:
        print('COMPILE_START', family, entry.name, entry_bd, flush=True)
        start = time.monotonic()
        op = compile_kernel(entry, device='a5', block_dim=entry_bd, backend='cce')
        builds.append(dict(family=family, entry=entry.name, block_dim=entry_bd, signature=op.signature,
                           seconds=time.monotonic()-start,
                           vendor_files={str(p.relative_to(op.vendor_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in op.vendor_dir.rglob('*') if p.is_file() and p.suffix in ('.so', '.o', '.json')}))
        write('compile', dict(complete=len(builds) == 50, entries=builds))
    prepare(block_dim=bd, backward=True, decode=True, decode_block_dim=decode_bd)
    torch.npu.set_device(0)
    check = load('_bf07_existing_checks', ROOT.parent / 'kda_layout/native_support.py')
    import native_audit
    native_audit.REPO = REPO
    environment['audit_driver_sha256'] = hashlib.sha256(Path(native_audit.__file__).read_bytes()).hexdigest()
    check.Audit = native_audit.Audit
    precision = load('_bf07_precision', ROOT / 'ref/calibrate.py')
    old, old_auto, old_decode = predecessors(chunk, auto, fused_recurrent)
    real, reference = check.reference_modules()
    fla_path = Path(os.environ['FLA_KDA_NAIVE'])
    assert hashlib.sha256(fla_path.read_bytes()).hexdigest() == '60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016'
    fla = load('_bf07_fla', fla_path)
    limits = json.loads((ROOT / 'budgets.json').read_text())['limits']
    flags = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    options = dict(block_dim=bd, layout_device='npu')
    case = dict(id='full_kimi_t4096_all_flags', B=1, T=4096, H=32, HV=32, K=128,
                raw_dtype='bf16', parameter_dtype='f32', flags=[True, True, True])
    try:
        print('GENERATE_FULL_KIMI', flush=True)
        x = real.make_inputs(B=1, H=32, HV=32, C=64, span=46, want_grads=True)
        generator = torch.Generator().manual_seed(7007)
        raw = {name: torch.randn((1,4096,32,128), generator=generator).bfloat16() for name in ('q','k')}
        raw['g'] = (torch.randn((1,4096,32,128), generator=generator)*.1).bfloat16()
        raw['beta'] = torch.randn((1,4096,32), generator=generator).bfloat16()
        raw['A_log'] = torch.linspace(-.2, .2, 32)
        raw['dt_bias'] = torch.linspace(-.3, -.1, 32*128)
        for name in ('v','h0','do','dht'):
            raw[name] = x[name]
        dev = {name: value.npu() for name, value in raw.items()}
        print('FULL_WORKLOAD_PUBLIC_CACHED_BACKWARD', flush=True)
        with torch.no_grad(), check.instrument() as (audit, launches):
            plain = auto.chunk_kda(dev['q'], dev['k'], dev['v'], dev['g'], dev['beta'],
                A_log=dev['A_log'], dt_bias=dev['dt_bias'], initial_state=dev['h0'],
                output_final_state=True, **flags, **options)
            prepared = chunk._prepare_kernel_inputs(dev['q'], dev['k'], dev['g'], dev['beta'],
                A_log=dev['A_log'], dt_bias=dev['dt_bias'], **flags, block_dim=bd)
            q, k, g, beta = prepared
            o, state, caches = chunk.chunk_kda_fwd_with_caches(q,k,dev['v'],g,beta,
                initial_state=dev['h0'], **options)
            beta_bf16 = chunk._layout_runtime().cast(beta, torch.bfloat16, block_dim=bd)
            grads = chunk_bwd.chunk_kda_bwd(q=q,k=k,v=dev['v'],beta=beta_bf16,
                do=dev['do'],dht=dev['dht'],caches=caches,**options)
        torch.npu.synchronize()
        write('full-audit', dict(operations=audit.report(), unexpected=audit.unexpected(), launches=launches))
        got = check.cpu(dict(o=o,final_state=state,**{'cache_'+n:v for n,v in caches.items()},**grads))
        candidate_prep = check.cpu(prepared)
        host_cpu = old._prepare_inputs(*(raw[n] for n in ('q','k','g','beta')),
            A_log=raw['A_log'],dt_bias=raw['dt_bias'],**flags)
        high = (precision.high_precision_norm(raw['q']), precision.high_precision_norm(raw['k']),
                precision.high_precision_gate(raw['g'],raw['A_log'],raw['dt_bias']),
                precision.high_precision_beta(raw['beta']))
        prep = {}
        for name,key,actual,cpu,fp64 in zip(('q','k','g','beta'),
                ('norm:bf16_bf16','norm:bf16_bf16','gate:bf16_f32_f32','beta:bf16'),
                candidate_prep,host_cpu,high):
            to64 = precision.metrics(actual,fp64)
            to_host = precision.metrics(actual,cpu)
            budget = limits[key]
            passed = (to64['finite_pairs'] == actual.numel()
                and to64['relative_l2'] is not None and to64['relative_l2'] <= budget['relative_l2']
                and to64['max_relative_nonzero'] <= budget['max_relative_nonzero']
                and to64['zero_reference_nonzero_actual'] == 0)
            if actual.dtype == torch.bfloat16:
                passed = passed and all(m['rounded_reference_over_one_ulp'] == 0 for m in (to64,to_host))
            prep[name] = dict(to_fp64=to64,to_predecessor_cpu_fp32=to_host,budget=budget,passed=passed)
            write('full-prep',dict(case=case,outputs=prep))
        del high
        with torch.no_grad():
            old_plain = old_auto.chunk_kda(dev['q'],dev['k'],dev['v'],dev['g'],dev['beta'],
                A_log=dev['A_log'],dt_bias=dev['dt_bias'],initial_state=dev['h0'],
                output_final_state=True,**flags,**options)
            old_prep = old._prepare_inputs(*(dev[n] for n in ('q','k','g','beta')),
                A_log=dev['A_log'],dt_bias=dev['dt_bias'],**flags)
            old_o,old_state,old_caches = old.chunk_kda_fwd_with_caches(
                old_prep[0],old_prep[1],dev['v'],old_prep[2],old_prep[3],initial_state=dev['h0'],**options)
            old_grads = chunk_bwd.chunk_kda_bwd(q=old_prep[0],k=old_prep[1],v=dev['v'],
                beta=old_prep[3].bfloat16(),do=dev['do'],dht=dev['dht'],caches=old_caches,**options)
        torch.npu.synchronize()
        before = check.cpu(dict(o=old_o,final_state=old_state,**{'cache_'+n:v for n,v in old_caches.items()},**old_grads))
        row = dict(case=case,outputs={n:check.digest(t) for n,t in got.items()},
            unchanged={n:check.digest(dev[n])==check.digest(value) for n,value in raw.items()},
            plain_cached_exact=[check.digest(a)==check.digest(b) for a,b in zip(plain,(o,state))],
            finite={n:bool(t.isfinite().all()) for n,t in got.items()},prep_passed=all(p['passed'] for p in prep.values()),
            predecessor_npu_difference={n:check.metrics(t,before[n],real.BUDGET.get(n,.05)) for n,t in got.items()},
            cpu_references={},passed=False)
        write('full',row)
        print('CPU_FP32_REFERENCES',flush=True)
        for prep_label,prep_values in (('predecessor_cpu_fp32',host_cpu),('actual_native_preparation',candidate_prep)):
            oracle_x = dict(zip(('q','k','g','beta'),prep_values))
            oracle_x.update(v=raw['v'],h0=raw['h0'],do=raw['do'],dht=raw['dht'])
            for label,function in (('independent',real.kda_recurrent_ref),('fla',fla.naive_recurrent_kda)):
                fwd = reference.independent_reference(oracle_x) if label=='independent' else reference.fla_reference(oracle_x)
                metrics = {n:reference.metrics(got[n],fwd[n]) for n in ('o','final_state')}
                gradient_ref = check.gradients(oracle_x,function)
                metrics['gradients'] = {n:check.metrics(got[n],t,real.BUDGET[n]) for n,t in gradient_ref.items()}
                metrics['per_head_chunk'] = [dict(chunk=c,head=h,**reference.metrics(
                    got['o'][:,c*64:(c+1)*64,h],fwd['o'][:,c*64:(c+1)*64,h])) for c in range(64) for h in range(32)]
                row['cpu_references'][prep_label+'_'+label] = metrics
                write('full',row)
        row['passed'] = (row['prep_passed'] and all(row['unchanged'].values()) and all(row['finite'].values())
            and all(row['plain_cached_exact']) and not audit.unexpected()
            and all(m[n]['passed'] for m in row['cpu_references'].values() for n in ('o','final_state'))
            and all(g['passed'] for m in row['cpu_references'].values() for g in m['gradients'].values())
            and all(t['passed'] for m in row['cpu_references'].values() for t in m['per_head_chunk']))
        write('full',row)
        write('summary',dict(complete=True,passed=row['passed'],cases=[case],scope='full workload only'))
        if not row['passed']:
            torch.save(dict(raw=raw,actual=got,predecessor=before,prepared=candidate_prep),args.output/'failure.private.pt')
            raise AssertionError('full BF-07 workload did not meet acceptance; inspect individual receipts')
        print('FULL_NATIVE_PASS',bd,flush=True)
    except BaseException as error:
        if 'got' in locals():
            torch.save(dict(raw=raw,actual=got,prepared=locals().get('candidate_prep')),
                       args.output/'failure.private.pt')
        write('failure',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
