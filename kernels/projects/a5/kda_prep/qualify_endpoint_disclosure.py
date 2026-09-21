"""Apply D-PM-52 disclosure scope to saved native outputs, never numerical PASS."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

OWNER = 'https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5753209528'


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def ordered(tensor):
    narrow = tensor.dtype == torch.bfloat16
    bits = tensor.contiguous().view(torch.int16 if narrow else torch.int32).long()
    offset = 32768 if narrow else 2147483648
    return torch.where(bits < 0, -bits-offset, bits)


def classify(route, case, actual, old, references):
    if route not in ('chunk', 'decode') or not case.startswith('nearzero_'):
        return dict(in_disclosure_class=False, reason='outside named boundary scope')
    if actual.dtype != old.dtype or actual.dtype not in (torch.bfloat16, torch.float32):
        return dict(in_disclosure_class=False, reason='unsupported output dtype')
    if route == 'chunk' and actual.dtype != torch.bfloat16:
        return dict(in_disclosure_class=False, reason='chunk output must be BF16')
    if set(references) != {'independent', 'fla'}:
        return dict(in_disclosure_class=False, reason='both CPU references required')
    tensors = [actual, old, *references.values()]
    if any(t.device.type != 'cpu' or t.shape != actual.shape or not bool(t.isfinite().all()) for t in tensors):
        return dict(in_disclosure_class=False, reason='shape, CPU or finite requirement')
    if any(t.dtype != torch.float32 for t in references.values()):
        return dict(in_disclosure_class=False, reason='CPU FP32 goldens required')
    maxima = {n: float(t.abs().max()) for n, t in references.items()}
    # BF16 -> FP32 is exact, including signed zero; decode uses FP32 ULP units.
    a, b = (actual.float(), old.float()) if route == 'decode' else (actual, old)
    ulp = int((ordered(a)-ordered(b)).abs().max())
    same = digest(actual) == digest(old)
    bound = 32 * torch.finfo(torch.float32).tiny
    within = all(v <= bound for v in maxima.values()) and (same or ulp <= (1 if route == 'chunk' else 4))
    return dict(in_disclosure_class=within, golden_max_both_references=maxima,
                golden_max_limit=bound, candidate_old_bitwise=same,
                candidate_old_ulp=ulp, ulp_dtype=str(a.dtype),
                reference_sha256={n: digest(t) for n, t in references.items()},
                actual_sha256=digest(actual), old_npu_sha256=digest(old))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensor-root', type=Path, required=True)
    parser.add_argument('--tensor-manifest', type=Path, required=True)
    parser.add_argument('--case-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    table = json.loads(args.case_table.read_text())
    manifest = json.loads(args.tensor_manifest.read_text())['files']
    seen, loaded, rows = set(), {}, []
    torch.set_num_threads(1)
    for entry in table['failed_slices']:
        route, label, scope, c, h = (entry[k] for k in ('route', 'case', 'scope', 'chunk', 'head'))
        key = (route, label, scope, c, h)
        if key in seen:
            continue
        seen.add(key)
        assert scope == 'head_chunk', 'D-PM-52 applies to head/chunk slices only'
        run = 'nearzero-observe-v1-chunk-bd1' if route == 'chunk' else 'nearzero-dpm51-v1-decode-bd1'
        relative = run + '/' + label + '.private.pt'
        path = args.tensor_root / relative
        if (route, label) not in loaded:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[relative]['sha256']
            loaded[route, label] = torch.load(path, map_location='cpu', weights_only=True)
        saved = loaded[route, label]
        refs = saved['reference'] if route == 'chunk' else saved['native_preparation_reference']
        idx = (slice(None), slice(c*64, (c+1)*64), h)
        result = classify(route, label, saved['actual']['o'][idx], saved['before']['o'][idx],
                          {n: refs[n]['o'][idx] for n in ('independent', 'fla')})
        rows.append(dict(route=route, case=label, scope=scope, chunk=c, head=h,
                         block_dims=entry['block_dims'], native_tensor_file=relative,
                         native_tensor_sha256=manifest[relative]['sha256'],
                         numerical_acceptance='failed', **result))
    summary = {}
    for route in ('chunk', 'decode'):
        selected = [x for x in rows if x['route'] == route]
        summary[route] = dict(locations=len(selected),
                             disclosed_locations=sum(x['in_disclosure_class'] for x in selected),
                             cases=sorted({x['case'] for x in selected}))
    complete = len(rows) == 188 and all(x['in_disclosure_class'] for x in rows)
    report = dict(decision='D-PM-52', owner_comment=OWNER, user_approved=True,
                  disclosure_scope_verified=complete, numerical_acceptance_passed=False,
                  meaning='No regression is not CPU correctness PASS. Original numerical failures remain unchanged.',
                  outside_class_failures_remain_failures=True,
                  source_case_table_sha256=hashlib.sha256(args.case_table.read_bytes()).hexdigest(),
                  classifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, device='cpu'),
                  cross_bd_basis='Original classification/metrics and actual output hashes agree across all listed block dimensions; retained bd1 tensors used for this offline scope audit.',
                  summary=summary, locations=rows)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(disclosure_scope_verified=complete, locations=len(rows), numerical_acceptance_passed=False)))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
