"""Rebuild fixed scan inputs from a public seed, verify hashes, then replay.

Run the original full workload first with this source identity and hold the
native verifier's external device locks. Historic raw tensors remain private;
this script retains every newly observed result without claiming a scan repair.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='plain JSON seed-manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-label', choices=('dev-A', 'dev-B'), required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'retain previous evidence'
    manifest = json.loads(args.bundle.read_bytes())
    assert manifest['schema'] == 'fmt02.scan-seeded-inputs/1'
    sys.argv = ['verify_native.py', '--block-dim', '4', '--mode', 'compile',
                '--output', str(args.output)]
    import verify_native
    verify_native.main()  # all 22 vendors, including every backward entry
    import torch
    import native_support as check
    from ascend_fla.ops.kda import chunk, chunk_bwd
    torch.npu.set_device(0)

    def sha(tensor):
        tensor = check.cpu(tensor)
        return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()

    generator = manifest['generator']
    path = check.REPO / generator['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == generator['sha256']
    real = check.load_file('_fmt02_seed_generator', path)
    x = getattr(real, generator['entry'])(**generator['parameters'])
    reproduction = {'public_inputs': {n: dict(actual=sha(t), expected=manifest['public_inputs'][n]['sha256'])
                                      for n, t in x.items()}}
    receipt = args.output / 'input-reproduction.json'
    receipt.write_text(json.dumps(reproduction, indent=2) + '\n')
    assert all(r['actual'] == r['expected'] for r in reproduction['public_inputs'].values()), 'seed input hash mismatch; see retained receipt'
    dev = {n: t.npu() for n, t in x.items()}
    _, _, caches = chunk.chunk_kda_fwd_with_caches(
        **{n: dev[n] for n in ('q', 'k', 'v', 'g', 'beta')},
        initial_state=dev['h0'], block_dim=4, layout_device='npu')
    b, hv, c, k, v, t = (manifest['scalars'][n] for n in ('B', 'HV', 'C', 'K', 'V', 'T'))
    gate_source = caches['g_cumsum'].view(-1)[63*hv*k:]
    g_last = chunk._layout_runtime().move(gate_source, (b,c,hv,1,k),
        (t*hv*k,64*hv*k,k,0,1), block_dim=4).view(b,c,hv,k)
    inputs = {n: caches[n] for n in ('kg', 'qg', 'w', 'Aqk', 'v_new')}
    inputs.update(g_last=g_last, grad_out=dev['do'], dht=dev['dht'].view(b,hv,k//2,2*v))
    reproduction['scan_inputs'] = {n: dict(actual=sha(tensor), expected=manifest['scan_inputs'][n]['sha256'])
                                   for n, tensor in inputs.items()}
    receipt.write_text(json.dumps(reproduction, indent=2) + '\n')
    assert all(r['actual'] == r['expected'] for r in reproduction['scan_inputs'].values()), 'rebuilt scan input hash mismatch; see retained receipt'
    op = chunk_bwd._compiled_chain('a5', 4, 'stable')['scan_fused']
    scalars = {n: manifest['scalars'][n] for n in op.scalar_names}
    repeats = []
    for _ in range(12):
        outputs = {n: torch.empty(s['shape'], dtype=getattr(torch, s['dtype']), device='npu')
                   for n, s in manifest['scan_outputs'].items()}
        with check.instrument(audit=False):
            op(inputs, scalars, outputs)
        torch.npu.synchronize()
        repeats.append(check.cpu(outputs))
    # No new layout kernel ran inside this direct scan repeat loop.
    input_unchanged = all(sha(tensor) == manifest['scan_inputs'][n]['sha256'] for n, tensor in inputs.items())
    hashes = [{n: sha(tensor) for n, tensor in row.items()} for row in repeats]
    distributions = {n: dict(collections.Counter(row[n] for row in hashes)) for n in hashes[0]}
    widening = []
    for index, replay in enumerate(repeats):
        cpu = replay['dh0']
        device = cpu.npu()
        with check.instrument(audit=False):
            native = chunk._layout_runtime().cast(device, torch.float32, block_dim=4)
        old = device.float()
        torch.npu.synchronize()
        widening.append(dict(sample=index, bf16_sha256=sha(cpu), cpu_fp32_sha256=sha(cpu.float()),
                             old_fp32_sha256=sha(old), new_fp32_sha256=sha(native),
                             input_unchanged=sha(device) == sha(cpu)))
    passed = input_unchanged and all(len(distributions[n]) == 1 for n in ('dAqk', 'dh', 'dv'))
    passed = passed and all(row['input_unchanged'] and
                           row['cpu_fp32_sha256'] == row['old_fp32_sha256'] == row['new_fp32_sha256']
                           for row in widening)
    torch.save(repeats, args.output / 'actual-scan-replays.pt')  # ignored private output only
    report = dict(device=args.device_label, block_dim=4, repeats=hashes,
                  distributions=distributions, same_tensor_widening=widening, passed=passed,
                  seed_manifest_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                  input_unchanged=input_unchanged, scan_dh0_fixed=False,
                  raw_capture_sha256=hashlib.sha256((args.output / 'actual-scan-replays.pt').read_bytes()).hexdigest(),
                  interpretation='D-PM-44; no acceptance of unstable h0 endpoint')
    (args.output / 'replay.json').write_text(json.dumps(report, indent=2) + '\n')
    assert passed
    print(json.dumps(dict(device=args.device_label, repeats=12, widening_samples=12,
                          passed=True, dh0_unique=len(distributions['dh0']))), flush=True)


if __name__ == '__main__':
    main()
