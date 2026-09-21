"""Losslessly archive original native JSON receipts in bounded gzip JSONL shards.

Receipts retain their complete original environment. Hash verification recreates
exact original pretty-printed bytes; archived source is never imported or run.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

LIMIT = 4_000_000
FORMAT = 'bf08-original-json-gzip-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(x in ('.', '..') for x in path.parts) or '\\' in name:
        raise ValueError('Unsafe archive member')
    return path


def original(receipt):
    if not isinstance(receipt, dict) or not isinstance(receipt.get('environment'), dict):
        raise ValueError('Original per-receipt environment is required')
    return (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    paths = sorted(source.rglob('*.json'))
    if not paths:
        raise ValueError('No original JSON receipts')
    destination.mkdir(parents=True, exist_ok=False)
    entries, shards, lines, raw_size = [], [], [], 0
    def flush():
        nonlocal lines, raw_size
        if not lines:
            return
        # The original summary may exceed the usual uncompressed shard target;
        # its compressed size must still satisfy the committed-blob bound.
        data = gzip.compress(b''.join(lines), mtime=0)
        if len(data) > LIMIT:
            raise ValueError('Compressed shard exceeds committed blob bound')
        name = f'receipts-{len(shards):04d}.jsonl.gz'
        (destination/name).write_bytes(data)
        shards.append(dict(path=name, bytes=len(data), sha256=sha(data), records=len(lines)))
        lines, raw_size = [], 0
    for path in paths:
        data = path.read_bytes()
        receipt = json.loads(data)
        if original(receipt) != data:
            raise ValueError('Unsupported original formatting: '+path.name)
        name = path.relative_to(source).as_posix()
        safe(name)
        row = dict(path=name, bytes=len(data), sha256=sha(data), receipt=receipt)
        line = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if lines and raw_size+len(line) > LIMIT:
            flush()
        entries.append(dict(path=name, bytes=len(data), sha256=sha(data),
            shard=f'receipts-{len(shards):04d}.jsonl.gz', line=len(lines)+1))
        lines.append(line)
        raw_size += len(line)
    flush()
    manifest = dict(format=FORMAT, original_files=len(entries), original_bytes=sum(e['bytes'] for e in entries),
        compressed_limit_bytes=LIMIT, shards=shards, files=entries)
    data = (json.dumps(manifest, indent=2)+'\n').encode()
    if len(data) > LIMIT:
        raise ValueError('Manifest exceeds committed blob bound')
    (destination/'manifest.json').write_bytes(data)
    return verify(destination)


def verify(archive, restore=None):
    archive = Path(archive)
    manifest = json.loads((archive/'manifest.json').read_text())
    assert manifest['format'] == FORMAT
    expected = {e['path']: e for e in manifest['files']}
    assert len(expected) == len(manifest['files'])
    if restore is not None:
        restore = Path(restore)
        restore.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    for shard in manifest['shards']:
        data = (archive/safe(shard['path'])).read_bytes()
        assert len(data) == shard['bytes'] <= LIMIT and sha(data) == shard['sha256']
        count = 0
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                name = row['path']
                relative = safe(name)
                assert name not in seen
                entry = expected[name]
                assert entry['shard'] == shard['path'] and entry['line'] == count
                raw = original(row['receipt'])
                assert len(raw) == entry['bytes'] == row['bytes'] and sha(raw) == entry['sha256'] == row['sha256']
                if restore is not None:
                    target = restore/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                seen.add(name)
                total += len(raw)
        assert count == shard['records']
    assert seen == set(expected) and len(seen) == manifest['original_files'] and total == manifest['original_bytes']
    return dict(passed=True, original_files=len(seen), original_bytes=total,
        compressed_bytes=sum(s['bytes'] for s in manifest['shards']), shards=len(manifest['shards']), restored=restore is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='action', required=True)
    create = modes.add_parser('pack')
    create.add_argument('source', type=Path)
    create.add_argument('destination', type=Path)
    check = modes.add_parser('verify')
    check.add_argument('archive', type=Path)
    check.add_argument('--restore', type=Path)
    args = parser.parse_args()
    result = pack(args.source, args.destination) if args.action == 'pack' else verify(args.archive, args.restore)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
