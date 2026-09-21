"""Lossless, bounded text shards for self-contained native JSON receipts.

Each JSONL record contains the complete original receipt, including its own
environment. Verification reconstructs and hashes the original pretty-printed
bytes; it does not execute any archived source or omit repeated environment data.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


SHARD_LIMIT = 2_500_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_bytes(receipt):
    return (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def pack(source, destination):
    source, destination = Path(source), Path(destination)
    files = sorted(source.rglob("*.json"))
    if not files:
        raise ValueError("No JSON receipts")
    destination.mkdir(parents=True, exist_ok=False)
    entries, shards, lines, size = [], [], [], 0

    def flush():
        nonlocal lines, size
        if not lines:
            return
        name = f"receipts-{len(shards):04d}.jsonl"
        raw = b"".join(lines)
        (destination / name).write_bytes(raw)
        shards.append(dict(path=name, bytes=len(raw), sha256=digest(raw), records=len(lines)))
        lines, size = [], 0

    for path in files:
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if not isinstance(receipt, dict) or not isinstance(receipt.get("environment"), dict):
            raise ValueError(f"Missing original per-receipt environment: {path.name}")
        if original_bytes(receipt) != raw:
            raise ValueError(f"Unsupported original formatting: {path.name}")
        name = path.relative_to(source).as_posix()
        safe_name(name)
        record = dict(path=name, bytes=len(raw), sha256=digest(raw), receipt=receipt)
        line = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(line) > SHARD_LIMIT:
            raise ValueError(f"Individual record exceeds shard bound: {name}")
        if size + len(line) > SHARD_LIMIT:
            flush()
        entries.append(dict(path=name, bytes=len(raw), sha256=record["sha256"],
                            shard=f"receipts-{len(shards):04d}.jsonl", line=len(lines) + 1))
        lines.append(line)
        size += len(line)
    flush()
    manifest = dict(format="bf07-complete-json-receipts-v1", source_label=source.name,
                    original_files=len(entries), original_bytes=sum(e["bytes"] for e in entries),
                    shard_limit_bytes=SHARD_LIMIT, shards=shards, files=entries)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    if len(raw) > SHARD_LIMIT:
        raise ValueError("Manifest exceeds shard bound")
    (destination / "manifest.json").write_bytes(raw)
    return verify(destination)


def verify(archive, restore=None):
    archive = Path(archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["format"] == "bf07-complete-json-receipts-v1"
    expected = {e["path"]: e for e in manifest["files"]}
    assert len(expected) == len(manifest["files"])
    if restore is not None:
        restore = Path(restore)
        restore.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    for shard in manifest["shards"]:
        raw = (archive / safe_name(shard["path"])).read_bytes()
        assert len(raw) == shard["bytes"] <= SHARD_LIMIT
        assert digest(raw) == shard["sha256"]
        records = raw.splitlines()
        assert len(records) == shard["records"]
        for number, line in enumerate(records, 1):
            record = json.loads(line)
            name = record["path"]
            relative = safe_name(name)
            assert name not in seen
            entry = expected[name]
            assert entry["shard"] == shard["path"] and entry["line"] == number
            assert isinstance(record["receipt"].get("environment"), dict)
            original = original_bytes(record["receipt"])
            assert len(original) == entry["bytes"] == record["bytes"]
            assert digest(original) == entry["sha256"] == record["sha256"]
            if restore is not None:
                target = restore / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original)
            seen.add(name)
            total += len(original)
    assert seen == set(expected)
    assert len(seen) == manifest["original_files"] and total == manifest["original_bytes"]
    return dict(passed=True, files=len(seen), original_bytes=total,
                shards=len(manifest["shards"]), restored=restore is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("pack")
    create.add_argument("source", type=Path)
    create.add_argument("destination", type=Path)
    check = actions.add_parser("verify")
    check.add_argument("archive", type=Path)
    check.add_argument("--restore", type=Path)
    args = parser.parse_args()
    result = (pack(args.source, args.destination) if args.action == "pack"
              else verify(args.archive, args.restore))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
