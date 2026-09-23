"""Restore complete validation receipts, including identical cross-bd data.

The bounded archive stores each identical JSON once. The only non-identical
case substitution supported here is a compiled launch signature. All restored
original bytes are checked against the original receipt's length and SHA-256.
No archived code is imported or executed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_archive import digest, original_bytes, safe_name, verify


def restore(archive: Path, destination: Path) -> dict:
    archive, destination = Path(archive), Path(destination)
    manifest = json.loads((archive / "reconstruction.json").read_text())
    assert manifest["format"] == "bf09-cross-bd-json-v1"
    result = verify(archive / "original-receipts", destination)
    for row in manifest["reconstructions"]:
        target = destination / safe_name(row["path"])
        source = destination / safe_name(row["base"])
        assert not target.exists() and source.is_file()
        raw = source.read_bytes()
        if "launch_signatures" in row:
            receipt = json.loads(raw)
            signatures = row["launch_signatures"]
            assert len(receipt["launches"]) == len(signatures)
            for launch, signature in zip(receipt["launches"], signatures):
                assert isinstance(signature, str)
                launch["signature"] = signature
            raw = original_bytes(receipt)
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
        assert isinstance(json.loads(raw).get("environment"), dict)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    actual = {p.relative_to(destination).as_posix(): digest(p.read_bytes())
              for p in destination.rglob("*.json")}
    assert actual == manifest["original_sha256"]
    assert len(actual) == manifest["original_files"]
    return dict(passed=True, original_files=len(actual),
                packed_files=result["files"],
                reconstructed_files=len(manifest["reconstructions"]),
                all_original_bytes_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--restore", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.restore)))


if __name__ == "__main__":
    main()
