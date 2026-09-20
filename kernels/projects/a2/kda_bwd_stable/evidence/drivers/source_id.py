"""Identity of the unit source a run used: sha256 of each file plus one combined digest."""
import hashlib, json, pathlib, sys

U = pathlib.Path(sys.argv[1])
NAMES = ([f"kernels/{n}.py" for n in ("scan_fused", "inverse_mm", "inverse_epilogue", "inverse_dainv",
                                      "inverse_dakk_fused", "finalize_pre", "finalize_pair",
                                      "finalize_post", "finalize_reduce")]
         + ["unit.py", "reference.py", "contract.json", "_unit_runner.py", "run.py"])


def strip_comments(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        out.append(line)
    return "\n".join(out)


files, combined = {}, hashlib.sha256()
for name in NAMES:
    data = (U / name).read_bytes()
    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if name.endswith(".py"):
        files[name]["code_sha256"] = hashlib.sha256(strip_comments(data.decode()).encode()).hexdigest()
    combined.update(name.encode() + b"\0" + data)
print(json.dumps({"unit_source_sha256": combined.hexdigest(), "files": files}, indent=1, sort_keys=True))
