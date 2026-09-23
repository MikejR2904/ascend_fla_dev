#!/bin/bash
# msprof PipeUtilization profile to find the GDN-2 kernel bottleneck pipe.
REPO="${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
export PYTHONPATH="${ASCRIPTOR_WORKSPACE:?set ASCRIPTOR_WORKSPACE}/library:$REPO"

export ASCEND_RT_VISIBLE_DEVICES=7
MSPROF="${MSPROF:-msprof}"
ROOT="${ROOT:-$(pwd)/gdn2_pipe}"
rm -rf $ROOT; mkdir -p $ROOT

for CASE in "fwd train" "bwd train"; do
  set -- $CASE; KERN=$1; SHP=$2
  OUT=$ROOT/${KERN}_${SHP}; mkdir -p $OUT
  echo "=== PipeUtilization $KERN $SHP ==="
  timeout 900 $MSPROF --application="python3 "$(dirname "$0")/bench_msprof.py" --kernel $KERN --shape $SHP --iters 40 --warmup 8" \
    --output=$OUT --ai-core=on --aic-metrics=PipeUtilization --task-time=on \
    > $OUT/msprof.log 2>&1
  python3 - "$OUT" "$KERN" <<'PY'
import sys, glob, csv, os
out, kern = sys.argv[1], sys.argv[2]
cands = glob.glob(out + "/**/op_summary_*.csv", recursive=True)
if not cands:
    print("  no op_summary; log tail:", open(glob.glob(out+"/msprof.log")[0]).read().splitlines()[-4:]); sys.exit(0)
rows = [r for r in csv.DictReader(open(sorted(cands)[-1])) if "gdn2" in str(r).lower()]
if not rows:
    print("  no gdn2 rows"); sys.exit(0)
# average the ratio-like columns
keys = [k for k in rows[0] if any(t in k.lower() for t in
        ["ratio","vec_","mac_","mte","scalar","cube","aic","aiv","duration","bound"])]
import statistics
print(f"  {len(rows)} launches, averaged pipe columns:")
for k in keys:
    vals = []
    for r in rows:
        try: vals.append(float(r[k]))
        except: pass
    if vals:
        print(f"    {k:32s} mean={statistics.mean(vals):.4f} max={max(vals):.4f}")
PY
done
echo PIPE_DONE
