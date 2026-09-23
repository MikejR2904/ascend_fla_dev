#!/bin/bash
# Drive msprof over the GDN-2 a2 kernels and parse per-op device time.
REPO="${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
export PYTHONPATH="${ASCRIPTOR_WORKSPACE:?set ASCRIPTOR_WORKSPACE}/library:$REPO"

export ASCEND_RT_VISIBLE_DEVICES=7
MSPROF="${MSPROF:-msprof}"
ROOT="${ROOT:-$(pwd)/gdn2_prof}"
rm -rf $ROOT; mkdir -p $ROOT
ITERS=${ITERS:-60}

for KERN in fwd bwd; do for SHP in decode train batch8; do
  OUT=$ROOT/${KERN}_${SHP}
  mkdir -p $OUT
  echo "=== msprof $KERN $SHP (iters=$ITERS) ==="
  timeout 900 $MSPROF --application="python3 "$(dirname "$0")/bench_msprof.py" --kernel $KERN --shape $SHP --iters $ITERS --warmup 10" \
    --output=$OUT --ai-core=on --aicpu=off --runtime-api=off --task-time=on --l2=off \
    > $OUT/msprof.log 2>&1
  # parse the op_summary csv
  python3 - "$OUT" "$KERN" "$SHP" "$ITERS" <<'PY'
import sys, glob, csv, os
out, kern, shp, iters = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cands = glob.glob(out + "/**/op_summary_*.csv", recursive=True)
if not cands:
    print(f"  [{kern} {shp}] no op_summary csv; log tail:")
    log = glob.glob(out + "/msprof.log")
    if log: print("   ", open(log[0]).read().splitlines()[-3:])
    sys.exit(0)
csvf = sorted(cands)[-1]
rows = list(csv.DictReader(open(csvf)))
# find duration + name columns (version-tolerant)
def col(keys):
    for k in rows[0].keys():
        kl = k.lower().replace(" ", "").replace("(us)", "").replace("(ns)", "")
        if any(t in kl for t in keys): return k
    return None
namec = col(["opname", "name"])
durc  = col(["taskduration", "aicoretime", "totaltime", "duration"])
unit_ns = durc and "ns" in durc.lower()
mine = [r for r in rows if "gdn2" in str(r.get(namec, "")).lower()]
tot = sum(float(r[durc]) for r in mine) if mine else 0.0
if unit_ns: tot /= 1000.0
per = tot / iters if iters else 0.0
print(f"  [{kern} {shp}] op_summary={os.path.basename(csvf)} kernel-op-launches={len(mine)} "
      f"device_time_total={tot:.1f}us  per-call={per:.2f}us  (dur col: {durc})")
PY
done; done
echo "ALL_MSPROF_DONE"
