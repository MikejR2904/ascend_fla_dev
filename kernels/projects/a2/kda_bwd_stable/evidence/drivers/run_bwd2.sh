#!/bin/bash
# $1 = card, $2 = case list file. Records the unit source identity next to every case output.
B=<task-tmp>
U=$B/repo/kernels/projects/a2/kda_bwd_stable
cd $U
while read cid; do
  [ -z "$cid" ] && continue
  out=$B/scratch/dev2/aclnn/$cid
  mkdir -p $out
  $B/venv/bin/python $B/scratch/source_id.py $U > $out/source_id.json
  PYTHONPATH=$B/workspace/library ASCEND_RT_VISIBLE_DEVICES=$1 ASCRIPTOR_NPU_LOCK=$B/../npu-locks/npu$1.lock \
    timeout 5400 $B/venv/bin/python run.py check --device a2 --backend cce --launcher aclnn --timeout 1800 --case $cid \
    --output $out > $B/scratch/logs2/aclnn_$cid.log 2>&1
  rc=$?
  $B/venv/bin/python $B/scratch/source_id.py $U > $out/source_id_after.json
  echo "$cid card$1 exit=$rc $(tail -1 $B/scratch/logs2/aclnn_$cid.log | cut -c1-180)"
done < $2
