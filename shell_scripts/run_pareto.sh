#!/usr/bin/env bash
# Drive 5 Stage-2 finetuning runs (Pareto curve) from a single Stage-1 checkpoint.
# Required env: STAGE1_AVG (path to averaged Stage-1 checkpoint)
# Optional env: POINTS=p1,p3,p5  (comma-separated subset to run; default: all 5)
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS=$PROJECT/shell_scripts
STAGE1_AVG=${STAGE1_AVG:?STAGE1_AVG env var required (e.g. work/checkpoints/iwslt14_s1/avg5.pt)}
[[ -f "$STAGE1_AVG" ]] || { echo "STAGE1_AVG not found: $STAGE1_AVG" >&2; exit 1; }

declare -A WAIT_UNTIL=( [p1]=1 [p2]=3 [p3]=5 [p4]=7 [p5]=9 )
declare -A LAT_FACTOR=( [p1]=1.0 [p2]=1.0 [p3]=0.5 [p4]=0.3 [p5]=0.0 )
declare -A LAT_THRESH=( [p1]=3.0 [p2]=4.5 [p3]=5.0 [p4]=6.0 [p5]=0.0 )

POINTS=${POINTS:-p1,p2,p3,p4,p5}
IFS=',' read -ra PLIST <<<"$POINTS"

for p in "${PLIST[@]}"; do
  if [[ -z "${WAIT_UNTIL[$p]+x}" ]]; then
    echo "unknown point: $p (must be one of p1..p5)" >&2
    exit 1
  fi
  EXP=iwslt14_s2_${p} \
  PRETRAIN=$STAGE1_AVG \
  WAIT_UNTIL=${WAIT_UNTIL[$p]} \
  LATENCY_FACTOR=${LAT_FACTOR[$p]} \
  LATENCY_THRESHOLD=${LAT_THRESH[$p]} \
    bash "$SCRIPTS/train_stage2.sh"
done
