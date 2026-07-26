#!/usr/bin/env bash
# P0-B formal launch: 104 runs, 5-way parallel, grid8 -> grid32
set -uo pipefail

REPO=/root/mamba-scan-study
DATA=/root/autodl-tmp/datasets
LOGDIR=/root/autodl-tmp/p0b_launch_logs
PARALLEL=5
export OMP_NUM_THREADS=1

EXP_IDS="GEO_SG1 GEO_SG2 GEO_SG3 GEO_SG4 GEO_DIV RND_S1 RND_S2 RND_S3 RND_D1 RND_D2 RND_D3 LOC_S LOC_D"
SEEDS="0 1 2 3"

mkdir -p "$LOGDIR"
TSV="$LOGDIR/run_timing.tsv"
[ -f "$TSV" ] || printf 'exp_id\tgrid\tseed\tstatus\tstart\tend\tdur_s\texit\n' > "$TSV"

reliance_of() { if [ "$1" = "8" ]; then echo R_low; else echo R_high; fi; }

run_one() {
  local exp=$1 grid=$2 seed=$3
  local rel dir tag t0 t1 dur rc
  rel=$(reliance_of "$grid")
  dir="$REPO/outputs/p0b_${exp}_${rel}_seed${seed}"
  tag="${exp}_g${grid}_s${seed}"

  if [ -f "$dir/completed.json" ]; then
    printf '%s\t%s\t%s\tSKIP\t-\t-\t0\t0\n' "$exp" "$grid" "$seed" >> "$TSV"
    return 0
  fi

  t0=$(date +%s)
  cd "$REPO" || return 1
  PYTHONPATH=. python -u -m mamba_scan_study.experiments.run_p0b_feasibility \
    --exp-id "$exp" --grid "$grid" --training-seed "$seed" \
    --data-root "$DATA" --mode formal --execute \
    > "$LOGDIR/$tag.log" 2>&1
  rc=$?
  t1=$(date +%s)
  dur=$((t1-t0))

  if [ $rc -eq 0 ] && [ -f "$dir/completed.json" ]; then
    printf '%s\t%s\t%s\tOK\t%s\t%s\t%d\t%d\n' "$exp" "$grid" "$seed" \
      "$(date -d @$t0 -Iseconds)" "$(date -d @$t1 -Iseconds)" $dur $rc >> "$TSV"
  else
    printf '%s\t%s\t%s\tFAIL\t%s\t%s\t%d\t%d\n' "$exp" "$grid" "$seed" \
      "$(date -d @$t0 -Iseconds)" "$(date -d @$t1 -Iseconds)" $dur $rc >> "$TSV"
    echo "[FAIL] $tag rc=$rc -> $LOGDIR/$tag.log" >&2
  fi
  return 0
}
export -f run_one reliance_of
export REPO DATA LOGDIR TSV

manifest() { for e in $EXP_IDS; do for s in $SEEDS; do echo "$e $1 $s"; done; done; }

echo "=== 前置检查 ==="
cd "$REPO"
echo "HEAD: $(git rev-parse --short HEAD)"
echo "dirty: $(git status --short | wc -l) 行"
echo "nproc: $(nproc)   free disk: $(df -h "$REPO" | awk 'NR==2{print $4}')"
PYTHONPATH=. python -m mamba_scan_study.experiments.run_p0b_feasibility \
  --exp-id GEO_SG1 --grid 8 --training-seed 0 --dry-run || { echo "DRY-RUN 失败,中止"; exit 1; }

echo
echo "=== Phase 0: canary (GEO_SG1 grid8 seed0) $(date -Iseconds) ==="
run_one GEO_SG1 8 0
if ! [ -f "$REPO/outputs/p0b_GEO_SG1_R_low_seed0/completed.json" ]; then
  echo "CANARY 失败,中止。见 $LOGDIR/GEO_SG1_g8_s0.log"; exit 1
fi
echo "canary OK,用时 $(tail -1 "$TSV" | cut -f7) s"

echo
echo "=== Phase 1: grid8 剩余 51 runs, ${PARALLEL} 并行  $(date -Iseconds) ==="
manifest 8 | grep -vx 'GEO_SG1 8 0' | xargs -P $PARALLEL -n 3 bash -c 'run_one "$0" "$1" "$2"'
echo "grid8 完成: OK=$(awk -F'\t' '$2==8&&$4=="OK"' "$TSV" | wc -l)  SKIP=$(awk -F'\t' '$2==8&&$4=="SKIP"' "$TSV" | wc -l)  FAIL=$(awk -F'\t' '$2==8&&$4=="FAIL"' "$TSV" | wc -l)"

echo
echo "=== Phase 2: grid32 52 runs, ${PARALLEL} 并行  $(date -Iseconds) ==="
manifest 32 | xargs -P $PARALLEL -n 3 bash -c 'run_one "$0" "$1" "$2"'

echo
echo "=== 全部结束 $(date -Iseconds) ==="
awk -F'\t' 'NR>1{c[$4]++} END{for(k in c) printf "%s=%d\n",k,c[k]}' "$TSV"
echo "完成目录数: $(ls -d "$REPO"/outputs/p0b_*_seed* 2>/dev/null | wc -l) / 104"
