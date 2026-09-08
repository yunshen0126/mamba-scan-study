#!/usr/bin/env bash
# launch_fused.sh -- the 32 fused-block runs.
#
#   bash launch_fused.sh selftest     # do this first, costs nothing
#   bash launch_fused.sh one          # one short run, checks the whole pipeline
#   bash launch_fused.sh all          # the 32 runs
#
set -euo pipefail

REPO=/root/mamba-scan-study
BANK=$REPO/P0B_R_PATH_BANK_FROZEN.json
LBANK=$REPO/P0B_L_PATH_BANK_FROZEN.json
CFG=/root/autodl-tmp/outputs_main/p0b_cifar10_main_uniform_mamba_GEO_DIV_R_high_seed0/metadata.json
OUT=/root/autodl-tmp/outputs_fused
DATA=/root/autodl-tmp/datasets
PAR=3

mkdir -p "$OUT" logs

case "${1:-selftest}" in

selftest)
  python3 run_fused_scan.py --selftest --bank "$BANK" --lbank "$LBANK"
  ;;

gpu)
  nvidia-smi || echo "nvidia-smi not found"
  python3 - <<'PY'
import sys, torch
print("python          :", sys.executable)
print("torch           :", torch.__version__)
print("built with CUDA :", torch.version.cuda)
print("cuda available  :", torch.cuda.is_available())
print("devices         :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0        :", torch.cuda.get_device_name(0))
PY
  ;;

one)
  # ten epochs, no batch cap, on the seed and condition that collapsed under
  # fp16. Under bf16 the training accuracy must be climbing, not stuck at 0.10.
  python3 run_fused_scan.py --bank "$BANK" --lbank "$LBANK" \
      --data-root "$DATA" --out "$OUT" \
      --dataset cifar10 --exp-id RND_D1 --seed 1 --epochs 10
  ;;

all)
  echo "40 runs: CIFAR-10 six conditions, CIFAR-100 four, four seeds each"
  i=0
  for DS in cifar10 cifar100; do
    if [ "$DS" = "cifar10" ]; then
      CONDS="GEO_SG1 GEO_DIV RND_S1 RND_D1 LOC_S1 LOC_D1"
    else
      CONDS="GEO_SG1 GEO_DIV RND_S1 RND_D1"
    fi
    for EXP in $CONDS; do
      for SEED in 0 1 2 3; do
        NAME="${DS}_${EXP}_seed${SEED}"
        if [ -f "$OUT/p0b_${DS}_fused_scan_mamba_${EXP}_R_high_seed${SEED}/metadata.json" ]; then
          echo "skip $NAME (already done)"
          continue
        fi
        echo "launch $NAME"
        OMP_NUM_THREADS=1 python3 run_fused_scan.py \
            --config-from "$CFG" --bank "$BANK" --lbank "$LBANK" \
            --data-root "$DATA" --out "$OUT" \
            --dataset "$DS" --exp-id "$EXP" --seed "$SEED" \
            > "logs/${NAME}.log" 2>&1 &
        i=$((i+1))
        if [ $((i % PAR)) -eq 0 ]; then wait; fi
      done
    done
  done
  wait
  echo "done. analyse with:"
  echo "  python3 compute_2x2.py $OUT --tail 20"
  ;;

loc)
  # the locality-matched arm, run after 'all' finishes.
  # CIFAR-10, fine patch, four seeds: 8 runs.
  echo "8 runs: locality-matched pair on CIFAR-10"
  i=0
  for EXP in LOC_S1 LOC_D1; do
    for SEED in 0 1 2 3; do
      NAME="cifar10_${EXP}_seed${SEED}"
      if [ -f "$OUT/p0b_cifar10_fused_scan_mamba_${EXP}_R_high_seed${SEED}/metadata.json" ]; then
        echo "skip $NAME (already done)"; continue
      fi
      echo "launch $NAME"
      OMP_NUM_THREADS=1 python3 run_fused_scan.py \
          --config-from "$CFG" --bank "$BANK" --lbank "$LBANK" \
          --data-root "$DATA" --out "$OUT" \
          --dataset cifar10 --exp-id "$EXP" --seed "$SEED" \
          > "logs/${NAME}.log" 2>&1 &
      i=$((i+1))
      if [ $((i % PAR)) -eq 0 ]; then wait; fi
    done
  done
  wait
  echo "done."
  ;;

*)
  echo "usage: bash launch_fused.sh {selftest|gpu|one|all|loc}" ; exit 1 ;;
esac
