#!/bin/bash
set -u
export OMP_NUM_THREADS=4
mkdir -p logs
run_seed () {
  local SEED=$1
  env PYTHONPATH=. OMP_NUM_THREADS=4 python -u \
    -m mamba_scan_study.experiments.run_stage1_seed0 \
    --arch channel_split --dataset cifar10 \
    --data-root /root/autodl-tmp/datasets \
    --outdir mamba_scan_study/outputs/csplit_d256_seed${SEED} \
    --microbatch-csv mamba_scan_study/outputs/cloud_microbatch_pilot/best_microbatch.csv \
    --epochs 100 --warmup-epochs 5 --effective-batch 128 \
    --d-model 256 --n-layers 2 --pos-mode xy_learned \
    --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 \
    --num-workers 4 --seed ${SEED} \
    > logs/csplit_d256_seed${SEED}.log 2>&1
  echo "=== d256 seed ${SEED} DONE $(date) ==="
}
echo "=== batch C d256 START $(date) ==="
for S in 0 1 2 3; do run_seed $S & done
wait
run_seed 4
echo "=== batch C d256 ALL DONE $(date) ==="
