#!/bin/bash
# Batch D: CIFAR-100 channel-split 2x2 factorial, d256, 5 seeds in parallel.
set -e
mkdir -p logs
pids=()
for SEED in 0 1 2 3 4; do
  OUT=mamba_scan_study/outputs/csplit_c100_d256_seed${SEED}
  LOG=logs/csplit_c100_d256_seed${SEED}.log
  echo "===== d=256 seed=${SEED} START $(date) ====="
  env PYTHONPATH=. python -u \
    -m mamba_scan_study.experiments.run_stage1_seed0 \
    --arch channel_split \
    --dataset cifar100 \
    --data-root /root/autodl-tmp/datasets \
    --outdir ${OUT} \
    --microbatch-csv mamba_scan_study/outputs/cloud_microbatch_pilot/best_microbatch.csv \
    --epochs 100 --warmup-epochs 5 --effective-batch 128 \
    --d-model 256 --n-layers 2 --pos-mode xy_learned \
    --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 \
    --num-workers 4 --seed ${SEED} \
    > ${LOG} 2>&1 &
  pids+=("$!")
done
for PID in "${pids[@]}"; do
  wait "${PID}"
done
