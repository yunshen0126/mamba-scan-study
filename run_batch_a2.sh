#!/bin/bash
set -e
for SEED in 1 2 3 4; do
  echo "===== SEED $SEED START $(date) ====="
  env PYTHONPATH=. python -u \
    -m mamba_scan_study.experiments.run_stage1_seed0 \
    --dataset cifar10 \
    --data-root /root/autodl-tmp/datasets \
    --outdir mamba_scan_study/outputs/cloud_seed${SEED} \
    --microbatch-csv mamba_scan_study/outputs/cloud_microbatch_pilot/best_microbatch.csv \
    --epochs 100 --warmup-epochs 5 --effective-batch 128 \
    --d-model 64 --n-layers 2 --pos-mode xy_learned \
    --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 \
    --num-workers 8 --seed ${SEED}
  echo "===== SEED $SEED DONE $(date) ====="
done
