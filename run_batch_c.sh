#!/bin/bash
# Batch C: channel-split 2x2 factorial, both scales, 5 seeds.
# runner internally loops 2block x 3grid x 4variant per call,
# so we only iterate (d_model, seed). One log per call, unique name.
set -e
mkdir -p logs
for D in 64 256; do
  for SEED in 0 1 2 3 4; do
    OUT=mamba_scan_study/outputs/csplit_d${D}_seed${SEED}
    LOG=logs/csplit_d${D}_seed${SEED}.log
    echo "===== d=${D} seed=${SEED} START $(date) ====="
    env PYTHONPATH=. python -u \
      -m mamba_scan_study.experiments.run_stage1_seed0 \
      --arch channel_split \
      --dataset cifar10 \
      --data-root /root/autodl-tmp/datasets \
      --outdir ${OUT} \
      --microbatch-csv mamba_scan_study/outputs/cloud_microbatch_pilot/best_microbatch.csv \
      --epochs 100 --warmup-epochs 5 --effective-batch 128 \
      --d-model ${D} --n-layers 2 --pos-mode xy_learned \
      --base-lr 0.001 --weight-decay 0.05 --grad-clip 1.0 \
      --num-workers 4 --seed ${SEED} \
      > ${LOG} 2>&1
    echo "===== d=${D} seed=${SEED} DONE $(date) ====="
  done
done
