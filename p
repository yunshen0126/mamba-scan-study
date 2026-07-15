#!/bin/bash
cd /root/mamba-scan-study
T=0
for S in 0 1 2 3 4; do
  N=$(python -c "import json;print(len(json.load(open('mamba_scan_study/outputs/cloud_seed${S}/stage1_results.json'))['results']))" 2>/dev/null || echo 0)
  T=$((T+N))
  echo "seed${S}: ${N}/30"
done
echo "-----------"
echo "total: ${T}/150"
tail -1 logs/batch_a2.log
