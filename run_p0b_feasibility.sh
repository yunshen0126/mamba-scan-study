#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--dry-run" && "$#" == "1" ]]; then
  python -m mamba_scan_study.experiments.run_p0b_preflight
  exit 0
fi

if [[ "$#" != "3" ]]; then
  echo "usage: $0 --dry-run | <exp_id> <grid> <training_seed>" >&2
  exit 2
fi

python -m mamba_scan_study.experiments.run_p0b_feasibility \
  --exp-id "$1" --grid "$2" --training-seed "$3" --dry-run

echo "The wrapper only validates one explicit design cell. It never starts training." >&2
