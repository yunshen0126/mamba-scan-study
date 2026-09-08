#!/usr/bin/env bash
# commit_repo.sh -- tidy the repository and commit the analysis code and results.
# Run from the repository root. Review `git status` before pushing.
set -euo pipefail

REPO=$(pwd)
MAIN=/root/autodl-tmp/outputs_main
EARLY=$REPO/mamba_scan_study/outputs
FUSED=/root/autodl-tmp/outputs_fused

echo "== 1. the tidy result table (this is what reviewers will read)"
mkdir -p results
python3 export_all_results.py --main "$MAIN" --earlier "$EARLY" --fused "$FUSED" \
        --out results/results_all.csv

echo
echo "== 2. per-run metadata, without the checkpoints"
mkdir -p results/metadata
for SRC in "$MAIN" "$FUSED"; do
  [ -d "$SRC" ] || continue
  BASE=$(basename "$SRC")
  find "$SRC" -name metadata.json -printf '%h\n' | while read -r d; do
    dest="results/metadata/$BASE/$(basename "$d")"
    mkdir -p "$dest"
    cp "$d/metadata.json" "$dest/"
  done
done
du -sh results/metadata

echo
echo "== 3. ignore rules"
if [ -f gitignore_additions.txt ]; then
  cat gitignore_additions.txt >> .gitignore
  rm gitignore_additions.txt
  sort -u .gitignore -o .gitignore
fi

echo
echo "== 4. stage"
git add README.md .gitignore results/
git add scan_geometry.py export_all_results.py export_main_cells.py \
        compute_2x2.py make_fig_28.py mkbbl.py \
        run_fused_scan.py launch_fused.sh 2>/dev/null || true
git add P0B_R_PATH_BANK_FROZEN.json P0B_L_PATH_BANK_FROZEN.json \
        P0B_EUROSAT_SPLIT_FROZEN.json P0B_RUN_LEDGER_104.csv 2>/dev/null || true

echo
echo "== 5. review, then commit"
git status --short
echo
echo "if this looks right:"
echo "  git commit -m 'analysis scripts, fused-scan arm, per-run records and README'"
echo "  git push"
