#!/usr/bin/env bash
# Golden check (rule 4). Runs the committed SUPPORT2 pipeline in the HEAD worktree
# ($GOLDEN) against the *current* src/ (editable install), then compares every
# MANIFEST artifact. Usage: golden.sh TAG [example|nb02|all]
set -u
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
TAG=$1; WHAT=${2:-all}
LOG=$REPO/hardening/logs
cd "$GOLDEN" && git checkout -q -- results notebooks && git clean -fdq results
# Remove the example's outputs so a failed example run cannot leave the committed files to be compared.
rm -rf results/support2_audit_report.html results/support2_audit_report.pdf results/support2_audit_report_figures
t0=$(date +%s)
if [[ $WHAT == example || $WHAT == all ]]; then
  echo "== example start $(date -Is)"
  "$PY" "$REPO/hardening/tools/run_cells.py" examples/support2_mortality_audit.py --cwd "$GOLDEN" \
     --timings "$LOG/golden_${TAG}_example_cells.csv" || echo "EXAMPLE FAILED rc=$?"
  echo "== example end $(date -Is) ($(( $(date +%s) - t0 )) s)"
fi
if [[ $WHAT == nb02 || $WHAT == all ]]; then
  t1=$(date +%s); echo "== nb02 start $(date -Is)"
  cd "$GOLDEN/notebooks" && "$PY" -m jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=-1 02_paper_analysis_support2.ipynb \
     --output-dir "$LOG" --output "golden_${TAG}_nb02.ipynb" || echo "NB02 FAILED rc=$?"
  cd "$GOLDEN"; echo "== nb02 end $(date -Is) ($(( $(date +%s) - t1 )) s)"
fi
echo "== compare"
"$PY" "$REPO/hardening/tools/golden_compare.py" "$GOLDEN"
echo "== git status of worktree results/"
git -C "$GOLDEN" status --porcelain results | head -100
echo "== total $(( $(date +%s) - t0 )) s"
