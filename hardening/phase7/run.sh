#!/usr/bin/env bash
# Phase 7 acceptance: nbconvert every hardening notebook (cell timeout 3600 s,
# HARDEN_STRICT=1, B = 50, n_jobs = -1), then pytest -q, then the golden check.
# One step at a time; each step appends a row to STATUS.md.
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
export HARDEN_STRICT=1 HARDEN_B=50 HARDEN_NJOBS=-1
S=$REPO/hardening/STATUS.md; L=$REPO/hardening/logs
row() { echo "| P7 $1 | $2 | $(date '+%Y-%m-%d %H:%M:%S') | $3 | | $4 |" >> $S; }
mkdir -p $REPO/hardening/notebooks/executed
for n in support2 compas german heart_cleveland oracle_sparse oracle_calib adult bank diabetes130; do
  st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s); echo "== $n start $st"
  (cd $REPO/hardening/notebooks && "$PY" -m jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=3600 $n.ipynb --output-dir executed --output ${n}_phase7.ipynb) > $L/p7_nb_$n.log 2>&1
  rc=$?; echo "== $n end rc=$rc $(( $(date +%s) - t )) s"
  row "nbconvert $n" "$st" "$(( $(date +%s) - t ))$([[ $rc != 0 ]] && echo " (rc=$rc)")" "logs/p7_nb_$n.log"
done
st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
(cd $REPO && "$PY" -m pytest -q) > $L/p7_pytest.log 2>&1; rc=$?
echo "== pytest rc=$rc"; row "pytest -q" "$st" "$(( $(date +%s) - t ))$([[ $rc != 0 ]] && echo " (rc=$rc)")" "$(tail -1 $L/p7_pytest.log)"
st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
bash $REPO/hardening/tools/golden.sh phase7 all > $L/p7_golden.log 2>&1
"$PY" $REPO/hardening/tools/golden_narrow.py "$GOLDEN" > $L/p7_golden_narrow.log 2>&1; rc=$?
echo "== golden rc=$rc"; row "golden check (narrowed)" "$st" "$(( $(date +%s) - t ))" "$(tail -1 $L/p7_golden_narrow.log)"
echo "== PHASE 7 DONE $(date -Is)"
