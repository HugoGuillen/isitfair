#!/usr/bin/env bash
# After E07/E08: full pytest, golden check (narrowed), then the Phase 1 sweep on the new code.
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
S=$REPO/hardening/STATUS.md; L=$REPO/hardening/logs
row() { echo "| $1 | $2 | $(date '+%Y-%m-%d %H:%M:%S') | $3 | | $4 |" >> $S; }
st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
(cd $REPO && "$PY" -m pytest -q) > $L/e07e08_pytest.log 2>&1; rc=$?
echo "== pytest rc=$rc"; row "E07/E08 pytest -q" "$st" "$(( $(date +%s) - t ))" "$(tail -1 $L/e07e08_pytest.log)"
st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
bash $REPO/hardening/tools/golden.sh e07e08 all > $L/e07e08_golden.log 2>&1
"$PY" $REPO/hardening/tools/golden_narrow.py "$GOLDEN" > $L/e07e08_golden_narrow.log 2>&1; rc=$?
echo "== golden rc=$rc"; row "E07/E08 golden check (narrowed)" "$st" "$(( $(date +%s) - t ))" "$(tail -1 $L/e07e08_golden_narrow.log)"
for n in 20000 50000 100000; do
  st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
  "$PY" $REPO/hardening/phase1/bench.py $n $REPO/hardening/phase1/results/after_e07e08_$n.json --extras > $L/e07e08_bench_$n.log 2>&1; rc=$?
  echo "== bench $n rc=$rc"; row "E07/E08 Phase 1 bench n=$n" "$st" "$(( $(date +%s) - t ))$([[ $rc != 0 ]] && echo " (rc=$rc)")" "phase1/results/after_e07e08_$n.json"
done
echo "== E07E08 DONE $(date -Is)"
