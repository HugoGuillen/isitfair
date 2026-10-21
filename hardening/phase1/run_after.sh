#!/usr/bin/env bash
# Phase 1 sweep on the fixed src/ (no extrapolation at any n).
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
cd $REPO
"$PY" -c "import isitfair; print('isitfair from', isitfair.__file__)"
for n in 1000 5000 20000 50000 100000; do
  echo "== n=$n start $(date -Is)"
  "$PY" hardening/phase1/bench.py $n hardening/phase1/results/after_$n.json --extras
  echo "== n=$n end $(date -Is)"
done
