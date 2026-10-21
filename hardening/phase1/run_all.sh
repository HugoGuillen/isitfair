#!/usr/bin/env bash
# Phase 1 sweep, sequential so runs do not compete with each other.
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
TAG=${1:-before}; cd $REPO
for n in 1000 5000 20000 50000; do
  echo "== n=$n start $(date -Is)"
  "$PY" hardening/phase1/bench.py $n hardening/phase1/results/${TAG}_$n.json --extras
  echo "== n=$n end $(date -Is)"
done
echo "== n=100000 start $(date -Is)"
"$PY" hardening/phase1/bench.py 100000 hardening/phase1/results/${TAG}_100000.json --extras \
   ${EXTRAP_100K:---extrapolate calibration_curves report}
echo "== n=100000 end $(date -Is)"
