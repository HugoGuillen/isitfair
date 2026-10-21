#!/usr/bin/env bash
# After run.sh: re-execute diabetes130 (template now drops audited rows whose level the calibration half lacks).
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
export HARDEN_STRICT=1 HARDEN_B=50 HARDEN_NJOBS=-1
until grep -q "PHASE 7 DONE" $REPO/hardening/logs/phase7.log; do sleep 30; done
st=$(date '+%Y-%m-%d %H:%M:%S'); t=$(date +%s)
(cd $REPO/hardening/notebooks && "$PY" -m jupyter nbconvert --to notebook --execute \
   --ExecutePreprocessor.timeout=3600 diabetes130.ipynb --output-dir executed --output diabetes130_phase7.ipynb) > $REPO/hardening/logs/p7_nb_diabetes130_rerun.log 2>&1
rc=$?
echo "| P7 nbconvert diabetes130 rerun (unseen-level rows dropped) | $st | $(date '+%Y-%m-%d %H:%M:%S') | $(( $(date +%s) - t ))$([[ $rc != 0 ]] && echo " (rc=$rc)") | | logs/p7_nb_diabetes130_rerun.log |" >> $REPO/hardening/STATUS.md
echo "DIABETES RERUN rc=$rc"
