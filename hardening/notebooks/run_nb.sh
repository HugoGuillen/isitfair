#!/usr/bin/env bash
# Execute one hardening notebook. Usage: run_nb.sh NAME TAG B [pin-head] [timeout_s]
#  pin-head: import isitfair from the HEAD worktree (pre-fix code) instead of src/.
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
NAME=$1; TAG=$2; export HARDEN_B=$3; PIN=${4:-}; TMO=${5:-36000}
[[ $PIN == pin-head ]] && export PYTHONPATH='C:\Users\hugoa\Desktop\isitfair-golden\src'
mkdir -p $REPO/hardening/notebooks/executed
echo "== $NAME tag=$TAG B=$HARDEN_B pin=$PIN start $(date -Is)"
cd $REPO/hardening/notebooks && timeout $TMO "$PY" -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 $NAME.ipynb --output-dir executed --output ${NAME}_${TAG}.ipynb
echo "== $NAME end rc=$? $(date -Is)"
