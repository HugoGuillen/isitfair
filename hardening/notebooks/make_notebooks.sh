#!/usr/bin/env bash
# Stamp one jupytext percent notebook per dataset from _template.py and pair it to .ipynb.
source /c/Users/hugoa/Desktop/isitfair/hardening/tools/env.sh
cd $REPO/hardening/notebooks
for n in support2 diabetes130 adult compas bank german heart_cleveland oracle_calib oracle_sparse; do
  sed "s/__NAME__/$n/g" _template.py > $n.py
  "$PY" -m jupytext --to ipynb --update-metadata '{"jupytext":{"formats":"ipynb,py:percent"}}' -q $n.py
done
ls
