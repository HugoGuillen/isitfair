"""Phase 6: the full report pipeline for one dataset.

Usage: python hardening/phase6/pipeline.py NAME [B] [N_JOBS]

Same calls as examples/support2_mortality_audit.py (and notebook 01): audit,
discrimination, roc_curves, calibration, calibration_curves, hosmer_lemeshow
(+ detail), decision_curve(zero_crossing=True), estimability, an intersection
audit with cell_diagnostics, GroupRecalibration(isotonic) mitigation on a
held-out calibration split, and report(pdf=True). B defaults to 200 and
N_JOBS to -1 (results do not depend on N_JOBS).

Calibration split: support2 uses the committed calibration partition; every
other dataset splits its audited test half 50/50 (stratified, random_state=42)
and audits the evaluation half, so no row is scored by a calibrator fitted on it.
Writes hardening/reports/<NAME>_B<B>.html/.pdf and
hardening/phase6/results/<NAME>_B<B>.json (per-stage seconds).
"""

from __future__ import annotations

import faulthandler
import json
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hardening" / "datasets"))
import registry  # noqa: E402

import isitfair  # noqa: E402
from isitfair import FairnessAudit, GroupRecalibration  # noqa: E402

NAME = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else -1
SEED = 42
OUT = REPO / "hardening" / "phase6" / "results"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = REPO / "hardening" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
REC = {"dataset": NAME, "B": B, "n_jobs": NJOBS, "isitfair": isitfair.__version__,
       "isitfair_file": isitfair.__file__, "seconds": {}}
T0 = time.perf_counter()


def timed(stage, fn):
    t = time.perf_counter()
    print(f"[{stage}] start (elapsed {time.perf_counter() - T0:.1f}s)", flush=True)
    out = fn()
    REC["seconds"][stage] = round(time.perf_counter() - t, 2)
    print(f"[{stage}] done {REC['seconds'][stage]:.1f}s (elapsed {time.perf_counter() - T0:.1f}s)", flush=True)
    (OUT / f"{NAME}_B{B}.json").write_text(json.dumps(REC, indent=2))
    return out


attrs = registry.ATTRS[NAME]
threshold = registry.meta(NAME)["threshold"]
df = registry.load(NAME)
for a in attrs:
    df[a] = df[a].fillna("Unknown")
if NAME == "support2":
    cal_df = pd.read_csv(registry.PRED / "support2_calib.csv", dtype={a: object for a in attrs})
    ev_df = df
else:
    i_cal, i_ev = train_test_split(np.arange(len(df)), test_size=0.5, random_state=SEED,
                                   stratify=df["y_true"])
    cal_df, ev_df = df.iloc[i_cal].reset_index(drop=True), df.iloc[i_ev].reset_index(drop=True)
    # mitigate() rejects audited levels the calibration half never saw (diabetes130:
    # gender "Unknown/Invalid", 3 rows). Drop those audited rows and say so; a no-op
    # for every dataset that already ran.
    unseen = np.zeros(len(ev_df), bool)
    for a in attrs:
        unseen |= ~ev_df[a].isin(set(cal_df[a]))
    if unseen.any():
        print(f"dropping {int(unseen.sum())} audited rows with levels absent from the calibration half", flush=True)
        ev_df = ev_df[~unseen].reset_index(drop=True)
y, p = ev_df["y_true"].to_numpy(float), ev_df["y_proba_lr"].to_numpy(float)
sf = {a: ev_df[a].to_numpy(object) for a in attrs}
calib = (cal_df["y_true"].to_numpy(float), cal_df["y_proba_lr"].to_numpy(float),
         {a: cal_df[a].to_numpy(object) for a in attrs})
kw = dict(threshold=threshold, n_bootstrap=B, random_state=SEED, n_jobs=NJOBS)
print(f"{NAME}: audited n={len(y)} events={int(y.sum())} calib n={len(cal_df)} B={B} n_jobs={NJOBS}", flush=True)

audit = timed("construct", lambda: FairnessAudit(y_true=y, y_proba=p, sensitive_features=sf, **kw))
timed("discrimination", audit.discrimination)
timed("roc_curves", audit.roc_curves)
timed("calibration", audit.calibration)
timed("calibration_curves", audit.calibration_curves)
timed("hosmer_lemeshow", audit.hosmer_lemeshow)
timed("hosmer_lemeshow_detail", lambda: audit.hosmer_lemeshow(detail=True))
timed("decision_curve_zero_crossing", lambda: audit.decision_curve(zero_crossing=True))
timed("estimability", audit.estimability)
if len(attrs) >= 2:
    inter = FairnessAudit(y_true=y, y_proba=p, sensitive_features={**sf, tuple(attrs[:2]): None}, **kw)
    timed("cell_diagnostics", inter.cell_diagnostics)
mitigated = timed("mitigate", lambda: audit.mitigate(GroupRecalibration(method="isotonic"),
                                                    calibration_data=calib))
report = timed("report_html_pdf", lambda: audit.report(
    out=REPORTS / f"{NAME}_B{B}.html",
    title=f"{NAME}: subgroup audit (B={B})",
    metadata={"Dataset": NAME, "Model": "LogisticRegression (hardening registry)",
              "Audited rows": f"n={len(y)}", "Threshold": str(threshold),
              "Uncertainty": f"patient bootstrap, B={B}, random_state={SEED}",
              "isitfair version": isitfair.__version__},
    mitigation_audits=mitigated, pdf=True))
REC["total_seconds"] = round(time.perf_counter() - T0, 1)
(OUT / f"{NAME}_B{B}.json").write_text(json.dumps(REC, indent=2))
print(f"report: {report}  total {REC['total_seconds']}s", flush=True)
