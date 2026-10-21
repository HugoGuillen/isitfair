# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # isitfair hardening: `diabetes130`
#
# Step-by-step audit of the `diabetes130` prediction table built by
# `hardening/datasets/registry.py` (see `hardening/datasets/CARDS.md`). The
# audited model is the logistic-regression column `y_proba_lr`; every stage is
# its own cell, timed with `perf_counter`. Warnings and exceptions are recorded
# per stage in `RECORD` and written to `hardening/notebooks/results/`.
#
# Environment knobs: `HARDEN_B` (bootstrap replicates, default 50) and
# `HARDEN_STRICT` (`1` makes the last cell fail on any unexpected error).

# %%
import faulthandler
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

for _c in (Path.cwd(), *Path.cwd().parents):
    if (_c / "hardening" / "datasets" / "registry.py").exists():
        REPO = _c
        break
sys.path.insert(0, str(REPO / "hardening" / "datasets"))
# Inside a Jupyter kernel sys.stderr has no file descriptor; give the watchdog a real file.
_FH = open(REPO / "hardening" / "logs" / "faulthandler_diabetes130.log", "w")
faulthandler.dump_traceback_later(600, repeat=True, file=_FH)
import registry  # noqa: E402

import isitfair  # noqa: E402
from isitfair import FairnessAudit, GroupRecalibration, WassersteinPostprocessing, report_to_pdf  # noqa: E402

NAME = "diabetes130"
B = int(os.environ.get("HARDEN_B", "50"))
STRICT = os.environ.get("HARDEN_STRICT", "0") == "1"
# Bootstrap workers for discrimination/calibration; results do not depend on it.
NJOBS = int(os.environ["HARDEN_NJOBS"]) if os.environ.get("HARDEN_NJOBS") else None
SEED = 42
OUT = REPO / "hardening" / "notebooks" / "results"
REPORTS = REPO / "hardening" / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
META = registry.meta(NAME)
THRESHOLD = META["threshold"]
RECORD = {"dataset": NAME, "B": B, "isitfair": isitfair.__version__,
          "isitfair_file": isitfair.__file__, "n_jobs": NJOBS, "stages": {}, "variants": {}}
print(f"isitfair {isitfair.__version__} from {isitfair.__file__}")
print(f"{NAME}: B={B} threshold={THRESHOLD} strict={STRICT} n_jobs={NJOBS}")


@contextmanager
def stage(name, expect_error=False):
    """Time a stage; record its warnings and any exception, then carry on."""
    entry = {"seconds": None, "warnings": [], "error": None, "expected_error": expect_error}
    RECORD["stages"][name] = entry
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield entry
        except Exception as exc:  # recorded, not raised, so later stages still run
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
            print(f"!! {name} raised {entry['error']}")
    entry["seconds"] = round(time.perf_counter() - t0, 3)
    seen = {}
    for w in caught:
        key = f"{w.category.__name__}: {str(w.message)[:300]}"
        seen[key] = seen.get(key, 0) + 1
    entry["warnings"] = [f"{k} (x{v})" for k, v in seen.items()]
    print(f"[{name}] {entry['seconds']:.2f} s, {len(caught)} warning(s)"
          + (f", ERROR {entry['error']}" if entry["error"] else ""))
    for k in entry["warnings"][:8]:
        print("   ", k)
    (OUT / f"{NAME}_B{B}.json").write_text(json.dumps(RECORD, indent=2, default=str))


# %% [markdown]
# ## 1. Load and validate inputs

# %%
with stage("load_validate"):
    df = registry.load(NAME)
    ATTRS = registry.ATTRS[NAME]
    checks = {
        "n": len(df),
        "index_is_range": isinstance(df.index, pd.RangeIndex),
        "y_true_dtype": str(df["y_true"].dtype),
        "y_true_values": sorted(df["y_true"].dropna().unique().tolist()),
        "y_true_missing": int(df["y_true"].isna().sum()),
        "y_proba_range": [float(df["y_proba_lr"].min()), float(df["y_proba_lr"].max())],
        "y_proba_missing": int(df["y_proba_lr"].isna().sum()),
        "attr_dtypes": {a: str(df[a].dtype) for a in ATTRS},
        "attr_missing": {a: int(df[a].isna().sum()) for a in ATTRS},
        "attr_levels": {a: int(df[a].nunique()) for a in ATTRS},
    }
    RECORD["checks"] = checks
    print(json.dumps(checks, indent=1))

# %% [markdown]
# A missing subgroup label must be rejected with an actionable message. If any
# attribute has missing labels, try the raw column first (recorded), then
# recode to an explicit `"Unknown"` category for the rest of the notebook.

# %%
missing_attrs = [a for a in ATTRS if df[a].isna().any()]
if missing_attrs:
    with stage("raw_missing_labels", expect_error=True):
        FairnessAudit(y_true=df["y_true"], y_proba=df["y_proba_lr"],
                      sensitive_features={a: df[a] for a in ATTRS},
                      threshold=THRESHOLD, n_bootstrap=B, random_state=SEED, n_jobs=NJOBS)
    for a in missing_attrs:
        df[a] = df[a].fillna("Unknown")
print("recoded to 'Unknown':", missing_attrs or "none")

# %%
y = df["y_true"].to_numpy(dtype=float)
p = df["y_proba_lr"].to_numpy(dtype=float)
marginals = {a: df[a].to_numpy(dtype=object) for a in ATTRS}
INTERSECTION = tuple(ATTRS[:2]) if len(ATTRS) >= 2 else None
sf = dict(marginals)
if INTERSECTION:
    sf[INTERSECTION] = None
print("intersection:", INTERSECTION)

# %% [markdown]
# ## 2. Construct the audit

# %%
with stage("construct"):
    audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features=sf, threshold=THRESHOLD,
                          n_bootstrap=B, random_state=SEED, n_jobs=NJOBS)
    print(audit)

# %% [markdown]
# ## 3. Discrimination

# %%
with stage("discrimination"):
    disc = audit.discrimination()
    print(disc[disc.subgroup != "GAP"][["attribute", "subgroup", "n", "n_events", "auroc",
                                         "auroc_ci_low", "auroc_ci_high"]].to_string(index=False))

# %% [markdown]
# ## 4. Calibration

# %%
with stage("calibration"):
    cal = audit.calibration()
    print(cal[cal.subgroup != "GAP"][["attribute", "subgroup", "n", "calibration_intercept",
                                       "calibration_slope", "brier", "ece", "ici"]].to_string(index=False))

# %% [markdown]
# ## 5. Calibration curves

# %%
with stage("calibration_curves"):
    curves = audit.calibration_curves()
    print(curves.shape)

# %% [markdown]
# ## 6. ROC curves

# %%
with stage("roc_curves"):
    roc = audit.roc_curves()
    print(roc.shape)

# %% [markdown]
# ## 7. Decision curve

# %%
with stage("decision_curve"):
    dc = audit.decision_curve()
    print(dc[dc.subgroup == "GAP"][["attribute", "gap_value", "gap_ci_low", "gap_ci_high"]].to_string(index=False))

# %% [markdown]
# ## 8. Estimability

# %%
with stage("estimability"):
    est = audit.estimability()
    print(f"{int(est.not_estimable_final.sum())} of {len(est)} cell-metrics not reportable")

# %% [markdown]
# ## 9. Hosmer-Lemeshow

# %%
with stage("hosmer_lemeshow"):
    hl = audit.hosmer_lemeshow()
    print(hl[hl.subgroup != "GAP"][["attribute", "subgroup", "n", "hl_statistic", "hl_df",
                                     "hl_p_value", "n_bins_used"]].to_string(index=False))

# %% [markdown]
# ## 10. Cell diagnostics for one intersection

# %%
with stage("cell_diagnostics"):
    if INTERSECTION:
        cells = audit.cell_diagnostics()
        print(cells.to_string(index=False))
    else:
        print("single attribute; no intersection to diagnose")

# %% [markdown]
# ## 11. Recalibration ladder on the attribute with the largest intercept gap

# %%
gaps = cal[(cal.subgroup == "GAP") & (cal.gap_metric == "max_intercept_gap")
           & cal.attribute.isin(ATTRS)]
RANKED = [str(a) for a in gaps.sort_values("gap_value", ascending=False).attribute]


def zero_event_levels(attr):
    ev = pd.Series(y).groupby(marginals[attr]).sum()
    return [str(k) for k, v in ev.items() if v == 0]


LADDER_BY = RANKED[0]
ZERO = zero_event_levels(LADDER_BY)
print("largest intercept gap:", LADDER_BY, "| zero-event levels:", ZERO or "none")
# A level without events cannot be recalibrated; isitfair refuses (by design).
with stage("recalibration_ladder", expect_error=bool(ZERO)):
    ladder = audit.recalibration_ladder(by=LADDER_BY)
    print(ladder.to_string(index=False)[:3000])

# %%
FALLBACK = next((a for a in RANKED[1:] if not zero_event_levels(a)), None) if ZERO else None
if FALLBACK:
    with stage("recalibration_ladder_fallback"):
        ladder = audit.recalibration_ladder(by=FALLBACK)
        print("by", FALLBACK)
        print(ladder.to_string(index=False)[:3000])
else:
    print("no fallback needed" if not ZERO else "no attribute without zero-event levels")

# %% [markdown]
# ## 12. Mitigation on a held-out calibration split
#
# support2 uses the committed calibration partition. Every other dataset splits
# its audited test half 50/50 (stratified on the outcome, `random_state=42`):
# the calibrators are fitted on one half and the before/after audit runs on the
# other, so no row is scored by a calibrator fitted on it.

# %%
with stage("mitigation_split"):
    if NAME == "support2":
        cal_df = pd.read_csv(registry.PRED / "support2_calib.csv", dtype={a: object for a in ATTRS})
        eval_df = df
    else:
        from sklearn.model_selection import train_test_split
        i_cal, i_eval = train_test_split(np.arange(len(df)), test_size=0.5, random_state=SEED,
                                         stratify=df["y_true"])
        cal_df, eval_df = df.iloc[i_cal].reset_index(drop=True), df.iloc[i_eval].reset_index(drop=True)
    for a in ATTRS:
        cal_df[a] = cal_df[a].fillna("Unknown")
    # mitigate() rejects audited levels the calibration half never saw (diabetes130:
    # gender "Unknown/Invalid"). Drop those audited rows and say so; a no-op elsewhere.
    unseen = np.zeros(len(eval_df), bool)
    for a in ATTRS:
        unseen |= ~eval_df[a].fillna("Unknown").isin(set(cal_df[a]))
    if unseen.any():
        print(f"dropping {int(unseen.sum())} audited rows with levels absent from the calibration half")
        eval_df = eval_df[~unseen].reset_index(drop=True)
    calib_data = (cal_df["y_true"].to_numpy(float), cal_df["y_proba_lr"].to_numpy(float),
                  {a: cal_df[a].to_numpy(object) for a in ATTRS})
    sf_eval = {a: eval_df[a].to_numpy(object) for a in ATTRS}
    if INTERSECTION:
        sf_eval[INTERSECTION] = None
    audit_eval = audit if NAME == "support2" else FairnessAudit(
        y_true=eval_df["y_true"].to_numpy(float), y_proba=eval_df["y_proba_lr"].to_numpy(float),
        sensitive_features=sf_eval, threshold=THRESHOLD, n_bootstrap=B, random_state=SEED, n_jobs=NJOBS)
    print(f"calibration n={len(cal_df)}, evaluation n={len(eval_df)}")

# %%
with stage("mitigate_group_recalibration"):
    mitigated_gr = audit_eval.mitigate(method=GroupRecalibration(method="isotonic"),
                                       calibration_data=calib_data)
    a0 = ATTRS[0]
    keep = ["subgroup", "calibration_intercept", "calibration_slope", "brier", "ece"]
    before, after = audit_eval.calibration(), mitigated_gr[a0].calibration()
    print("BEFORE", a0); print(before[before.attribute == a0][keep].to_string(index=False))
    print("AFTER (group isotonic)"); print(after[after.attribute == a0][keep].to_string(index=False))

# %%
with stage("mitigate_wasserstein"):
    mitigated_w = audit_eval.mitigate(method=WassersteinPostprocessing(random_state=SEED),
                                      calibration_data=calib_data)
    d_after = mitigated_w[a0].discrimination()
    print(d_after[(d_after.attribute == a0) & (d_after.subgroup == "GAP")][
        ["gap_metric", "gap_value", "gap_ci_low", "gap_ci_high"]].to_string(index=False))

# %% [markdown]
# ## 13. Report: HTML, then PDF

# %%
REPORT = REPORTS / f"{NAME}_B{B}.html"
with stage("report_html"):
    audit_eval.report(out=REPORT, title=f"{NAME}: hardening audit (B={B})",
                      metadata={"Dataset": NAME, "Model": "LogisticRegression (registry)",
                                "Threshold": str(THRESHOLD), "Bootstrap": f"B={B}, random_state={SEED}"},
                      mitigation_audits=mitigated_gr)
    print(REPORT, REPORT.stat().st_size, "bytes")

# %%
with stage("report_pdf"):
    pdf_path = report_to_pdf(REPORT)
    print(pdf_path, pdf_path.stat().st_size, "bytes")

# %% [markdown]
# ## 14. Input-type variants
#
# The same audit (B = 20 to keep this cheap) with inputs passed in the forms
# real pipelines produce. Each variant is classified as `raises`, `warns`,
# `ok-identical` (results equal the plain-numpy baseline exactly), or
# `ok-DIFFERENT` (runs silently but gives different numbers: a silent
# misbehaviour unless the input itself changed, as in the ties variant).

# %%
VB = 20


def run_variant(name, y_in, p_in, sf_in, compare=True, ref=None):
    out = {"status": None, "warnings": [], "error": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            a = FairnessAudit(y_true=y_in, y_proba=p_in, sensitive_features=sf_in,
                              threshold=THRESHOLD, n_bootstrap=VB, random_state=SEED, n_jobs=NJOBS)
            frames = {"discrimination": a.discrimination(), "calibration": a.calibration(),
                      "hosmer_lemeshow": a.hosmer_lemeshow(), "decision_curve": a.decision_curve()}
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            frames = None
    out["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})
    if frames is None:
        out["status"] = "raises"
    else:
        num_bad = {k: int(np.isinf(f.select_dtypes("number").to_numpy(float)).sum()) for k, f in frames.items()}
        out["inf_values"] = num_bad
        if ref is not None and compare:
            same = all(frames[k].equals(ref[k]) for k in frames)
            out["identical_to_baseline"] = same
            out["status"] = ("warns, " if out["warnings"] else "") + ("ok-identical" if same else "ok-DIFFERENT")
            if not same:
                diffs = {}
                for k in frames:
                    fa, fb = frames[k], ref[k]
                    if fa.shape == fb.shape:
                        na = fa.select_dtypes("number").to_numpy(float)
                        nb = fb.select_dtypes("number").to_numpy(float)
                        diffs[k] = float(np.nanmax(np.abs(na - nb))) if na.size else 0.0
                    else:
                        diffs[k] = f"shape {fa.shape} vs {fb.shape}"
                out["max_abs_diff"] = diffs
        else:
            out["status"] = ("warns, " if out["warnings"] else "") + "ok"
    RECORD["variants"][name] = out
    print(f"{name:<28} {out['status']}" + (f"  {out['error']}" if out["error"] else ""))
    for w in out["warnings"][:5]:
        print("     ", w)
    (OUT / f"{NAME}_B{B}.json").write_text(json.dumps(RECORD, indent=2, default=str))
    return frames


with stage("variants"):
    base = run_variant("baseline_numpy", y, p, sf, compare=False)
    s_attr = {a: pd.Series(v, dtype="category") for a, v in marginals.items()}
    run_variant("category_attributes", y, p, {**s_attr, **({INTERSECTION: None} if INTERSECTION else {})}, ref=base)
    run_variant("Int64_outcome", pd.Series(y.astype(int), dtype="Int64"), p, sf, ref=base)
    run_variant("bool_outcome", y.astype(bool), p, sf, ref=base)
    run_variant("bool_outcome_series", pd.Series(y.astype(bool)), p, sf, ref=base)
    perm = np.random.default_rng(SEED).permutation(len(y))
    idx = pd.Index(perm + 10_000)
    run_variant("shuffled_index_consistent",
                pd.Series(y, index=idx), pd.Series(p, index=idx),
                {**{a: pd.Series(v, index=idx) for a, v in marginals.items()},
                 **({INTERSECTION: None} if INTERSECTION else {})}, ref=base)
    # y sorted/shuffled with its labels, p left in the original order: the
    # classic silent misalignment, which isitfair is documented to reject.
    run_variant("shuffled_index_misaligned",
                pd.Series(y).iloc[perm], pd.Series(p), sf, ref=base)
    p_ties = np.round(p, 2)
    k = max(2, len(p) // 50)  # ~2% of rows exactly 0 or 1, never all of them
    pos = np.random.default_rng(SEED).choice(len(p), size=k, replace=False)
    p_ties[pos[: k // 2]] = 0.0
    p_ties[pos[k // 2 :]] = 1.0
    run_variant("ties_and_exact_0_1", y, p_ties, sf, compare=False)

# %% [markdown]
# ## 15. Summary

# %%
unexpected = {k: v["error"] for k, v in RECORD["stages"].items()
              if v["error"] and not v["expected_error"]}
RECORD["unexpected_errors"] = unexpected
RECORD["total_seconds"] = round(sum(v["seconds"] or 0 for v in RECORD["stages"].values()), 2)
(OUT / f"{NAME}_B{B}.json").write_text(json.dumps(RECORD, indent=2, default=str))
print(pd.DataFrame({k: {"seconds": v["seconds"], "n_warnings": len(v["warnings"]),
                        "error": v["error"]} for k, v in RECORD["stages"].items()}).T.to_string())
print("unexpected errors:", unexpected or "none")
if STRICT:
    assert not unexpected, unexpected
