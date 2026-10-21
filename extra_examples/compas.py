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
# # Auditing a model on `compas`
#
# A full isitfair audit of a logistic regression fitted to the public
# `compas` dataset: discrimination, calibration, decision curves,
# Hosmer-Lemeshow, estimability, a recalibration ladder, mitigation and an
# HTML report.
#
# Before running, build the prediction table once from the repository root:
#
#     python -I extra_examples/datasets.py compas
#
# `B = 50` bootstrap replicates keep the run short; use 2,000 for results you
# intend to report.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

for _c in (Path.cwd(), *Path.cwd().parents):
    if (_c / "extra_examples" / "datasets.py").exists():
        HERE = _c / "extra_examples"
        break
sys.path.insert(0, str(HERE))
import datasets  # noqa: E402

import isitfair  # noqa: E402
from isitfair import FairnessAudit, GroupRecalibration, WassersteinPostprocessing  # noqa: E402

NAME = "compas"
B = 50
SEED = 42
META = datasets.meta(NAME)
THRESHOLD = META["threshold"]
ATTRS = datasets.ATTRS[NAME]
print(f"isitfair {isitfair.__version__}")
print(f"{NAME}: outcome {META['outcome']}; threshold {THRESHOLD}; B = {B}")

# %% [markdown]
# ## 1. Load the predictions
#
# One row per test-set patient or person: the outcome, the model's predicted
# probability, and the attributes to audit.

# %%
df = datasets.load(NAME)
print(f"n = {len(df)}, prevalence = {df['y_true'].mean():.3f}")
for a in ATTRS:
    print(f"{a}: {df[a].nunique()} levels, {int(df[a].isna().sum())} missing")

# %% [markdown]
# isitfair refuses missing subgroup labels rather than silently dropping those
# rows. If any attribute has missing labels, this cell shows the error and then
# recodes them to an explicit `"Unknown"` level.

# %%
missing_attrs = [a for a in ATTRS if df[a].isna().any()]
if missing_attrs:
    try:
        FairnessAudit(y_true=df["y_true"], y_proba=df["y_proba"],
                      sensitive_features={a: df[a] for a in ATTRS},
                      threshold=THRESHOLD, n_bootstrap=B, random_state=SEED)
    except ValueError as exc:
        print("ValueError:", exc)
    for a in missing_attrs:
        df[a] = df[a].fillna("Unknown")
print("recoded to 'Unknown':", missing_attrs or "none")

# %% [markdown]
# ## 2. Split off a calibration half for mitigation
#
# The test half is split 50/50, stratified on the outcome. Mitigation
# calibrators are fitted on one half and every audit runs on the other, so no
# row is scored by a calibrator fitted on it.

# %%
i_cal, i_eval = train_test_split(np.arange(len(df)), test_size=0.5, random_state=SEED,
                                 stratify=df["y_true"])
cal_df, eval_df = df.iloc[i_cal].reset_index(drop=True), df.iloc[i_eval].reset_index(drop=True)
# mitigate() rejects audited levels that the calibration half never saw; drop
# such rows (for example diabetes130's single "Unknown/Invalid" gender).
unseen = np.zeros(len(eval_df), bool)
for a in ATTRS:
    unseen |= ~eval_df[a].isin(set(cal_df[a]))
if unseen.any():
    print(f"dropping {int(unseen.sum())} audited row(s) with levels absent from the calibration half")
    eval_df = eval_df[~unseen].reset_index(drop=True)
print(f"calibration n = {len(cal_df)}, audit n = {len(eval_df)}")

# %% [markdown]
# ## 3. Construct the audit
#
# Each attribute is audited on its own, and the first two are also crossed
# (passing `None` for a tuple key asks isitfair to build the intersection).

# %%
y = eval_df["y_true"].to_numpy(dtype=float)
p = eval_df["y_proba"].to_numpy(dtype=float)
marginals = {a: eval_df[a].to_numpy(dtype=object) for a in ATTRS}
INTERSECTION = tuple(ATTRS[:2]) if len(ATTRS) >= 2 else None
sf = dict(marginals)
if INTERSECTION:
    sf[INTERSECTION] = None

audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features=sf, threshold=THRESHOLD,
                      n_bootstrap=B, random_state=SEED)
audit

# %% [markdown]
# ## 4. Discrimination

# %%
disc = audit.discrimination()
disc[disc.subgroup != "GAP"][["attribute", "subgroup", "n", "n_events", "auroc",
                              "auroc_ci_low", "auroc_ci_high"]]

# %% [markdown]
# ## 5. Calibration

# %%
cal = audit.calibration()
cal[cal.subgroup != "GAP"][["attribute", "subgroup", "n", "calibration_intercept",
                            "calibration_slope", "brier", "ece", "ici"]]

# %%
curves = audit.calibration_curves()
roc = audit.roc_curves()
print("calibration curves:", curves.shape, "| ROC curves:", roc.shape)

# %% [markdown]
# ## 6. Decision curve: gaps in net benefit between subgroups

# %%
dc = audit.decision_curve()
dc[dc.subgroup == "GAP"][["attribute", "gap_value", "gap_ci_low", "gap_ci_high"]]

# %% [markdown]
# ## 7. Estimability
#
# Cells with too few events for a metric are flagged rather than reported.

# %%
est = audit.estimability()
print(f"{int(est.not_estimable_final.sum())} of {len(est)} cell-metrics not reportable")

# %% [markdown]
# ## 8. Hosmer-Lemeshow

# %%
hl = audit.hosmer_lemeshow()
hl[hl.subgroup != "GAP"][["attribute", "subgroup", "n", "hl_statistic", "hl_df",
                          "hl_p_value", "n_bins_used"]]

# %% [markdown]
# ## 9. Intersection cells

# %%
audit.cell_diagnostics() if INTERSECTION else "single attribute; no intersection"

# %% [markdown]
# ## 10. Recalibration ladder on the attribute with the largest intercept gap
#
# A level with no events at all cannot be recalibrated, and isitfair says so.
# In that case the ladder moves on to the next attribute.

# %%
gaps = cal[(cal.subgroup == "GAP") & (cal.gap_metric == "max_intercept_gap")
           & cal.attribute.isin(ATTRS)]
RANKED = [str(a) for a in gaps.sort_values("gap_value", ascending=False).attribute]
ladder = None
for by in RANKED:
    try:
        ladder = audit.recalibration_ladder(by=by)
        print("ladder by", by)
        break
    except ValueError as exc:
        print(f"{by}: {exc}")
ladder

# %% [markdown]
# ## 11. Mitigation
#
# Group-wise isotonic recalibration, then Wasserstein post-processing, both
# fitted on the calibration half.

# %%
calib_data = (cal_df["y_true"].to_numpy(float), cal_df["y_proba"].to_numpy(float),
              {a: cal_df[a].to_numpy(object) for a in ATTRS})
mitigated_gr = audit.mitigate(method=GroupRecalibration(method="isotonic"), calibration_data=calib_data)
a0 = ATTRS[0]
keep = ["subgroup", "calibration_intercept", "calibration_slope", "brier", "ece"]
before, after = audit.calibration(), mitigated_gr[a0].calibration()
print("Before,", a0)
print(before[before.attribute == a0][keep].to_string(index=False))
print("\nAfter group isotonic recalibration,", a0)
print(after[after.attribute == a0][keep].to_string(index=False))

# %%
mitigated_w = audit.mitigate(method=WassersteinPostprocessing(random_state=SEED), calibration_data=calib_data)
d_after = mitigated_w[a0].discrimination()
d_after[(d_after.attribute == a0) & (d_after.subgroup == "GAP")][
    ["gap_metric", "gap_value", "gap_ci_low", "gap_ci_high"]]

# %% [markdown]
# ## 12. HTML report

# %%
REPORT = HERE / "reports" / f"{NAME}.html"
REPORT.parent.mkdir(exist_ok=True)
audit.report(out=REPORT, title=f"{NAME}: isitfair audit",
             metadata={"Dataset": NAME, "Model": "LogisticRegression",
                       "Threshold": str(THRESHOLD), "Bootstrap": f"B={B}, random_state={SEED}"},
             mitigation_audits=mitigated_gr)
print("wrote", REPORT.relative_to(HERE.parent))
