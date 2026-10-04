# ---
# jupyter:
#   jupytext:
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
# # Generating an audit report — SUPPORT2 ICU mortality
#
# This is the example script the *isitfair* manuscript refers to. It runs a
# complete subgroup deployment-readiness audit of an in-hospital mortality model
# and writes a self-contained TRIPOD+AI HTML report.
#
# **If you want to audit your own model, start here.** Replace section 2 with
# your own outcome, predictions and subgroup labels; the rest runs unchanged.
#
# ## The cohort
#
# SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks
# of Treatment) is 9,105 critically ill adults from five US medical centres,
# distributed by the UCI Machine Learning Repository under CC BY 4.0 with no
# credentialing. The outcome is in-hospital mortality (`hospdead`).
#
# The audit runs on a held-out **test partition of n = 1,821**. A separate
# calibration partition is held back so recalibration is fitted on data the
# audit never sees — the leak-free arrangement any mitigation comparison needs.
#
# The three audit axes are **sex**, **age group** and **race/ethnicity**.
# SUPPORT2 records all three, so a single audit exercises a sociodemographic
# axis, a clinical one, and the intersection of two of them.
#
# Every random operation is seeded, so re-running this reproduces the shipped
# report exactly.
#
# ## Running it
#
# ```bash
# pip install -e ".[examples]"
# python examples/support2_mortality_audit.py          # ~12 min at B = 2000
# ISITFAIR_FAST=1 python examples/support2_mortality_audit.py   # ~1 min at B = 200
# ```

# %% [markdown]
# ## 1. Setup

# %%
import importlib.util
import sys
from pathlib import Path

# Make examples/ importable whether this runs as a script or as the paired notebook.
_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "examples" / "support2_data.py").exists():
        sys.path.insert(0, str(_candidate / "examples"))
        break

import isitfair
from isitfair import FairnessAudit, GroupRecalibration
from support2_data import (
    SEED,
    THRESHOLD,
    build_model_matrix,
    fit_model,
    load_support2,
    make_sensitive_attributes,
    n_bootstrap,
    repo_root,
    results_dir,
    split_cohort,
)

B = n_bootstrap()
RESULTS = results_dir()

print(f"isitfair {isitfair.__version__}")
print(f"seed {SEED} · threshold {THRESHOLD} · bootstrap replicates {B}")
print(f"results -> {RESULTS.relative_to(repo_root())}/")

# %% [markdown]
# ## 2. The four things you must bring
#
# An audit needs four inputs. The package works from predictions, so your
# feature matrix stays with you.
#
# | Input | Here |
# |---|---|
# | `y_true` — binary outcome | in-hospital mortality |
# | `y_proba` — predicted risk, **out of fold** | logistic regression fitted on a disjoint training partition |
# | `sensitive_features` — subgroup labels | sex, age group, race/ethnicity |
# | `threshold` — a prespecified operating point | 0.30 |
#
# The predictions must be out of fold. In-sample predictions are optimistic,
# and *differentially* so: the optimism is largest in the sparsest subgroups,
# which is exactly where an audit is trying to look.
#
# Below, each stage of the pipeline is one call. The implementations live in
# `examples/support2_data.py` so that this script, the paper-analysis notebook
# and the tutorial notebook all audit an identical test partition.

# %%
df = load_support2()
print(f"N = {len(df)}   in-hospital mortality = {df['hospdead'].mean():.3f}")

# %%
# Sex and race are audit axes; the design matrix is built without them.
attrs = make_sensitive_attributes(df)
for name, values in attrs.items():
    import numpy as np

    labels, counts = np.unique(values, return_counts=True)
    print(f"{name:>15}: " + ", ".join(f"{lab} n={cnt}" for lab, cnt in zip(labels, counts)))

# %%
X = build_model_matrix(df)
y = df["hospdead"].to_numpy().astype(float)
print(f"design matrix: {X.shape[1]} columns, {int(X.isnull().sum().sum())} missing")
print(f"sex in features?  {'sex' in X.columns}")
print(f"race in features? {'race' in X.columns}")

# %%
idx_train, idx_calib, idx_test = split_cohort(X, y, seed=SEED)
model, scaler = fit_model(X, y, idx_train, seed=SEED)


def predict(idx):
    return model.predict_proba(scaler.transform(X.iloc[idx]))[:, 1]


y_test, p_test = y[idx_test], predict(idx_test)
y_calib, p_calib = y[idx_calib], predict(idx_calib)
attrs_test = {k: v[idx_test] for k, v in attrs.items()}
attrs_calib = {k: v[idx_calib] for k, v in attrs.items()}

for label, part in (("train", idx_train), ("calib", idx_calib), ("test", idx_test)):
    print(f"{label:>5}: n={len(part):>5}  events={int(y[part].sum()):>4}  prev={y[part].mean():.3f}")

# %% [markdown]
# ## 3. Constructing the audit
#
# `FairnessAudit` is lazy: the bootstrap runs the first time you ask for an
# axis, and every axis afterwards reuses the same replicate index array. That is what makes a gap statistic a *reduction over*
# the per-subgroup distributions rather than an independent resample of them.

# %%
audit = FairnessAudit(
    y_true=y_test,
    y_proba=p_test,
    sensitive_features=attrs_test,
    threshold=THRESHOLD,
    n_bootstrap=B,
    random_state=SEED,
)
print(audit)

# %% [markdown]
# ## 4. Discrimination — can the model tell them apart?
#
# AUROC and AUPRC are threshold-free; the rates below them are properties of
# the decision rule at t = 0.30. Note that AUPRC's no-skill baseline is each
# subgroup's own prevalence, not 0.5, so ranking subgroups by AUPRC largely
# ranks their prevalences.

# %%
disc = audit.discrimination()
cols = ["attribute", "subgroup", "n", "n_events", "prevalence", "auroc", "auroc_ci_low", "auroc_ci_high"]
print(disc[disc.subgroup != "GAP"][cols].to_string(index=False))

# %% [markdown]
# ### The four parity gaps
#
# Equal opportunity (sensitivity), predictive equality (FPR), statistical parity
# (flag rate) and equalized odds (the max of the first two). All four are
# properties of the decision rule at one threshold — move the threshold and they
# all move.
#
# Read each gap next to the within-group numbers above it: parity is equally
# consistent with both groups being served well and with both being served
# badly, and only the within-group rows tell you which. Every frame carries
# both.

# %%
gaps = disc[disc.subgroup == "GAP"][["attribute", "gap_metric", "gap_value", "gap_ci_low", "gap_ci_high"]]
print(gaps.to_string(index=False))

# %% [markdown]
# ### ROC curves
#
# Per-subgroup ROC on a fixed false-positive-rate grid with bootstrap bands.
# Where curves cross, the ranking between subgroups changes with the operating
# point — detail a single summary AUROC cannot show.

# %%
roc = audit.roc_curves()
print(f"{roc.shape[0]} rows across {roc['attribute'].nunique()} axes")

# %% [markdown]
# ## 5. Calibration — is the number right?
#
# Calibration is where subgroup disparities most often hide in clinical models.
# A model that never saw race as a feature can still be miscalibrated across
# racial groups, because the features it *did* see are distributed differently
# (Obermeyer et al., *Science* 2019).
#
# Read the intercept and slope together. The intercept α is
# calibration-in-the-large with the slope pinned at 1; it is prevalence
# dependent, so it is only interpretable against a stated population. The slope
# β is fitted with both terms free: β < 1 means the predictions are spread too
# wide, β > 1 that they are too compressed.

# %%
cal = audit.calibration()
print(
    cal[["attribute", "subgroup", "n", "calibration_intercept", "calibration_slope", "brier", "ece", "ici"]]
    .to_string(index=False)
)

# %%
curves = audit.calibration_curves()
print(f"LOESS calibration curves: {curves.shape[0]} rows across {curves['attribute'].nunique()} axes")

# %% [markdown]
# ### Hosmer–Lemeshow
#
# Reported with everything Kramer & Zimmerman (2007) require alongside it: the
# subgroup's *n*, the full per-decile observed-versus-expected table, and the
# aggregate scores above.
#
# Read each statistic beside its own *n*. The value scales with sample size, so
# it compares a subgroup against its own expected counts rather than against
# another subgroup. And a large p-value means miscalibration was not detected,
# which in a small subgroup is the expected outcome either way — the decile
# table below is what carries the detail.
#
# `df_mode` defaults to `"validation"`, i.e. df = g rather than g − 2, because
# out-of-fold predictions estimate nothing on this data. Using g − 2 here
# roughly doubles the false miscalibration rate.

# %%
hl = audit.hosmer_lemeshow()
print(
    hl[hl.subgroup != "GAP"][
        ["attribute", "subgroup", "n", "n_events", "hl_statistic", "hl_df",
         "hl_p_value", "n_bins_used", "bins_reduced", "decile_slope", "decile_intercept"]
    ].to_string(index=False)
)

# %%
hl_deciles = audit.hosmer_lemeshow(detail=True)
print(f"per-decile observed vs expected: {hl_deciles.shape[0]} rows")
print(hl_deciles.head(10).to_string(index=False))

# %% [markdown]
# ## 6. Clinical utility — is it worth acting on?
#
# A decision curve asks whether acting on the model beats the two strategies
# that need no model at all: treat everyone, or treat no one. Net benefit
# subtracts false positives at the exchange rate the threshold implies —
# t/(1−t), which at t = 0.30 is about 0.43. That ratio is arithmetic from the
# threshold, not an independently elicited preference.
#
# The treat-all reference uses **each subgroup's own prevalence**, so every
# subgroup is compared against the no-model strategy as it would actually
# perform in that subgroup.

# %%
dc = audit.decision_curve(zero_crossing=True)
print(
    dc[dc.subgroup != "GAP"][
        ["attribute", "subgroup", "threshold", "net_benefit", "standardized_net_benefit", "treat_all_nb"]
    ].head(12).to_string(index=False)
)

# %% [markdown]
# ## 7. How much of this is noise?
#
# `estimability()` is the single gate, and it answers the question in one
# table: which estimates this cohort supports an interval for. A cell is
# reportable when it clears **both** rules — the pre-registered event floor for
# that metric, and the bootstrap replicate rule (at most 10% of replicates
# non-estimable). A gap inherits the verdict of both constituent subgroups, and
# suppression withholds the **interval, keeping the point estimate**: the value
# is real, it simply has no defensible uncertainty around it.

# %%
est = audit.estimability()
unreportable = est[est.not_estimable_final]
print(f"{len(unreportable)} of {len(est)} cell-metrics are not reportable")
print(
    unreportable[["attribute", "subgroup", "metric", "n", "n_events", "event_floor", "below_floor", "proportion_nonestimable"]]
    .to_string(index=False)
)

# %% [markdown]
# ## 8. Intersectional cells
#
# Passing a tuple key builds the interaction. Sex × race/ethnicity gives 10
# cells across a wide range of sizes, which is exactly the case empirical-Bayes
# shrinkage exists for: cells below the event threshold are shrunk toward the
# first constituent's marginal rates, only rate metrics are shrunk (AUROC and
# AUPRC keep their own values), and shrunken cells carry NaN intervals so the
# borrowing is visible.

# %%
audit_inter = FairnessAudit(
    y_true=y_test,
    y_proba=p_test,
    sensitive_features={**attrs_test, ("sex", "race_ethnicity"): None},
    threshold=THRESHOLD,
    n_bootstrap=B,
    random_state=SEED,
)
cells = audit_inter.cell_diagnostics()
print(cells.to_string(index=False))
print(f"\n{int(cells['shrunk'].sum())} of {len(cells)} cells were shrunk")

# %% [markdown]
# ## 9. Mitigation
#
# Group-specific isotonic recalibration, fitted on the held-back calibration
# partition. Isotonic is a monotone step function, so it preserves rank within
# a subgroup — AUROC is untouched by construction — and only moves the numbers.
#
# The check that matters is **levelling down** (Pfohl et al. 2021): confirm the
# disadvantaged group actually improved, rather than the well-calibrated groups
# converging downward to meet it.

# %%
audit_pre = FairnessAudit(
    y_true=y_test,
    y_proba=p_test,
    sensitive_features=attrs_test,
    threshold=THRESHOLD,
    n_bootstrap=B,
    random_state=SEED,
)
mitigated = audit_pre.mitigate(
    method=GroupRecalibration(method="isotonic"),
    calibration_data=(y_calib, p_calib, attrs_calib),
)

keep = ["subgroup", "calibration_intercept", "calibration_slope", "brier", "ece"]
before = audit_pre.calibration()
after = mitigated["race_ethnicity"].calibration()
print("BEFORE (race_ethnicity):")
print(before[before.attribute == "race_ethnicity"][keep].to_string(index=False))
print("\nAFTER group-specific isotonic recalibration:")
print(after[after.attribute == "race_ethnicity"][keep].to_string(index=False))

# %% [markdown]
# ## 10. The report
#
# One self-contained HTML file, no JavaScript, with 300-DPI PNGs written to a
# sibling `_figures/` directory. It carries all three axes, the gap statistics,
# the per-decile H–L tables, the intersectional cell diagnostics, the
# before/after mitigation section and a TRIPOD+AI appendix.
#
# `pdf=True` also typesets it as a PDF beside the HTML. That needs the `pdf`
# extra (`pip install -e ".[pdf]"`, which installs WeasyPrint), so the cell
# asks for the PDF only when WeasyPrint is importable.

# %%
write_pdf = importlib.util.find_spec("weasyprint") is not None
report_path = audit.report(
    out=RESULTS / "support2_audit_report.html",
    title="ICU mortality (SUPPORT2) — subgroup deployment-readiness audit",
    metadata={
        "Model": "Logistic regression (13 clinical features + disease class, cancer status)",
        "Cohort": f"SUPPORT2, UCI id 880 (N=9,105); audited on held-out test partition n={len(y_test)}",
        "Outcome": f"In-hospital mortality ({y_test.mean():.1%} in the test partition)",
        "Threshold": str(THRESHOLD),
        "Predictions": "Out of fold — fitted on a disjoint 60% training partition",
        "Uncertainty": f"Patient-level bootstrap, B={B}, random_state={SEED}",
        "isitfair version": isitfair.__version__,
    },
    mitigation_audits=mitigated,
    pdf=write_pdf,
)
print(f"report written to {report_path.relative_to(repo_root())}")
if write_pdf:
    print(f"PDF written to {report_path.with_suffix('.pdf').relative_to(repo_root())}")
else:
    print("PDF skipped: install the pdf extra to write one")

# %% [markdown]
# ## 11. Adapting this to your own model
#
# 1. Replace section 2 with your data. You need `y_true`, out-of-fold
#    `y_proba`, and a dict of subgroup label arrays.
# 2. Set `threshold` to your clinical operating point, and prespecify it —
#    every rate metric and the whole utility axis move with it.
# 3. Set `event_floors` for your setting if the defaults (AUROC 25,
#    calibration intercept 25, slope 100, net benefit 50) do not fit.
# 4. Run sections 3–10 unchanged.
#
# The next two notebooks go further: `02_paper_analysis_support2.ipynb`
# works through every analysis the package supports on this cohort, and
# `03_tutorial_support2.ipynb` is the tutorial deck made runnable.
#
# ### References
#
# - Knaus WA, et al. The SUPPORT prognostic model. *Ann Intern Med* 1995;122:191–203.
# - Obermeyer Z, et al. Dissecting racial bias in an algorithm used to manage
#   the health of populations. *Science* 2019;366:447–53.
# - Kramer AA, Zimmerman JE. Assessing the calibration of mortality benchmarks
#   in critical care. *Crit Care Med* 2007;35:2052–6.
# - Pfohl SR, et al. An empirical characterization of fair machine learning for
#   clinical risk prediction. *J Biomed Inform* 2021;113:103621.
