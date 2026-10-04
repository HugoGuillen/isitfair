# Auditing your own model

This page takes you from a table of your own predictions to an audit report.
A ready-to-edit script that does every step below is in
[`examples/audit_your_own_data.py`](https://github.com/HugoGuillen/isitfair/blob/main/examples/audit_your_own_data.py).

## What you need

isitfair does not train or run your model. It evaluates predictions you have
already made. You need one row per patient with:

| Input | What it is | Typical source |
|---|---|---|
| `y_true` | the observed outcome, coded 0/1 | a column of your cohort table |
| `y_proba` | the model's predicted **probability** of the outcome, between 0 and 1 | `model.predict_proba(X_test)[:, 1]`, or a risk column exported from another tool |
| `sensitive_features` | one array of group labels per audit axis | columns such as sex, age group, ethnicity |
| `threshold` | the risk above which a patient would be acted on | a clinical decision, fixed before you look at the results |

**The predictions must come from patients the model was not trained on**: a
held-out test set, or out-of-fold predictions from cross-validation. In-sample
predictions are optimistic, and most optimistic in the smallest subgroups, which
are exactly the ones an audit is trying to look at.

The model can be anything (logistic regression, gradient boosting, a deep
network, a published score, a model in R or SAS) as long as you can export its
predicted probabilities.

## From a CSV to an audit

```python
import pandas as pd
from isitfair import FairnessAudit

df = pd.read_csv("my_predictions.csv")

# 1. Rows without an outcome or a prediction cannot be audited.
df = df.dropna(subset=["outcome", "predicted_risk"]).reset_index(drop=True)

# 2. Continuous attributes must be binned into groups.
df["age_group"] = pd.cut(df["age"], [0, 45, 65, 80, 120], right=False)

# 3. Missing group labels are not allowed: drop the rows or name the category.
#    astype("string") keeps a missing value missing (astype(str) would turn it
#    into the text "nan", which would then be audited as a subgroup).
for col in ["sex", "age_group", "ethnicity"]:
    df[col] = df[col].astype("string").fillna("unknown")

audit = FairnessAudit(
    y_true=df["outcome"],
    y_proba=df["predicted_risk"],
    sensitive_features={
        "sex": df["sex"],
        "age_group": df["age_group"],
        "ethnicity": df["ethnicity"],
        ("sex", "age_group"): None,      # optional: an intersection
    },
    threshold=0.10,
    n_bootstrap=200,                     # quick first pass; use 2000 for the final run
    random_state=42,
)

print(audit)
disc = audit.discrimination()
cal = audit.calibration()
dca = audit.decision_curve()
audit.report("audit_report.html", title="My model: subgroup audit")
```

## Input rules

- **Outcome**: binary, coded 0/1 (booleans work too). Recode text outcomes
  first, e.g. `(df["status"] == "dead").astype(int)`. Multiclass, survival and
  continuous outcomes are not supported.
- **Predictions**: probabilities in [0, 1]. Pass the positive-class column of
  `predict_proba`, not the two-column array and not hard 0/1 labels (the audit
  warns if it receives only 0s and 1s). Logits or scores must be converted to
  probabilities first.
- **No missing values** in any input. The audit raises an error naming the input
  and the number of missing values.
- **Alignment is by position.** Element *i* of every input must be the same
  patient. Taking every input as a column of one DataFrame is the simplest way
  to guarantee this. Passing pandas Series that hold the same index in different
  orders raises an error rather than silently mismatching rows.
- **Group labels** can be strings, numbers or categories. Numeric attributes
  with more than 20 distinct values trigger a warning, because every value would
  become its own subgroup.
- **Intersections** are tuple keys whose constituents are also given as
  separate keys; `None` builds the cell labels for you (see
  {doc}`intersectional`).

## Choosing the threshold

The threshold sets sensitivity, specificity, PPV, NPV and the single-threshold
net benefit. Choose it from the clinical decision the model supports (for
example, the risk at which a patient would get a preventive treatment), and fix
it before looking at the audit. Decision curves cover a range of thresholds
regardless; the default range is 0.02 to 0.40, which suits low-prevalence
outcomes. Pass `audit.decision_curve(thresholds=...)` to change it.

## Runtime

The bootstrap dominates the cost. As a guide, on a laptop:

| Patients | Axes | `n_bootstrap` | Time for the full report |
|---:|---:|---:|---|
| 1,000 | 1 | 200 | under 10 seconds |
| 1,821 | 3 + 1 intersection | 200 | about 1 minute |
| 1,821 | 3 + 1 intersection | 2,000 | about 12 minutes |

Time grows with the number of patients, axes and replicates. Start with
`n_bootstrap=200` to check that everything runs, then use 2,000 for the
results you report. `n_jobs=-1` spreads the discrimination and calibration
bootstrap over all CPU cores; results are identical whatever `n_jobs` is.

## Reading the output

- Every result is a long-format DataFrame with `attribute` and `subgroup`
  columns. Each attribute has an `Overall` row, one row per subgroup, and `GAP`
  rows holding the between-subgroup gap statistics.
- `audit.estimability()` says which estimates carry a confidence interval. A
  subgroup with too few events keeps its point estimate but loses its interval;
  the `rule` column says why (see {doc}`estimability`). With small subgroups,
  expect many cells without an interval: this is the audit telling you what your
  data can support.
- The default event floors (25 events for AUROC and the calibration intercept,
  100 for the calibration slope, 50 for net benefit) were pre-registered for the
  companion study. Change them with `event_floors=` if your setting justifies
  it, and report the values you used.
- In intersections, cells with fewer than `intersectional_min_events` events
  (default 50) are shrunk towards their marginal subgroup; the `shrunk` column
  marks them.

## Mitigation needs a separate calibration set

Post-hoc mitigation (`GroupRecalibration`, `WassersteinPostprocessing`) is
fitted on data other than the data being audited:

```python
from isitfair import GroupRecalibration

mitigated = audit.mitigate(
    GroupRecalibration(method="isotonic"),
    calibration_data=(y_calib, p_calib, {"sex": sex_calib, "age_group": age_calib}),
)
audit.report("audit_report.html", mitigation_audits=mitigated)
```

`y_calib`, `p_calib` and the group arrays come from a calibration split that is
neither the model's training data nor the audited test set, and the dict keys
must match the audit's attributes. See {doc}`mitigation`.

## Troubleshooting

| Message | Fix |
|---|---|
| `` `sensitive_features['x']` has N missing value(s) `` | `df[col].fillna("unknown")`, or drop those rows from every input |
| `` `y_proba` has N missing value(s) `` / `` `y_true` has N missing value(s) `` | drop those rows from every input |
| `` `y_true` must be numeric `` / `` must contain only 0 and 1 `` | recode the outcome to 0/1 |
| `` `y_proba` must be 1-dimensional, got shape (n, 2) `` | pass `predict_proba(X)[:, 1]` |
| `` `y_proba` values must be in [0, 1] `` | convert scores or logits to probabilities |
| `` ... same index labels in a different order `` | build every input from one DataFrame, or reorder with `.loc[...]` |
| warning: `` is numeric with N distinct values `` | bin the variable with `pd.cut` |
| `` cell label(s) ... cover more than one combination `` | pass `None` for the intersection and let isitfair build the labels |
| many cells "reported without an interval" | expected with few events per subgroup; see {doc}`estimability` |
