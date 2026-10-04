# Intersectional analysis

Marginal subgroup analysis (e.g., by sex alone or age alone) can mask disparities that appear only at intersections. An infection model might perform well for women overall and for elderly patients overall, but poorly for *elderly women* specifically — because the training data contained few elderly women with infections.

isitfair supports intersectional analysis natively through tuple keys in `sensitive_features`.

## Specifying intersections

Pass a tuple of marginal attribute names as a key, with `None` as the value to auto-compute labels:

```python
audit = FairnessAudit(
    y_true=y_true,
    y_proba=y_proba,
    sensitive_features={
        "sex": sex,
        "age_group": age,
        "frailty": frailty,
        ("sex", "age_group"): None,          # 2-way interaction
        ("sex", "age_group", "frailty"): None,  # 3-way interaction
    },
    threshold=0.15,
    n_bootstrap=200,
    random_state=42,
)
```

When the value is `None`, isitfair joins the marginal arrays into composite labels like `"female / 65-79"`. You can also pass a pre-computed label array if you want custom labels.

**Requirement**: all marginal attributes referenced in a tuple key must also be present as string keys. Passing `("sex", "ethnicity"): None` without a `"ethnicity"` key raises `ValueError`.

## How intersections appear in outputs

Interaction attributes follow the same structure as marginals. In `discrimination()`, `calibration()`, and `decision_curve()` outputs:

- The `attribute` column shows the interaction name: `"sex × age_group"`.
- The `subgroup` column shows the cell label: `"female / 65-79"`.
- Overall and GAP rows are computed for the interaction attribute just like for marginals.

```python
disc = audit.discrimination()
interaction_rows = disc[disc["attribute"] == "sex × age_group"]
```

## The sparse-cell problem

Intersections create many small cells. A 2-way cross of sex (2 levels) × age (3 levels) produces 6 cells; adding frailty (3 levels) yields 18. Some cells may contain very few events (e.g., 3 infections among 40 frail elderly women), making point estimates unreliable and bootstrap CIs meaningless.

isitfair handles this with **empirical Bayes shrinkage** for rate metrics.

## Shrinkage

When a cell has fewer events than `intersectional_min_events` (default: 50), isitfair shrinks rate metrics (prevalence, sensitivity, specificity, PPV, NPV) toward the corresponding marginal estimate using a Beta-Binomial posterior:

$$\hat{p}_{\text{shrunk}} = \frac{\alpha + \text{successes}}{\alpha + \beta + \text{trials}}$$

where α = target × prior_strength, β = (1 − target) × prior_strength, the target is the marginal subgroup's rate, and prior_strength is calibrated from the data.

This means:
- **Large cells**: shrinkage has negligible effect — the cell's own data dominates.
- **Small cells**: the estimate is pulled toward the marginal rate, reducing variance.
- **AUROC and AUPRC are never shrunk** — ranking metrics don't have a natural Beta-Binomial interpretation.
- **Bootstrap CIs are set to NaN** for shrunk cells, because the shrinkage already regularises the estimate.

The `shrunk` column in all output DataFrames indicates which rows were shrunk (`True`/`False`).

## Cell diagnostics

`FairnessAudit.cell_diagnostics()` returns a diagnostic DataFrame for interaction cells:

```python
diag = audit.cell_diagnostics()
print(diag[["attribute", "subgroup", "n", "n_events",
            "shrunk", "shrinkage_factor"]].to_string(index=False))
```

| Column | Description |
|---|---|
| `attribute` | Interaction attribute name |
| `subgroup` | Cell label |
| `n` | Cell sample size |
| `n_events` | Number of events in cell |
| `shrunk` | Whether shrinkage was applied |
| `marginal_rate` | Prevalence from the first constituent's matching marginal subgroup |
| `cell_rate` | Raw prevalence in the cell (n_events / n) |
| `shrinkage_factor` | prior_strength / (prior_strength + n); 0 = no shrinkage, 1 = fully shrunk to marginal |

## Controlling the threshold

Set `intersectional_min_events` in the constructor to control when shrinkage activates:

```python
# More aggressive shrinkage (shrink cells with < 100 events)
audit = FairnessAudit(..., intersectional_min_events=100)

# No shrinkage (not recommended for small cells)
audit = FairnessAudit(..., intersectional_min_events=0)
```

## Determinism guarantee

Adding interaction attributes does **not** change marginal results. The RNG state is managed so that marginal bootstrap CIs remain bit-identical whether or not intersections are included. This makes intersectional analysis safe to add to an existing audit without invalidating previous results.
