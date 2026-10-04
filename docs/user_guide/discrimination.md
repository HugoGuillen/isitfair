# Discrimination

The discrimination axis measures how well the model separates positive from negative cases within each subgroup.

## Metrics

`FairnessAudit.discrimination()` returns a long-format DataFrame with these columns:

| Metric | What it measures | Range |
|---|---|---|
| `auroc` | Ranking ability (threshold-independent) | 0–1; 0.5 = random |
| `auprc` | Ranking with prevalence sensitivity | 0–1; baseline = prevalence |
| `sensitivity` | True positive rate at the threshold | 0–1 |
| `specificity` | True negative rate at the threshold | 0–1 |
| `ppv` | Positive predictive value at the threshold | 0–1 |
| `npv` | Negative predictive value at the threshold | 0–1 |

Each metric has accompanying `_ci_low` and `_ci_high` columns from stratified bootstrap resampling.

## Example

```python
import numpy as np
from isitfair import FairnessAudit

rng = np.random.default_rng(42)
n = 2000
y_true = rng.binomial(1, 0.12, n).astype(float)
y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.25, n), 0.01, 0.99)
sex = rng.choice(["female", "male"], n)
age = rng.choice(["18-64", "65-79", "80+"], n, p=[0.5, 0.35, 0.15])

audit = FairnessAudit(
    y_true=y_true,
    y_proba=y_proba,
    sensitive_features={"sex": sex, "age_group": age},
    threshold=0.15,
    n_bootstrap=200,
    random_state=42,
)

disc = audit.discrimination()
```

## Reading the output

The DataFrame has three kinds of rows per attribute:

1. **Overall** — metrics computed on the entire cohort. This is the reference.
2. **Subgroup rows** — one per group (e.g., "female", "male"). Compare these to Overall and to each other.
3. **GAP rows** — summary gap statistics. `subgroup == "GAP"` rows have `gap_metric` and `gap_value` columns.

### Gap statistics

| Gap metric | Definition |
|---|---|
| `equal_opportunity_difference` | Max range of sensitivity across subgroups |
| `predictive_equality` | Max range of FPR across subgroups |
| `equalized_odds_gap` | Max of equal opportunity and predictive equality |
| `statistical_parity` | Max range of positive prediction rate |

## When to worry

- **AUROC difference > 0.05 between subgroups**: investigate whether data quality or sample size differs.
- **Sensitivity gap at the clinical threshold**: a subgroup with lower sensitivity will have more missed cases.
- **AUPRC much lower than AUROC in low-prevalence groups**: AUPRC is the more honest metric when events are rare.
- **Wide confidence intervals**: the subgroup may be too small for reliable estimates. Consider whether intersectional analysis with shrinkage is appropriate.

## Forest plot

The HTML report includes a forest plot showing AUROC and AUPRC per subgroup with confidence intervals. The dashed vertical line is the Overall estimate. Subgroups whose CI does not overlap the overall line warrant attention.

## ROC curves

`FairnessAudit.roc_curves()` returns per-subgroup ROC curves with bootstrap confidence bands:

```python
roc = audit.roc_curves()
# roc has columns: attribute, subgroup, shrunk, fpr, tpr, ci_low, ci_high
```

The true-positive rate is evaluated at 100 evenly spaced false-positive rates spanning `[0, 1]` (vertical averaging), so curves from different subgroups and from different bootstrap resamples share an x-axis and can be compared directly. The grid is fixed rather than data-dependent — a ROC curve lives on the unit square by construction — which also makes curves comparable between a baseline and a mitigated audit.

The HTML report plots these per attribute, with the chance diagonal and AUROC in the legend. Reading them:

- **A curve well above the diagonal** discriminates well; the diagonal itself is chance.
- **Curves that cross** mean no subgroup dominates at every operating point. The subgroup with the higher AUROC is not necessarily the better-served one at *your* threshold — check where the curves sit near the false-positive rate you actually operate at.
- **A wide band** means the subgroup is small; the curve's shape is not reliable.

```{note}
AUROC in the report legend is taken from `discrimination()`, not integrated from the plotted curve. The area under the 100-point grid curve is a trapezoid approximation and differs in the third or fourth decimal. Always report the value from `discrimination()`.

The bands are **pointwise** 95% percentile intervals, not simultaneous — they do not support the claim that the entire true curve lies inside the band.
```
