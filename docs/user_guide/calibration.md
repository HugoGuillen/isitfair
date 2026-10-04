# Calibration

The calibration axis measures whether predicted probabilities match observed event rates. A well-calibrated model predicting 15% risk should be correct about 15% of the time.

## Why calibration matters

Discrimination (AUROC) tells you whether the model ranks patients correctly. Calibration tells you whether the numbers it outputs are trustworthy. A model with perfect AUROC but poor calibration will rank patients correctly but give clinicians wrong risk estimates — leading to over- or under-treatment.

In clinical prediction, calibration often differs across subgroups even when discrimination is similar. A model trained primarily on younger patients may systematically underestimate risk in elderly patients. This is a calibration fairness problem that AUROC-only audits miss entirely.

## Metrics

`FairnessAudit.calibration()` returns:

| Metric | What it measures | Ideal value |
|---|---|---|
| `calibration_intercept` | Systematic over/under-prediction | 0 |
| `calibration_slope` | Spread of predictions | 1 |
| `brier` | Overall probability accuracy | 0 (lower is better) |
| `ece` | Average bin-wise calibration error | 0 |
| `ici` | Smooth (LOESS) calibration error | 0 |

A separate goodness-of-fit test, `FairnessAudit.hosmer_lemeshow()`, is documented below.

The intercept and slope follow Steyerberg (2009): a logistic regression of `y_true` on `logit(y_proba)`. A negative intercept means the model over-predicts risk on average; a slope below 1 means predictions are too extreme.

ECE uses equal-frequency (quantile-based) bins, which is more robust than equal-width bins when the prediction distribution is skewed. ICI is the integrated calibration index (Austin & Steyerberg 2019), computed from a LOESS smoother — it avoids the arbitrary choice of bin count.

## Example

```python
cal = audit.calibration()
print(cal[["attribute", "subgroup", "calibration_intercept",
           "calibration_slope", "brier"]].to_string(index=False))
```

## Calibration curves

`FairnessAudit.calibration_curves()` returns LOESS-smoothed calibration curves evaluated at 100 points, with bootstrap confidence intervals:

```python
curves = audit.calibration_curves()
# curves has columns: attribute, subgroup, predicted_prob,
#                     observed_prob, ci_low, ci_high
```

The HTML report plots these curves per attribute. A perfectly calibrated subgroup follows the 45-degree diagonal. Deviations indicate:

- **Curve above the diagonal**: the model underestimates risk (observed > predicted).
- **Curve below the diagonal**: the model overestimates risk (observed < predicted).
- **S-shaped curve**: predictions are too extreme in the tails.

## Hosmer–Lemeshow goodness-of-fit

`FairnessAudit.hosmer_lemeshow()` returns the Hosmer–Lemeshow C statistic per subgroup, together with the supplementary information that makes it interpretable:

```python
hl = audit.hosmer_lemeshow()
# attribute, subgroup, shrunk, n, n_events, hl_statistic, hl_df, hl_p_value,
# n_bins_used, bins_reduced, decile_slope, decile_intercept

deciles = audit.hosmer_lemeshow(detail=True)
# attribute, subgroup, decile, n, observed, expected,
# observed_rate, expected_rate, difference
```

The primitive is also available standalone as `isitfair.hosmer_lemeshow(y_true, y_proba)` for a single slice.

### Why the statistic is never reported alone

The Hosmer–Lemeshow test is widely criticised for depending on sample size, and that criticism is correct: at a fixed degree of lack of fit, the statistic grows roughly linearly with *n*. Kramer & Zimmerman (2007) quantified this — in their simulations a 0.4% average deviation between observed and predicted mortality was significant at p<0.05 in 9.7% of samples at n=5,000, 34.3% at n=10,000, and **100%** at n=50,000.

Their conclusion, however, is not that the test should be dropped. It is that the statistic must be reported *with*:

1. the number of patients,
2. the full per-decile table of observed versus expected events,
3. adjunct measures of calibration.

This package implements exactly that. `n` and `n_events` sit beside the statistic in the summary frame, `detail=True` gives the decile table, and the adjunct measures are the intercept/slope/Brier/ECE/ICI from `calibration()`.

```{warning}
Hosmer–Lemeshow statistics are **not comparable across subgroups of different size**. In a fairness audit this matters more than usual: a large subgroup can be flagged for a clinically irrelevant deviation while a small subgroup with genuinely worse fit is not. Do not rank subgroups by `hl_p_value`. The `max_hl_gap` row exists for consistency with the other axes and inherits the same caveat.
```

Running the test across every subgroup of every attribute is also a family of tests. No multiplicity adjustment is applied; use {func}`isitfair.fdr_correct` if you want one, and adjust within an attribute rather than pooling the `Overall` rows with the subgroup rows.

### Bins, ties, and reduced group counts

Groups are equal-frequency deciles of predicted risk, with cut points snapped to tie boundaries so that identical predicted probabilities always land in the same group. This matters for any tree ensemble, which produces many tied scores: with naive equal-count splitting, bin membership depends on the input row order, and simply permuting the rows moves the p-value. Snapping makes the statistic permutation-invariant and matches R's `ResourceSelection::hoslem.test`.

Because ties are never split, and because small subgroups cannot support ten groups, `n_bins_used` may be lower than `n_bins`. When it is, `bins_reduced` is `True` and the degrees of freedom follow the reduced count. Nothing is silently dropped.

### Degrees of freedom

`df_mode` controls the reference distribution:

| `df_mode` | Degrees of freedom | Use when |
|---|---|---|
| `"validation"` (default) | *g* | The predictions come from a model fitted on other data — always the case for `isitfair` |
| `"development"` | *g* − 2 | The model's intercept and slope were estimated on the very data being tested |

The classical textbook convention is *g* − 2, which is the penalty for estimating two coefficients on the data under test. This package evaluates predictions supplied by the user and never fits a model, so those degrees of freedom are not spent. Simulation on well-calibrated data with *g* = 10 confirms it: the statistic's mean and variance track chi-square with 10 df, and the type-I error rate at the nominal 5% level is 4.6–6.5% under `df = g` against 10.6–14.2% under `df = g − 2`.

Both conventions are available, and `hl_df` is always reported, so the choice is never hidden.

## Interpreting subgroup differences

When comparing calibration across subgroups, focus on:

1. **Intercept sign and magnitude**: if one group has a large negative intercept and another is near zero, the model is systematically biased for that group.
2. **Slope differences**: a slope below 1 in one group but near 1 in another suggests the model was primarily trained on the second group.
3. **ICI**: the integrated measure is the most robust single summary. ICI > 0.05 is generally considered clinically concerning.
4. **Visual inspection of calibration curves**: the curve shape reveals *where* in the risk spectrum the miscalibration occurs. A model may be well-calibrated at low risk but poorly calibrated at high risk in a specific subgroup.

## Connection to mitigation

Calibration differences across subgroups are directly addressable via group-specific recalibration (see {doc}`mitigation`). The `GroupRecalibration` class fits a separate isotonic, Platt, or beta calibrator per subgroup. After mitigation, re-run the calibration audit to verify improvement.
