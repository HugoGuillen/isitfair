# Phase 5: isitfair against independent oracles

All checks use the code with E01, E03, E04 and E05 applied. E07 and E08, applied later, change no output (bit-identical by construction and by test).

- **Scripts**: `hardening/phase5/oracles.py`, `oracle_definitions.py` and
  `oracle_calib_coverage.py`.
- **Results**: `hardening/phase5/results/oracles_final.json`,
  `oracle_definitions.json` and `oracle_calib_coverage.json`.
- **Logs**: `hardening/logs/phase5_*.log`.
- **Inputs**: 18 runs (9 datasets x {LR, HGB} predictions on the audited test
  split from `hardening/datasets/registry.py`). Every subgroup row (Overall,
  each marginal level, and the first two attributes crossed) is compared.
- **Tolerance**: 1e-9 absolute.

## Oracles

| isitfair quantity | oracle |
|---|---|
| AUROC, AUPRC | `sklearn.metrics.roc_auc_score`, `average_precision_score` |
| sensitivity, specificity, PPV, NPV | `sklearn.metrics.confusion_matrix` at the audit threshold |
| Brier | `sklearn.metrics.brier_score_loss` |
| calibration intercept | statsmodels Binomial GLM, intercept only, offset logit(p), `tol=1e-12` |
| calibration slope | statsmodels Binomial GLM on [1, logit(p)], `tol=1e-12` |
| ECE, Hosmer-Lemeshow C | from scratch, 10 equal-count bins (two versions, see below) |
| net benefit | `dcurves.dca` and the formula TP/n - FP/n x t/(1-t) |
| gaps (EO, PE, EOdds, SP) | `fairlearn.metrics.MetricFrame(...).difference("between_groups")` |

## Result

| metric | worst abs. diff, all 18 runs | within 1e-9? |
|---|---:|---|
| AUROC, AUPRC, sensitivity, specificity, PPV, NPV | 0 | yes |
| Brier | 5.6e-17 | yes |
| HL df | 0 | yes |
| net benefit vs dcurves | 0 | yes |
| net benefit vs formula | 1.7e-16 | yes |
| gap: predictive equality, statistical parity (marginal) | 1.1e-16 | yes |
| ECE, HL statistic, HL p-value vs the tie-snapped oracle | 0, 2.1e-14, 7.8e-16 | yes |
| gap: equal opportunity, equalized odds vs fairlearn with undefined rates left out | 1.1e-16 | yes |
| calibration intercept | 1.7e-6 | **no**: optimizer tolerance |
| calibration slope, cells with >= 3 events in 11 rows | 1.2e-4 | **no**: optimizer tolerance |
| calibration slope, quasi-separated cells | up to 1,663 | **no**: not identified; all below floor |
| ECE, HL vs the row-order oracle (ties split by position) | 0.033, 3.18 | definitional; see below |
| gap: EO, EOdds vs plain fairlearn | 0.61 | definitional; see below |
| intersection gaps | up to 0.91 | by design (shrunk cell rates) |

Statistical parity agrees to 1.1e-16 only because of E04. Before that fix,
COMPAS race differed from fairlearn by 0.113 (0.337 vs 0.451; `oracles.json`
holds the pre-fix run).

### Differences above 1e-9, explained

1. **Calibration intercept, max 1.7e-6 (every run).** isitfair's fitting
   routine stops at a looser convergence tolerance than the GLM run at
   `tol=1e-12`. Across all 18 runs the difference never exceeds 1.7e-6. The
   committed SUPPORT2 tables round these to 3 or 4 decimals. Tightening the
   tolerance would move the committed SUPPORT2 numbers in the 6th decimal and
   would breach the golden rule, so it was not changed (ERRORLOG E06, won't fix).

2. **Calibration slope.** There are two kinds of difference:
   - **Optimizer tolerance.** The largest is 1.2e-4 on COMPAS race = Asian
     (n = 11, 3 events, slope 7.56). Elsewhere it is 1e-6 to 2e-5. Same cause
     and decision as item 1 (E06).
   - **Quasi-separation.** The cells are bank 40-49/student (n = 4, 1 event),
     bank 60+/entrepreneur (7, 2), german female/50+ (13, 1) and
     heart_cleveland female/65+ (4, 1). With one or two events the likelihood
     is nearly flat in the slope, so both optimizers stop at arbitrary large
     values (for example -1,184 vs -2,846). Every one of these cells is far
     below the 100-event slope floor, so `estimability()` already marks it
     not estimable (`not_estimable_final = True`). This is not a defect.

3. **ECE and HL on tied predictions** (adult HGB, COMPAS LR and HGB, bank HGB,
   diabetes130 HGB).
   - **Why the first oracle differs.** It splits sorted predictions into
     equal-count bins by position, so tied predictions straddle bin edges
     depending on row order. That is exactly the dependence that E05 removed.
   - **Agreement with isitfair's definition.** isitfair moves each cut to the
     nearer end of a run of ties; this was already HL's documented definition,
     and E05 applied it to ECE. A second from-scratch oracle that does the
     same (`oracle_definitions.py`, written without importing isitfair) agrees
     on all 8 tie-affected runs: ECE to 0, HL statistic to 2.1e-14, HL
     p-value to 7.8e-16.

4. **Equal-opportunity and equalized-odds gaps** (diabetes130 gender, COMPAS
   race, oracle_sparse level).
   - **What fairlearn does.** It reports TPR = 0 for a subgroup with no
     events: diabetes130 "Unknown/Invalid" (n = 3), COMPAS "Native American"
     (n = 2) and oracle_sparse L6. The gap then becomes the largest observed
     TPR.
   - **What isitfair does.** It treats TPR as undefined there, and FPR as
     undefined for a subgroup without non-events, and leaves them out of the
     range.
   - **Check.** Recomputing the fairlearn gaps over the defined rates only
     agrees with isitfair to 1.1e-16 on all six affected runs.
   - **Why this differs from E04.** A subgroup with no events still has a
     defined flag rate, so it does stay in statistical parity.

5. **Intersection gaps.** isitfair forms intersection-cell rates with
   empirical-Bayes shrinkage toward the marginal rates, by design and as
   documented, so they cannot equal fairlearn's raw cell rates. The marginal
   gaps, which use the same code without shrinkage, agree to 1.1e-16.

## oracle_calib: planted values inside the CIs

- **Setup**: n = 20,000 in two groups. Group A is calibrated (intercept 0,
  slope 1). Group B has planted logit miscalibration a = -0.8, b = 0.7.
- **Single run** (B = 2,000, `random_state=42`):

| group | target | value | 95% CI | contains |
|---|---|---:|---|---|
| A | slope 1.0 | 0.965 | [0.919, 1.013] | yes |
| A | intercept 0 | -0.047 | [-0.094, 0.003] | yes |
| B | slope 0.7 | 0.695 | [0.652, 0.740] | yes |
| B | intercept estimand -0.663 | -0.682 | [-0.739, -0.631] | yes |
| B | planted a = -0.8 | -0.682 | [-0.739, -0.631] | no (not the estimand) |

The calibration intercept is calibration-in-the-large: the GLM is fitted with
the slope fixed at 1. Under a planted slope of 0.7, its population value is
not a but the solution of E[y - expit(logit p + alpha)] = 0. By Gauss-Hermite
quadrature over the planted risk distribution that solution is
alpha* = -0.6634. The interval contains alpha* and should not contain -0.8.
In group A the two targets coincide because the slope is 1.

**Coverage over 200 simulated datasets, B = 200 each** (3,415 s; target 93% to 97%):

| group | slope | intercept vs estimand | intercept vs planted a |
|---|---:|---:|---:|
| A | 91.5% | 95.0% | 95.0% |
| B | 92.0% | 98.0% | 0% (expected; see above) |

Mean estimates are unbiased: slope 0.9997 (A) and 0.7024 (B); intercept
-0.0005 (A) and -0.6642 (B, estimand -0.6634).

Slope coverage is 1.0 and 1.5 points below the 93% lower bound. With 200
simulations the Monte Carlo standard error of a 95% coverage is about 1.5
points, so 91.5% is 2.3 standard errors below nominal: weak but real evidence
of undercoverage. This is logged as ERRORLOG E09 and marked won't fix by the maintainer, because a different interval method would change the committed
SUPPORT2 CIs.

## oracle_sparse: 0-event and 3-event levels not estimable

Log: `hardening/logs/phase5_oracle_sparse_estimability.log`. The L5 level has
n = 300 and 3 events; L6 has n = 200 and 0 events.

| level | AUROC, sensitivity, intercept, slope | `not_estimable_final` | CIs |
|---|---|---|---|
| L5 (3 events) | below the 25/100-event floors; 1 of 20 replicates non-estimable | True for every metric except prevalence | NaN |
| L6 (0 events) | 20 of 20 replicates non-estimable | True for every metric except prevalence | NaN |
| L4 (83 events) | slope only, below the 100-event floor | True for slope only | slope CI NaN |
| GAP rows | inherit the floor from L5 and L6 | True | NaN |

The requirement is met. L5 still prints point estimates (AUROC 0.806,
intercept -2.86), but every one is flagged `not_estimable_final` and carries no
CI.
