# Analysis statistics

This page covers the small statistical primitives that `isitfair` ships alongside the main audit pipeline. They live in five sub-modules — `isitfair.effect_sizes`, `isitfair.inference`, `isitfair.metrics`, `isitfair.table_one`, and `isitfair.plots` — and are all re-exported from the top-level `isitfair` namespace.

These helpers are used internally by the report generator (the Table 1 section and the rate-metric effect sizes in the discrimination gap table) and are also fine to call standalone.

## When to reach for which

| You want to … | Use |
|---|---|
| Compare two groups on a continuous variable, with sign | `cohens_d` (+ Mann–Whitney p-value) |
| Compare two groups on a binary variable, with sign | `cohens_h` |
| Compare any two categorical variables (k ≥ 2 levels) | `cramers_v` |
| Rank-based effect size for non-normal continuous | `rank_biserial_r` |
| Adjust a set of p-values | `fdr_correct` (BH default) |
| Test whether sex × covariate predicts an outcome | `interaction_test` |
| Bootstrap CI for a between-group metric difference | `bootstrap_metric_difference` |
| 2 × 2 odds ratio with Wald CI | `odds_ratio_2x2` |
| Quick AUROC/AUPRC/sens/spec/PPV/NPV on one slice | `subgroup_metrics` |
| Cohort characterisation table | `table_one` |
| Plot effect size vs. q-value | `volcano_plot` |
| Per-feature OR/CI plot | `forest_plot` |

## Effect sizes

All four primitives return a plain float. The sign convention for the two-sample statistics is **positive when `group_a` is larger** (matching Cohen's d):

```python
from isitfair import cohens_d, cohens_h, cramers_v, rank_biserial_r, effect_size_label

cohens_d([5, 6, 7], [1, 2, 3])           # +1.27 — group_a is much larger
cohens_h(0.20, 0.10)                      # +0.28 — proportion in group_a is higher
cramers_v(['a', 'a', 'b'], ['x', 'y', 'x'])  # in [0, 1]
rank_biserial_r([5, 6, 7], [1, 2, 3])    # +1.0 — completely separated
```

The qualitative `effect_size_label` helper applies Cohen's conventional cutoffs and is kind-aware:

```python
effect_size_label(0.6, kind="d")   # 'medium'   (0.5 ≤ |d| < 0.8)
effect_size_label(0.15, kind="v")  # 'small'    (0.1 ≤ V < 0.3)
```

```{warning}
The 0.2 / 0.5 / 0.8 cutoffs were derived by Cohen for `d` and `h`. For Cramér's V and rank-biserial r the conventional cutoffs are 0.1 / 0.3 / 0.5 — use `kind="v"` or `kind="r"` to get those.
```

## BH-FDR vs. Bonferroni

`fdr_correct` is a thin wrapper over `statsmodels.stats.multitest.multipletests`. Default is Benjamini–Hochberg (`fdr_bh`), which controls the expected false discovery rate under independence and positive regression dependence. Use it when screening many features (e.g. all comorbidities against a sensitive attribute).

```python
from isitfair import fdr_correct

q = fdr_correct([0.001, 0.01, 0.04, 0.5])
# array([0.004, 0.02, 0.053..., 0.5])
```

For a small, pre-specified set of hypotheses you may prefer Bonferroni (`method="bonferroni"`) — it's stricter and easier to defend in a confirmatory context. For arbitrary dependence between tests, use Benjamini–Yekutieli (`method="fdr_by"`).

## Interaction tests

`interaction_test` runs a likelihood-ratio test of whether a sensitive attribute interacts with a covariate on a binary outcome:

```python
from isitfair import interaction_test

res = interaction_test(
    df,
    outcome="ssi",
    sensitive="sex",
    covariate="comorbidity",
    adjust_for=("age", "asa"),
)
print(res.p_value, res.statistic, res.df, res.delta_r2)
```

For multi-level factors, `df` is `(n_sens_levels - 1) × (n_cov_levels - 1)`. The Cox–Snell pseudo-R² (`delta_r2`) is a magnitude companion to the p-value — a tiny p-value with a tiny ΔR² often means a real but clinically minor interaction.

`interaction_test` is **never** added to the auto-generated HTML report; whether to test a particular interaction is a user decision that depends on the clinical question.

## Bootstrap metric-difference test

`bootstrap_metric_difference` returns a CI and a percentile bootstrap p-value for the between-group difference of any metric (AUC by default):

```python
from isitfair import bootstrap_metric_difference

res = bootstrap_metric_difference(
    y_true=oof["y_true"], y_score=oof["y_prob"], group=oof["sex"],
    metric="auc", n_boot=1000, random_state=42, paired=True,
)
print(res.delta, res.ci_low, res.ci_high, res.p_value)
```

The **paired** scheme (default) resamples patients from the pooled cohort and re-stratifies by group, so the same resampled cohort flows through every group's metric. This is the right scheme for fairness audits where the same source population is being evaluated. The independent scheme (`paired=False`) resamples each group separately — useful when the comparison is between truly independent samples.

For more than two groups, pass `reference_group=...` and the result's `delta` becomes the max-pairwise gap against the reference, with `pairwise_deltas` exposing the per-group differences.

Custom metrics: pass any callable `(y_true, y_score) -> float`.

## Odds ratios

```python
from isitfair import odds_ratio_2x2

res = odds_ratio_2x2(exposure, outcome, continuity="haldane")
print(res.odds_ratio, res.ci_low, res.ci_high, res.p_value)
```

`continuity="haldane"` adds 0.5 to every cell of the 2 × 2 table when any cell is zero (Haldane–Anscombe correction). Set `continuity="none"` for stricter behaviour — the OR is `nan` when a cell is empty.

## Per-subgroup metrics

`subgroup_metrics` is the primitive `FairnessAudit` uses internally and is exposed so you can compute the same set on ad-hoc slices (e.g. out-of-fold predictions stratified by sensitive attribute):

```python
from isitfair import subgroup_metrics

m = subgroup_metrics(y_true_slice, y_proba_slice, threshold=0.15)
# {'n': 1234.0, 'n_events': 134.0, 'prevalence': 0.108,
#  'auroc': 0.78, 'auprc': 0.42, 'sensitivity': 0.81, ...}
```

## Table 1

`table_one` wraps the `tableone` package for the heavy lifting (categorical detection, χ²-or-Fisher switching, multi-group handling, HTML/LaTeX rendering) and adds two things `tableone` doesn't provide:

1. **Benjamini–Hochberg FDR** adjustment (`tableone`'s `pval_adjust` does not support `fdr_bh`).
2. **Per-variable effect sizes** with magnitude labels.

Minimal example:

```python
from isitfair import table_one

res = table_one(
    df,
    group_col="sex",
    group_labels={0: "Female", 1: "Male"},
)
res.effect_sizes  # DataFrame: variable, p_value, q_fdr, effect_size, effect_kind, magnitude, n
res.to_html()     # rendered Table 1 with BH-FDR overlay
```

If you want to render Table 1 *inside* a fairness report, pass it through:

```python
audit.report("audit.html", table_one_data=res)        # pre-built TableOneResult
audit.report("audit.html", table_one_data=df,         # or let the report build it
             table_one_kwargs={"group_col": "sex"})
```

Multi-group cohorts (k > 2) are supported transparently: the effect-size column switches to η² for continuous variables and Cramér's V for categorical / binary variables.

## Plot helpers

All three return a `matplotlib.axes.Axes` and never call `plt.show()` or `plt.savefig()` — composition and persistence are the caller's job.

**Volcano plot** (two-group only, requires a signed effect size):

```python
from isitfair import volcano_plot

ax = volcano_plot(
    results,
    effect_col="effect_size",
    q_col="q_fdr",
    label_col="variable",
    positive_label="Higher in group A",
    negative_label="Higher in group B",
)
```

Install `pip install isitfair[plots]` to get `adjustText` for label repulsion. Without it the function falls back to static placement.

**Lollipop plot** is the same input but rendered as horizontal direction-coded sticks, sorted by effect size.

**Forest plot** is for point estimates with confidence intervals (e.g. odds ratios by group):

```python
from isitfair import forest_plot

ax = forest_plot(
    or_df,
    estimate_col="odds_ratio",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    group_col="sex",
    q_col="q_fdr",
    log_scale=True,   # appropriate for ratio statistics
    null_value=1.0,
)
```

## Sign-convention note for `rank_biserial_r`

`isitfair.rank_biserial_r(a, b)` uses `r = 2U/(n_a · n_b) − 1`, which is positive when `group_a` tends to be larger and so matches the sign convention of Cohen's d. The opposite convention, `1 − 2U/(n_a · n_b)`, is also in circulation; if you are comparing against a value computed that way, flip its sign first.
