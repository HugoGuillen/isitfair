# Estimability: which estimates carry an interval

An audit's job is to look inside subgroups, and subgroups are small. This page
describes the rule `isitfair` uses to decide, in advance, which estimates the
data supports an interval for — so that every number in a report arrives with
its own evidence base attached.

## The rule

A cell-metric carries an interval when it clears **both** tests:

> **reportable = at or above the event floor **AND** at most 10% of bootstrap
> replicates non-estimable**

The union is conservative by design, because each test catches what the other
misses: the floor catches a subgroup that is small in a way the bootstrap cannot
see, and the replicate rule catches a subgroup whose events are distributed so
that resampling often produces a degenerate estimate.

```python
est = audit.estimability()
est[est.not_estimable_final]     # the cells reported without an interval
est["rule"]                      # which of the two tests fired
```

## Rule 1 — the replicate rule

Bootstrap replicates that produce no estimate are **not dropped**. Dropping them
conditions the interval on success and narrows it — the opposite of what the
interval is for. They are retained as NaN, and if more than
`nonestimable_threshold` (default 0.10) of replicates fail, the cell is not
estimable.

Note the asymmetry: a surviving interval is conditional on estimability from the
**first** failed replicate, not from the 10% mark. Ten percent is where the
verdict flips, not where the conditioning starts.

## Rule 2 — the event floors

Motivated by the standard error of each metric in a low-prevalence regime.
Import them rather than restating the numbers inline, so a change propagates
everywhere at once.

```python
from isitfair import DEFAULT_EVENT_FLOORS, EVENT_FLOOR_RATIONALE, floor_fails
```

| Metric | Floor | Why |
|---|---:|---|
| `auroc` | 25 events | SE ≈ 0.5/√E; 25 events holds the 95% interval to about ±0.20 — the widest interval that can still separate a strong c-statistic from chance. |
| `calibration_intercept` | 25 events | SE ≈ 1/√E; 25 events gives ±0.39 on the log-odds scale, small relative to the between-subgroup gradients an audit looks for. |
| `calibration_slope` | 100 events | Riley et al. (*Stat Med* 2021) require roughly 100 events *and* 100 non-events. The slope is the most sample-hungry of the four: it estimates a gradient across the risk range, not a level. |
| `net_benefit` | 50 events | Net benefit depends on the true positives among those flagged — a subset of the events. Below about 50, the flagged count falls into single digits and individual patients drive the estimate. |

Metrics share a floor with the one they are governed by: `auprc`,
`sensitivity`, `specificity`, `ppv` and `npv` inherit the AUROC floor; `brier`
inherits the intercept floor; `standardized_net_benefit` inherits the net
benefit floor. `FLOOR_FOR` is that map.

Override per audit when your setting warrants it:

```python
audit = FairnessAudit(
    ..., nonestimable_threshold=0.10,
    event_floors={"auroc": 25, "calibration_intercept": 25,
                  "calibration_slope": 100, "net_benefit": 50},
)
```

## Gaps are filed under their own name

A gap is a different quantity from the metric it is built from, so it gets its
own row and its own floor — `max_intercept_gap`, not `calibration_intercept`.
`GAP_SOURCE` maps each gap to the floor it inherits.

**A gap inherits the verdict of both constituent subgroups**, so a contrast
carries an interval only when both sides of it do.

## Suppression withholds the interval, keeping the point estimate

A suppressed point estimate is **still a real quantity** — the observed value in
that subgroup. Keeping it, and dropping only the interval, reports the subgroup
and its evidence base together. So:

- the point estimate is reported;
- the interval is withheld;
- plot helpers draw those points **dashed, with a hollow marker**
  (`mark_not_estimable=True`, the default).

Every subgroup keeps its row, which means a sparse subgroup stays visible in the
table rather than dropping out of it.

## Reporting a suppressed cell

State the evidence base alongside the estimate:

> In this stratum there were *E* events, below the pre-registered floor of *F*
> for this metric. The point estimate is shown without an interval.

That is a result in its own right. Which subgroups a model could be measured in,
and how precisely, belongs in the results section alongside the estimates.
