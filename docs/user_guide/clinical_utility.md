# Clinical utility

Discrimination and calibration answer *"how good is the model?"* Clinical utility answers *"should we use it?"*

A model with excellent AUROC and calibration may still not be worth acting on if the harm of a false positive outweighs the benefit of a true positive at the clinical decision threshold. Decision curve analysis (DCA) formalises this trade-off.

## Net benefit

Net benefit at threshold *t* is:

$$\text{NB}(t) = \frac{\text{TP}}{n} - \frac{\text{FP}}{n} \cdot \frac{t}{1-t}$$

The weighting factor *t / (1 − t)* encodes the clinician's harm-to-benefit ratio. At *t* = 0.15, a false positive is weighted 0.15/0.85 ≈ 0.18 times as harmful as a false negative is costly. This comes from decision theory: a rational clinician who would treat at probability 0.15 implicitly accepts up to ~5.7 false positives per true positive.

`FairnessAudit.decision_curve()` returns net benefit evaluated across a range of thresholds (default: 0.01 to 0.99), per subgroup.

## Reference strategies

Two reference strategies anchor the decision curve:

- **Treat all**: assume every patient is positive. NB = prevalence − (1 − prevalence) × t/(1−t).
- **Treat none**: assume no patient is positive. NB = 0 at all thresholds.

A model is clinically useful at threshold *t* only if its net benefit exceeds both reference lines. When net benefit drops below "treat none" (i.e., goes negative), the model is actively harmful at that threshold — it generates more weighted false positives than it catches true positives.

## Standardised net benefit

Comparing raw net benefit across subgroups with different prevalences is misleading. A subgroup with 20% prevalence has higher net benefit simply because there are more true positives to find.

**Standardised net benefit** (sNB, Naderalvojoud 2025) divides net benefit by subgroup prevalence, making the metric comparable across groups:

$$\text{sNB}(t) = \frac{\text{NB}(t)}{\text{prevalence}}$$

isitfair reports both. Use raw net benefit to judge whether the model helps a specific group; use standardised net benefit to compare whether the model helps groups *equally*.

## Example

```python
dc = audit.decision_curve()
print(dc[["attribute", "subgroup", "threshold",
          "net_benefit", "standardized_net_benefit"]].head(10))
```

## Reading the output

The returned DataFrame has these columns:

| Column | Description |
|---|---|
| `attribute` | Sensitive attribute name |
| `subgroup` | Group label (or "Overall") |
| `threshold` | Decision threshold (0.01–0.99) |
| `net_benefit` | Raw net benefit at this threshold |
| `net_benefit_ci_low`, `net_benefit_ci_high` | Bootstrap 95% CI |
| `standardized_net_benefit` | Prevalence-adjusted net benefit |
| `treat_all_nb` | Net benefit of the "treat all" strategy |
| `treat_none_nb` | Always 0 |

## Interpreting subgroup differences

When comparing decision curves across subgroups:

1. **Useful range**: identify the range of thresholds where each subgroup's net benefit exceeds both reference lines. If the useful range is much narrower for one group, the model has limited clinical value for that group.
2. **Crossover points**: thresholds where the model's curve crosses the "treat all" line differ across groups when prevalence or sensitivity differs. This affects the range of clinical scenarios where the model adds value.
3. **Standardised net benefit gaps**: if sNB is lower in one group across most thresholds, the model provides systematically less benefit to that group relative to its burden of disease.
4. **Confidence intervals**: wide CIs at clinically relevant thresholds suggest insufficient sample size for reliable utility claims.

## Connection to the decision threshold

The `threshold` parameter you pass to `FairnessAudit` defines where sensitivity, specificity, PPV, and NPV are evaluated. The decision curve is computed across *all* thresholds — it shows how the model performs across the full range of possible clinical scenarios. But the intersection of the decision curve with your chosen threshold tells you whether the model adds value at that specific operating point.
