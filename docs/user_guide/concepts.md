# Concepts: the three-axis framework

## Why not a single fairness metric?

A common pattern in the fairness literature is to pick one metric — say, equalized odds — and declare a model "fair" or "unfair" based on whether the gap exceeds some threshold. This is tempting but misleading in clinical settings.

Consider an infection prediction model deployed across age groups. If you only report that AUROC is similar across groups, you miss that the model may be systematically overconfident in elderly patients (a calibration problem). If you only report that calibration is good, you miss that the model may have lower sensitivity in a subgroup (a discrimination problem). And neither metric tells you whether acting on the model's predictions actually helps patients (a clinical utility problem).

isitfair reports all three axes because each captures a different failure mode:

- **Discrimination** answers: *can the model rank patients correctly within this subgroup?*
- **Calibration** answers: *are the predicted probabilities accurate in this subgroup?*
- **Clinical utility** answers: *does acting on these predictions benefit patients in this subgroup?*

A model can fail on any axis independently. Reporting only one creates a false sense of security.

## The three axes

### Discrimination

Discrimination metrics measure how well the model separates positive from negative cases. isitfair reports:

- **AUROC** (area under the ROC curve) — the probability that a randomly chosen positive case has a higher predicted probability than a randomly chosen negative case.
- **AUPRC** (area under the precision-recall curve) — more informative than AUROC when events are rare, which is typical in clinical prediction.
- **Sensitivity, specificity, PPV, NPV** — threshold-dependent metrics evaluated at the clinical decision threshold you specify.

These are reported per subgroup with bootstrap 95% confidence intervals and gap statistics (equalized odds, predictive equality, statistical parity).

### Calibration

Calibration measures whether the predicted probabilities match observed event rates. A model that predicts 15% infection risk should be right about 15% of the time. isitfair reports:

- **Calibration intercept** — deviation from zero indicates systematic over- or under-prediction (Steyerberg 2009).
- **Calibration slope** — deviation from one indicates the model's predictions are too extreme or too conservative.
- **Brier score** — overall probability accuracy (lower is better).
- **ECE** (expected calibration error) — average absolute difference between predicted and observed rates within equal-frequency bins.
- **ICI** (integrated calibration index) — smooth analogue of ECE using LOESS curves (Austin & Steyerberg 2019).

Calibration matters because clinical decisions depend on absolute risk levels. A model with good AUROC but poor calibration gives clinicians the wrong numbers.

### Clinical utility

Decision curve analysis (DCA) asks: *would patients be better off if clinicians used this model rather than treating everyone or treating no one?* This is measured by **net benefit**:

$$\text{NB}(t) = \frac{\text{TP}}{n} - \frac{\text{FP}}{n} \cdot \frac{t}{1-t}$$

where *t* is the threshold probability. Net benefit above the "treat all" and "treat none" reference lines means the model adds value.

isitfair also reports **standardized net benefit** (Naderalvojoud 2025), which divides NB by subgroup prevalence to facilitate comparison across groups with different event rates.

## Within-group performance AND gaps

isitfair always reports both. Here is why.

Gap statistics (e.g., "the AUROC difference between men and women is 0.03") are useful summaries, but they hide critical context. If both groups have AUROC of 0.60 vs 0.57, the model is poor for everyone — the gap is not the problem. If both groups have AUROC of 0.92 vs 0.89, the model is excellent and the gap may not be clinically meaningful.

By reporting within-group performance alongside gaps, clinicians can judge whether a disparity matters in practice.

## The levelling-down concern

Pfohl et al. (2021) warned that post-hoc mitigation can "level down" — it may equalize metrics across groups by degrading the best-performing group rather than improving the worst. A threshold optimizer that satisfies equalized odds by lowering sensitivity in the well-served group is technically "fair" but clinically harmful.

isitfair addresses this by:

1. Always showing before-and-after performance per group, so you can see whether mitigation helped the disadvantaged group or hurt the advantaged one.
2. Reporting all three axes after mitigation, because optimizing one metric (e.g., equalized odds) can damage another (e.g., calibration).
3. Not making mitigation automatic. The `mitigate()` method returns new audit objects for comparison — it does not silently replace your predictions.

The decision of whether to deploy a mitigated model is yours. isitfair gives you the data to make that decision.
