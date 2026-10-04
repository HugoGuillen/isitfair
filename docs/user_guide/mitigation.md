# Mitigation

isitfair provides three post-hoc mitigation strategies. Each addresses a different aspect of the fairness–performance trade-off. None is automatic — mitigation returns new audit objects for comparison, so you can judge whether the trade-off is acceptable before deploying.

## Available methods

| Method | Class | What it does | Requires retraining? |
|---|---|---|---|
| Group recalibration | `GroupRecalibration` | Fits a separate calibrator per subgroup | No |
| Wasserstein score post-processing | `WassersteinPostprocessing` | Aligns per-subgroup score distributions via quantile-to-quantile mapping onto their Wasserstein barycenter | No |
| Reweighing | `Reweighing` | Computes sample weights for fair retraining | Yes |

`GroupThresholdOptimization` is a deprecated alias for `WassersteinPostprocessing`. Use `WassersteinPostprocessing` in new code: it returns a calibrated risk score in `[0, 1]`, which is what the calibration and decision-curve axes require, whereas a randomized-classifier labelling probability is not a risk and does not support them.

### GroupRecalibration

Fits a separate calibration model per subgroup of each sensitive attribute. This directly addresses calibration differences without affecting discrimination.

Supported methods:
- `"isotonic"` — isotonic regression (non-parametric, default)
- `"platt"` — Platt scaling (logistic regression on the logit)
- `"beta"` — beta calibration (requires `betacal` package)

```python
from isitfair import GroupRecalibration

recalib = GroupRecalibration(method="isotonic")
mitigated = audit.mitigate(recalib, calibration_data=(y_calib, p_calib, sens_calib))

# mitigated is a dict: {"sex": FairnessAudit, "age_group": FairnessAudit}
for attr, new_audit in mitigated.items():
    print(f"--- {attr} ---")
    print(new_audit.calibration()[["subgroup", "ici"]])
```

**When to use**: when calibration differs across subgroups but discrimination is acceptable. This is the most common clinical fairness issue.

### WassersteinPostprocessing

Aligns per-subgroup score distributions through a per-group quantile-to-quantile transformation onto the 1-D Wasserstein barycenter of those distributions on the calibration set. For each subgroup `g` with empirical CDF `F_g`, the mitigated score for a test prediction `p` in `g` is `T_g(p) = Q_bary(F_g(p))`, where `Q_bary(u) = sum_g π_g · F_g^{-1}(u)`. The transformation is monotone non-decreasing within each subgroup, so **within-subgroup rank order (hence AUROC) is preserved exactly**, and the output remains a calibrated risk score in `[0, 1]`. On the calibration set the per-subgroup score distributions become identical, i.e. **exact demographic parity**.

References:

- Chzhen, Denis, Hebiri, Oneto, Pontil (NeurIPS 2020). *Fair Regression with Wasserstein Barycenters.*
- Jiang, Pacchiano, Stepleton, Jiang, Chiappa (UAI 2020). *Wasserstein Fair Classification.*
- Gordaliza, del Barrio, Gamboa, Loubes (ICML 2019). *Obtaining Fairness using Optimal Transport Theory.*

Only `constraint="demographic_parity"` is supported. By the impossibility theorem of Pleiss et al. (NeurIPS 2017), a post-processing scheme that preserves calibration cannot generally enforce equalized odds when group base rates differ; the class raises `NotImplementedError` for `constraint="equalized_odds"` or `"equal_opportunity"`.

```python
from isitfair import WassersteinPostprocessing

wass = WassersteinPostprocessing(constraint="demographic_parity")
mitigated = audit.mitigate(wass, calibration_data=(y_calib, p_calib, sens_calib))
```

**When to use**: when the model's predicted-risk distribution differs across subgroups in a way that downstream decisions (e.g. resource allocation) may amplify. Because within-group rank order is preserved, AUROC and ROC-based decisions within a subgroup are unaffected.

### Reweighing

Computes Kamiran–Calders sample weights that equalise the label distribution across subgroups. Unlike the other methods, reweighing requires **model retraining** — it cannot be applied post-hoc.

```python
from isitfair import Reweighing

rw = Reweighing()
weights = rw.fit_transform(y_train, sensitive_train)

# Use weights in your training pipeline
model.fit(X_train, y_train, sample_weight=weights["sex"])

# Re-evaluate on test set with the retrained model
new_proba = model.predict_proba(X_test)[:, 1]
new_audit = FairnessAudit(y_test, new_proba, {"sex": sex_test}, threshold=0.15)
```

**When to use**: when you have access to the training pipeline and want to address fairness at the source. Results depend on the model architecture and may not generalise.

## The `mitigate()` convenience method

`FairnessAudit.mitigate()` automates the fit-transform-reaudit cycle for post-hoc methods (`GroupRecalibration` and `WassersteinPostprocessing`):

```python
mitigated = audit.mitigate(method, calibration_data=(y_calib, p_calib, sens_calib))
```

Returns a dict mapping each attribute name to a new `FairnessAudit` with the same `y_true` but mitigated `y_proba`. You can then call `.discrimination()`, `.calibration()`, `.decision_curve()` on the new audits and compare.

**Important**: mitigated probabilities are attribute-specific. Recalibrating by sex produces different probabilities than recalibrating by age. The dict contains one audit per attribute, not a single joint mitigation.

## The levelling-down concern

Pfohl et al. (2021) warned that post-hoc mitigation can "level down" — equalising metrics by degrading the best-performing group rather than improving the worst. A score post-processor that aligns subgroup distributions by pulling the well-served group's scores toward the disadvantaged group's is technically "fair" but clinically harmful.

isitfair addresses this by:

1. **Showing before-and-after per group.** The `report()` method includes a mitigation comparison section when you pass `mitigation_audits`.
2. **Reporting all three axes.** Optimising equalized odds may damage calibration. You will see this in the post-mitigation calibration table.
3. **Never auto-deploying.** `mitigate()` returns new audit objects for inspection, not a silently modified model.

Always compare the full before-and-after audit. If the disadvantaged group did not improve — or the advantaged group was harmed without the disadvantaged group benefiting — the mitigation is not worth deploying.

## Reporting mitigation results

Pass mitigated audits to `report()` for a side-by-side comparison:

```python
mitigated = audit.mitigate(GroupRecalibration(), calibration_data=calib_data)
audit.report(
    "fairness_report.html",
    title="My Audit",
    mitigation_audits=mitigated,
)
```

The HTML report will include a dedicated mitigation section showing before-and-after metric tables per attribute.
