# isitfair

**Fairness evaluation toolkit for clinical prediction models.**

isitfair evaluates whether a trained prediction model performs equitably across patient subgroups. It reports three axes of fairness — discrimination, calibration, and clinical utility — because no single metric captures whether a model is fair.

The package is designed for clinical researchers who have a trained binary-classification model and want to audit its predictions before deployment.

## Key features

- **Three-axis evaluation**: discrimination (AUROC, AUPRC, sensitivity, specificity), calibration (intercept, slope, Brier, ECE, ICI, Hosmer–Lemeshow), and clinical utility (decision curves with net benefit).
- **Intersectional analysis**: evaluate fairness for interaction subgroups (e.g., elderly frail women) with empirical Bayes shrinkage for sparse cells.
- **Mitigation**: group-specific recalibration, Wasserstein score post-processing, and reweighing.
- **TRIPOD+AI reports**: self-contained HTML reports with embedded plots, with an optional typeset PDF copy.
- **Deterministic**: fixed seeds produce bit-identical results across runs.

## Installation

```bash
pip install git+https://github.com/HugoGuillen/isitfair.git
```

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
user_guide/index
api/index
examples/index
```
