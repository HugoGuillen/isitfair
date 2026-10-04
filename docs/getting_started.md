# Getting started

## Installation

isitfair needs Python 3.10 to 3.13. Install it into a fresh environment
from GitHub:

```bash
python -m venv isitfair-env               # or: conda create -n isitfair python=3.12
source isitfair-env/bin/activate          # Windows: isitfair-env\Scripts\activate
pip install git+https://github.com/HugoGuillen/isitfair.git@v0.4.0
python -c "import isitfair; print(isitfair.__version__)"
```

To run the examples and notebooks, or to write PDF reports, clone the
repository and install the extras:

```bash
git clone https://github.com/HugoGuillen/isitfair.git
cd isitfair
pip install -e ".[examples,plots,pdf]"
```

The `pdf` extra installs WeasyPrint, which also needs the Pango system library
(see [WeasyPrint's install notes](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)).
`GroupRecalibration(method="beta")` needs the `beta` extra.

To build the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Requirements

- Python 3.10 to 3.13
- numpy, pandas, scipy, scikit-learn, joblib, statsmodels, dcurves, matplotlib, jinja2, tableone

All dependencies are installed automatically.

## Minimal example

```python
import numpy as np
from isitfair import FairnessAudit

# Simulate a clinical prediction model's output
rng = np.random.default_rng(42)
n = 1000
y_true = rng.binomial(1, 0.15, n).astype(float)
y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
sex = rng.choice(["female", "male"], n)

# Create an audit
audit = FairnessAudit(
    y_true=y_true,
    y_proba=y_proba,
    sensitive_features={"sex": sex},
    threshold=0.15,
    n_bootstrap=200,
    random_state=42,
)

# Run all three evaluation axes
results = audit.run_all()
print(results["discrimination"].head())
print(results["calibration"].head())
print(results["decision_curve"].head())

# Generate an HTML report
audit.report("fairness_report.html", title="My Model Audit")
```

The `threshold` parameter is the clinical decision threshold — the probability above which a patient would receive treatment. This is required because sensitivity, specificity, PPV, NPV, and decision-curve analysis depend on it.

## What happens when you call `run_all()`

Three DataFrames are computed and cached:

1. **`discrimination`** — one row per (attribute, subgroup) with AUROC, AUPRC, sensitivity, specificity, PPV, NPV, plus bootstrap 95% CIs and gap statistics.
2. **`calibration`** — calibration intercept, slope, Brier score, ECE, and ICI per subgroup. `hosmer_lemeshow` adds the Hosmer–Lemeshow goodness-of-fit test with its per-decile table.
3. **`decision_curve`** — net benefit and standardized net benefit at each threshold, per subgroup.

All results are long-format DataFrames, suitable for filtering, plotting, or export.

## Next steps

- To audit your own model's predictions, read {doc}`user_guide/your_own_data`.
- Read the {doc}`user_guide/concepts` page to understand the three-axis framework.
- See the {doc}`user_guide/mitigation` page for post-hoc fairness mitigation.
- Browse the [example audit](https://github.com/HugoGuillen/isitfair/blob/main/examples/support2_mortality_audit.py) for a complete walkthrough on the public SUPPORT2 cohort.
