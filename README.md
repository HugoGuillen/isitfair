<img src="./logo/isitfair-logo.svg">

# isitfair

**Fairness evaluation for clinical prediction models** — subgroup-stratified
discrimination, calibration and decision-curve analysis with TRIPOD+AI-aligned
reporting.

`isitfair` works from *probabilities* rather than hard classifications. You
bring a trained model's predictions; it reports how they behave across patient
subgroups, and how much of that behaviour the data actually supports.

This is the software companion to:

> Guillen-Ramirez H, Lucas KL, Wintsch YM, Blatter TU, Triep K, Endrich O, Beldi G.
> **Subgroup deployment-readiness audit of calibration and clinical utility in
> perioperative infection prediction.** Under review, *JAMIA*.

---

## Install

Python 3.10 to 3.13. Use a fresh environment:

```bash
python -m venv isitfair-env               # or: conda create -n isitfair python=3.12
source isitfair-env/bin/activate          # Windows: isitfair-env\Scripts\activate
pip install git+https://github.com/HugoGuillen/isitfair.git
python -c "import isitfair; print(isitfair.__version__)"
```


## Quick start

Copy and run; it simulates a model's predictions, so it needs nothing else.

```python
import numpy as np
from isitfair import FairnessAudit

rng = np.random.default_rng(42)
n = 1000
y = rng.binomial(1, 0.15, n)                                     # 0/1 outcomes
p = np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)        # predicted probabilities
sex = rng.choice(["female", "male"], n)
age_group = rng.choice(["<65", "65-79", "80+"], n)

audit = FairnessAudit(
    y_true=y,
    y_proba=p,
    sensitive_features={"sex": sex, "age_group": age_group},
    threshold=0.15,           # prespecified operating point
    n_bootstrap=200,          # 2000 for reported results
    random_state=42,
)

disc = audit.discrimination()   # AUROC, AUPRC, sens/spec/PPV/NPV + four parity gaps
cal  = audit.calibration()      # intercept, slope, Brier, ECE, ICI
dca  = audit.decision_curve()   # net benefit, standardized net benefit
est  = audit.estimability()     # what you are allowed to put an interval on

audit.report("audit_report.html", title="My model — deployment-readiness audit")
```

The report is a single HTML file; 300-DPI copies of its figures are written to
`audit_report_figures/` beside it.

## Use it on your own data

You need one row per patient with the observed **0/1 outcome**, your model's
**predicted probability** for patients it was *not* trained on (held-out or
out-of-fold), and the **group labels** to audit across. Any model works, in any
language, as long as you can export its probabilities.

```python
import pandas as pd
from isitfair import FairnessAudit

df = pd.read_csv("my_predictions.csv").dropna(subset=["outcome", "risk"]).reset_index(drop=True)
df["age_group"] = pd.cut(df["age"], [0, 65, 80, 120], right=False)   # bin continuous columns
groups = {col: df[col].astype("string").fillna("unknown") for col in ["sex", "age_group"]}

audit = FairnessAudit(
    y_true=df["outcome"], y_proba=df["risk"], sensitive_features=groups,
    threshold=0.10, n_bootstrap=2000, random_state=42,
)
audit.report("audit_report.html")
```

- **Guide:** [`docs/user_guide/your_own_data.md`](docs/user_guide/your_own_data.md):
  input rules, choosing the threshold, runtime, reading the output, and a
  troubleshooting table for every error message.
- **Template script:** [`examples/audit_your_own_data.py`](examples/audit_your_own_data.py):
  set your column names at the top and run
  `python examples/audit_your_own_data.py my_predictions.csv`.

Expect about 1 minute at `n_bootstrap=200` and 10–15 minutes at 2000 for a
cohort of ~2,000 patients and three attributes.

---

## Finished products

Everything below is committed, already executed, and reproducible from this
repository. GitHub shows `.html` files as raw text, so each one has a rendered
link next to it.

| | What it is | Rendered |
|---|---|---|
| 📓 [`notebooks/01_audit_report_support2.ipynb`](notebooks/01_audit_report_support2.ipynb) | **Start here.** An audit end to end, producing the HTML report below. | [HTML](https://htmlpreview.github.io/?https://github.com/HugoGuillen/isitfair/blob/main/results/notebooks_html/01_audit_report_support2.html) |
| 📄 `results/support2_audit_report.html` | The **generated audit report** — three axes, gap statistics, per-decile H–L tables, intersectional diagnostics, before/after mitigation, TRIPOD+AI appendix. Self-contained, no JavaScript. | [view report](https://htmlpreview.github.io/?https://github.com/HugoGuillen/isitfair/blob/main/results/support2_audit_report.html) |
| 📄 [`results/support2_audit_report.pdf`](results/support2_audit_report.pdf) | The same report typeset for print. | — |
| 📓 [`notebooks/02_paper_analysis_support2.ipynb`](notebooks/02_paper_analysis_support2.ipynb) | The **complete analysis**, section by section — cohort table through mitigation and interaction screens. Writes 20 tables and 8 figures. | [HTML](https://htmlpreview.github.io/?https://github.com/HugoGuillen/isitfair/blob/main/results/notebooks_html/02_paper_analysis_support2.html) |
| 📓 [`notebooks/03_tutorial_support2.ipynb`](notebooks/03_tutorial_support2.ipynb) | The **tutorial**, runnable. Nine sections, every slide's code executing on real data. | [HTML](https://htmlpreview.github.io/?https://github.com/HugoGuillen/isitfair/blob/main/results/notebooks_html/03_tutorial_support2.html) |
| 🎞️ [`slides/isitfair_audit_tutorial.pdf`](slides/isitfair_audit_tutorial.pdf) | The tutorial deck the notebook above is built from. 61 frames. | — |
| 📁 [`results/tables/`](results/tables) · [`results/figures/`](results/figures) | Every table as `.csv` **and** `.md`; every figure as `.png` **and** `.pdf`. | — |
| 🏥 [`results/bern/`](results/bern) | **The paper's clinical cohort, aggregated**: its audit reports and subgroup tables for 2014–2022 and the 2018–2022 window. Aggregates only, some of them small; see its README. | — |

`results/MANIFEST.csv` records the producing notebook, seed, replicate count,
package versions and a content hash for each SUPPORT2 artifact.
`examples/make_manifest.py` rebuilds it after a re-run.

The clinical cohort in the paper is patient-level hospital data and cannot be
shared. [`results/bern/`](results/bern) holds its aggregated subgroup outputs;
its README describes each file.

---

## The data

Everything here runs on **SUPPORT2** — 9,105 critically ill adults from five US
medical centres, published by the UCI Machine Learning Repository (dataset 880,
CC BY 4.0, no credentialing required). A copy is bundled at
`examples/data/support2.csv.gz`, so the examples run offline as well as online.

It suits a fairness audit well: it records **race/ethnicity** alongside sex and
age, so a single run exercises a sociodemographic axis, a clinical one, and the
intersection of two of them — including the sparse cells that make subgroup
estimation interesting.

---

## Reproducing everything

```bash
git clone https://github.com/HugoGuillen/isitfair.git && cd isitfair
pip install -e ".[examples,plots,pdf]"

python examples/support2_mortality_audit.py         # the audit report  (~12 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
python examples/make_manifest.py                    # refresh results/MANIFEST.csv
```

The `plots` extra installs adjustText, which places the labels of the volcano
figure (Figure 6A); without it the labels are drawn unadjusted and that one
figure will not match its committed hash.

The `pdf` extra installs [WeasyPrint](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html),
which also needs the Pango system library. Without it, everything runs and only
the PDF copy of the report is skipped.

Set `ISITFAIR_FAST=1` to drop the bootstrap from B = 2000 to B = 200 for a
quick pass while you are editing. Committed artifacts are built at B = 2000,
except the tutorial notebook, which runs at B = 200 so every cell stays quick.

---

## What this package is

- A **three-axis evaluator**: discrimination, calibration and clinical utility,
  reported jointly and never one alone.
- A **subgroup reporter** that always returns within-group performance *and* gap
  statistics. There is no gaps-only mode.
- An **intersectional analysis tool** with empirical-Bayes shrinkage for sparse
  cells.
- A **mitigation toolkit** — the recalibration ladder (isotonic/Platt/beta),
  Wasserstein score post-processing, Kamiran–Calders reweighing.
- A **report generator** producing self-contained, print-styled HTML.
- A set of **analysis-statistics primitives** clinical fairness audits keep
  needing: effect sizes, BH-FDR, likelihood-ratio interaction tests, a paired
  bootstrap for metric differences, 2×2 odds ratios, a multi-group Table 1, and
  volcano / forest / lollipop helpers.

## What it is not

- **Not a model trainer.** Models are your input; we evaluate the predictions.
- **Not an imaging tool.** Tabular data only — see MEDFAIR and FairMedFM.
- **Not a reimplementation of Fairlearn or AIF360.** It complements them with
  probability-based clinical evaluation rather than duplicating their metrics.
- **Not a single-metric tool.** Within-group performance and gap statistics are
  always reported together, on all three axes.
- **Not an inference engine.** No causal or counterfactual fairness.

---

## Five commitments worth knowing before you use it

**Within-group *and* gap, always.** Parity is equally consistent with both
groups being served well and with both being served badly, so every frame
carries the subgroup rows and the `GAP` row together and you can tell which.

**One definition of "not estimable".** A cell-metric is reportable only if it
clears *both* the pre-registered event floor for that metric and the bootstrap
replicate rule (at most 10% non-estimable replicates). The union is conservative by design, and a gap
inherits the verdict of both constituent subgroups. Import the floors from
`isitfair.estimability` so a change to them propagates everywhere at once.

**Suppression withholds the interval, keeping the point estimate.** The value
is real; what the data cannot support is an interval around it. Figures mark
those points dashed, with a hollow marker, so the distinction is visible.

**Resampling is patient-level by default.** `resample_scheme="patient"`
resamples patients within each subgroup, preserving the subgroup's size while
letting its event count vary — which is what gives calibration intercept and
slope, Brier, PPV/NPV and net benefit intervals that reflect the uncertainty in
prevalence too. The older `"case_control"` scheme holds the event count fixed
and so estimates uncertainty conditional on the observed prevalence; it stays
available, unchanged, for reproducing analyses that used it.

**Deterministic.** Identical data plus `random_state=42` gives bit-identical
output. `requirements-lock.txt` pins the versions the committed artifacts were
built with, because that guarantee is only meaningful against fixed dependencies.

---

## API at a glance

### The audit

| Method | Returns |
|---|---|
| `.discrimination()` | AUROC, AUPRC, sens/spec/PPV/NPV per subgroup + four parity gaps, with bootstrap CIs |
| `.calibration()` | intercept, slope, Brier, ECE, ICI + gap statistics |
| `.calibration_curves()` | LOESS calibration curves with bootstrap bands |
| `.roc_curves()` | per-subgroup ROC on a fixed FPR grid |
| `.hosmer_lemeshow(detail=..., df_mode=...)` | H–L statistic, df, p, decile regression; `detail=True` gives per-decile observed vs expected |
| `.decision_curve(thresholds=..., zero_crossing=True)` | net benefit, standardized net benefit, per-subgroup treat-all reference, zero crossing |
| `.estimability()` | the reportability verdict for every cell-metric, and which rule produced it |
| `.recalibration_ladder(by=..., method=..., folds=...)` | rungs none / common / group-specific, cross-fitted |
| `.cell_diagnostics()` | shrinkage diagnostics for intersectional cells |
| `.run_all()` | all axes in one call |
| `.report(out, title=..., metadata=..., mitigation_audits=..., table_one_data=..., pdf=False)` | the HTML report; `pdf=True` also writes a PDF beside it (`isitfair[pdf]`) |
| `.mitigate(method, calibration_data)` | apply a mitigation, get new audits back |

### Mitigation

| Class | Description |
|---|---|
| `GroupRecalibration(method="isotonic"\|"platt"\|"beta")` | per-subgroup recalibration; `"beta"` needs `isitfair[beta]` |
| `WassersteinPostprocessing(constraint="demographic_parity")` | Wasserstein-barycenter score post-processing (Chzhen 2020). Exact demographic parity, within-subgroup AUROC preserved by construction |
| `Reweighing()` | Kamiran–Calders sample weights for retraining |

### Statistics and plotting

| Function | Description |
|---|---|
| `cohens_d` · `cohens_h` · `cramers_v` · `eta_squared` · `rank_biserial_r` | effect sizes, with `effect_size_label` for Cohen's bands |
| `fdr_correct(pvals)` | Benjamini–Hochberg q-values |
| `lrt` · `interaction_test(df, outcome, sensitive, covariate, adjust_for=...)` | nested-model likelihood-ratio tests; reports `p_value` **and** `delta_r2` |
| `bootstrap_metric_difference(...)` | paired-bootstrap CI and p for a between-group metric gap |
| `odds_ratio_2x2(exposure, outcome)` | crude OR with Wald CI |
| `subgroup_metrics` · `hosmer_lemeshow` | standalone metric helpers |
| `table_one(df, group_col=...)` | multi-group Table 1 with BH-FDR and per-variable effect sizes |
| `compare_periods({label: audit, ...})` | one axis side by side across periods |
| `report_to_pdf(html, out=None)` | typeset an existing HTML report as a PDF, without re-running the audit (`isitfair[pdf]`) |
| `volcano_plot` · `forest_plot` · `lollipop_plot` · `subgroup_forest_plot` · `reliability_plot` · `roc_plot` · `decision_curve_plot` · `recalibration_ladder_plot` | matplotlib helpers |

Only the names in `isitfair.__init__` are public.

---

## Relationship to Fairlearn

`isitfair` and [Fairlearn](https://fairlearn.org/) are complementary.
Fairlearn provides general-purpose fairness metrics and mitigation for binary
*predictions*; `isitfair` provides clinical-prediction evaluation built around
*probability* inputs — calibration fairness, decision-curve fairness, and
TRIPOD+AI reporting. The mitigation methods shipped here (the recalibration
ladder, Wasserstein post-processing and Kamiran–Calders reweighing) are
implemented in the package, so Fairlearn is not a dependency.

`dcurves` provides decision curve analysis, `tableone` the Table 1 base,
`statsmodels` the LOESS smoothing and BH-FDR, and `sklearn` the isotonic
regression. The value here is the integration, the per-subgroup machinery around
them, and the estimability contract.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## Documentation

`docs/` holds the Sphinx sources — concepts, a guide per axis, intersectional
analysis, mitigation, reporting, and the analysis-statistics primitives.

```bash
pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html
```

## Citation

```bibtex
@article{guillenramirez2026isitfair,
  title   = {Subgroup deployment-readiness audit of calibration and clinical
             utility in perioperative infection prediction},
  author  = {Guillen-Ramirez, Hugo and Lucas, Katharina Lucia and
             Wintsch, Yves Max and Blatter, Tobias Ueli and Triep, Karen and
             Endrich, Olga and Beldi, Guido},
  journal = {Journal of the American Medical Informatics Association},
  year    = {2026},
  note    = {Under review. Software:
             https://github.com/HugoGuillen/isitfair},
}
```

Please also cite the SUPPORT2 cohort if you use the shipped examples: Knaus WA,
Harrell FE, Lynn J, et al. *Ann Intern Med* 1995;122:191–203.

## License

MIT — see [LICENSE](LICENSE). The bundled SUPPORT2 extract is redistributed
under CC BY 4.0 from the UCI Machine Learning Repository.
