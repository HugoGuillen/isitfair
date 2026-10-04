# Examples

Three runnable artifacts ship with the package, all on the public **SUPPORT2**
cohort (UCI Machine Learning Repository dataset 880, CC BY 4.0): 9,105 critically
ill adults, audited on a held-out test partition of 1,821.

```bash
pip install -e ".[examples,plots]"
```

Set `ISITFAIR_FAST=1` on any of them to drop the bootstrap from B = 2000 to
B = 200 for a quick pass while you are editing.

## 1. Generating an audit report

**Source**: [`examples/support2_mortality_audit.py`](https://github.com/HugoGuillen/isitfair/blob/main/examples/support2_mortality_audit.py),
paired with `notebooks/01_audit_report_support2.ipynb`.

The place to start. A complete
subgroup deployment-readiness audit of an in-hospital mortality model, producing
the self-contained TRIPOD+AI HTML report.

```bash
python examples/support2_mortality_audit.py
```

Covers: the four required inputs · constructing the audit · discrimination and
the four parity gaps · calibration, curves and Hosmer–Lemeshow · decision curves
· the estimability verdict · intersectional cells with empirical-Bayes shrinkage
· group-specific isotonic recalibration · report generation.

Writes `results/support2_audit_report.html` and its sibling
`support2_audit_report_figures/` directory (15 PNGs at 300 DPI). With the `pdf`
extra installed it also writes `results/support2_audit_report.pdf`.

## Your own data: a template

**Source**: [`examples/audit_your_own_data.py`](https://github.com/HugoGuillen/isitfair/blob/main/examples/audit_your_own_data.py).

A short script to copy and edit: set your column names at the top, point it at
a CSV of your own predictions, and it writes the audit report. Run without a
file, it demonstrates itself on SUPPORT2. See {doc}`../user_guide/your_own_data`.

```bash
python examples/audit_your_own_data.py my_predictions.csv --fast
```

## 2. The manuscript's analysis structure

**Source**: `notebooks/02_paper_analysis_support2.ipynb`.

Works through every analysis the package supports, section by section: Table 1
with BH-FDR and effect sizes, subgroup discrimination and gaps, calibration and
Hosmer–Lemeshow, decision curves with a threshold-band sensitivity, the
estimability verdict, the recalibration ladder, the Wasserstein comparator,
intersectional cells, a feature-by-sex effect-size scan, and a comorbidity × sex
interaction screen. Section headings name the corresponding element of the
companion paper.

Writes 20 tables (`.csv` and `.md`) to `results/tables/` and 8 figures (`.png`
and `.pdf`) to `results/figures/`.


## 3. The tutorial, runnable

**Source**: `notebooks/03_tutorial_support2.ipynb`, built from
`slides/isitfair_audit_tutorial.pdf`.

The nine-section tutorial deck with every code block executing against real
data: why audit at all · the three axes · setting up · discrimination ·
calibration · clinical utility · uncertainty, floors and multiplicity ·
mitigation · reporting.

Figures carried over from the tutorial's worked example are marked `[study]`;
everything else on the page is computed in the cell above it.

## The clinical cohort's outputs

The paper's clinical cohort is patient-level hospital data and cannot be
shared. Its aggregated subgroup outputs (two audit reports and the manuscript's
subgroup tables) are in
[`results/bern/`](https://github.com/HugoGuillen/isitfair/tree/main/results/bern),
with a README describing each file.

## The cohort module

`examples/support2_data.py` holds the cohort definition — loading, the audit
axes, the model matrix, the 60/20/20 split and the reference model. All three
artifacts above build their cohort through it, so they audit an identical test
partition under the same seed.

A copy of SUPPORT2 is bundled at `examples/data/support2.csv.gz`, so the
examples run offline; the loader tries UCI first and falls back to it.

Its doctests are a quick end-to-end check:

```bash
python -m doctest examples/support2_data.py && echo OK
```
