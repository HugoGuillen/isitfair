# Examples

## `support2_mortality_audit.py`

A complete subgroup deployment-readiness audit of an in-hospital mortality model
on the public SUPPORT2 cohort, producing the HTML report in `results/`.

This is the script the manuscript's Supplementary Methods refers to, and the
path the tutorial deck's final slide points at, so its filename is a stable
reference.

```bash
pip install -e ".[examples,plots]"
python examples/support2_mortality_audit.py            # B = 2000, ~12 min
ISITFAIR_FAST=1 python examples/support2_mortality_audit.py   # B = 200, ~1 min
```

It is written in the [jupytext](https://jupytext.readthedocs.io/) `py:percent`
format, so it opens directly in Jupyter as well as running as a plain script.
`notebooks/01_audit_report_support2.ipynb` is generated from it, so the two
share one source. After editing the script, regenerate the notebook with:

```bash
jupytext --to ipynb --output notebooks/01_audit_report_support2.ipynb \
    examples/support2_mortality_audit.py
```

With the `pdf` extra installed (`pip install -e ".[pdf]"`), the script also
writes `results/support2_audit_report.pdf`.

## `audit_your_own_data.py`

A template for auditing **your own** model's predictions. Edit the column names
in its configuration block, then:

```bash
python examples/audit_your_own_data.py path/to/your_predictions.csv          # B = 2000
python examples/audit_your_own_data.py path/to/your_predictions.csv --fast   # B = 200
```

Without a file it runs on a demonstration CSV built from SUPPORT2. It writes
`audit_report.html` and `audit_report_figures/` to the current directory
(`--out` changes that). See `docs/user_guide/your_own_data.md`.

## `support2_data.py`

The cohort definition: data loading, the audit axes, the model matrix, the
60/20/20 split and the reference logistic regression.

Every runnable artifact in the repository builds its cohort through this module,
so the script and all three notebooks audit an *identical* test partition under
the same seed. Each function covers one stage, so the example script can call them in sequence
with the pipeline visible step by step, while the notebooks call
`fit_reference_model()`, which chains them in one line.

Its doctests are a quick end-to-end check:

```bash
python -m doctest examples/support2_data.py && echo OK
```

## `data/support2.csv.gz`

The SUPPORT2 cohort as distributed by the UCI Machine Learning Repository
(dataset 880), redistributed here under CC BY 4.0. 9,105 rows x 45 columns — the
concatenation of the UCI feature and target frames, unmodified.

`load_support2()` fetches from UCI and falls back to this copy, so the examples
run offline as well as online. Pass `prefer_cache=True` to read the bundled copy
directly.

> Knaus WA, Harrell FE, Lynn J, et al. The SUPPORT prognostic model: objective
> estimates of survival for seriously ill hospitalized adults.
> *Ann Intern Med* 1995;122:191-203.

## `make_manifest.py`

Rebuilds `results/MANIFEST.csv` after a re-run: one row per executed artifact,
with the producing notebook, seed, replicate count, package versions and a
content hash.

```bash
python examples/make_manifest.py
```
