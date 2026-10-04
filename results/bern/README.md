# Clinical cohort: aggregated outputs

These are the aggregated subgroup outputs of the audit reported in the paper:
an end-of-surgery postoperative-infection model, audited on surgical
procedures at Inselspital, Bern University Hospital. The underlying data are
patient-level hospital records and cannot be shared. The files here are
aggregated: they report subgroup counts, rates, metrics and verdicts, not
patient records. Some aggregates are very small, however: the per-decile
Hosmer–Lemeshow tables in the audit reports split small subgroups (for example
pediatric procedures and some intersectional cells) into bins of one to five
procedures.

The code that produced them is the `isitfair` package in this repository. The
cohort-assembly scripts read the hospital extract and are not included.

## Two windows

Every result comes in two versions, and each file names its window.

| Window | Procedures | Infection events | Role in the paper |
|---|---:|---:|---|
| 2014–2022 | 10,719 | 504 | Primary audit: the cohort the deployed model was developed on |
| 2018–2022 | 6,051 | 501 | Ascertainment window, where the present-on-admission code the outcome needs is available |

Before 2018 the extract carries almost no present-on-admission coding, so
infections whose timing could not be established left the cohort at
construction. Coded prevalence is therefore 4.70% over 2014–2022 and 8.28% in
2018–2022. Discrimination is stable across the two windows. Calibration-in-the-
large, PPV and net benefit depend on prevalence, so read them against the window
named in the file.

## Files

| File | What it is | In the paper |
|---|---|---|
| `audit_report_bern-full.html` · `.pdf` | The full audit report, 2014–2022: all three axes, gap statistics, per-decile Hosmer–Lemeshow tables, intersectional diagnostics and a TRIPOD+AI appendix. | Supplementary Materials, Subgroup Audit Reports |
| `audit_report_bern-poa.html` · `.pdf` | The same report for 2018–2022. | — |
| `estimability_bern-full.csv` · `estimability_bern-poa.csv` | The estimability verdict for every cell in the corresponding report. | — |
| `tables_2014_2022/table1.*` | Cohort characterisation by recorded sex | Table 1 |
| `tables_2014_2022/table2_discrimination.*` | Discrimination, prespecified axes | Table 2 |
| `tables_2014_2022/table3_calibration.*` | Calibration, prespecified axes | Table 3 |
| `tables_2014_2022/table_s2_discrimination.*` | Discrimination, case-mix axes and intersections | Supplementary Table 19 |
| `tables_2014_2022/table_s3_calibration.*` | Calibration, case-mix axes and intersections | — |
| `tables_2014_2022/table_s4_intersectional.*` | Intersectional cell sparsity and shrinkage | Supplementary Table 12 |
| `tables_2018_2022/table2_discrimination.*` | Discrimination, prespecified axes | Supplementary Table 15 |
| `tables_2018_2022/table3_calibration.*` | Calibration, prespecified axes | Supplementary Table 16 |
| `tables_2018_2022/table_s4_intersectional.*` | Intersectional cell sparsity and shrinkage | Supplementary Table 17 |
| `tables_2018_2022/table1.*`, `table_s2_*`, `table_s3_*` | The remaining tables for the 2018–2022 window | — |
| `tables_*/estimability_enforced.csv` | The estimability verdict for every cell in the tables beside it | Supplementary Table 10 (2014–2022) |

The table captions use the numbering of the analysis that produced them, so a
caption's "Supplementary Table *n*" can differ from the paper. The last column
above gives the paper's number. Each table ships as `.csv` and `.md`.

## How they were produced

- **Predictions:** the published model's out-of-fold predictions from 5-fold
  stratified cross-validation.
- **Uncertainty:** 2,000 bootstrap resamples, drawn by resampling patients within
  each subgroup (`resample_scheme="patient"`), `random_state=42`.
- **Threshold:** t = 0.10, a reference operating point.
- **Estimability:** the package's event floors (AUROC 25, calibration intercept
  25, calibration slope 100, net benefit 50 events) together with the bootstrap
  replicate rule. A cell that fails either keeps its point estimate and loses its
  interval. The `estimability*.csv` files record which cells that applies to,
  and why.
- **Package version:** `isitfair` 0.4.0.

Several strata are small: the non-frail stratum has 6 infection events, and
some specialty, procedure-cluster and intersectional cells have fewer. Their
point estimates are shown for completeness and carry no interval.
