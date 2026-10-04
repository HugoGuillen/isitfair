**Supplementary Table 7. Two-way intersectional cell sparsity and gap statistics (audited Bern cohort, *n* = 10,719; 504 infection events).** For each of the three pre-specified two-way intersections, every cell's size, outcome count, and empirical-Bayes (EB) shrinkage status is reported. Cells with fewer than 50 outcome events (marked †) were estimated with EB shrinkage of the cell-level coded-infection rate toward the marginal subgroup rate; the shrinkage factor (0 = no shrinkage, 1 = full shrinkage to the margin) is shown for those cells. The per-intersection header row gives the number of cells, the number shrunk, the equalized-odds gap (discrimination), and the maximum net-benefit gap with 95% bootstrap CI (clinical utility). Primary analysis: the published model's out-of-fold predictions on the audited 2014-2022 cohort, which is the cohort the deployed model was fitted on. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (98 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Outcome ascertainment is not constant across this span. The present-on-admission coding the endpoint requires is effectively absent before 2018, so infection cases whose postoperative timing could not be established left the cohort at construction rather than being recorded as negatives, and 2014-2017 contributes 4,668 procedures against 3 coded events. Prevalence is therefore 4.70% here against 8.28% in 2018-2022, and prevalence-dependent quantities -- calibration-in-the-large, PPV and net benefit -- differ accordingly; the temporal-validation analyses report both.

| Cell | *n* | Events | Cell infection rate (%) | Marginal rate (%) | EB shrinkage factor |
|---|---:|---:|---:|---:|---:|
| **Sex × age group** — 10 cells, 6 EB-shrunk · equalized-odds gap 0.13 · max net-benefit gap 0.052 (0.044, 0.102) | | | | | |
| female / 0-17† | 175 | 13 | 7.4 | 5.2 | 0.50 |
| female / 18-44† | 408 | 13 | 3.2 | 5.2 | 0.30 |
| female / 45-64 | 1,179 | 52 | 4.4 | 5.2 | — |
| female / 65-79 | 1,470 | 93 | 6.3 | 5.2 | — |
| female / 80+† | 457 | 19 | 4.2 | 5.2 | 0.28 |
| male / 0-17† | 192 | 8 | 4.2 | 4.5 | 0.48 |
| male / 18-44† | 533 | 21 | 3.9 | 4.5 | 0.25 |
| male / 45-64 | 2,536 | 95 | 3.7 | 4.5 | — |
| male / 65-79 | 3,214 | 156 | 4.9 | 4.5 | — |
| male / 80+† | 555 | 34 | 6.1 | 4.5 | 0.24 |
| **Sex × physiological reserve** — 6 cells, 3 EB-shrunk · equalized-odds gap 0.25 · max net-benefit gap 0.073 (0.061, 0.087) | | | | | |
| female / frail | 1,666 | 129 | 7.7 | 5.2 | — |
| female / non_frail† | 522 | 1 | 0.2 | 5.2 | 0.29 |
| female / pre_frail | 1,501 | 60 | 4.0 | 5.2 | — |
| male / frail | 4,473 | 271 | 6.1 | 4.5 | — |
| male / non_frail† | 500 | 5 | 1.0 | 4.5 | 0.30 |
| male / pre_frail† | 2,057 | 38 | 1.8 | 4.5 | 0.09 |
| **Age group × physiological reserve** — 15 cells, 13 EB-shrunk · equalized-odds gap 0.40 · max net-benefit gap 0.091 (0.074, 0.137) | | | | | |
| 0-17 / frail† | 211 | 19 | 9.0 | 5.7 | 0.39 |
| 0-17 / non_frail† | 45 | 0 | 0.0 | 5.7 | 0.75 |
| 0-17 / pre_frail† | 111 | 2 | 1.8 | 5.7 | 0.55 |
| 18-44 / frail† | 300 | 21 | 7.0 | 3.6 | 0.31 |
| 18-44 / non_frail† | 331 | 2 | 0.6 | 3.6 | 0.29 |
| 18-44 / pre_frail† | 310 | 11 | 3.5 | 3.6 | 0.30 |
| 45-64 / frail | 1,967 | 111 | 5.6 | 4.0 | — |
| 45-64 / non_frail† | 410 | 3 | 0.7 | 4.0 | 0.25 |
| 45-64 / pre_frail† | 1,338 | 33 | 2.5 | 4.0 | 0.09 |
| 65-79 / frail | 3,032 | 204 | 6.7 | 5.3 | — |
| 65-79 / non_frail† | 218 | 1 | 0.5 | 5.3 | 0.38 |
| 65-79 / pre_frail† | 1,434 | 44 | 3.1 | 5.3 | 0.09 |
| 80+ / frail† | 629 | 45 | 7.2 | 5.2 | 0.18 |
| 80+ / non_frail† | 18 | 0 | 0.0 | 5.2 | 0.88 |
| 80+ / pre_frail† | 365 | 8 | 2.2 | 5.2 | 0.27 |
