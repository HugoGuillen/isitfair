**Supplementary Table 6.** Per-subgroup calibration on the audited cohort (*n* = 10,719; 504 infection events) for the case-mix reference axes (surgical specialty, procedure cluster) and the three pre-specified pairwise intersections. Within-subgroup calibration intercept (calibration-in-the-large; fixed-slope logit recalibration) and calibration slope (logistic regression of outcome on linear predictor) with 95% bootstrap CIs (2,000 patient-level resamples). Brier score and 95% CI, expected calibration error (ECE; 10 quantile bins), and integrated calibration index (ICI). Per-axis maximum intercept- and slope-gaps with 95% bootstrap CIs are reported in the section header rows. A maximum gap is reported as not estimable where a stratum it is measured against falls below the pre-registered event floor; its point estimate is shown and its interval withheld. A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Primary analysis: the published model's out-of-fold predictions on the audited 2014-2022 cohort, which is the cohort the deployed model was fitted on. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (98 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Outcome ascertainment is not constant across this span. The present-on-admission coding the endpoint requires is effectively absent before 2018, so infection cases whose postoperative timing could not be established left the cohort at construction rather than being recorded as negatives, and 2014-2017 contributes 4,668 procedures against 3 coded events. Prevalence is therefore 4.70% here against 8.28% in 2018-2022, and prevalence-dependent quantities -- calibration-in-the-large, PPV and net benefit -- differ accordingly; the temporal-validation analyses report both.

| Subgroup | *n* | Events (%) | Intercept (95% CI) | Slope (95% CI) | Brier (95% CI) | ECE | ICI |
|---|---:|---:|---|---|---|---:|---:|
| **Surgical specialty (reference)** — max ΔIntercept 0.89 (not estimable) · max ΔSlope 1.27 (not estimable) | | | | | | | |
| Overall | 10,719 | 504 (4.7) | -1.02 (-1.12, -0.94) | 1.57 (1.45, 1.69) | 0.041 (0.039, 0.043) | 0.062 | 0.061 |
| ORL | 233 | 15 (6.4) | -1.15 (not estimable) | 1.30 (not estimable) | 0.068 (not estimable) | 0.098 | 0.098 |
| gynecology | 452 | 24 (5.3) | -0.69 (not estimable) | 1.41 (not estimable) | 0.044 (not estimable) | 0.042 | 0.045 |
| neuro | 1,117 | 26 (2.3) | -1.31 (-1.77, -0.96) | 1.86 (not estimable) | 0.022 (0.017, 0.028) | 0.052 | 0.054 |
| ortho | 1,690 | 76 (4.5) | -1.07 (-1.33, -0.87) | 1.32 (not estimable) | 0.045 (0.039, 0.051) | 0.065 | 0.065 |
| pediatric | 65 | 3 (4.6) | -1.22 (not estimable) | 0.71 (not estimable) | 0.054 (not estimable) | 0.107 | 0.089 |
| thoracic | 723 | 14 (1.9) | -1.54 (not estimable) | 1.98 (not estimable) | 0.020 (not estimable) | 0.056 | 0.055 |
| urology | 349 | 19 (5.4) | -0.78 (not estimable) | 1.29 (not estimable) | 0.048 (not estimable) | 0.055 | 0.051 |
| vascular | 4,852 | 184 (3.8) | -1.15 (-1.30, -1.01) | 1.66 (1.44, 1.89) | 0.036 (0.033, 0.039) | 0.064 | 0.062 |
| visceral | 1,238 | 143 (11.6) | -0.66 (-0.83, -0.48) | 1.57 (1.34, 1.85) | 0.078 (0.071, 0.086) | 0.071 | 0.067 |
| **Procedure cluster (reference)** — max ΔIntercept 1.24 (not estimable) · max ΔSlope 0.85 (not estimable) | | | | | | | |
| Overall | 10,719 | 504 (4.7) | -1.02 (-1.12, -0.94) | 1.57 (1.45, 1.69) | 0.041 (0.039, 0.043) | 0.062 | 0.061 |
| chopcluster_1 (NervousSystem) | 903 | 22 (2.4) | -1.32 (not estimable) | 1.83 (not estimable) | 0.023 (not estimable) | 0.055 | 0.055 |
| chopcluster_10 (UroAndrological+Urologic) | 346 | 28 (8.1) | -0.49 (-0.95, -0.14) | 1.36 (not estimable) | 0.064 (0.047, 0.082) | 0.060 | 0.052 |
| chopcluster_11 (Gynecological) | 303 | 9 (3.0) | -1.11 (not estimable) | 1.12 (not estimable) | 0.030 (not estimable) | 0.053 | 0.050 |
| chopcluster_12 (Musculoskeletal) | 1,623 | 62 (3.8) | -1.17 (-1.45, -0.94) | 1.34 (not estimable) | 0.040 (0.034, 0.046) | 0.066 | 0.065 |
| chopcluster_13 (Cutaneous) | 213 | 43 (20.2) | -0.11 (-0.45, 0.21) | 1.38 (not estimable) | 0.127 (0.103, 0.156) | 0.058 | 0.032 |
| chopcluster_3 (Thoracic) | 807 | 22 (2.7) | -1.35 (not estimable) | 1.94 (not estimable) | 0.026 (not estimable) | 0.059 | 0.058 |
| chopcluster_4 (Cardiac) | 3,764 | 118 (3.1) | -1.27 (-1.47, -1.09) | 1.58 (1.31, 1.89) | 0.032 (0.029, 0.035) | 0.064 | 0.063 |
| chopcluster_5 (Vascular) | 769 | 30 (3.9) | -1.20 (-1.60, -0.88) | 1.73 (not estimable) | 0.036 (0.029, 0.044) | 0.068 | 0.067 |
| chopcluster_6 (AbdominalWall) | 530 | 67 (12.6) | -0.57 (-0.84, -0.32) | 1.62 (not estimable) | 0.078 (0.066, 0.091) | 0.067 | 0.065 |
| chopcluster_7 (GastroIntestinal) | 298 | 36 (12.1) | -0.79 (-1.18, -0.46) | 1.75 (not estimable) | 0.090 (0.074, 0.105) | 0.093 | 0.093 |
| chopcluster_8 (Hepatobiliary) | 306 | 23 (7.5) | -0.90 (not estimable) | 1.08 (not estimable) | 0.068 (not estimable) | 0.082 | 0.074 |
| none | 857 | 44 (5.1) | -1.01 (-1.37, -0.72) | 1.58 (not estimable) | 0.045 (0.038, 0.053) | 0.064 | 0.063 |
| **Sex × age group** — max ΔIntercept 0.64 (not estimable) · max ΔSlope 1.45 (not estimable) | | | | | | | |
| Overall | 10,719 | 504 (4.7) | -1.02 (-1.12, -0.94) | 1.57 (1.45, 1.69) | 0.041 (0.039, 0.043) | 0.062 | 0.061 |
| female / 0-17† | 175 | 13 (7.4) | -0.76 (not estimable) | 2.24 (not estimable) | 0.061 (not estimable) | 0.077 | 0.064 |
| female / 18-44† | 408 | 13 (3.2) | -1.40 (not estimable) | 1.53 (not estimable) | 0.034 (not estimable) | 0.073 | 0.072 |
| female / 45-64 | 1,179 | 52 (4.4) | -1.06 (-1.38, -0.78) | 1.36 (not estimable) | 0.039 (0.032, 0.046) | 0.061 | 0.059 |
| female / 65-79 | 1,470 | 93 (6.3) | -0.75 (-0.97, -0.55) | 1.81 (not estimable) | 0.046 (0.040, 0.053) | 0.052 | 0.056 |
| female / 80+† | 457 | 19 (4.2) | -1.31 (not estimable) | 1.18 (not estimable) | 0.048 (not estimable) | 0.083 | 0.084 |
| male / 0-17† | 192 | 8 (4.2) | -1.22 (not estimable) | 0.79 (not estimable) | 0.049 (not estimable) | 0.086 | 0.080 |
| male / 18-44† | 533 | 21 (3.9) | -1.07 (not estimable) | 1.33 (not estimable) | 0.038 (not estimable) | 0.058 | 0.058 |
| male / 45-64 | 2,536 | 95 (3.7) | -1.21 (-1.42, -1.02) | 1.75 (not estimable) | 0.035 (0.031, 0.039) | 0.067 | 0.065 |
| male / 65-79 | 3,214 | 156 (4.9) | -0.96 (-1.12, -0.81) | 1.61 (1.40, 1.86) | 0.041 (0.037, 0.045) | 0.059 | 0.057 |
| male / 80+† | 555 | 34 (6.1) | -0.90 | 1.17 (not estimable) | 0.058 | 0.067 | 0.068 |
| **Sex × frailty** — max ΔIntercept 2.87 (not estimable) · max ΔSlope 3.17 (not estimable) | | | | | | | |
| Overall | 10,719 | 504 (4.7) | -1.02 (-1.12, -0.94) | 1.57 (1.45, 1.69) | 0.041 (0.039, 0.043) | 0.062 | 0.061 |
| female / frail | 1,666 | 129 (7.7) | -0.82 (-1.01, -0.65) | 1.68 (1.44, 1.98) | 0.058 (0.052, 0.065) | 0.068 | 0.067 |
| female / non_frail† | 522 | 1 (0.2) | -3.69 (not estimable) | 4.40 (not estimable) | 0.009 (not estimable) | 0.064 | 0.064 |
| female / pre_frail | 1,501 | 60 (4.0) | -0.98 (-1.26, -0.74) | 1.23 (not estimable) | 0.039 (0.033, 0.046) | 0.053 | 0.053 |
| male / frail | 4,473 | 271 (6.1) | -0.87 (-1.00, -0.76) | 1.56 (1.40, 1.74) | 0.050 (0.046, 0.053) | 0.062 | 0.060 |
| male / non_frail† | 500 | 5 (1.0) | -1.83 (not estimable) | 1.37 (not estimable) | 0.014 (not estimable) | 0.046 | 0.046 |
| male / pre_frail† | 2,057 | 38 (1.8) | -1.73 | 1.33 (not estimable) | 0.025 | 0.069 | 0.069 |
| **Age group × frailty** — max ΔIntercept 2.00 (not estimable) · max ΔSlope 2.59 (not estimable) | | | | | | | |
| Overall | 10,719 | 504 (4.7) | -1.02 (-1.12, -0.94) | 1.57 (1.45, 1.69) | 0.041 (0.039, 0.043) | 0.062 | 0.061 |
| 0-17 / frail† | 211 | 19 (9.0) | -0.69 (not estimable) | 1.44 (not estimable) | 0.079 (not estimable) | 0.070 | 0.066 |
| 0-17 / non_frail† | 45 | 0 (0.0) | — | — | 0.007 (not estimable) | 0.072 | 0.072 |
| 0-17 / pre_frail† | 111 | 2 (1.8) | -1.86 (not estimable) | 0.24 (not estimable) | 0.028 (not estimable) | 0.093 | 0.084 |
| 18-44 / frail† | 300 | 21 (7.0) | -0.99 (not estimable) | 1.60 (not estimable) | 0.060 (not estimable) | 0.083 | 0.081 |
| 18-44 / non_frail† | 331 | 2 (0.6) | -2.47 (not estimable) | 1.06 (not estimable) | 0.013 (not estimable) | 0.056 | 0.056 |
| 18-44 / pre_frail† | 310 | 11 (3.5) | -1.08 (not estimable) | 0.97 (not estimable) | 0.039 (not estimable) | 0.056 | 0.057 |
| 45-64 / frail | 1,967 | 111 (5.6) | -1.00 (-1.20, -0.83) | 1.63 (1.40, 1.92) | 0.046 (0.041, 0.051) | 0.069 | 0.067 |
| 45-64 / non_frail† | 410 | 3 (0.7) | -2.22 (not estimable) | 2.79 (not estimable) | 0.011 (not estimable) | 0.051 | 0.051 |
| 45-64 / pre_frail† | 1,338 | 33 (2.5) | -1.42 | 1.25 (not estimable) | 0.029 | 0.063 | 0.062 |
| 65-79 / frail | 3,032 | 204 (6.7) | -0.76 (-0.91, -0.63) | 1.73 (1.52, 1.98) | 0.050 (0.045, 0.054) | 0.057 | 0.058 |
| 65-79 / non_frail† | 218 | 1 (0.5) | -2.69 (not estimable) | 0.20 (not estimable) | 0.011 (not estimable) | 0.055 | 0.055 |
| 65-79 / pre_frail† | 1,434 | 44 (3.1) | -1.24 | 1.44 (not estimable) | 0.032 | 0.059 | 0.058 |
| 80+ / frail† | 629 | 45 (7.2) | -0.90 | 1.13 (not estimable) | 0.068 | 0.075 | 0.076 |
| 80+ / non_frail† | 18 | 0 (0.0) | — | — | 0.012 (not estimable) | 0.078 | 0.078 |
| 80+ / pre_frail† | 365 | 8 (2.2) | -1.64 (not estimable) | 0.98 (not estimable) | 0.030 (not estimable) | 0.074 | 0.073 |
