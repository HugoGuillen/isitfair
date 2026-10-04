**Supplementary Table 6.** Per-subgroup calibration on the audited cohort (*n* = 6,051; 501 infection events) for the case-mix reference axes (surgical specialty, procedure cluster) and the three pre-specified pairwise intersections. Within-subgroup calibration intercept (calibration-in-the-large; fixed-slope logit recalibration) and calibration slope (logistic regression of outcome on linear predictor) with 95% bootstrap CIs (2,000 patient-level resamples). Brier score and 95% CI, expected calibration error (ECE; 10 quantile bins), and integrated calibration index (ICI). Per-axis maximum intercept- and slope-gaps with 95% bootstrap CIs are reported in the section header rows. A maximum gap is reported as not estimable where a stratum it is measured against falls below the pre-registered event floor; its point estimate is shown and its interval withheld. A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Ascertainment-window analysis: the published model's out-of-fold predictions restricted to 2018-2022, the period in which the present-on-admission coding the endpoint requires is available. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (96 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Ascertainment within this window remains incomplete: 22.9% of cohort-eligible infected patients become labelled cases in 2018, stepping to a 31-35% plateau from 2019 onward, and it differs by physiological reserve (Supplementary Table 18 of the paper).

| Subgroup | *n* | Events (%) | Intercept (95% CI) | Slope (95% CI) | Brier (95% CI) | ECE | ICI |
|---|---:|---:|---|---|---|---:|---:|
| **Surgical specialty (reference)** — max ΔIntercept 0.91 (not estimable) · max ΔSlope 1.32 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| ORL | 118 | 14 (11.9) | -0.78 (not estimable) | 0.95 (not estimable) | 0.110 (not estimable) | 0.104 | 0.097 |
| gynecology | 274 | 24 (8.8) | -0.26 (not estimable) | 1.22 (not estimable) | 0.066 (not estimable) | 0.031 | 0.028 |
| neuro | 488 | 25 (5.1) | -0.72 (-1.20, -0.37) | 1.76 (not estimable) | 0.040 (0.029, 0.053) | 0.047 | 0.053 |
| ortho | 971 | 76 (7.8) | -0.64 (-0.89, -0.41) | 1.17 (not estimable) | 0.067 (0.058, 0.078) | 0.051 | 0.051 |
| pediatric | 37 | 3 (8.1) | -0.68 (not estimable) | 0.57 (not estimable) | 0.080 (not estimable) | 0.123 | 0.078 |
| thoracic | 403 | 14 (3.5) | -1.07 (not estimable) | 1.89 (not estimable) | 0.029 (not estimable) | 0.051 | 0.050 |
| urology | 200 | 19 (9.5) | -0.30 (not estimable) | 1.11 (not estimable) | 0.074 (not estimable) | 0.069 | 0.041 |
| vascular | 2,838 | 183 (6.4) | -0.74 (-0.90, -0.60) | 1.54 (1.31, 1.80) | 0.053 (0.048, 0.058) | 0.054 | 0.053 |
| visceral | 722 | 143 (19.8) | -0.17 (-0.36, 0.03) | 1.43 (1.21, 1.75) | 0.111 (0.099, 0.124) | 0.052 | 0.043 |
| **Procedure cluster (reference)** — max ΔIntercept 1.07 (not estimable) · max ΔSlope 0.85 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| chopcluster_1 (NervousSystem) | 395 | 21 (5.3) | -0.76 (not estimable) | 1.69 (not estimable) | 0.041 (not estimable) | 0.052 | 0.053 |
| chopcluster_10 (UroAndrological+Urologic) | 186 | 28 (15.1) | 0.11 (-0.35, 0.48) | 1.20 (not estimable) | 0.105 (0.074, 0.135) | 0.095 | 0.057 |
| chopcluster_11 (Gynecological) | 178 | 9 (5.1) | -0.68 (not estimable) | 0.95 (not estimable) | 0.047 (not estimable) | 0.042 | 0.040 |
| chopcluster_12 (Musculoskeletal) | 826 | 61 (7.4) | -0.63 (-0.91, -0.38) | 1.18 (not estimable) | 0.064 (0.054, 0.075) | 0.048 | 0.048 |
| chopcluster_13 (Cutaneous) | 154 | 43 (27.9) | 0.18 (-0.21, 0.53) | 1.11 (not estimable) | 0.167 (0.135, 0.200) | 0.057 | 0.032 |
| chopcluster_3 (Thoracic) | 414 | 22 (5.3) | -0.81 (not estimable) | 1.76 (not estimable) | 0.039 (not estimable) | 0.048 | 0.049 |
| chopcluster_4 (Cardiac) | 2,245 | 118 (5.3) | -0.87 (-1.07, -0.69) | 1.46 (1.20, 1.79) | 0.047 (0.041, 0.052) | 0.058 | 0.056 |
| chopcluster_5 (Vascular) | 328 | 30 (9.1) | -0.57 (-1.00, -0.24) | 1.51 (not estimable) | 0.068 (0.052, 0.085) | 0.056 | 0.051 |
| chopcluster_6 (AbdominalWall) | 373 | 66 (17.7) | -0.24 (-0.52, 0.01) | 1.62 (not estimable) | 0.098 (0.082, 0.115) | 0.072 | 0.057 |
| chopcluster_7 (GastroIntestinal) | 143 | 36 (25.2) | -0.08 (-0.48, 0.27) | 1.52 (not estimable) | 0.141 (0.113, 0.171) | 0.078 | 0.041 |
| chopcluster_8 (Hepatobiliary) | 142 | 23 (16.2) | -0.16 (not estimable) | 0.91 (not estimable) | 0.112 (not estimable) | 0.057 | 0.029 |
| none | 667 | 44 (6.6) | -0.89 (-1.25, -0.60) | 1.51 (not estimable) | 0.056 (0.046, 0.066) | 0.068 | 0.066 |
| **Sex × age group** — max ΔIntercept 0.64 (not estimable) · max ΔSlope 1.30 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| female / 0-17† | 109 | 13 (11.9) | -0.32 (not estimable) | 2.09 (not estimable) | 0.085 (not estimable) | 0.068 | 0.068 |
| female / 18-44† | 220 | 13 (5.9) | -0.96 (not estimable) | 1.28 (not estimable) | 0.054 (not estimable) | 0.072 | 0.068 |
| female / 45-64 | 683 | 52 (7.6) | -0.67 (-1.01, -0.39) | 1.18 (not estimable) | 0.059 (0.049, 0.071) | 0.054 | 0.050 |
| female / 65-79 | 839 | 92 (11.0) | -0.33 (-0.55, -0.12) | 1.69 (not estimable) | 0.069 (0.059, 0.080) | 0.047 | 0.047 |
| female / 80+† | 239 | 19 (7.9) | -0.83 (not estimable) | 0.96 (not estimable) | 0.076 (not estimable) | 0.076 | 0.078 |
| male / 0-17† | 114 | 8 (7.0) | -0.66 (not estimable) | 0.79 (not estimable) | 0.068 (not estimable) | 0.078 | 0.053 |
| male / 18-44† | 298 | 21 (7.0) | -0.62 (not estimable) | 1.23 (not estimable) | 0.058 (not estimable) | 0.047 | 0.048 |
| male / 45-64 | 1,366 | 94 (6.9) | -0.73 (-0.95, -0.54) | 1.61 (not estimable) | 0.053 (0.046, 0.060) | 0.054 | 0.054 |
| male / 65-79 | 1,859 | 155 (8.3) | -0.54 (-0.71, -0.38) | 1.54 (1.31, 1.81) | 0.060 (0.053, 0.067) | 0.049 | 0.049 |
| male / 80+† | 324 | 34 (10.5) | -0.45 | 1.04 (not estimable) | 0.086 | 0.046 | 0.045 |
| **Sex × frailty** — max ΔIntercept 2.81 (not estimable) · max ΔSlope 3.10 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| female / frail | 1,010 | 129 (12.8) | -0.41 (-0.60, -0.24) | 1.50 (1.25, 1.81) | 0.083 (0.073, 0.093) | 0.055 | 0.045 |
| female / non_frail† | 275 | 1 (0.4) | -3.22 (not estimable) | 4.17 (not estimable) | 0.013 (not estimable) | 0.072 | 0.072 |
| female / pre_frail | 805 | 59 (7.3) | -0.54 (-0.84, -0.29) | 1.06 (not estimable) | 0.063 (0.051, 0.075) | 0.041 | 0.041 |
| male / frail | 2,674 | 270 (10.1) | -0.45 (-0.58, -0.33) | 1.49 (1.33, 1.67) | 0.072 (0.066, 0.078) | 0.044 | 0.045 |
| male / non_frail† | 272 | 5 (1.8) | -1.35 (not estimable) | 1.28 (not estimable) | 0.021 (not estimable) | 0.045 | 0.045 |
| male / pre_frail† | 1,015 | 37 (3.6) | -1.19 | 1.15 (not estimable) | 0.039 | 0.065 | 0.065 |
| **Age group × frailty** — max ΔIntercept 2.02 (not estimable) · max ΔSlope 2.57 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| 0-17 / frail† | 128 | 19 (14.8) | -0.18 (not estimable) | 1.35 (not estimable) | 0.111 (not estimable) | 0.095 | 0.046 |
| 0-17 / non_frail† | 28 | 0 (0.0) | — | — | 0.007 (not estimable) | 0.073 | 0.073 |
| 0-17 / pre_frail† | 67 | 2 (3.0) | -1.35 (not estimable) | 0.20 (not estimable) | 0.038 (not estimable) | 0.091 | 0.076 |
| 18-44 / frail† | 183 | 21 (11.5) | -0.61 (not estimable) | 1.46 (not estimable) | 0.084 (not estimable) | 0.065 | 0.068 |
| 18-44 / non_frail† | 182 | 2 (1.1) | -2.07 (not estimable) | 0.85 (not estimable) | 0.020 (not estimable) | 0.064 | 0.064 |
| 18-44 / pre_frail† | 153 | 11 (7.2) | -0.46 (not estimable) | 0.79 (not estimable) | 0.067 (not estimable) | 0.061 | 0.037 |
| 45-64 / frail | 1,167 | 110 (9.4) | -0.59 (-0.80, -0.42) | 1.50 (1.26, 1.79) | 0.067 (0.058, 0.075) | 0.052 | 0.050 |
| 45-64 / non_frail† | 210 | 3 (1.4) | -1.66 (not estimable) | 2.71 (not estimable) | 0.016 (not estimable) | 0.051 | 0.051 |
| 45-64 / pre_frail† | 672 | 33 (4.9) | -0.89 | 1.06 (not estimable) | 0.048 | 0.060 | 0.055 |
| 65-79 / frail | 1,846 | 204 (11.1) | -0.35 (-0.50, -0.21) | 1.67 (1.45, 1.93) | 0.071 (0.064, 0.079) | 0.056 | 0.052 |
| 65-79 / non_frail† | 118 | 1 (0.8) | -2.20 (not estimable) | 0.13 (not estimable) | 0.016 (not estimable) | 0.067 | 0.059 |
| 65-79 / pre_frail† | 734 | 42 (5.7) | -0.79 | 1.31 (not estimable) | 0.050 | 0.053 | 0.051 |
| 80+ / frail† | 360 | 45 (12.5) | -0.45 | 0.95 (not estimable) | 0.103 | 0.061 | 0.052 |
| 80+ / non_frail† | 9 | 0 (0.0) | — | — | 0.020 (not estimable) | 0.109 | 0.109 |
| 80+ / pre_frail† | 194 | 8 (4.1) | -1.12 (not estimable) | 0.82 (not estimable) | 0.046 (not estimable) | 0.073 | 0.067 |
