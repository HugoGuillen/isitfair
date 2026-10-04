**Table 3.** Per-subgroup calibration on the audited cohort (*n* = 6,051; 501 infection events) for the three pre-specified protected attributes. Within-subgroup calibration intercept (calibration-in-the-large; fixed-slope logit recalibration) and calibration slope (logistic regression of outcome on linear predictor) with 95% bootstrap CIs (2,000 patient-level resamples). Brier score and 95% CI, expected calibration error (ECE; 10 quantile bins), and integrated calibration index (ICI). Per-axis maximum intercept- and slope-gaps with 95% bootstrap CIs are reported in the section header rows. A maximum gap is reported as not estimable where a stratum it is measured against falls below the pre-registered event floor; its point estimate is shown and its interval withheld. A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Ascertainment-window analysis: the published model's out-of-fold predictions restricted to 2018-2022, the period in which the present-on-admission coding the endpoint requires is available. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (96 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Ascertainment within this window remains incomplete: 22.9% of cohort-eligible infected patients become labelled cases in 2018, stepping to a 31-35% plateau from 2019 onward, and it differs by physiological reserve (Supplementary Table 18 of the paper).

| Subgroup | *n* | Events (%) | Intercept (95% CI) | Slope (95% CI) | Brier (95% CI) | ECE | ICI |
|---|---:|---:|---|---|---|---:|---:|
| **Sex** — max ΔIntercept 0.06 (0.00, 0.25) · max ΔSlope 0.07 (0.00, 0.33) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| Female | 2,090 | 189 (9.0) | -0.54 (-0.70, -0.39) | 1.40 (1.21, 1.61) | 0.066 (0.059, 0.073) | 0.047 | 0.045 |
| Male | 3,961 | 312 (7.9) | -0.60 (-0.72, -0.49) | 1.47 (1.32, 1.64) | 0.060 (0.055, 0.065) | 0.048 | 0.048 |
| **Age group** — max ΔIntercept 0.30 (not estimable) · max ΔSlope 0.60 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| 0-17 | 223 | 21 (9.4) | -0.46 (not estimable) | 1.47 (not estimable) | 0.076 (not estimable) | 0.052 | 0.047 |
| 18-44 | 518 | 34 (6.6) | -0.76 (-1.18, -0.44) | 1.24 (not estimable) | 0.056 (0.044, 0.069) | 0.057 | 0.055 |
| 45-64 | 2,049 | 146 (7.1) | -0.71 (-0.88, -0.55) | 1.43 (1.22, 1.69) | 0.055 (0.049, 0.061) | 0.053 | 0.051 |
| 65-79 | 2,698 | 247 (9.2) | -0.47 (-0.60, -0.34) | 1.60 (1.41, 1.82) | 0.063 (0.057, 0.069) | 0.046 | 0.048 |
| 80+ | 563 | 53 (9.4) | -0.60 (-0.91, -0.31) | 1.00 (not estimable) | 0.082 (0.068, 0.097) | 0.057 | 0.059 |
| **Physiological reserve (frailty)** — max ΔIntercept 1.56 (not estimable) · max ΔSlope 0.38 (not estimable) | | | | | | | |
| Overall | 6,051 | 501 (8.3) | -0.58 (-0.67, -0.49) | 1.44 (1.32, 1.58) | 0.062 (0.058, 0.066) | 0.047 | 0.047 |
| Frail | 3,684 | 399 (10.8) | -0.44 (-0.55, -0.34) | 1.49 (1.35, 1.65) | 0.075 (0.069, 0.080) | 0.044 | 0.045 |
| Non-frail | 547 | 6 (1.1) | -2.00 (not estimable) | 1.36 (not estimable) | 0.017 (not estimable) | 0.059 | 0.059 |
| Pre-frail | 1,820 | 96 (5.3) | -0.84 (-1.07, -0.65) | 1.11 (not estimable) | 0.050 (0.043, 0.057) | 0.055 | 0.054 |
