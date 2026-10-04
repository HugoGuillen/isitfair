**Table 2.** Per-subgroup discrimination on the audited cohort (*n* = 6,051; 501 infection events). Within-subgroup AUROC, AUPRC, sensitivity, and specificity at the *t* = 0.10 operating threshold, with 95% bootstrap CIs (2,000 patient-level resamples). Gap statistics per axis are reported in the section header rows (EOD = equal-opportunity difference, PE = predictive-equality difference, EOG = equalized-odds gap, SP = statistical parity). A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Ascertainment-window analysis: the published model's out-of-fold predictions restricted to 2018-2022, the period in which the present-on-admission coding the endpoint requires is available. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (96 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Ascertainment within this window remains incomplete: 22.9% of cohort-eligible infected patients become labelled cases in 2018, stepping to a 31-35% plateau from 2019 onward, and it differs by physiological reserve (Supplementary Table 18 of the paper).

| Subgroup | *n* | Events (%) | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---:|---:|---|---|---|---|
| **Sex** — EOD 0.007 · PE 0.008 · EOG 0.008 · SP 0.013 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| Female | 2,090 | 189 (9.0) | 0.857 (0.826–0.886) | 0.456 (0.385–0.530) | 0.905 (0.862–0.944) | 0.584 (0.563–0.606) |
| Male | 3,961 | 312 (7.9) | 0.853 (0.830–0.876) | 0.430 (0.372–0.487) | 0.897 (0.862–0.930) | 0.592 (0.576–0.608) |
| **Age group** — EOD 0.094 · PE 0.179 · EOG 0.179 · SP 0.178 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| 0-17 | 223 | 21 (9.4) | 0.803 (not estimable) | 0.350 (not estimable) | 0.857 (not estimable) | 0.441 (not estimable) |
| 18-44 | 518 | 34 (6.6) | 0.823 (0.756–0.884) | 0.396 (0.243–0.540) | 0.824 (0.692–0.938) | 0.620 (0.574–0.662) |
| 45-64 | 2,049 | 146 (7.1) | 0.867 (0.831–0.898) | 0.427 (0.347–0.511) | 0.918 (0.872–0.959) | 0.610 (0.588–0.632) |
| 65-79 | 2,698 | 247 (9.2) | 0.872 (0.845–0.896) | 0.515 (0.450–0.580) | 0.915 (0.878–0.948) | 0.598 (0.579–0.617) |
| 80+ | 563 | 53 (9.4) | 0.768 (0.699–0.829) | 0.288 (0.191–0.411) | 0.849 (0.741–0.938) | 0.502 (0.457–0.546) |
| **Physiological reserve (frailty)** — EOD 0.136 · PE 0.249 · EOG 0.249 · SP 0.292 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| Frail | 3,684 | 399 (10.8) | 0.860 (0.839–0.879) | 0.495 (0.442–0.548) | 0.927 (0.901–0.952) | 0.533 (0.516–0.551) |
| Non-frail | 547 | 6 (1.1) | 0.840 (not estimable) | 0.099 (not estimable) | 0.833 (not estimable) | 0.782 (not estimable) |
| Pre-frail | 1,820 | 96 (5.3) | 0.791 (0.737–0.837) | 0.241 (0.168–0.332) | 0.792 (0.701–0.868) | 0.636 (0.614–0.659) |
