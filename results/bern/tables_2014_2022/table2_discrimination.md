**Table 2.** Per-subgroup discrimination on the audited cohort (*n* = 10,719; 504 infection events). Within-subgroup AUROC, AUPRC, sensitivity, and specificity at the *t* = 0.10 operating threshold, with 95% bootstrap CIs (2,000 patient-level resamples). Gap statistics per axis are reported in the section header rows (EOD = equal-opportunity difference, PE = predictive-equality difference, EOG = equalized-odds gap, SP = statistical parity). A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Primary analysis: the published model's out-of-fold predictions on the audited 2014-2022 cohort, which is the cohort the deployed model was fitted on. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (98 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Outcome ascertainment is not constant across this span. The present-on-admission coding the endpoint requires is effectively absent before 2018, so infection cases whose postoperative timing could not be established left the cohort at construction rather than being recorded as negatives, and 2014-2017 contributes 4,668 procedures against 3 coded events. Prevalence is therefore 4.70% here against 8.28% in 2018-2022, and prevalence-dependent quantities -- calibration-in-the-large, PPV and net benefit -- differ accordingly; the temporal-validation analyses report both.

| Subgroup | *n* | Events (%) | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---:|---:|---|---|---|---|
| **Sex** — EOD 0.002 · PE 0.010 · EOG 0.010 · SP 0.014 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| Female | 3,689 | 190 (5.2) | 0.878 (0.849–0.903) | 0.408 (0.341–0.478) | 0.900 (0.856–0.940) | 0.640 (0.624–0.656) |
| Male | 7,030 | 314 (4.5) | 0.875 (0.854–0.895) | 0.356 (0.303–0.411) | 0.898 (0.863–0.930) | 0.650 (0.639–0.662) |
| **Age group** — EOD 0.095 · PE 0.220 · EOG 0.220 · SP 0.220 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| 0-17 | 367 | 21 (5.7) | 0.807 (not estimable) | 0.246 (not estimable) | 0.857 (not estimable) | 0.462 (not estimable) |
| 18-44 | 941 | 34 (3.6) | 0.854 (0.792–0.908) | 0.329 (0.175–0.505) | 0.824 (0.679–0.939) | 0.682 (0.653–0.712) |
| 45-64 | 3,715 | 147 (4.0) | 0.888 (0.856–0.916) | 0.382 (0.303–0.468) | 0.918 (0.872–0.959) | 0.666 (0.650–0.682) |
| 65-79 | 4,684 | 249 (5.3) | 0.889 (0.865–0.910) | 0.438 (0.371–0.509) | 0.912 (0.875–0.944) | 0.654 (0.639–0.668) |
| 80+ | 1,012 | 53 (5.2) | 0.805 (0.744–0.863) | 0.227 (0.145–0.344) | 0.849 (0.746–0.940) | 0.577 (0.545–0.607) |
| **Physiological reserve (frailty)** — EOD 0.142 · PE 0.232 · EOG 0.232 · SP 0.262 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| Frail | 6,139 | 400 (6.5) | 0.878 (0.860–0.896) | 0.428 (0.379–0.481) | 0.927 (0.901–0.951) | 0.584 (0.571–0.597) |
| Non-frail | 1,022 | 6 (0.6) | 0.868 (not estimable) | 0.084 (not estimable) | 0.833 (not estimable) | 0.816 (not estimable) |
| Pre-frail | 3,558 | 98 (2.8) | 0.821 (0.774–0.865) | 0.190 (0.127–0.274) | 0.786 (0.701–0.863) | 0.702 (0.687–0.716) |
