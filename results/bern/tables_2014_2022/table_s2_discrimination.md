**Supplementary Table 5.** Per-subgroup discrimination on the audited cohort (*n* = 10,719; 504 infection events) for the case-mix reference axes (surgical specialty, procedure cluster) and the three pre-specified pairwise intersections. Within-subgroup AUROC, AUPRC, sensitivity, and specificity at the *t* = 0.10 operating threshold, with 95% bootstrap CIs (2,000 patient-level resamples). Gap statistics per axis are reported in the section header rows (EOD = equal-opportunity difference, PE = predictive-equality difference, EOG = equalized-odds gap, SP = statistical parity). A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Primary analysis: the published model's out-of-fold predictions on the audited 2014-2022 cohort, which is the cohort the deployed model was fitted on. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (98 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Outcome ascertainment is not constant across this span. The present-on-admission coding the endpoint requires is effectively absent before 2018, so infection cases whose postoperative timing could not be established left the cohort at construction rather than being recorded as negatives, and 2014-2017 contributes 4,668 procedures against 3 coded events. Prevalence is therefore 4.70% here against 8.28% in 2018-2022, and prevalence-dependent quantities -- calibration-in-the-large, PPV and net benefit -- differ accordingly; the temporal-validation analyses report both.

| Subgroup | *n* | Events (%) | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---:|---:|---|---|---|---|
| **Surgical specialty (reference)** — EOD 0.333 · PE 0.373 · EOG 0.373 · SP 0.386 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| ORL | 233 | 15 (6.4) | 0.820 (not estimable) | 0.221 (not estimable) | 1.000 (not estimable) | 0.408 (not estimable) |
| gynecology | 452 | 24 (5.3) | 0.821 (not estimable) | 0.395 (not estimable) | 0.750 (not estimable) | 0.689 (not estimable) |
| neuro | 1,117 | 26 (2.3) | 0.871 (0.785–0.939) | 0.434 (0.240–0.621) | 0.808 (0.643–0.955) | 0.756 (0.731–0.780) |
| ortho | 1,690 | 76 (4.5) | 0.852 (0.808–0.887) | 0.232 (0.160–0.332) | 0.908 (0.835–0.965) | 0.633 (0.610–0.657) |
| pediatric | 65 | 3 (4.6) | 0.653 (not estimable) | 0.181 (not estimable) | 0.667 (not estimable) | 0.452 (not estimable) |
| thoracic | 723 | 14 (1.9) | 0.945 (not estimable) | 0.411 (not estimable) | 0.929 (not estimable) | 0.781 (not estimable) |
| urology | 349 | 19 (5.4) | 0.785 (not estimable) | 0.331 (not estimable) | 0.737 (not estimable) | 0.624 (not estimable) |
| vascular | 4,852 | 184 (3.8) | 0.866 (0.834–0.896) | 0.323 (0.257–0.395) | 0.886 (0.837–0.929) | 0.656 (0.643–0.670) |
| visceral | 1,238 | 143 (11.6) | 0.883 (0.854–0.909) | 0.557 (0.472–0.640) | 0.965 (0.932–0.993) | 0.482 (0.453–0.510) |
| **Procedure cluster (reference)** — EOD 0.444 · PE 0.516 · EOG 0.516 · SP 0.543 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| chopcluster_1 (NervousSystem) | 903 | 22 (2.4) | 0.879 (not estimable) | 0.460 (not estimable) | 0.818 (not estimable) | 0.751 (not estimable) |
| chopcluster_10 (UroAndrological+Urologic) | 346 | 28 (8.1) | 0.789 (0.673–0.889) | 0.383 (0.233–0.594) | 0.786 (0.625–0.933) | 0.560 (0.505–0.614) |
| chopcluster_11 (Gynecological) | 303 | 9 (3.0) | 0.761 (not estimable) | 0.239 (not estimable) | 0.556 (not estimable) | 0.769 (not estimable) |
| chopcluster_12 (Musculoskeletal) | 1,623 | 62 (3.8) | 0.856 (0.810–0.894) | 0.192 (0.135–0.294) | 0.919 (0.844–0.981) | 0.654 (0.631–0.676) |
| chopcluster_13 (Cutaneous) | 213 | 43 (20.2) | 0.796 (0.716–0.871) | 0.561 (0.419–0.700) | 0.930 (0.846–1.000) | 0.253 (0.192–0.318) |
| chopcluster_3 (Thoracic) | 807 | 22 (2.7) | 0.941 (not estimable) | 0.482 (not estimable) | 0.955 (not estimable) | 0.753 (not estimable) |
| chopcluster_4 (Cardiac) | 3,764 | 118 (3.1) | 0.841 (0.793–0.883) | 0.251 (0.188–0.338) | 0.839 (0.767–0.902) | 0.673 (0.658–0.687) |
| chopcluster_5 (Vascular) | 769 | 30 (3.9) | 0.893 (0.827–0.943) | 0.392 (0.225–0.565) | 0.933 (0.823–1.000) | 0.636 (0.602–0.672) |
| chopcluster_6 (AbdominalWall) | 530 | 67 (12.6) | 0.889 (0.844–0.927) | 0.625 (0.516–0.728) | 0.955 (0.901–1.000) | 0.471 (0.425–0.514) |
| chopcluster_7 (GastroIntestinal) | 298 | 36 (12.1) | 0.864 (0.813–0.913) | 0.521 (0.365–0.681) | 1.000 (1.000–1.000) | 0.309 (0.256–0.366) |
| chopcluster_8 (Hepatobiliary) | 306 | 23 (7.5) | 0.802 (not estimable) | 0.327 (not estimable) | 0.870 (not estimable) | 0.551 (not estimable) |
| none | 857 | 44 (5.1) | 0.900 (0.866–0.932) | 0.388 (0.256–0.531) | 0.977 (0.927–1.000) | 0.625 (0.590–0.658) |
| **Sex × age group** — EOD 0.112 · PE 0.130 · EOG 0.130 · SP 0.142 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| female / 0-17† | 175 | 13 (7.4) | 0.902 (not estimable) | 0.408 (not estimable) | 0.907 (not estimable) | 0.549 (not estimable) |
| female / 18-44† | 408 | 13 (3.2) | 0.873 (not estimable) | 0.340 (not estimable) | 0.896 (not estimable) | 0.663 (not estimable) |
| female / 45-64 | 1,179 | 52 (4.4) | 0.860 (0.788–0.919) | 0.377 (0.248–0.517) | 0.846 (0.738–0.941) | 0.674 (0.647–0.702) |
| female / 65-79 | 1,470 | 93 (6.3) | 0.900 (0.865–0.931) | 0.545 (0.451–0.640) | 0.935 (0.882–0.980) | 0.645 (0.620–0.670) |
| female / 80+† | 457 | 19 (4.2) | 0.814 (not estimable) | 0.166 (not estimable) | 0.894 (not estimable) | 0.596 (not estimable) |
| male / 0-17† | 192 | 8 (4.2) | 0.648 (not estimable) | 0.119 (not estimable) | 0.886 (not estimable) | 0.560 (not estimable) |
| male / 18-44† | 533 | 21 (3.9) | 0.844 (not estimable) | 0.329 (not estimable) | 0.889 (not estimable) | 0.679 (not estimable) |
| male / 45-64 | 2,536 | 95 (3.7) | 0.903 (0.873–0.929) | 0.392 (0.301–0.494) | 0.958 (0.915–0.991) | 0.662 (0.644–0.681) |
| male / 65-79 | 3,214 | 156 (4.9) | 0.882 (0.848–0.911) | 0.378 (0.307–0.463) | 0.897 (0.843–0.942) | 0.658 (0.641–0.675) |
| male / 80+† | 555 | 34 (6.1) | 0.798 | 0.269 | 0.891 | 0.595 |
| **Sex × frailty** — EOD 0.121 · PE 0.255 · EOG 0.255 · SP 0.284 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| female / frail | 1,666 | 129 (7.7) | 0.888 (0.855–0.918) | 0.492 (0.408–0.577) | 0.938 (0.892–0.977) | 0.530 (0.505–0.556) |
| female / non_frail† | 522 | 1 (0.2) | 0.996 (not estimable) | 0.333 (not estimable) | 0.900 (not estimable) | 0.748 (not estimable) |
| female / pre_frail | 1,501 | 60 (4.0) | 0.823 (0.761–0.877) | 0.226 (0.148–0.345) | 0.817 (0.710–0.907) | 0.703 (0.679–0.726) |
| male / frail | 4,473 | 271 (6.1) | 0.874 (0.850–0.896) | 0.395 (0.335–0.459) | 0.923 (0.891–0.954) | 0.604 (0.589–0.619) |
| male / non_frail† | 500 | 5 (1.0) | 0.857 (not estimable) | 0.071 (not estimable) | 0.896 (not estimable) | 0.785 (not estimable) |
| male / pre_frail† | 2,057 | 38 (1.8) | 0.816 | 0.169 | 0.873 | 0.696 |
| **Age group × frailty** — EOD 0.155 · PE 0.397 · EOG 0.397 · SP 0.416 | | | | | | |
| Overall | 10,719 | 504 (4.7) | 0.876 (0.859–0.892) | 0.376 (0.332–0.422) | 0.899 (0.871–0.925) | 0.647 (0.638–0.656) |
| 0-17 / frail† | 211 | 19 (9.0) | 0.791 (not estimable) | 0.282 (not estimable) | 0.862 (not estimable) | 0.393 (not estimable) |
| 0-17 / non_frail† | 45 | 0 (0.0) | — | — | — | 0.711 (not estimable) |
| 0-17 / pre_frail† | 111 | 2 (1.8) | 0.539 (not estimable) | 0.042 (not estimable) | 0.852 (not estimable) | 0.510 (not estimable) |
| 18-44 / frail† | 300 | 21 (7.0) | 0.864 (not estimable) | 0.463 (not estimable) | 0.835 (not estimable) | 0.565 (not estimable) |
| 18-44 / non_frail† | 331 | 2 (0.6) | 0.869 (not estimable) | 0.031 (not estimable) | 0.826 (not estimable) | 0.789 (not estimable) |
| 18-44 / pre_frail† | 310 | 11 (3.5) | 0.744 (not estimable) | 0.141 (not estimable) | 0.809 (not estimable) | 0.680 (not estimable) |
| 45-64 / frail | 1,967 | 111 (5.6) | 0.898 (0.869–0.926) | 0.444 (0.354–0.540) | 0.964 (0.927–0.992) | 0.603 (0.581–0.625) |
| 45-64 / non_frail† | 410 | 3 (0.7) | 0.971 (not estimable) | 0.307 (not estimable) | 0.920 (not estimable) | 0.790 (not estimable) |
| 45-64 / pre_frail† | 1,338 | 33 (2.5) | 0.808 | 0.175 | 0.887 | 0.701 |
| 65-79 / frail | 3,032 | 204 (6.7) | 0.889 (0.862–0.914) | 0.487 (0.416–0.560) | 0.922 (0.884–0.958) | 0.610 (0.592–0.627) |
| 65-79 / non_frail† | 218 | 1 (0.5) | 0.592 (not estimable) | 0.010 (not estimable) | 0.905 (not estimable) | 0.742 (not estimable) |
| 65-79 / pre_frail† | 1,434 | 44 (3.1) | 0.870 | 0.247 | 0.905 | 0.714 |
| 80+ / frail† | 629 | 45 (7.2) | 0.796 | 0.253 | 0.859 | 0.523 |
| 80+ / non_frail† | 18 | 0 (0.0) | — | — | — | 0.667 (not estimable) |
| 80+ / pre_frail† | 365 | 8 (2.2) | 0.740 (not estimable) | 0.209 (not estimable) | 0.836 (not estimable) | 0.652 (not estimable) |
