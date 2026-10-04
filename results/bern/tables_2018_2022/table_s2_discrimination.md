**Supplementary Table 5.** Per-subgroup discrimination on the audited cohort (*n* = 6,051; 501 infection events) for the case-mix reference axes (surgical specialty, procedure cluster) and the three pre-specified pairwise intersections. Within-subgroup AUROC, AUPRC, sensitivity, and specificity at the *t* = 0.10 operating threshold, with 95% bootstrap CIs (2,000 patient-level resamples). Gap statistics per axis are reported in the section header rows (EOD = equal-opportunity difference, PE = predictive-equality difference, EOG = equalized-odds gap, SP = statistical parity). A subgroup estimate is labelled not estimable where the stratum falls below the pre-registered event floor for that metric (see Methods); its point estimate is shown and its interval withheld. Subgroups flagged with † used empirical-Bayes shrinkage toward the marginal rate (cell *n* events < 50). Ascertainment-window analysis: the published model's out-of-fold predictions restricted to 2018-2022, the period in which the present-on-admission coding the endpoint requires is available. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (96 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Ascertainment within this window remains incomplete: 22.9% of cohort-eligible infected patients become labelled cases in 2018, stepping to a 31-35% plateau from 2019 onward, and it differs by physiological reserve (Supplementary Table 18 of the paper).

| Subgroup | *n* | Events (%) | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---:|---:|---|---|---|---|
| **Surgical specialty (reference)** — EOD 0.333 · PE 0.474 · EOG 0.474 · SP 0.484 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| ORL | 118 | 14 (11.9) | 0.737 (not estimable) | 0.250 (not estimable) | 1.000 (not estimable) | 0.279 (not estimable) |
| gynecology | 274 | 24 (8.8) | 0.805 (not estimable) | 0.425 (not estimable) | 0.750 (not estimable) | 0.668 (not estimable) |
| neuro | 488 | 25 (5.1) | 0.852 (0.752–0.930) | 0.512 (0.317–0.692) | 0.840 (0.682–0.963) | 0.678 (0.635–0.719) |
| ortho | 971 | 76 (7.8) | 0.822 (0.778–0.862) | 0.282 (0.206–0.387) | 0.908 (0.837–0.967) | 0.581 (0.548–0.613) |
| pediatric | 37 | 3 (8.1) | 0.652 (not estimable) | 0.227 (not estimable) | 0.667 (not estimable) | 0.412 (not estimable) |
| thoracic | 403 | 14 (3.5) | 0.936 (not estimable) | 0.479 (not estimable) | 0.929 (not estimable) | 0.753 (not estimable) |
| urology | 200 | 19 (9.5) | 0.762 (not estimable) | 0.374 (not estimable) | 0.737 (not estimable) | 0.564 (not estimable) |
| vascular | 2,838 | 183 (6.4) | 0.844 (0.809–0.878) | 0.371 (0.298–0.444) | 0.885 (0.837–0.932) | 0.591 (0.573–0.610) |
| visceral | 722 | 143 (19.8) | 0.862 (0.831–0.893) | 0.642 (0.562–0.719) | 0.965 (0.934–0.993) | 0.454 (0.416–0.496) |
| **Procedure cluster (reference)** — EOD 0.444 · PE 0.535 · EOG 0.535 · SP 0.561 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| chopcluster_1 (NervousSystem) | 395 | 21 (5.3) | 0.868 (not estimable) | 0.525 (not estimable) | 0.857 (not estimable) | 0.679 (not estimable) |
| chopcluster_10 (UroAndrological+Urologic) | 186 | 28 (15.1) | 0.775 (0.655–0.883) | 0.474 (0.308–0.692) | 0.786 (0.625–0.929) | 0.500 (0.422–0.574) |
| chopcluster_11 (Gynecological) | 178 | 9 (5.1) | 0.739 (not estimable) | 0.260 (not estimable) | 0.556 (not estimable) | 0.751 (not estimable) |
| chopcluster_12 (Musculoskeletal) | 826 | 61 (7.4) | 0.825 (0.773–0.872) | 0.259 (0.182–0.380) | 0.918 (0.842–0.981) | 0.603 (0.567–0.639) |
| chopcluster_13 (Cutaneous) | 154 | 43 (27.9) | 0.752 (0.664–0.833) | 0.584 (0.446–0.722) | 0.930 (0.846–1.000) | 0.216 (0.142–0.292) |
| chopcluster_3 (Thoracic) | 414 | 22 (5.3) | 0.932 (not estimable) | 0.547 (not estimable) | 0.955 (not estimable) | 0.730 (not estimable) |
| chopcluster_4 (Cardiac) | 2,245 | 118 (5.3) | 0.819 (0.767–0.865) | 0.299 (0.232–0.391) | 0.839 (0.770–0.903) | 0.613 (0.593–0.633) |
| chopcluster_5 (Vascular) | 328 | 30 (9.1) | 0.856 (0.781–0.920) | 0.465 (0.297–0.642) | 0.933 (0.833–1.000) | 0.527 (0.472–0.581) |
| chopcluster_6 (AbdominalWall) | 373 | 66 (17.7) | 0.877 (0.829–0.920) | 0.691 (0.583–0.784) | 0.955 (0.900–1.000) | 0.463 (0.407–0.518) |
| chopcluster_7 (GastroIntestinal) | 143 | 36 (25.2) | 0.826 (0.755–0.890) | 0.634 (0.479–0.777) | 1.000 (1.000–1.000) | 0.271 (0.192–0.357) |
| chopcluster_8 (Hepatobiliary) | 142 | 23 (16.2) | 0.785 (not estimable) | 0.439 (not estimable) | 0.870 (not estimable) | 0.513 (not estimable) |
| none | 667 | 44 (6.6) | 0.880 (0.836–0.919) | 0.395 (0.259–0.546) | 0.977 (0.921–1.000) | 0.559 (0.521–0.599) |
| **Sex × age group** — EOD 0.111 · PE 0.099 · EOG 0.111 · SP 0.116 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| female / 0-17† | 109 | 13 (11.9) | 0.893 (not estimable) | 0.491 (not estimable) | 0.911 (not estimable) | 0.525 (not estimable) |
| female / 18-44† | 220 | 13 (5.9) | 0.840 (not estimable) | 0.363 (not estimable) | 0.901 (not estimable) | 0.600 (not estimable) |
| female / 45-64 | 683 | 52 (7.6) | 0.838 (0.764–0.899) | 0.413 (0.292–0.553) | 0.846 (0.738–0.939) | 0.624 (0.586–0.660) |
| female / 65-79 | 839 | 92 (11.0) | 0.885 (0.848–0.920) | 0.601 (0.502–0.694) | 0.946 (0.892–0.989) | 0.592 (0.557–0.628) |
| female / 80+† | 239 | 19 (7.9) | 0.773 (not estimable) | 0.220 (not estimable) | 0.899 (not estimable) | 0.532 (not estimable) |
| male / 0-17† | 114 | 8 (7.0) | 0.647 (not estimable) | 0.175 (not estimable) | 0.886 (not estimable) | 0.543 (not estimable) |
| male / 18-44† | 298 | 21 (7.0) | 0.811 (not estimable) | 0.438 (not estimable) | 0.888 (not estimable) | 0.612 (not estimable) |
| male / 45-64 | 1,366 | 94 (6.9) | 0.883 (0.850–0.915) | 0.448 (0.351–0.551) | 0.957 (0.910–0.990) | 0.603 (0.575–0.629) |
| male / 65-79 | 1,859 | 155 (8.3) | 0.863 (0.827–0.895) | 0.464 (0.387–0.545) | 0.897 (0.848–0.942) | 0.601 (0.577–0.624) |
| male / 80+† | 324 | 34 (10.5) | 0.765 | 0.335 | 0.890 | 0.541 |
| **Sex × frailty** — EOD 0.107 · PE 0.220 · EOG 0.220 · SP 0.264 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| female / frail | 1,010 | 129 (12.8) | 0.869 (0.833–0.902) | 0.540 (0.457–0.624) | 0.938 (0.893–0.977) | 0.491 (0.460–0.526) |
| female / non_frail† | 275 | 1 (0.4) | 0.993 (not estimable) | 0.333 (not estimable) | 0.905 (not estimable) | 0.684 (not estimable) |
| female / pre_frail | 805 | 59 (7.3) | 0.791 (0.721–0.850) | 0.277 (0.183–0.405) | 0.831 (0.729–0.921) | 0.630 (0.595–0.662) |
| male / frail | 2,674 | 270 (10.1) | 0.854 (0.830–0.878) | 0.472 (0.410–0.537) | 0.922 (0.890–0.953) | 0.549 (0.529–0.567) |
| male / non_frail† | 272 | 5 (1.8) | 0.825 (not estimable) | 0.120 (not estimable) | 0.895 (not estimable) | 0.712 (not estimable) |
| male / pre_frail† | 1,015 | 37 (3.6) | 0.785 | 0.218 | 0.872 | 0.633 |
| **Age group × frailty** — EOD 0.154 · PE 0.351 · EOG 0.351 · SP 0.377 | | | | | | |
| Overall | 6,051 | 501 (8.3) | 0.855 (0.836–0.873) | 0.440 (0.394–0.486) | 0.900 (0.873–0.926) | 0.590 (0.576–0.602) |
| 0-17 / frail† | 128 | 19 (14.8) | 0.779 (not estimable) | 0.398 (not estimable) | 0.862 (not estimable) | 0.379 (not estimable) |
| 0-17 / non_frail† | 28 | 0 (0.0) | — | — | — | 0.714 (not estimable) |
| 0-17 / pre_frail† | 67 | 2 (3.0) | 0.538 (not estimable) | 0.061 (not estimable) | 0.852 (not estimable) | 0.478 (not estimable) |
| 18-44 / frail† | 183 | 21 (11.5) | 0.836 (not estimable) | 0.549 (not estimable) | 0.835 (not estimable) | 0.517 (not estimable) |
| 18-44 / non_frail† | 182 | 2 (1.1) | 0.819 (not estimable) | 0.042 (not estimable) | 0.826 (not estimable) | 0.717 (not estimable) |
| 18-44 / pre_frail† | 153 | 11 (7.2) | 0.706 (not estimable) | 0.186 (not estimable) | 0.809 (not estimable) | 0.620 (not estimable) |
| 45-64 / frail | 1,167 | 110 (9.4) | 0.880 (0.848–0.912) | 0.489 (0.399–0.588) | 0.964 (0.927–0.993) | 0.556 (0.526–0.586) |
| 45-64 / non_frail† | 210 | 3 (1.4) | 0.960 (not estimable) | 0.373 (not estimable) | 0.920 (not estimable) | 0.730 (not estimable) |
| 45-64 / pre_frail† | 672 | 33 (4.9) | 0.775 | 0.222 | 0.886 | 0.631 |
| 65-79 / frail | 1,846 | 204 (11.1) | 0.874 (0.844–0.903) | 0.566 (0.495–0.638) | 0.922 (0.883–0.958) | 0.563 (0.540–0.587) |
| 65-79 / non_frail† | 118 | 1 (0.8) | 0.517 (not estimable) | 0.017 (not estimable) | 0.908 (not estimable) | 0.678 (not estimable) |
| 65-79 / pre_frail† | 734 | 42 (5.7) | 0.847 | 0.308 | 0.913 | 0.644 |
| 80+ / frail† | 360 | 45 (12.5) | 0.751 | 0.318 | 0.859 | 0.455 |
| 80+ / non_frail† | 9 | 0 (0.0) | — | — | — | 0.444 (not estimable) |
| 80+ / pre_frail† | 194 | 8 (4.1) | 0.714 (not estimable) | 0.244 (not estimable) | 0.836 (not estimable) | 0.570 (not estimable) |
