**Table 1. Cohort characterisation by sex (audited Bern cohort, *n* = 10,719; 504 infection events).** Counts (column percentages) for categorical variables and median [Q1, Q3] for continuous variables, stratified by administratively recorded sex. p-values are from the χ² test for categorical variables and the Mann–Whitney U test for continuous variables. The FDR column gives Benjamini–Hochberg-adjusted q-values across the eight tested variables, controlling the false discovery rate at q < 0.05. Frailty is the three-level composite of ASA Physical Status, the Hospital Frailty Risk Score, and the four-item modified Frailty Index defined in Methods. Primary analysis: the published model's out-of-fold predictions on the audited 2014-2022 cohort, which is the cohort the deployed model was fitted on. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (98 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Outcome ascertainment is not constant across this span. The present-on-admission coding the endpoint requires is effectively absent before 2018, so infection cases whose postoperative timing could not be established left the cohort at construction rather than being recorded as negatives, and 2014-2017 contributes 4,668 procedures against 3 coded events. Prevalence is therefore 4.70% here against 8.28% in 2018-2022, and prevalence-dependent quantities -- calibration-in-the-large, PPV and net benefit -- differ accordingly; the temporal-validation analyses report both.

| Variable | Level | Overall | Female | Male | p-value | FDR |
|---|---|---:|---:|---:|---:|---:|
| n |  | 10719 | 3689 | 7030 |  |  |
| Age, median [Q1,Q3] |  | 66.0 [55.0,74.0] | 66.0 [53.0,75.0] | 66.0 [56.0,73.0] | 0.467 | 0.467 |
| Age group, n (%) | 0-17 | 367 (3.4) | 175 (4.7) | 192 (2.7) | <0.001 | 0.0016 |
|  | 18-44 | 941 (8.8) | 408 (11.1) | 533 (7.6) |  |  |
|  | 45-64 | 3715 (34.7) | 1179 (32.0) | 2536 (36.1) |  |  |
|  | 65-79 | 4684 (43.7) | 1470 (39.8) | 3214 (45.7) |  |  |
|  | 80+ | 1012 (9.4) | 457 (12.4) | 555 (7.9) |  |  |
| ASA score, n (%) | 1 | 1227 (11.4) | 630 (17.1) | 597 (8.5) | <0.001 | 0.0016 |
|  | 2 | 213 (2.0) | 78 (2.1) | 135 (1.9) |  |  |
|  | 3 | 3803 (35.5) | 1549 (42.0) | 2254 (32.1) |  |  |
|  | 4 | 5476 (51.1) | 1432 (38.8) | 4044 (57.5) |  |  |
| Frailty, n (%) | Frail | 6139 (57.3) | 1666 (45.2) | 4473 (63.6) | <0.001 | 0.0016 |
|  | Non-frail | 1022 (9.5) | 522 (14.2) | 500 (7.1) |  |  |
|  | Pre-frail | 3558 (33.2) | 1501 (40.7) | 2057 (29.3) |  |  |
| Emergency surgery, n (%) | No | 9250 (86.3) | 3221 (87.3) | 6029 (85.8) | 0.028 | 0.0373 |
|  | Yes | 1469 (13.7) | 468 (12.7) | 1001 (14.2) |  |  |
| Surgery duration, median [Q1,Q3] |  | 213.0 [147.0,281.0] | 193.0 [135.0,267.0] | 221.0 [156.0,285.0] | <0.001 | 0.0016 |
| Procedure cluster, n (%) | NervousSystem (c1) | 903 (8.4) | 365 (9.9) | 538 (7.7) | <0.001 | 0.0016 |
|  | Thoracic (c3) | 807 (7.5) | 317 (8.6) | 490 (7.0) |  |  |
|  | Cardiac (c4) | 3764 (35.1) | 836 (22.7) | 2928 (41.7) |  |  |
|  | Vascular (c5) | 769 (7.2) | 189 (5.1) | 580 (8.3) |  |  |
|  | AbdominalWall (c6) | 530 (4.9) | 249 (6.7) | 281 (4.0) |  |  |
|  | GastroIntestinal (c7) | 298 (2.8) | 105 (2.8) | 193 (2.7) |  |  |
|  | Hepatobiliary (c8) | 306 (2.9) | 95 (2.6) | 211 (3.0) |  |  |
|  | UroAndrological (c10) | 346 (3.2) | 98 (2.7) | 248 (3.5) |  |  |
|  | Gynecological (c11) | 303 (2.8) | 303 (8.2) | 0 (0.0) |  |  |
|  | Musculoskeletal (c12) | 1623 (15.1) | 711 (19.3) | 912 (13.0) |  |  |
|  | Cutaneous (c13) | 213 (2.0) | 95 (2.6) | 118 (1.7) |  |  |
|  | Unclassified | 857 (8.0) | 326 (8.8) | 531 (7.6) |  |  |
| Coded infection, n (%) | No | 10215 (95.3) | 3499 (94.8) | 6716 (95.5) | 0.123 | 0.141 |
|  | Yes | 504 (4.7) | 190 (5.2) | 314 (4.5) |  |  |
