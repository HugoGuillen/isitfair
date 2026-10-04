**Supplementary Table 7. Two-way intersectional cell sparsity and gap statistics (audited Bern cohort, *n* = 6,051; 501 infection events).** For each of the three pre-specified two-way intersections, every cell's size, outcome count, and empirical-Bayes (EB) shrinkage status is reported. Cells with fewer than 50 outcome events (marked †) were estimated with EB shrinkage of the cell-level coded-infection rate toward the marginal subgroup rate; the shrinkage factor (0 = no shrinkage, 1 = full shrinkage to the margin) is shown for those cells. The per-intersection header row gives the number of cells, the number shrunk, the equalized-odds gap (discrimination), and the maximum net-benefit gap with 95% bootstrap CI (clinical utility). Ascertainment-window analysis: the published model's out-of-fold predictions restricted to 2018-2022, the period in which the present-on-admission coding the endpoint requires is available. Rows that do not meet the pre-registered event floors are shown for completeness but are not estimable and must not be interpreted: the non-frail stratum on every metric (6 events) and the pre-frail calibration slope (96 events against a floor of 100); the verdict for every cell is in estimability_enforced.csv beside this table. Ascertainment within this window remains incomplete: 22.9% of cohort-eligible infected patients become labelled cases in 2018, stepping to a 31-35% plateau from 2019 onward, and it differs by physiological reserve (Supplementary Table 18 of the paper).

| Cell | *n* | Events | Cell infection rate (%) | Marginal rate (%) | EB shrinkage factor |
|---|---:|---:|---:|---:|---:|
| **Sex × age group** — 10 cells, 6 EB-shrunk · equalized-odds gap 0.11 · max net-benefit gap 0.079 (0.069, 0.165) | | | | | |
| female / 0-17† | 109 | 13 | 11.9 | 9.0 | 0.62 |
| female / 18-44† | 220 | 13 | 5.9 | 9.0 | 0.44 |
| female / 45-64 | 683 | 52 | 7.6 | 9.0 | — |
| female / 65-79 | 839 | 92 | 11.0 | 9.0 | — |
| female / 80+† | 239 | 19 | 7.9 | 9.0 | 0.42 |
| male / 0-17† | 114 | 8 | 7.0 | 7.9 | 0.61 |
| male / 18-44† | 298 | 21 | 7.0 | 7.9 | 0.37 |
| male / 45-64 | 1,366 | 94 | 6.9 | 7.9 | — |
| male / 65-79 | 1,859 | 155 | 8.3 | 7.9 | — |
| male / 80+† | 324 | 34 | 10.5 | 7.9 | 0.35 |
| **Sex × physiological reserve** — 6 cells, 3 EB-shrunk · equalized-odds gap 0.22 · max net-benefit gap 0.123 (0.102, 0.146) | | | | | |
| female / frail | 1,010 | 129 | 12.8 | 9.0 | — |
| female / non_frail† | 275 | 1 | 0.4 | 9.0 | 0.43 |
| female / pre_frail | 805 | 59 | 7.3 | 9.0 | — |
| male / frail | 2,674 | 270 | 10.1 | 7.9 | — |
| male / non_frail† | 272 | 5 | 1.8 | 7.9 | 0.43 |
| male / pre_frail† | 1,015 | 37 | 3.6 | 7.9 | 0.17 |
| **Age group × physiological reserve** — 15 cells, 13 EB-shrunk · equalized-odds gap 0.35 · max net-benefit gap 0.151 (0.128, 0.246) | | | | | |
| 0-17 / frail† | 128 | 19 | 14.8 | 9.4 | 0.51 |
| 0-17 / non_frail† | 28 | 0 | 0.0 | 9.4 | 0.83 |
| 0-17 / pre_frail† | 67 | 2 | 3.0 | 9.4 | 0.67 |
| 18-44 / frail† | 183 | 21 | 11.5 | 6.6 | 0.42 |
| 18-44 / non_frail† | 182 | 2 | 1.1 | 6.6 | 0.42 |
| 18-44 / pre_frail† | 153 | 11 | 7.2 | 6.6 | 0.47 |
| 45-64 / frail | 1,167 | 110 | 9.4 | 7.1 | — |
| 45-64 / non_frail† | 210 | 3 | 1.4 | 7.1 | 0.39 |
| 45-64 / pre_frail† | 672 | 33 | 4.9 | 7.1 | 0.17 |
| 65-79 / frail | 1,846 | 204 | 11.1 | 9.2 | — |
| 65-79 / non_frail† | 118 | 1 | 0.8 | 9.2 | 0.53 |
| 65-79 / pre_frail† | 734 | 42 | 5.7 | 9.2 | 0.15 |
| 80+ / frail† | 360 | 45 | 12.5 | 9.4 | 0.27 |
| 80+ / non_frail† | 9 | 0 | 0.0 | 9.4 | 0.94 |
| 80+ / pre_frail† | 194 | 8 | 4.1 | 9.4 | 0.41 |
