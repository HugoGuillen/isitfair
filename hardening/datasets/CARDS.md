# Dataset cards

Built by `hardening/datasets/registry.py`; numbers below are read from
`predictions/<name>.json`. Audited model for every real dataset except
support2: `Pipeline(ColumnTransformer(median-impute + scale | mode-impute +
one-hot(min_frequency=50)), LogisticRegression(max_iter=2000))` and, separately,
`HistGradientBoostingClassifier(random_state=42)`; stratified 70/30 split with
`random_state=42`; predictions on the 30% test half only. Sex and race are held out
of the features (as in `examples/support2_data.py`); age stays in.

## support2

- **Source**: https://archive.ics.uci.edu/dataset/880 (bundled copy examples/data/support2.csv.gz)
- **License**: CC BY 4.0
- **Downloaded**: n/a (in repository / generated)
- **Outcome**: hospdead == 1 (in-hospital death)
- **Attributes and recoding**: sex as recorded; age_group = age binned [18,45) [45,65) [65,80) [80,inf) (examples/support2_data.py AGE_BINS); race_ethnicity = race with missing filled 'other' (0.5% of rows)
- **Features**: examples/support2_data.py reference model (LR, seed 99, 60/20/20); the 20% calibration partition is written to support2_calib.csv
- **Sizes**: total 9,105; train 5,463; audited test 1,821; committed reference split, seed 99, 60/20/20 (train/calibration/test); the calibration partition is predictions/support2_calib.csv
- **Test prevalence**: 0.2592 (472 events); audit threshold 0.3
- **Stress condition**: control; must match results/MANIFEST.csv

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| sex | female | 787 | 199 | 0.2529 |
| sex | male | 1,034 | 273 | 0.2640 |
| age_group | 18-44 | 257 | 65 | 0.2529 |
| age_group | 45-64 | 684 | 171 | 0.2500 |
| age_group | 65-79 | 677 | 172 | 0.2541 |
| age_group | 80+ | 203 | 64 | 0.3153 |
| race_ethnicity | asian | 13 | 5 | 0.3846 |
| race_ethnicity | black | 266 | 82 | 0.3083 |
| race_ethnicity | hispanic | 57 | 10 | 0.1754 |
| race_ethnicity | other | 32 | 9 | 0.2812 |
| race_ethnicity | white | 1,453 | 366 | 0.2519 |

## diabetes130

- **Source**: https://archive.ics.uci.edu/dataset/296
- **License**: CC BY 4.0
- **Downloaded**: 2026-10-20T20:10:13+02:00
- **Outcome**: readmitted == '<30' (readmission within 30 days)
- **Attributes and recoding**: race as written by UCI (the current static CSV writes the original '?' as the literal text 'NaN', which pandas reads as missing; kept missing so the audit's validation is exercised); gender as recorded (includes 'Unknown/Invalid'); age as UCI's 10-year bins '[0-10)'...'[90-100)'. Rows are encounters, not patients (patient_nbr repeats); the split is by row as specified, so a patient can sit in both halves
- **Features**: all columns except race, gender, ids, and >39%-missing weight, payer_code, medical_specialty; age kept as a categorical feature; diag_1..3 one-hot with min_frequency=50
- **Sizes**: total 101,766; train 71,236; audited test 30,530
- **Test prevalence**: 0.1116 (3,407 events); audit threshold 0.15
- **Stress condition**: large n, LOESS scaling, missing label, likely reproduces the hang

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| race | <missing> | 676 | 63 | 0.0932 |
| race | AfricanAmerican | 5,770 | 643 | 0.1114 |
| race | Asian | 174 | 16 | 0.0920 |
| race | Caucasian | 22,893 | 2,578 | 0.1126 |
| race | Hispanic | 594 | 71 | 0.1195 |
| race | Other | 423 | 36 | 0.0851 |
| gender | Female | 16,427 | 1,875 | 0.1141 |
| gender | Male | 14,102 | 1,532 | 0.1086 |
| gender | Unknown/Invalid | 1 | 0 | 0.0000 |
| age | [0-10) | 44 | 0 | 0.0000 |
| age | [10-20) | 199 | 9 | 0.0452 |
| age | [20-30) | 502 | 63 | 0.1255 |
| age | [30-40) | 1,125 | 115 | 0.1022 |
| age | [40-50) | 2,850 | 296 | 0.1039 |
| age | [50-60) | 5,259 | 478 | 0.0909 |
| age | [60-70) | 6,714 | 773 | 0.1151 |
| age | [70-80) | 7,849 | 930 | 0.1185 |
| age | [80-90) | 5,145 | 644 | 0.1252 |
| age | [90-100) | 843 | 99 | 0.1174 |

## adult

- **Source**: https://archive.ics.uci.edu/dataset/2
- **License**: CC BY 4.0
- **Downloaded**: 2026-10-20T20:10:51+02:00
- **Outcome**: income in {'>50K', '>50K.'} (the test file's labels carry a trailing period)
- **Attributes and recoding**: sex, race (5 levels) as recorded; age_group = [17,30) [30,45) [45,60) [60,inf); income labels '>50K' and '>50K.' (the original test file's trailing period) both count as the event
- **Features**: all columns except income, sex, race; age kept numeric
- **Sizes**: total 48,842; train 34,189; audited test 14,653
- **Test prevalence**: 0.2393 (3,506 events); audit threshold 0.3
- **Stress condition**: large n, multi-level attribute

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| sex | Female | 4,917 | 544 | 0.1106 |
| sex | Male | 9,736 | 2,962 | 0.3042 |
| race | Amer-Indian-Eskimo | 146 | 14 | 0.0959 |
| race | Asian-Pac-Islander | 445 | 112 | 0.2517 |
| race | Black | 1,433 | 177 | 0.1235 |
| race | Other | 105 | 11 | 0.1048 |
| race | White | 12,524 | 3,192 | 0.2549 |
| age_group | 17-29 | 4,376 | 218 | 0.0498 |
| age_group | 30-44 | 5,583 | 1,647 | 0.2950 |
| age_group | 45-59 | 3,492 | 1,335 | 0.3823 |
| age_group | 60+ | 1,202 | 306 | 0.2546 |

## compas

- **Source**: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
- **License**: no license file in propublica/compas-analysis (GitHub API reports license: null); used for research under ProPublica's publication of the data
- **Downloaded**: 2026-10-20T20:10:56+02:00
- **Outcome**: two_year_recid == 1
- **Attributes and recoding**: race (6 levels), sex, age_cat as recorded; rows filtered as in ProPublica's analysis (|days_b_screening_arrest| <= 30, is_recid != -1, c_charge_degree != 'O', score_text != 'N/A'), 7,214 -> 6,172
- **Features**: age, priors_count, juv_fel/misd/other_count, c_charge_degree (the COMPAS decile score itself is not a feature); rows filtered with ProPublica's days_b_screening_arrest/is_recid/charge/score filter
- **Sizes**: total 6,172; train 4,320; audited test 1,852
- **Test prevalence**: 0.4552 (843 events); audit threshold 0.5
- **Stress condition**: canonical fairness set, moderate n

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| race | African-American | 956 | 511 | 0.5345 |
| race | Asian | 11 | 3 | 0.2727 |
| race | Caucasian | 645 | 240 | 0.3721 |
| race | Hispanic | 141 | 60 | 0.4255 |
| race | Native American | 2 | 0 | 0.0000 |
| race | Other | 97 | 29 | 0.2990 |
| sex | Female | 360 | 127 | 0.3528 |
| sex | Male | 1,492 | 716 | 0.4799 |
| age_cat | 25 - 45 | 1,027 | 481 | 0.4684 |
| age_cat | Greater than 45 | 403 | 124 | 0.3077 |
| age_cat | Less than 25 | 422 | 238 | 0.5640 |

## bank

- **Source**: https://archive.ics.uci.edu/dataset/222
- **License**: CC BY 4.0
- **Downloaded**: 2026-10-20T20:11:07+02:00
- **Outcome**: y == 'yes' (term deposit subscribed)
- **Attributes and recoding**: age_group = [18,30) [30,40) [40,50) [50,60) [60,inf); job (12 levels; UCI's 'unknown' arrives as missing and is recoded to 'unknown'); marital as recorded
- **Features**: all columns except y and duration (duration is only known after the call, a documented leak); job and marital stay in as features
- **Sizes**: total 45,211; train 31,647; audited test 13,564
- **Test prevalence**: 0.1170 (1,587 events); audit threshold 0.2
- **Stress condition**: 11.7% prevalence, 12-level attribute, intersections with empty cells

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| age_group | 18-29 | 1,503 | 255 | 0.1697 |
| age_group | 30-39 | 5,463 | 577 | 0.1056 |
| age_group | 40-49 | 3,457 | 302 | 0.0874 |
| age_group | 50-59 | 2,615 | 268 | 0.1025 |
| age_group | 60+ | 526 | 185 | 0.3517 |
| job | admin. | 1,509 | 179 | 0.1186 |
| job | blue-collar | 2,866 | 217 | 0.0757 |
| job | entrepreneur | 443 | 31 | 0.0700 |
| job | housemaid | 347 | 36 | 0.1037 |
| job | management | 2,928 | 381 | 0.1301 |
| job | retired | 674 | 165 | 0.2448 |
| job | self-employed | 503 | 69 | 0.1372 |
| job | services | 1,275 | 106 | 0.0831 |
| job | student | 261 | 65 | 0.2490 |
| job | technician | 2,291 | 268 | 0.1170 |
| job | unemployed | 385 | 63 | 0.1636 |
| job | unknown | 82 | 7 | 0.0854 |
| marital | divorced | 1,575 | 175 | 0.1111 |
| marital | married | 8,165 | 858 | 0.1051 |
| marital | single | 3,824 | 554 | 0.1449 |

## german

- **Source**: https://archive.ics.uci.edu/dataset/144
- **License**: CC BY 4.0
- **Downloaded**: 2026-10-20T20:11:15+02:00
- **Outcome**: class == 2 (bad credit risk)
- **Attributes and recoding**: sex from Attribute9 (A91/A93/A94 male; A92/A95 female); age_group from Attribute13 = [19,25) [25,35) [35,50) [50,inf); outcome class 2 = bad
- **Features**: all columns except class and Attribute9 (personal status and sex)
- **Sizes**: total 1,000; train 700; audited test 300
- **Test prevalence**: 0.3000 (90 events); audit threshold 0.3
- **Stress condition**: every event floor fails, `not estimable` paths

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| sex | female | 101 | 35 | 0.3465 |
| sex | male | 199 | 55 | 0.2764 |
| age_group | 19-24 | 59 | 28 | 0.4746 |
| age_group | 25-34 | 104 | 32 | 0.3077 |
| age_group | 35-49 | 95 | 21 | 0.2211 |
| age_group | 50+ | 42 | 9 | 0.2143 |

## heart_cleveland

- **Source**: https://archive.ics.uci.edu/dataset/45
- **License**: CC BY 4.0
- **Downloaded**: 2026-10-20T20:11:21+02:00
- **Outcome**: num > 0 (any angiographic disease)
- **Attributes and recoding**: sex 1 male / 0 female; age_group = [0,45) [45,55) [55,65) [65,inf)
- **Features**: all columns except num and sex; ca/thal missing values median/mode-imputed
- **Sizes**: total 303; train 212; audited test 91
- **Test prevalence**: 0.4615 (42 events); audit threshold 0.5
- **Stress condition**: tiny n, degenerate bootstrap replicates

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| sex | female | 31 | 11 | 0.3548 |
| sex | male | 60 | 31 | 0.5167 |
| age_group | 45-54 | 21 | 6 | 0.2857 |
| age_group | 55-64 | 41 | 25 | 0.6098 |
| age_group | 65+ | 10 | 5 | 0.5000 |
| age_group | <45 | 19 | 6 | 0.3158 |

## oracle_calib

- **Source**: synthetic (registry.make_oracle_calib, seed 42)
- **License**: n/a
- **Downloaded**: n/a (in repository / generated)
- **Outcome**: synthetic, y ~ Bernoulli(expit(a_g + b_g logit p))
- **Attributes and recoding**: group A/B with equal probability; logit(p) ~ N(-1, 1.2^2); y ~ Bernoulli(expit(a_g + b_g logit p)); both prediction columns are p
- **Features**: no model; p is the prediction
- **Sizes**: total 20,000; train 0; audited test 20,000
- **Test prevalence**: 0.2609 (5,218 events); audit threshold 0.3
- **Stress condition**: planted subgroup calibration (a, b) = (0, 1) in A and (-0.8, 0.7) in B

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| group | A | 10,072 | 3,091 | 0.3069 |
| group | B | 9,928 | 2,127 | 0.2142 |

## oracle_sparse

- **Source**: synthetic (registry.make_oracle_sparse, seed 42)
- **License**: n/a
- **Downloaded**: n/a (in repository / generated)
- **Outcome**: synthetic, y ~ Bernoulli(p) with L5 forced to 3 events and L6 to 0
- **Attributes and recoding**: level L1..L6 with sizes 1800/1200/900/600/300/200; logit(p) ~ N(-2.2, 1); y ~ Bernoulli(p), then L5 forced to exactly 3 events and L6 to 0
- **Features**: no model; p is the prediction
- **Sizes**: total 5,000; train 0; audited test 5,000
- **Test prevalence**: 0.1186 (593 events); audit threshold 0.15
- **Stress condition**: a level with exactly 3 events (L5) and a level with 0 (L6): must be not estimable

| attribute | subgroup | n | events | prevalence |
|---|---|---:|---:|---:|
| level | L1 | 1,800 | 211 | 0.1172 |
| level | L2 | 1,200 | 176 | 0.1467 |
| level | L3 | 900 | 120 | 0.1333 |
| level | L4 | 600 | 83 | 0.1383 |
| level | L5 | 300 | 3 | 0.0100 |
| level | L6 | 200 | 0 | 0.0000 |
