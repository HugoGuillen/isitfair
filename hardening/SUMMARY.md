# Hardening summary

Branch `hardening`. Code changes are confined to `src/isitfair/audit.py` (6
fixes, 116 added lines in total, none above the 40-line limit; E07 about 37,
E08 about 16). Supporting
changes:

- `tests/test_hardening_regressions.py`
- `CHANGELOG.md` ("Unreleased")
- two troubleshooting rows in `docs/user_guide/your_own_data.md`

Committed `results/` were never written.

## The hang

The hang was `calibration_curves()`. It refit an O(n^2) LOWESS for every
bootstrap replicate of every subgroup, so it cost B x n^2 and printed nothing
(E01). For example, synthetic n = 20,000 at B = 20 dropped from 914 s to 45 s,
and n = 100,000 from about 7 h to 13 min. diabetes130 (n = 30,530) now runs
the whole Phase 3 notebook in 31 min at B = 50, and the full Phase 6 report in
6 min at B = 200 on half of it. It was not a joblib deadlock, and dcurves is
not involved (`diagnosis.md`).

## Findings

| id | what | status |
|---|---|---|
| E01 | `calibration_curves()` stalls on large cohorts (the hang) | fixed, tested |
| E02 | harmless loky `KeyError` tracebacks in Jupyter on Windows | deferred (third-party) |
| E03 | `recalibration_ladder()` says "use fewer folds" for a level with no events at all | fixed, tested, documented |
| E04 | statistical-parity gap dropped subgroups with no events (COMPAS: 0.337 instead of 0.451) | fixed, tested |
| E05 | ECE depended on row order and CPU when predictions are tied | fixed, tested |
| E06 | intercept and slope agree with a GLM to about 1e-6, not 1e-9 | won't fix (maintainer) |
| E07 | point-estimate LOWESS (ICI, curves) refitted by every stage, O(n^2) each | fixed, tested (one fit per subgroup, identical results) |
| E08 | `decision_curve(zero_crossing=True)` O(U x n) | fixed, tested (cumulative counts, identical results) |
| E09 | slope CI coverage 91.5% and 92% vs the 93% to 97% target | won't fix (maintainer) |

Details, reproducers and before-fix failing output are in `ERRORLOG.md`.

## Golden check (SUPPORT2)

- **Narrowed rule.** HEAD itself does not reproduce the committed report on
  this machine (`baseline.md`). Seven mitigation-section ECE values differ
  because of how the CPU breaks ties in an unstable sort. With the user's
  approval, the rule therefore exempts the ECE cells built from
  isotonic-recalibrated predictions: in the report's "After" section and in
  three tables.
- **Everything else must match.**
  - Tables: within 1e-12.
  - Report HTML: identical line for line.
  - Figures: same pixels, up to sub-visual noise.
- **Results.**
  - Notebook 02 tables, all fixes applied: PASS.
  - Example and report, Phase 7 run (E01 to E05): PASS. 10 report ECE cells and 33 table ECE cells differ, all exempt; the figures differ only sub-visually (at most 0.008 intensity in 0.02% of pixels).
  - Rerun after E07 and E08 (`logs/e07e08_golden_narrow.log`): PASS, with the same exempt cells and figure noise. `pytest -q`: 51 passed.
  - E04 changed no committed SUPPORT2 number, because SUPPORT2 has no
    subgroup without events.

## Oracles (`oracle_checks.md`)

- **Exact agreement (within 1e-14) across all 18 runs:**
  - against sklearn: AUROC, AUPRC, the four threshold rates and Brier;
  - against dcurves and the closed formula: net benefit;
  - against fairlearn: the marginal gaps;
  - against from-scratch oracles: ECE and Hosmer-Lemeshow, using isitfair's
    documented tie handling.
- **Equal-opportunity and equalized-odds gaps** match fairlearn once
  subgroups whose TPR or FPR is undefined are excluded. fairlearn counts those
  rates as 0.
- **Remaining differences:**
  - optimizer tolerance on intercept and slope (E06);
  - quasi-separated cells with 1 or 2 events, which are already flagged not
    estimable;
  - intersection gaps, which are shrunk by design.
- **oracle_calib:** at B = 2,000 every CI contains its target. The intercept's
  target is the calibration-in-the-large estimand, -0.663, not the planted
  -0.8.
- **Coverage over 200 x B = 200:**
  - intercept: 95% (group A) and 98% (group B);
  - slope: 91.5% and 92% (E09).
- **oracle_sparse:** the 3-event and 0-event levels are flagged
  `not_estimable_final` for every metric and carry no CI.

## Real datasets, full pipeline at B = 200 (`hardening/reports/`)

Sequential runs with `n_jobs=-1` and an HTML+PDF report each. Wall seconds:

| dataset | audited n | total | calibration_curves | report (HTML+PDF) |
|---|---:|---:|---:|---:|
| support2 | 1,821 | 121 | 60 | 41 |
| compas | 926 | 66 | 14 | 34 |
| german | 150 | 33 | 1 | 17 |
| heart_cleveland | 46 | 32 | 0 | 17 |
| adult | 7,327 | 187 | 44 | 101 |
| bank | 6,782 | 238 | 86 | 107 |
| diabetes130 | 15,264 | 360 | 72 | 176 |

The datasets other than SUPPORT2 audit the evaluation half of the test split.
The other half fits the mitigation calibrators, so no row is scored by a
calibrator fitted on it.

The first diabetes130 run stopped in `mitigate()` with a clear `ValueError`.
One gender level occurred in the audited half but never in the calibration
half. That is the documented behaviour, so the run script now drops such rows
and says so (1 row).

Input variants (Phase 3, every dataset): categorical attributes, `Int64`, bool
and bool-Series outcomes, and a consistently shuffled index all give
bit-identical results. A misaligned index raises the documented error. Tied
predictions and exact 0/1 predictions run without error.

## Phase 7 acceptance

All steps on the final code, one at a time, 2026-10-21 (`hardening/phase7/run.sh`, rows in STATUS.md):

| step | wall s | result |
|---|---:|---|
| nbconvert support2 | 111 | pass |
| nbconvert compas | 109 | pass |
| nbconvert german | 74 | pass |
| nbconvert heart_cleveland | 76 | pass |
| nbconvert oracle_sparse | 73 | pass |
| nbconvert oracle_calib | 370 | pass |
| nbconvert adult | 672 | pass |
| nbconvert bank | 649 | pass |
| nbconvert diabetes130 | 1,864 | fail: mitigation split in the hardening template (see below) |
| nbconvert diabetes130, rerun after the template fix | 2,175 | pass |
| `pytest -q` | 65 | 42 passed |
| golden check (example 1,531 s, notebook 02 1,601 s) | 3,134 | PASS (narrowed rule) |

Each notebook ran with nbconvert (cell timeout 3,600 s), `HARDEN_STRICT=1`,
B = 50 and `n_jobs=-1`. Strict mode fails a notebook on any error the
notebook does not expect. The executed copies are in
`hardening/notebooks/executed/*_phase7.ipynb`.

The diabetes130 failure was in my notebook template, not in isitfair. One
audited row had the gender level "Unknown/Invalid", which never appears in
the calibration half, so `mitigate()` raised its documented error. The
overnight Phase 3 run of diabetes130 hit the same error without stopping,
because strict mode was off. The template now drops such rows and prints
that it did; for every other dataset it changes nothing.

## Runtime guidance for the docs

From the Phase 1 sweep (synthetic, B = 20, one process): totals for all
stages including `report()`.

| n | after E01 | after E07 and E08 |
|---:|---:|---:|
| 1,000 | about 20 s | not rerun |
| 5,000 | about 40 s | not rerun |
| 20,000 | about 3.5 min | about 1.2 min |
| 50,000 | about 17.5 min | about 4.3 min |
| 100,000 | about 65 min | about 14.3 min |

At B = 2,000 only the bootstrap parts scale up. The exact point-estimate
LOWESS is still O(n^2), but it now runs twice per subgroup (once in `calibration()` for ICI
and once in `calibration_curves()` for the curve) instead of in every stage;
per-stage numbers are in `diagnosis.md`.
