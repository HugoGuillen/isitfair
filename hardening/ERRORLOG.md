# Error log

Schema per entry: id, dataset, stage, minimal reproducer, observed, expected,
root cause (file:line), fix, regression test, status, golden check after fix.

## E01: calibration_curves() stalls on large cohorts (the reported hang)

- **dataset**: diabetes130 (n = 30,530); reproduced on synthetic n >= 20,000
- **stage**: `calibration_curves()` (also inside `report()`, once per mitigated audit)
- **minimal reproducer**:
  ```python
  import numpy as np; from scipy.special import expit; from isitfair import FairnessAudit
  rng = np.random.default_rng(42); n = 20_000
  p = expit(rng.normal(-2.6, 1, n)); y = rng.binomial(1, p).astype(float)
  g = rng.choice(["a", "b"], n)
  FairnessAudit(y_true=y, y_proba=p, sensitive_features={"g": g}, threshold=0.2,
                n_bootstrap=20, random_state=42).calibration_curves()   # minutes; hours at B=2000
  ```
- **observed**: no output, no error; CPU at 100% for hours. Watchdog stack on
  diabetes130 (pre-fix, `HEAD` source): `audit.py:2762 calibration_curves` ->
  `audit.py:483 _loess_curve_at_grid` -> `statsmodels/.../smoothers_lowess.py:226 lowess`.
  Synthetic n = 20,000, B = 20: 914 s of 1,156 s total; 609 `lowess` calls hold
  882 of 884 profiled seconds (`hardening/profiles/`).
- **expected**: runtime roughly linear in n x B.
- **root cause**: `src/isitfair/audit.py:2762` (pre-fix line numbers) calls
  `_loess_curve_at_grid` (`audit.py:483`) for every bootstrap replicate of every
  subgroup; it fits LOWESS with `frac=2/3, delta=0` at every data point, which is
  O(n^2) per call. See `hardening/diagnosis.md`.
- **fix**: replicates of more than `_LOESS_EXACT_MAX_N = 2_000` rows evaluate the
  same smoother at the 100 grid points (`lowess(..., xvals=grid)`, grid clipped
  to the data range so the curve is held flat beyond it, as `np.interp` does).
  Point estimates and replicates of <= 2,000 rows unchanged. Diff:
  `git diff src/isitfair/audit.py` (hunks at `_LOESS_EXACT_MAX_N`,
  `_loess_curve_at_grid`, `calibration_curves`); 31 lines.
- **regression test**: `tests/test_hardening_regressions.py::test_e01_*`
  (fails before: `hardening/logs/regression/E01_before.log`):
  ```
  >       assert full_fits_above_cutoff == [3_000]
  E       assert [3000, 3000, ...0, 3000, 3000] == [3000]
  E         Left contains 5 more items, first extra item: 3000
  FAILED tests/test_hardening_regressions.py::test_e01_large_replicates_do_not_refit_loess_at_every_point
  ```
- **status**: fixed
- **golden check after fix**: notebook 02 tables (all fixes together, run 2026-10-20 22:26, `logs/golden_postfix.log`, compare `logs/golden_postfix_narrow_v2.log`): PASS under the narrowed rule. Every number equal to the committed one within 1e-12 except the ECE columns of the three tables built from isotonic-recalibrated predictions (33 cells; exempted with the user's approval; see E05 and `baseline.md`). The example/report part of that run did not execute (harness bug, see STATUS.md note); it is re-checked in Phase 7: PASS (Phase 7, 2026-10-21 11:24, `logs/p7_golden.log`, compare `logs/p7_golden_narrow_v2.log`): example and notebook 02 both executed on the final code; every table within 1e-12 outside the exempted ECE columns, the report HTML identical except 10 ECE cells in the "After" (mitigation) section, figures differ by at most 0.008 intensity in at most 0.02% of pixels. No committed SUPPORT2 number other than tied-prediction ECE changed

## E02: loky resource-tracker tracebacks inside Jupyter on Windows

- **dataset**: synthetic n = 5,000 (`hardening/phase1/hyp_b_njobs.py`)
- **stage**: `discrimination()` / `calibration()` with `n_jobs=2` or `-1`, in a Jupyter kernel
- **minimal reproducer**: run `hardening/phase1/hyp_b_njobs.ipynb` with nbconvert on Windows.
- **observed**: dozens of `KeyError: '...\joblib_memmapping_folder_...'` tracebacks
  from `joblib/externals/loky/backend/resource_tracker.py:319` on stderr. Results
  identical to `n_jobs=None`; no stall. Not printed in a plain script.
- **expected**: no tracebacks for a successful run.
- **root cause**: third-party (joblib 1.5.3 loky resource tracker cleanup of memmap
  folders on Windows), not in `src/isitfair/`.
- **fix**: none in isitfair.
- **regression test**: n/a
- **status**: deferred (third-party, harmless; results verified identical). Proposed
  doc note: "on Windows in Jupyter, `n_jobs != None` may print harmless joblib
  `resource_tracker` KeyError tracebacks".
- **golden check after fix**: n/a

## E03: recalibration_ladder() advises "fewer folds" for a level with zero events

- **dataset**: compas (race 'Native American', n = 2, 0 events); oracle_sparse (L6)
- **stage**: `recalibration_ladder(by=...)`
- **minimal reproducer**:
  ```python
  audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features={"race": race}, threshold=0.5)
  audit.recalibration_ladder(by="race")   # 'race' has a level with no events at all
  ```
- **observed**: `ValueError: fold 0, stratum 'Native American': zero events in the training
  union. ... Use fewer folds, or drop the stratum.`
- **expected**: an error that says the level has no events anywhere, so no fold choice helps.
- **root cause**: `src/isitfair/audit.py:3614-3621` (pre-fix) checks per fold only.
- **fix**: whole-audit check before the fold loop with an actionable message (9 lines);
  both messages added to the troubleshooting table in `docs/user_guide/your_own_data.md`.
- **regression test**: `tests/test_hardening_regressions.py::test_e03_zero_event_stratum_gets_an_actionable_message`
  (fails before: `hardening/logs/regression/E03_before.log`):
  ```
  E       AssertionError: Regex pattern did not match.
  E         Expected regex: 'zero events in the whole audit'
  E         Actual message: "fold 0, stratum 'Empty': zero events in the training union. ... Use fewer folds, or drop the stratum."
  ```
- **status**: fixed
- **golden check after fix**: notebook 02 tables (all fixes together, run 2026-10-20 22:26, `logs/golden_postfix.log`, compare `logs/golden_postfix_narrow_v2.log`): PASS under the narrowed rule. Every number equal to the committed one within 1e-12 except the ECE columns of the three tables built from isotonic-recalibrated predictions (33 cells; exempted with the user's approval; see E05 and `baseline.md`). The example/report part of that run did not execute (harness bug, see STATUS.md note); it is re-checked in Phase 7: PASS (Phase 7, 2026-10-21 11:24, `logs/p7_golden.log`, compare `logs/p7_golden_narrow_v2.log`): example and notebook 02 both executed on the final code; every table within 1e-12 outside the exempted ECE columns, the report HTML identical except 10 ECE cells in the "After" (mitigation) section, figures differ by at most 0.008 intensity in at most 0.02% of pixels. No committed SUPPORT2 number other than tied-prediction ECE changed

## E04: statistical-parity gap drops subgroups with no events (or no non-events)

- **dataset**: compas (race; found in Phase 5 against fairlearn)
- **stage**: `discrimination()` GAP row `statistical_parity`, point and bootstrap
- **minimal reproducer**:
  ```python
  # compas test half, y_proba_lr, threshold 0.5
  pd.Series(p >= 0.5).groupby(race).mean()
  # African-American 0.451, ..., Native American 0.000 (n = 2, 0 events)
  audit.discrimination()  # statistical_parity gap for race = 0.3374 (Native American omitted)
  ```
- **observed**: gap 0.33743; fairlearn `selection_rate` difference(between_groups) 0.45084.
- **expected**: 0.45084; the flag rate of an event-free subgroup is defined.
- **root cause**: `src/isitfair/audit.py:2272-2275` (point, `_compute_gaps`) reconstructs
  TP as `sensitivity * n_events`; sensitivity is NaN when `n_events == 0`, so the flag
  rate is NaN and `_range` drops the subgroup. Same for FP when every patient has the
  event. Bootstrap twin at `audit.py:1623` (`sens * prev + (1 - spec) * (1 - prev)`).
- **fix**: use 0 for the TP term without events and the FP term without non-events,
  in both places (12 lines).
- **regression test**: `tests/test_hardening_regressions.py::test_e04_*`
  (fails before: `hardening/logs/regression/E04_E05_before.log`):
  ```
  E       assert np.float64(0....6590700593915) == 0.563758389261745 ± 1.0e-12
  E         Obtained: 0.028506590700593915
  E         Expected: 0.563758389261745 ± 1.0e-12
  FAILED tests/test_hardening_regressions.py::test_e04_statistical_parity_keeps_zero_event_subgroups
  FAILED tests/test_hardening_regressions.py::test_e04_statistical_parity_keeps_all_event_subgroups
  ```
- **status**: fixed
- **golden check after fix**: notebook 02 tables (all fixes together, run 2026-10-20 22:26, `logs/golden_postfix.log`, compare `logs/golden_postfix_narrow_v2.log`): PASS under the narrowed rule. Every number equal to the committed one within 1e-12 except the ECE columns of the three tables built from isotonic-recalibrated predictions (33 cells; exempted with the user's approval; see E05 and `baseline.md`). The example/report part of that run did not execute (harness bug, see STATUS.md note); it is re-checked in Phase 7: PASS (Phase 7, 2026-10-21 11:24, `logs/p7_golden.log`, compare `logs/p7_golden_narrow_v2.log`): example and notebook 02 both executed on the final code; every table within 1e-12 outside the exempted ECE columns, the report HTML identical except 10 ECE cells in the "After" (mitigation) section, figures differ by at most 0.008 intensity in at most 0.02% of pixels. No committed SUPPORT2 number other than tied-prediction ECE changed

## E05: ECE depends on row order when predictions are tied

- **dataset**: compas (807 distinct predictions in 1,852 rows)
- **stage**: `calibration()` column `ece` (also `hosmer_lemeshow()` and the ladder, which call the same function)
- **minimal reproducer**:
  ```python
  # same compas inputs, rows permuted with np.random.default_rng(1).permutation(n)
  overall ECE: 0.040404 (original order) vs 0.041484 (permuted)
  ```
- **observed**: the value changes with row order.
- **expected**: invariant to row order (as `hosmer_lemeshow()` already is).
- **root cause**: `src/isitfair/audit.py:436-438` (pre-fix) `np.argsort(y_proba)` (unstable)
  followed by `np.array_split`, so tied rows straddle bin edges by position.
- **fix**: stable sort plus `isitfair.metrics._tie_snapped_bins`, the binning
  `hosmer_lemeshow()` uses (6 lines). For untied predictions the bins are unchanged.
- **regression test**: `tests/test_hardening_regressions.py::test_e05_*`
  (fails before: `hardening/logs/regression/E04_E05_before.log`):
  ```
  E           assert False   (ece by subgroup: original row order vs a permutation)
  E             Overall 0.021667 / a 0.041071 / b 0.050514  vs  Overall 0.024000 / a 0.047890 / b 0.056678
  FAILED tests/test_hardening_regressions.py::test_e05_ece_does_not_depend_on_row_order
  ```
- **status**: fixed
- **golden check after fix**: notebook 02 tables (all fixes together, run 2026-10-20 22:26, `logs/golden_postfix.log`, compare `logs/golden_postfix_narrow_v2.log`): PASS under the narrowed rule. Every number equal to the committed one within 1e-12 except the ECE columns of the three tables built from isotonic-recalibrated predictions (33 cells; exempted with the user's approval; see E05 and `baseline.md`). The example/report part of that run did not execute (harness bug, see STATUS.md note); it is re-checked in Phase 7: PASS (Phase 7, 2026-10-21 11:24, `logs/p7_golden.log`, compare `logs/p7_golden_narrow_v2.log`): example and notebook 02 both executed on the final code; every table within 1e-12 outside the exempted ECE columns, the report HTML identical except 10 ECE cells in the "After" (mitigation) section, figures differ by at most 0.008 intensity in at most 0.02% of pixels. No committed SUPPORT2 number other than tied-prediction ECE changed


## E06: calibration intercept and slope agree with a GLM only to about 1e-6

- **dataset**: all 18 oracle runs (Phase 5)
- **stage**: `calibration()` columns `calibration_intercept`, `calibration_slope`
- **minimal reproducer**: `hardening/phase5/oracles.py` (statsmodels Binomial GLM, `tol=1e-12`)
- **observed**: max abs. difference 1.7e-6 (intercept) and 1e-6 to 2e-5 (slope; 1.2e-4 on COMPAS
  race = Asian, n = 11, 3 events). Separately, cells with 1 or 2 events (bank 40-49/student,
  german female/50+, heart_cleveland female/65+) give arbitrary slopes in both tools (for example
  -1,184 vs -2,846): quasi-separation, not identified.
- **expected**: agreement within 1e-9 (the Phase 5 tolerance).
- **root cause**: isitfair's fitting routine stops at a looser convergence tolerance than the
  oracle. The quasi-separated cells are below the 100-event slope floor and already flagged
  `not_estimable_final`, so that part is not a defect.
- **fix**: none. Tightening the tolerance would change committed SUPPORT2 intercepts and slopes
  in about the 6th decimal, which the golden rule forbids.
- **regression test**: n/a
- **status**: deferred (decision for the maintainer; 1e-6 is far below every reported precision)
- **golden check after fix**: n/a

## E07: point-estimate LOWESS (ICI and calibration curves) is still O(n^2)

- **dataset**: synthetic n = 50,000 and 100,000 (Phase 1 "after" sweep); visible on diabetes130
- **stage**: `calibration()` (ICI), `calibration_curves()` (point curve), `hosmer_lemeshow()`,
  `report()`; each recomputes it
- **minimal reproducer**: `python hardening/phase1/bench.py --n 100000 --B 20` with the fixed code
- **observed**: about 13 minutes per stage at n = 100,000 (`calibration()` 793 s,
  `calibration_curves()` 799 s, `hosmer_lemeshow()` 776 s, `report()` 783 s); log-log slope
  about 1.9. py-spy on a live kernel: `calibration (audit.py:2593) -> _compute_ici (audit.py:554)
  -> _loess_smooth_at_data (audit.py:467) -> lowess` (`logs/pyspy_dump_phase7_kernel.txt`).
- **expected**: no hang (met: it is B-independent and finishes), ideally near-linear.
- **root cause**: `audit.py:467` and `audit.py:503` fit LOWESS with `delta=0` at every data point
  for the point estimate, once per subgroup per stage.
- **fix**: none here. Any faster exact-at-data method (statsmodels `delta`, or evaluating at a grid
  for ICI) changes the reported ICI and curve points, which the golden rule forbids. A
  number-preserving option is to cache the per-subgroup point fit on the audit object so the
  four stages share one fit (about 15 lines; it removes the three repeated fits); not done
  because it is an optimization, not a defect fix.
- **regression test**: n/a
- **status**: deferred (proposal for the maintainer)
- **golden check after fix**: n/a

## E08: decision_curve(zero_crossing=True) is O(U x n)

- **dataset**: synthetic (Phase 1); diabetes130 25 s at n = 30,530
- **stage**: `decision_curve(zero_crossing=True)`
- **minimal reproducer**: Phase 1 sweep; 27 s at 20k, 134 s at 50k, 498 s at 100k (slope 1.8)
- **observed**: quadratic growth for continuous predictions.
- **expected**: near-linear.
- **root cause**: `audit.py:621 _zero_crossings` loops over every distinct prediction (U of them,
  about n for continuous models) and recounts TP and FP with a full `y_proba >= hi` mask each time.
- **fix**: none here. An exact equivalent sorts once and uses cumulative sums (about 25 lines,
  identical output); it is an optimization, not a correctness fix, and the stage finishes.
- **regression test**: n/a
- **status**: deferred (proposal for the maintainer)
- **golden check after fix**: n/a

## E09: calibration-slope 95% CIs cover 91.5% to 92% in simulation

- **dataset**: oracle_calib (200 simulated datasets, n = 20,000, B = 200)
- **stage**: `calibration()` columns `slope_ci_low`, `slope_ci_high`
- **minimal reproducer**: `python hardening/phase5/oracle_calib_coverage.py`
- **observed**: coverage of the planted slope 91.5% (group A, slope 1) and 92.0% (group B, slope
  0.7); target band 93% to 97%. Intercept coverage of its estimand 95% and 98%. Point estimates
  unbiased (mean slope 0.9997 and 0.7024).
- **expected**: 93% to 97%.
- **root cause**: not established. 91.5% is 2.3 Monte Carlo standard errors below 95% (SE about
  1.5 points at 200 simulations), so it is weak evidence of slightly short percentile bootstrap
  intervals at B = 200 rather than a coding error; the B = 2,000 single run covers both slopes.
- **fix**: none. Changing the interval method would change committed SUPPORT2 CIs.
- **regression test**: n/a
- **status**: deferred (needs a larger simulation, for example 1,000 runs at B = 2,000, before any change)
- **golden check after fix**: n/a
