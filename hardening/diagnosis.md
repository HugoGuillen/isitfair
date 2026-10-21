# Phase 1 diagnosis: why isitfair "hangs" on datasets other than SUPPORT2

## Answer

It is not a deadlock. `calibration_curves()` fitted LOWESS (`frac=2/3, delta=0`)
at every data point inside every bootstrap replicate of every subgroup. One such
fit is O(n^2), so the stage cost grows as B x n^2 and prints nothing while it
runs. At B = 2,000 (the default used in the examples) a 30,000-row cohort such
as diabetes130 runs for many hours. `report()` calls `calibration_curves()`
again, once per mitigated audit, which multiplies the stall. Fixed as ERRORLOG
E01; regression test `tests/test_hardening_regressions.py::test_e01_*`.

## Timing sweep (synthetic, B = 20, seconds per stage)

`hardening/phase1/bench.py`; logit(p) ~ N(-2.6, 1), y ~ Bernoulli(p), two
attributes `sex` (2 levels), `group8` (8 levels, smallest 3%) and their
intersection, threshold 0.2, `random_state=42`; JSON per run in
`hardening/phase1/results/`. "Before" imports the HEAD worktree (`PYTHONPATH`
pinned to `../isitfair-golden/src`); those runs shared the machine with other
jobs, so they are upper bounds. "After" is the fixed `src/`, run alone
(`logs/phase1_after_uncontended.log`).

| stage | code | 1k | 5k | 20k | 50k | 100k | slope 20k-100k |
|---|---|---:|---:|---:|---:|---:|---:|
| `calibration_curves()` | before | 3.4 | 50.0 | 914.0 | 5,971.7 | 25,364.8 | 2.06 |
| | after | 2.4 | 7.1 | 45.0 | 216.2 | 799.0 | 1.78 |
| `report()` | before | 8.3 | 14.5 | 59.7 | 249.3 | 26,428.4 | 3.67 |
| | after | 6.0 | 8.3 | 39.9 | 207.3 | 783.4 | 1.85 |
| `calibration()` | before | 4.0 | 7.1 | 58.6 | 314.7 | 2,615.7 | 2.33 |
| | after | 3.2 | 5.6 | 38.4 | 208.2 | 793.2 | 1.88 |
| `hosmer_lemeshow()` | before | 0.4 | 4.3 | 50.4 | 237.8 | 934.4 | 1.81 |
| | after | 0.2 | 2.5 | 32.8 | 200.2 | 775.6 | 1.97 |
| `decision_curve(zero_crossing=True)` | before | 3.4 | 8.2 | 41.0 | 159.4 | 629.7 | 1.69 |
| | after | 2.4 | 4.6 | 27.1 | 134.3 | 497.8 | 1.81 |
| `recalibration_ladder()` | before | 4.1 | 6.5 | 24.2 | 92.6 | 337.1 | 1.63 |
| | after | 3.4 | 4.6 | 17.5 | 75.4 | 274.1 | 1.70 |
| `discrimination()` | before | 2.0 | 2.2 | 3.2 | 4.2 | 8.3 | 0.58 |
| | after | 1.4 | 1.6 | 2.0 | 2.9 | 4.5 | 0.49 |
| `decision_curve()` | after | 2.1 | 2.3 | 2.8 | 3.9 | 5.5 | 0.42 |
| `roc_curves()`, `estimability()`, construct | after | < 0.4 everywhere | | | | | |

Log-log slopes are least-squares fits of log(seconds) on log(n) over 20k, 50k
and 100k. A slope near 2 is quadratic. The 100k "before" `report()` is
inflated by contention and by the stalled `calibration_curves()` inside it.

What the fix changed: at 20k `calibration_curves()` fell from 914 s to 45 s and
at 100k from about 7 hours to 13 minutes. B now enters only linearly through
cheap grid evaluations.

What remains quadratic (not the hang, not fixed; ERRORLOG E07): every stage
that computes ICI or a LOESS curve for the point estimate still runs one exact
LOWESS per subgroup, O(n^2). That is what the "after" slopes of about 1.8 to 2
measure. It is B-independent, so it costs about 13 minutes per stage at
100,000 rows and under 4 minutes at 50,000, and it is repeated in each of
`calibration()`, `calibration_curves()`, `hosmer_lemeshow()` and `report()`.
`decision_curve(zero_crossing=True)` is quadratic for a separate reason (E08).

## Profile at n = 20,000 (pre-fix code)

`hardening/profiles/n20000_calibration_curves_cprofile.txt` and
`..._pyinstrument.txt`, `calibration_curves()` alone, B = 20:

```
883.725 s total
  609 calls  881.804 s  isitfair/audit.py:469(_loess_curve_at_grid)
  609 calls  881.804 s  statsmodels/nonparametric/smoothers_lowess.py:12(lowess)   844.2 s own time
  5000 calls   1.658 s  numpy nanpercentile   (the bootstrap CI step)
```

609 calls = 29 subgroup rows (3 for `sex`, 9 for `group8`, 17 for the
intersection) x (1 point estimate + 20 replicates). Of 884 s, 882 s are
inside `lowess`.

## Watchdogs and py-spy

- `faulthandler.dump_traceback_later(600, repeat=True)` in every bench and
  notebook run (`logs/faulthandler_*.log`). On diabetes130 with the pre-fix code
  the dumps all show the same stack:
  `audit.py:2762 calibration_curves -> audit.py:483 _loess_curve_at_grid ->
  statsmodels/.../smoothers_lowess.py:226 lowess`. The stack never moves out
  of `lowess`, so the process is computing, not waiting on a lock.
- py-spy 0.4.2 works on this Windows machine, with one caveat. The venv's
  `python.exe` is a launcher that starts the real interpreter as a child
  process, so `py-spy dump --pid` has to target the child (the interpreter
  under `AppData\Roaming\uv\python\...`); pointed at the launcher it finds no
  Python stack. Example from the running Phase 7 kernel
  (`logs/pyspy_dump_phase7_kernel.txt`), showing the remaining point-estimate
  LOWESS of E07:
  ```
  Thread 16500 (active+gil): "MainThread"
      lowess (statsmodels\nonparametric\smoothers_lowess.py:226)
      _loess_smooth_at_data (isitfair\audit.py:467)
      _compute_ici (isitfair\audit.py:554)
      calibration (isitfair\audit.py:2593)
  ```

## Hypotheses

**(a) Bootstrapped LOESS is quadratic and dominates: confirmed, with one
correction.** Confirmed for `calibration_curves()` (profile above; slope 2.06
before the fix). The correction is that ICI is not bootstrapped. `calibration()`
computes ICI once per subgroup for the point estimate only, so it is quadratic
in n but does not scale with B. That is the residual E07, not the hang.

**(b) joblib/loky parallelism deadlocks or oversubscribes: rejected.**
`hardening/phase1/hyp_b_njobs.py` (also run as a notebook) compares
`n_jobs=None`, 2 and -1 at n = 5,000. All finish, all results are bit-identical,
and nothing stalls. Inside a Jupyter kernel on Windows, loky prints dozens of
harmless `KeyError` tracebacks from its resource tracker during cleanup
(ERRORLOG E02, third-party, deferred). In a terminal, a kernel that prints
tracebacks and then goes quiet for hours looks like a deadlock. The quiet
period is (a).

**(c) `decision_curve()` calls dcurves once per threshold or per replicate:
rejected.** `hardening/phase1/hyp_c_dca.py` counts calls. There is one dcurves
call per subgroup with all thresholds at once, about 0.15 to 0.23 s each, and the
cost is linear in the number of subgroups (k = 2, 8, 32, 128 subgroups:
0.68, 1.63, 5.44, 19.36 s). The `zero_crossing=True` option is quadratic for
a different reason (E08).
