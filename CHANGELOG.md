# Changelog

## 0.5.0

The committed SUPPORT2 results are unchanged except the ECE of isotonic-recalibrated
(heavily tied) predictions in the mitigation tables and report, which now no
longer depends on row order or CPU.

### Fixes

- `calibration_curves()` no longer stalls on large cohorts. Every bootstrap
  replicate refitted LOESS (`frac=2/3`) at every data point, which is quadratic
  in n: about 15 minutes at n = 20,000 with 20 replicates, and days for a
  100,000-row audit at 2,000 replicates. Replicates of more than 2,000 patients
  now evaluate the same smoother directly at the 100 grid points. Point
  estimates, and every replicate of 2,000 patients or fewer, are computed
  exactly as before; above the cutoff the band edges move by about 1e-4.
- `recalibration_ladder()` on an attribute with a level that has no events at
  all now says so, instead of advising fewer folds, which cannot help.
- The statistical-parity gap now includes subgroups with no events (or no
  non-events). Their flag rate is defined, but it was reconstructed from an
  undefined sensitivity or specificity and silently dropped, understating the
  gap (COMPAS race: 0.337 reported, 0.451 true). The same applies to its
  bootstrap replicates.
- ECE no longer depends on row order when predicted probabilities are tied.
  Bins are tie-snapped exactly as in `hosmer_lemeshow()`; untied predictions
  get the same equal-frequency bins as before.

### Performance

- The exact point-estimate LOESS fits behind ICI and the calibration curve
  (O(n^2)) are now computed once per subgroup and reused by `calibration()`,
  `calibration_curves()`, `hosmer_lemeshow()`, `recalibration_ladder()` and
  `report()`, instead of once per method call. Results are identical; only
  inputs of more than 2,000 rows are memoized (at most 64 fits).
- `decision_curve(zero_crossing=True)` counts flagged events with one sort and
  cumulative sums instead of a full pass per distinct prediction. Results are
  identical; the search is O(n log n) instead of O(n^2).
- Together: a full audit of 100,000 patients with 20 bootstrap replicates
  takes about 14 minutes instead of about 65 (synthetic benchmark).

### Documentation

- New `extra_examples/`: audit notebooks for six public datasets (Adult, Bank
  Marketing, COMPAS, Diabetes 130-US hospitals, German Credit, Heart Disease
  Cleveland), with `datasets.py` to download them and fit the audited models.

- Two troubleshooting rows in `docs/user_guide/your_own_data.md` for the
  `recalibration_ladder()` messages: a level with no events at all, and a
  fold whose training data has no events for a level.

### Known issues

- Calibration intercept and slope agree with a tightly converged GLM to about
  1e-6, not 1e-9 (optimizer tolerance). Not changed, to keep published results
  stable.
- In simulation, 95% CIs for the calibration slope covered 91.5% to 92% of the
  time (200 datasets, 200 replicates each). Not changed, for the same reason.
- In Jupyter on Windows, joblib's loky backend can print harmless `KeyError`
  tracebacks from its resource tracker during cleanup.

## 0.4.0

Release accompanying the companion paper (accepted, *JAMIA*). Numerical
results are unchanged from the version that produced `results/`.

### Using the package on your own data

- New guide, `docs/user_guide/your_own_data.md`, and template script,
  `examples/audit_your_own_data.py`.
- README: install instructions for a fresh environment, a Quick start that runs
  as pasted, and a "Use it on your own data" section.

### Input validation

- Missing values in `y_true`, `y_proba` or any sensitive attribute now raise a
  `ValueError` naming the input and the count. Previously, a missing
  probability failed later inside scikit-learn. A missing string label crashed
  with an unrelated `TypeError`, and a missing numeric label silently dropped
  those patients from every subgroup.
- Clearer errors for a two-column `predict_proba` array, text outcomes, and
  probabilities outside [0, 1].
- pandas inputs holding the same index labels in different orders now raise,
  instead of being silently mismatched by position.
- Warnings for hard 0/1 labels passed as probabilities, for numeric attributes
  with more than 20 distinct values (unbinned continuous variables), and for
  labels that look like missing values converted to text (`"nan"`).

### Fixes

- Intersections: subgroup labels containing `" / "` (e.g. "Asian / Pacific
  Islander") and user-supplied interaction label arrays no longer raise
  `KeyError` when a cell is shrunk. Interaction labels that do not map to a
  single combination of their marginals are rejected with a clear message.
- `mitigate()` now carries custom `event_floors` into the audits it returns.
- `recalibration_ladder()` and `recalibration_ladder_scores()` default to the
  audit's `random_state` instead of an unseeded fold split.
- `compare_periods(attribute=...)` raises on an unknown attribute instead of
  returning an empty frame.
- `FairnessAudit` has an informative `repr`.
- Corrected docstring examples, including the value in the `cohens_h`
  example.

### Packaging and testing

- `requires-python = ">=3.10,<3.14"`, matching `dcurves`. Runtime dependencies
  have lower bounds at the tested versions. `tabulate` is in the `examples`
  extra, and there is a new `dev` extra. The reproduction instructions install
  the `plots` extra (adjustText), which the committed volcano figure was made with.
- `examples/make_manifest.py` records every package that can change an
  artifact's bytes (adds scipy, matplotlib, joblib, jinja2, tableone, tabulate,
  adjustText and WeasyPrint, writing "not installed" for absent optional ones).
  `requirements-lock.txt` pins tabulate and adjustText, recovered by
  reproducing the committed files byte for byte.
- Test suite in `tests/`, and GitHub Actions CI on Python 3.10–3.13
  (Linux and Windows).
