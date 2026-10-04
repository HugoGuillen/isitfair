# Changelog

## 0.4.0

Release accompanying the companion paper (under review, *JAMIA*). Numerical
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
