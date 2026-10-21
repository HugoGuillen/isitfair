"""Regression tests for defects found while auditing real classification datasets.

Each test names the hardening/ERRORLOG.md entry it guards.
"""

import numpy as np
import pytest
from scipy.special import expit

import isitfair.audit as audit_module
from isitfair import FairnessAudit


@pytest.fixture(autouse=True)
def _fresh_lowess_cache():
    # E07 memoizes exact LOWESS fits by content; start every test without hits.
    getattr(audit_module, "_LOWESS_CACHE", {}).clear()
    yield


def _large_cohort(n, seed=0):
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(-1.5, 1.0, n))
    y = rng.binomial(1, p).astype(float)
    return y, p, rng.choice(["a", "b"], n)


# E01: calibration_curves() fitted LOESS at every data point inside every
# bootstrap replicate, O(n^2) per subgroup per replicate, which stalled large
# audits for hours.


def test_e01_large_replicates_do_not_refit_loess_at_every_point(monkeypatch):
    y, p, g = _large_cohort(3_000)
    calls = []
    real = audit_module.lowess

    def spy(endog, exog, *args, **kwargs):
        calls.append((len(exog), kwargs.get("xvals") is not None))
        return real(endog, exog, *args, **kwargs)

    monkeypatch.setattr(audit_module, "lowess", spy)
    FairnessAudit(
        y_true=y, y_proba=p, sensitive_features={"g": g},
        threshold=0.2, n_bootstrap=5, random_state=1,
    ).calibration_curves()

    # Only the Overall point estimate (n = 3,000) may be fitted at every point;
    # its five replicates must evaluate on the grid.
    full_fits_above_cutoff = [n for n, on_grid in calls if not on_grid and n > 2_000]
    assert full_fits_above_cutoff == [3_000]


def test_e01_point_estimate_is_the_exact_fit():
    y, p, g = _large_cohort(3_000)
    curves = FairnessAudit(
        y_true=y, y_proba=p, sensitive_features={"g": g},
        threshold=0.2, n_bootstrap=3, random_state=1,
    ).calibration_curves()
    overall = curves[(curves.attribute == "g") & (curves.subgroup == "Overall")]
    grid = np.linspace(p.min(), p.max(), 100)
    expected = audit_module._loess_curve_at_grid(y, p, grid)
    np.testing.assert_array_equal(overall["observed_prob"].to_numpy(), expected)


def test_e01_grid_bands_match_exact_bands_closely(monkeypatch):
    y, p, g = _large_cohort(3_000)
    kw = dict(y_true=y, y_proba=p, sensitive_features={"g": g},
              threshold=0.2, n_bootstrap=20, random_state=1)
    fast = FairnessAudit(**kw).calibration_curves()
    monkeypatch.setattr(audit_module, "_LOESS_EXACT_MAX_N", 10**9, raising=False)
    exact = FairnessAudit(**kw).calibration_curves()
    np.testing.assert_array_equal(fast["observed_prob"], exact["observed_prob"])
    for col in ("ci_low", "ci_high"):
        assert np.nanmax(np.abs(fast[col] - exact[col])) < 2e-3


def test_e01_small_cohorts_are_unchanged(cohort, kw, monkeypatch):
    # SUPPORT2-sized strata (<= 2,000) keep the exact path, bit for bit.
    sf = {"sex": cohort["sex"], "race": cohort["race"]}
    base = FairnessAudit(y_true=cohort["y"], y_proba=cohort["p"], sensitive_features=sf, **kw)
    got = base.calibration_curves()
    monkeypatch.setattr(audit_module, "_LOESS_EXACT_MAX_N", 10**9, raising=False)
    ref = FairnessAudit(
        y_true=cohort["y"], y_proba=cohort["p"], sensitive_features=sf, **kw
    ).calibration_curves()
    assert got.equals(ref)


@pytest.mark.parametrize("n", [2_000, 2_001])
def test_e01_cutoff_boundary(n, monkeypatch):
    y, p, g = _large_cohort(n)
    seen = []
    real = audit_module._loess_curve_at_grid

    def spy(y_true, y_proba, eval_points, exact=True):
        seen.append((len(y_true), exact))
        return real(y_true, y_proba, eval_points, exact)

    monkeypatch.setattr(audit_module, "_loess_curve_at_grid", spy)
    FairnessAudit(
        y_true=y, y_proba=p, sensitive_features={"g": g},
        threshold=0.2, n_bootstrap=2, random_state=1,
    ).calibration_curves()
    overall_replicates = [exact for size, exact in seen if size == n][1:]
    assert overall_replicates == [n <= 2_000] * 2


# E03: recalibration_ladder() on an attribute with a zero-event level said
# "Use fewer folds", which cannot help when the level has no events at all.


def test_e03_zero_event_stratum_gets_an_actionable_message(cohort, kw):
    y = cohort["y"].copy()
    race = cohort["race"].copy()
    race[:40] = "Empty"
    y[:40] = 0.0
    audit = FairnessAudit(y_true=y, y_proba=cohort["p"], sensitive_features={"race": race}, **kw)
    with pytest.raises(ValueError, match="zero events in the whole audit") as err:
        audit.recalibration_ladder(by="race")
    assert "'Empty'" in str(err.value)
    assert "fewer folds will not help" in str(err.value)


# E04: the statistical-parity gap silently dropped subgroups whose sensitivity
# or specificity is undefined (no events, or no non-events), although their
# flag rate is defined. COMPAS race: 0.337 reported, 0.451 true.


def test_e04_statistical_parity_keeps_zero_event_subgroups(cohort, kw):
    y, p = cohort["y"].copy(), cohort["p"].copy()
    g = np.where(np.arange(len(y)) < 60, "none", cohort["sex"]).astype(object)
    y[:60] = 0.0
    p[:60] = 0.9  # every patient of the event-free subgroup is flagged
    audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features={"g": g}, **kw)
    disc = audit.discrimination()
    flag = {k: float(np.mean(p[g == k] >= kw["threshold"])) for k in ("none", "F", "M")}
    gap = disc[(disc.subgroup == "GAP") & (disc.gap_metric == "statistical_parity")].iloc[0]
    assert gap.gap_value == pytest.approx(max(flag.values()) - min(flag.values()), abs=1e-12)
    # Its interval is withheld (the subgroup is below the event floor), but the
    # replicates behind it must include the event-free subgroup too.
    reps = audit._boot_store["discrimination"][("g", "GAP", "statistical_parity")].values
    assert np.all(np.isfinite(reps))
    assert abs(np.median(reps) - gap.gap_value) < 0.05


def test_e04_statistical_parity_keeps_all_event_subgroups(cohort, kw):
    y, p = cohort["y"].copy(), cohort["p"].copy()
    g = np.where(np.arange(len(y)) < 60, "all", cohort["sex"]).astype(object)
    y[:60] = 1.0
    p[:60] = 0.01  # none flagged
    disc = FairnessAudit(y_true=y, y_proba=p, sensitive_features={"g": g}, **kw).discrimination()
    flag = {k: float(np.mean(p[g == k] >= kw["threshold"])) for k in ("all", "F", "M")}
    gap = disc[(disc.subgroup == "GAP") & (disc.gap_metric == "statistical_parity")].iloc[0]
    assert gap.gap_value == pytest.approx(max(flag.values()) - min(flag.values()), abs=1e-12)


# E05: ECE depended on row order when predictions were tied, because tied
# rows were split across bin edges by their position in an unstable argsort.


def test_e05_ece_does_not_depend_on_row_order(kw):
    rng = np.random.default_rng(3)
    n = 1200
    p = np.round(rng.uniform(0.05, 0.6, n), 1)  # six distinct values, heavy ties
    y = rng.binomial(1, p).astype(float)
    g = rng.choice(["a", "b"], n)

    def ece(order):
        cal = FairnessAudit(y_true=y[order], y_proba=p[order], sensitive_features={"g": g[order]},
                            **kw).calibration()
        return cal[cal.subgroup != "GAP"].set_index("subgroup")["ece"]

    base = ece(np.arange(n))
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(n)
        assert ece(perm).equals(base)


def test_e05_untied_predictions_keep_equal_frequency_bins():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.99, 997)
    y = rng.binomial(1, p).astype(float)
    order = np.argsort(p)
    expected = sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, 10)) / len(p)
    assert audit_module._compute_ece(y, p, 10) == expected


# E07: the exact point-estimate LOWESS (O(n^2)) was refitted for the same
# subgroup by every stage that needs ICI or the calibration curve.


def test_e07_point_estimate_lowess_is_fitted_once_per_subgroup(monkeypatch):
    y, p, g = _large_cohort(5_000)
    calls = []
    real = audit_module.lowess

    def spy(endog, exog, *args, **kwargs):
        if kwargs.get("xvals") is None and len(exog) > 2_000:
            calls.append((kwargs.get("return_sorted", True),
                          np.asarray(endog).tobytes(), np.asarray(exog).tobytes()))
        return real(endog, exog, *args, **kwargs)

    monkeypatch.setattr(audit_module, "lowess", spy)
    audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features={"g": g},
                          threshold=0.2, n_bootstrap=2, random_state=1)
    # Each of these stages needs ICI for Overall, a and b (the ladder also on its
    # recalibrated rungs, which are different inputs).
    audit.calibration()
    audit.hosmer_lemeshow()
    audit.recalibration_ladder(by="g")
    audit.calibration_curves()
    audit.calibration_curves()
    # No identical input is fitted twice.
    assert len(calls) == len(set(calls)) and len(calls) >= 6


def test_e07_cached_results_equal_uncached(monkeypatch):
    y, p, g = _large_cohort(5_000)
    kw = dict(y_true=y, y_proba=p, sensitive_features={"g": g},
              threshold=0.2, n_bootstrap=2, random_state=1)
    audit = FairnessAudit(**kw)
    first = (audit.calibration(), audit.hosmer_lemeshow(detail=True), audit.calibration_curves())
    second = (FairnessAudit(**kw).calibration(), audit.hosmer_lemeshow(detail=True),
              FairnessAudit(**kw).calibration_curves())
    monkeypatch.setattr(audit_module, "_LOWESS_CACHE_MAX", 0, raising=False)
    getattr(audit_module, "_LOWESS_CACHE", {}).clear()
    fresh = FairnessAudit(**kw)
    uncached = (fresh.calibration(), fresh.hosmer_lemeshow(detail=True), fresh.calibration_curves())
    for a, b, c in zip(first, second, uncached):
        assert a.equals(b) and a.equals(c)


# E08: _zero_crossings() rescanned every row for every distinct prediction,
# O(U x n), so decision_curve(zero_crossing=True) took minutes at 100,000 rows.


def _zero_crossings_reference(y_true, y_proba, t_lo, t_hi):
    # The pre-fix algorithm, kept as the oracle for the faster one.
    import itertools

    if len(y_true) == 0 or t_hi <= t_lo:
        return []
    edges = sorted({float(t_lo), float(t_hi)}
                   | {float(a) for a in np.unique(y_proba) if t_lo < a < t_hi})
    eps = 1e-9
    candidates = []
    for lo, hi in itertools.pairwise(edges):
        flagged = y_proba >= hi
        tp = float(np.sum(y_true[flagged] == 1))
        fp = float(np.sum(y_true[flagged] == 0))
        if tp + fp > 0:
            t_star = tp / (tp + fp)
            if lo < t_star <= hi:
                candidates.append(float(t_star))
        if hi < t_hi:
            candidates.append(float(hi))
    candidates.append(float(edges[0]))

    def nb_at(t):
        return float(audit_module._compute_net_benefits(y_true, y_proba, np.array([t]))[0])

    found = []
    for t in candidates:
        before, after = nb_at(t - eps), nb_at(t + eps)
        if np.isnan(before) or np.isnan(after) or np.sign(before) * np.sign(after) >= 0:
            continue
        found.append(t)
    return sorted(set(found))


def test_e08_zero_crossings_do_not_rescan_every_row(monkeypatch):
    y, p, _ = _large_cohort(3_000)
    scans = []
    real = audit_module._compute_net_benefits

    def spy(y_true, y_proba, thresholds):
        scans.append(len(y_true))
        return real(y_true, y_proba, thresholds)

    monkeypatch.setattr(audit_module, "_compute_net_benefits", spy)
    audit_module._zero_crossings(y, p, 0.01, 0.99)
    assert scans == []


@pytest.mark.parametrize("seed", range(6))
def test_e08_zero_crossings_match_the_reference(seed):
    rng = np.random.default_rng(seed)
    n = 400
    p = expit(rng.normal(-1.0, 1.5, n))
    if seed % 2:
        p = np.round(p, 2)  # heavy ties
    if seed == 5:
        p[:5], p[5:10] = 0.0, 1.0  # exact 0/1 predictions
    y = rng.binomial(1, np.clip(p * rng.uniform(0.3, 1.7), 0, 1)).astype(float)
    for t_lo, t_hi in [(0.01, 0.99), (0.05, 0.6), (0.2, 0.21)]:
        assert audit_module._zero_crossings(y, p, t_lo, t_hi) ==             _zero_crossings_reference(y, p, t_lo, t_hi)
