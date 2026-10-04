"""FairnessAudit: subgroup-stratified evaluation of clinical prediction models.

This module provides the core audit class. Each analysis axis (discrimination,
calibration, decision-curve) is a method that returns a long-format DataFrame
with one row per (attribute, subgroup) plus gap-statistic rows.

Discrimination metrics use sklearn directly. Mitigation methods live in
:mod:`isitfair.mitigation`.
"""

from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from dcurves import dca
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit
from sklearn.metrics import confusion_matrix, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

from isitfair.estimability import (
    FLOOR_FOR,
    GAP_SOURCE,
    floor_fails,
    resolve_event_floors,
)
from isitfair.metrics import hosmer_lemeshow as _hosmer_lemeshow
from isitfair.metrics import subgroup_metrics

# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _to_1d_float_array(x: Any, name: str) -> npt.NDArray[np.floating[Any]]:
    """Coerce input to a 1-d float64 numpy array."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        example = next(
            (v for v in np.asarray(x, dtype=object).ravel() if isinstance(v, str)), None
        )
        msg = f"`{name}` must be numeric, got non-numeric values (e.g. {example!r})."
        if name == "y_true":
            msg += " Recode the outcome to 0/1 first, e.g. `(df[col] == 'yes').astype(int)`."
        raise ValueError(msg) from exc
    if arr.ndim == 2 and arr.shape[1] == 2 and name == "y_proba":
        msg = (
            f"`y_proba` must be 1-dimensional, got shape {arr.shape}. This looks "
            "like the output of `predict_proba`; pass the positive-class column, "
            "`model.predict_proba(X)[:, 1]`."
        )
        raise ValueError(msg)
    if arr.ndim != 1:
        msg = f"`{name}` must be 1-dimensional, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _to_1d_array(x: Any, name: str) -> npt.NDArray[Any]:
    """Coerce input to a 1-d numpy array (any dtype), rejecting missing values."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        msg = f"`{name}` must be 1-dimensional, got shape {arr.shape}"
        raise ValueError(msg)
    # A missing label would either crash np.unique (str mixed with NaN) or
    # form a "nan" subgroup that no row ever matches, silently dropping those
    # patients from every subgroup while keeping them in Overall.
    n_missing = int(pd.isna(arr).sum())
    if n_missing:
        msg = (
            f"`{name}` has {n_missing} missing value(s). Subgroup labels cannot "
            "be missing: drop those rows from every input, or recode them as an "
            "explicit category, e.g. `df[col].fillna('unknown')`."
        )
        raise ValueError(msg)
    if arr.dtype.kind in "OUS":
        stringified = sorted({str(v) for v in arr} & {"nan", "NaN", "None", "<NA>"})
        if stringified:
            warnings.warn(
                f"`{name}` has the label(s) {stringified}, which look like missing "
                "values converted to text (e.g. by `.astype(str)`). They will be "
                "audited as a subgroup. Use `.astype('string').fillna('unknown')` "
                "if they are missing.",
                UserWarning,
                stacklevel=3,
            )
    return arr


def _check_index_alignment(named: dict[str, Any]) -> None:
    """Reject pandas inputs that carry the same index labels in different orders.

    Inputs are aligned by position, never by index. Two Series holding the same
    index labels in a different order are therefore almost certainly
    misaligned (one was sorted or shuffled), and auditing them would pair each
    outcome with another patient's prediction without any error.
    """
    indexed = [(k, v.index) for k, v in named.items() if isinstance(v, pd.Series)]
    if len(indexed) < 2:
        return
    ref_name, ref = indexed[0]
    for name, idx in indexed[1:]:
        if idx.equals(ref) or len(idx) != len(ref) or not idx.is_unique:
            continue
        try:
            same_labels = idx.sort_values().equals(ref.sort_values())
        except TypeError:  # unorderable mixed-type index; nothing to compare
            continue
        if same_labels:
            msg = (
                f"`{name}` and `{ref_name}` are pandas objects with the same index "
                "labels in a different order. FairnessAudit aligns inputs by "
                "position, not by index, so the rows would be mismatched. Reorder "
                "them consistently, e.g. `s.loc[other.index]`, or pass "
                "`.to_numpy()` arrays built from one DataFrame."
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Per-subgroup metric computation — discrimination (point estimates)
# ---------------------------------------------------------------------------


def _compute_subgroup_metrics(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    threshold: float,
) -> dict[str, float]:
    """Compute discrimination metrics for a single subgroup slice.

    Thin wrapper around :func:`isitfair.metrics.subgroup_metrics` kept for
    internal callers that pass positional arguments.
    """
    return subgroup_metrics(y_true, y_proba, threshold=threshold)


_CI_METRICS = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"]

# Retained per replicate but never emitted with an interval of its own. The
# statistical-parity gap is a range over the flag rate, which is not a stored
# metric; with prevalence in hand it is
# ``sensitivity * pi + (1 - specificity) * (1 - pi)`` and so becomes a
# reduction over the same replicates as every other gap, rather than a second
# resample that would not reconcile with its constituents.
_GAP_SUPPORT_METRICS = ["prevalence"]
_DISCRIMINATION_BOOT_METRICS = [*_CI_METRICS, *_GAP_SUPPORT_METRICS]
_RATE_METRICS = ["prevalence", "sensitivity", "specificity", "ppv", "npv"]

# ---------------------------------------------------------------------------
# Intersectional subgroup helpers
# ---------------------------------------------------------------------------

_INTERACTION_SEP = " × "


def _interaction_attr_name(constituents: tuple[str, ...]) -> str:
    """Build display name for an interaction attribute (e.g. 'sex × age')."""
    return _INTERACTION_SEP.join(constituents)


def _build_interaction_labels(
    arrays: list[npt.NDArray[Any]],
) -> npt.NDArray[Any]:
    """Join marginal label arrays into 'M / Young' style composite labels."""
    n = len(arrays[0])
    labels = np.array([str(arrays[0][i]) for i in range(n)], dtype=object)
    for arr in arrays[1:]:
        labels = np.array([f"{labels[i]} / {arr[i]}" for i in range(n)], dtype=object)
    return labels


def _shrink_rate(
    successes: int,
    trials: int,
    target: float,
    prior_strength: float,
) -> float:
    """Beta-Binomial posterior mean: (α + successes) / (α + β + trials).

    The prior is ``Beta(α, β)`` where ``α = target * prior_strength``
    and ``β = (1 - target) * prior_strength``.
    """
    alpha = target * prior_strength
    beta = (1.0 - target) * prior_strength
    return (alpha + successes) / (alpha + beta + trials)


# ---------------------------------------------------------------------------
# Per-subgroup metric computation — calibration
# ---------------------------------------------------------------------------

_LOGIT_EPS = 1e-10

# Bootstrap resampling schemes.
#
# "case_control" is the scheme used for the published analysis: positives and
# negatives are resampled separately at fixed sizes, so every replicate carries
# the subgroup's observed event count. That is defensible for rank-based
# metrics (AUROC, AUPRC are conditional-on-margins functionals) but removes the
# dominant variance component for prevalence-dependent ones -- calibration
# intercept and slope, Brier, PPV, NPV, prevalence, standardized net benefit.
# For the fixed-slope calibration intercept, which solves
# ``sum(y) = sum(expit(alpha + eta))``, pinning ``sum(y)`` makes alpha nearly
# deterministic and the interval collapses.
#
# The code path behind "case_control" is FROZEN: it exists so the published
# numbers of the companion paper's originally submitted analysis stay exactly
# reproducible.
# Do not refactor it, do not "improve" it, and do not change the order in which
# it consumes the shared Generator -- the reported intervals depend on the
# exact draw sequence, and Monte Carlo noise alone moves CI endpoints by
# 5-20x more than a 3-decimal-place tolerance allows.
#
# "patient" resamples patients within each subgroup stratum, preserving the
# subgroup's n while letting its event count vary. It is the statistically
# correct default for this package's metric set.
_RESAMPLE_SCHEMES = ("case_control", "patient")

# Columns appended to every bootstrap-bearing frame. Appended at the tail
# rather than inserted, so existing pinned column orders stay valid.
_ESTIMABILITY_COLUMNS = [
    "n_replicates",
    "n_estimable",
    "proportion_nonestimable",
    "not_estimable",
]

# Axes whose per-replicate distributions are retained. The curve axes are 100
# points wide and have no gap statistic, so they keep returning percentiles
# only rather than costing ~100x the memory.
_BOOTSTRAP_AXES = ("discrimination", "calibration")

# Independent RNG streams. Overall is keyed separately from the per-attribute
# work so the whole-cohort interval does not depend on which attributes exist.
_STREAM_OVERALL = 0
_STREAM_ATTRIBUTE = 1

# Which per-subgroup metric each calibration gap statistic reduces over.
_CALIBRATION_GAP_SOURCE = {
    "max_intercept_gap": "calibration_intercept",
    "max_slope_gap": "calibration_slope",
}

# Appended after _ESTIMABILITY_COLUMNS on every bootstrap-bearing frame.
# `event_floor` and `below_floor` describe the axis's *primary* metric;
# `not_estimable_final` is the conservative union over every metric the row
# carries, so it fires when any one of them is unreportable.
_FLOOR_COLUMNS = ["event_floor", "below_floor", "not_estimable_final"]

# Per axis: the primary metric whose floor is reported in `event_floor`, and
# every metric the frame carries with the interval columns a floor failure
# blanks. Point estimates are never blanked -- only intervals.
_AXIS_FLOOR_SPEC: dict[str, tuple[str, tuple[tuple[str, tuple[str, str]], ...]]] = {
    "discrimination": (
        "auroc",
        (
            ("auroc", ("auroc_ci_low", "auroc_ci_high")),
            ("auprc", ("auprc_ci_low", "auprc_ci_high")),
            ("sensitivity", ("sensitivity_ci_low", "sensitivity_ci_high")),
            ("specificity", ("specificity_ci_low", "specificity_ci_high")),
            ("ppv", ("ppv_ci_low", "ppv_ci_high")),
            ("npv", ("npv_ci_low", "npv_ci_high")),
        ),
    ),
    "calibration": (
        "calibration_intercept",
        (
            ("calibration_intercept", ("intercept_ci_low", "intercept_ci_high")),
            ("calibration_slope", ("slope_ci_low", "slope_ci_high")),
            ("brier", ("brier_ci_low", "brier_ci_high")),
        ),
    ),
    "decision_curve": (
        "net_benefit",
        (
            ("net_benefit", ("net_benefit_ci_low", "net_benefit_ci_high")),
            (
                "standardized_net_benefit",
                (
                    "standardized_net_benefit_ci_low",
                    "standardized_net_benefit_ci_high",
                ),
            ),
        ),
    ),
}

# Used for rows that carry no bootstrap distribution (e.g. gap rows on the
# frozen path). Estimability is *undefined* there, not zero.
_EMPTY_ESTIMABILITY: dict[str, Any] = {
    "n_replicates": float("nan"),
    "n_estimable": float("nan"),
    "proportion_nonestimable": float("nan"),
    "not_estimable": False,
}


def _floor_rule_text(floor_fail: bool, boot_flag: bool, threshold: float) -> str:
    """Say which estimability rule fired, in words a reader can check."""
    if floor_fail and boot_flag:
        return f"below its event floor, and >{threshold:.0%} of replicates non-estimable"
    if floor_fail:
        return "below its pre-registered event floor"
    if boot_flag:
        return f">{threshold:.0%} of bootstrap replicates non-estimable"
    return "reportable"


@dataclass(frozen=True)
class BootstrapCell:
    """Per-replicate values for one (attribute, subgroup, metric) cell.

    ``values`` always has length ``n_bootstrap``, with ``nan`` at replicates
    where the metric was not estimable. Replicates are never dropped: a metric
    that is undefined in 40% of replicates is a fact about the data that must
    survive into the report, not a silent reduction in sample size.

    Attributes
    ----------
    attribute : str
        Sensitive attribute name.
    subgroup : str
        Subgroup label, ``"Overall"``, or ``"GAP"``.
    metric : str
        Metric name; for ``"GAP"`` cells this is the gap statistic name.
    values : ndarray
        Shape ``(n_bootstrap,)``, ``nan`` where non-estimable.
    """

    attribute: str
    subgroup: str
    metric: str
    values: npt.NDArray[np.floating[Any]]

    @property
    def n_replicates(self) -> int:
        """Total replicates drawn, including non-estimable ones."""
        return int(self.values.size)

    @property
    def n_estimable(self) -> int:
        """Replicates in which the metric was defined."""
        return int(np.count_nonzero(~np.isnan(self.values)))

    @property
    def proportion_nonestimable(self) -> float:
        """Fraction of replicates in which the metric was undefined."""
        if self.n_replicates == 0:
            return float("nan")
        return 1.0 - self.n_estimable / self.n_replicates

    def not_estimable(self, threshold: float) -> bool:
        """Whether the non-estimable proportion exceeds *threshold*."""
        proportion = self.proportion_nonestimable
        return bool(proportion == proportion and proportion > threshold)

_CALIBRATION_CI_METRICS = ["calibration_intercept", "calibration_slope", "brier"]

_CALIBRATION_CI_COLUMNS: dict[str, tuple[str, str]] = {
    "calibration_intercept": ("intercept_ci_low", "intercept_ci_high"),
    "calibration_slope": ("slope_ci_low", "slope_ci_high"),
    "brier": ("brier_ci_low", "brier_ci_high"),
}


def _safe_logit(
    y_proba: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Logit transform with clipping to avoid -inf/+inf."""
    clipped: npt.NDArray[np.floating[Any]] = np.clip(y_proba, _LOGIT_EPS, 1 - _LOGIT_EPS)
    result: npt.NDArray[np.floating[Any]] = np.asarray(logit(clipped), dtype=np.float64)
    return result


def _fit_calibration_intercept(
    y_true: npt.NDArray[np.floating[Any]],
    lp: npt.NDArray[np.floating[Any]],
) -> float:
    """Calibration intercept: logistic regression with slope fixed to 1.

    Steyerberg 2009 convention. The intercept *a* satisfies
    logit(P(Y=1)) = a + logit(p). A perfectly calibrated model has a = 0.
    """

    def neg_ll(a: float) -> float:
        eta = a + lp
        return -float(np.sum(y_true * eta - np.logaddexp(0, eta)))

    result = minimize_scalar(neg_ll, bounds=(-20, 20), method="bounded")
    return float(result.x)


def _fit_calibration_slope(
    y_true: npt.NDArray[np.floating[Any]],
    lp: npt.NDArray[np.floating[Any]],
) -> float:
    """Calibration slope: logistic regression with free intercept and slope.

    Steyerberg 2009 convention. The slope *b* satisfies
    logit(P(Y=1)) = a + b * logit(p). A perfectly calibrated model has b = 1.
    Over-confident predictions yield b < 1; under-confident yield b > 1.
    """

    def neg_ll(params: npt.NDArray[np.floating[Any]]) -> float:
        a, b = float(params[0]), float(params[1])
        eta = a + b * lp
        return -float(np.sum(y_true * eta - np.logaddexp(0, eta)))

    result = minimize(neg_ll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    return float(result.x[1])


def _compute_ece(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    n_bins: int,
) -> float:
    """Expected calibration error with equal-frequency (quantile-based) bins.

    Bins are formed by sorting predicted probabilities and splitting into
    ``n_bins`` groups of approximately equal size. ECE is the weighted mean
    absolute difference between predicted and observed event rates across bins.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    n_bins_actual = min(n_bins, n)
    sorted_indices = np.argsort(y_proba)
    ece = 0.0
    for bin_indices in np.array_split(sorted_indices, n_bins_actual):
        if len(bin_indices) == 0:
            continue
        avg_pred = float(y_proba[bin_indices].mean())
        avg_true = float(y_true[bin_indices].mean())
        ece += len(bin_indices) * abs(avg_pred - avg_true)
    return ece / n


def _loess_smooth_at_data(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """LOESS-smoothed observed probabilities at data points.

    Uses ``statsmodels.nonparametric.smoothers_lowess.lowess`` with
    ``frac=2/3`` (default bandwidth) and ``it=0`` (no robustifying
    iterations — essential for binary outcomes where positive events
    would otherwise be downweighted as outliers). Returns smoothed
    values at the original ``y_proba`` positions.

    A stratum whose predictions are all equal gives LOWESS a zero bandwidth;
    its divide-by-zero warning is silenced, and the value it returns is
    unchanged.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = lowess(y_true, y_proba, frac=2 / 3, it=0, return_sorted=False)
    return np.asarray(result, dtype=np.float64)


def _loess_curve_at_grid(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    eval_points: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """LOESS-smoothed calibration curve evaluated at grid points.

    Fits LOESS at the data points with ``it=0`` (no robustifying
    iterations — essential for binary outcomes), then linearly
    interpolates to the evaluation grid.
    """
    if len(y_true) < 3:
        return np.full(len(eval_points), float("nan"))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = lowess(y_true, y_proba, frac=2 / 3, it=0, return_sorted=True)
    smoothed: npt.NDArray[np.floating[Any]] = np.interp(eval_points, result[:, 0], result[:, 1])
    return np.clip(smoothed, 0.0, 1.0)


def _roc_tpr_at_grid(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    fpr_grid: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """ROC curve evaluated at a fixed grid of false-positive rates.

    Vertical averaging (Fawcett 2006): the true-positive rate is read off at
    each requested false-positive rate, so curves from different subgroups and
    from different bootstrap resamples share an x-axis and can be averaged.

    Two details matter for correctness. ``roc_curve`` returns ``fpr``
    non-decreasing but *not* strictly increasing — tied scores produce vertical
    runs — and ``np.interp`` requires increasing ``xp``; with duplicates it does
    not raise but silently picks a searchsorted-dependent branch of the step
    function. Duplicate false-positive rates are therefore collapsed first,
    keeping the largest true-positive rate at each, which is the upper boundary
    of the achievable set. Interpolation between the retained points is linear
    because ``roc_auc_score`` integrates with the trapezoid rule; step
    interpolation would make the area under the drawn curve disagree with the
    AUROC reported beside it.
    """
    n_events = int(y_true.sum())
    if n_events == 0 or n_events == len(y_true):
        return np.full(len(fpr_grid), float("nan"))

    fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
    unique_fpr = np.unique(fpr)
    # `tpr` is non-decreasing, so the last index of each run holds its maximum.
    last_of_run = np.searchsorted(fpr, unique_fpr, side="right") - 1
    interpolated: npt.NDArray[np.floating[Any]] = np.interp(
        fpr_grid, unique_fpr, tpr[last_of_run]
    )
    return interpolated


def _compute_ici(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
) -> float:
    """Integrated calibration index (Austin & Steyerberg 2019).

    ICI = mean(|LOESS(y_true | p) - p|) evaluated at each observed p.
    """
    if len(y_true) < 3:
        return float("nan")
    smoothed = _loess_smooth_at_data(y_true, y_proba)
    return float(np.mean(np.abs(smoothed - y_proba)))


def _compute_calibration_boot_metrics(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compute calibration metrics that get bootstrap CIs (intercept, slope, Brier)."""
    n = len(y_true)
    n_events = int(y_true.sum())
    brier = float(np.mean((y_true - y_proba) ** 2))

    if n_events == 0 or n_events == n or n < 3:
        return {
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
            "brier": brier,
        }

    lp = _safe_logit(y_proba)

    # Constant logit predictions → slope undefined
    if np.ptp(lp) < 1e-8:
        return {
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
            "brier": brier,
        }

    return {
        "calibration_intercept": _fit_calibration_intercept(y_true, lp),
        "calibration_slope": _fit_calibration_slope(y_true, lp),
        "brier": brier,
    }


# ---------------------------------------------------------------------------
# Per-subgroup metric computation — decision curve
# ---------------------------------------------------------------------------


def _compute_net_benefits(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    thresholds: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Net benefit at each threshold: TP/N - FP/N * pt/(1-pt).

    Used for bootstrap resamples where calling ``dcurves.dca`` would be
    unnecessarily expensive.
    """
    n = len(y_true)
    nbs: npt.NDArray[np.floating[Any]] = np.empty(len(thresholds), dtype=np.float64)
    for i, pt in enumerate(thresholds):
        pred_pos = y_proba >= pt
        tp = float(np.sum((y_true == 1.0) & pred_pos))
        fp = float(np.sum((y_true == 0.0) & pred_pos))
        nbs[i] = tp / n - fp / n * float(pt) / (1.0 - float(pt))
    return nbs


# ---------------------------------------------------------------------------
# Bootstrap CI helpers (shared infrastructure)
# ---------------------------------------------------------------------------


def _zero_crossings(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    t_lo: float,
    t_hi: float,
) -> list[float]:
    """Every threshold in ``[t_lo, t_hi]`` at which net benefit changes sign.

    Solved exactly rather than scanned. Between two adjacent prediction atoms
    the flagged set is constant, so net benefit is the smooth function
    ``TP/n - (FP/n) * t/(1-t)``, which is zero at ``t* = TP/(TP + FP)``. Under
    the ``p >= t`` convention the flagged set is constant on the half-open
    interval ``(a, b]``, so a root counts when ``a < t* <= b``.

    The flagged set *shrinks* discontinuously as ``t`` passes each atom, so net
    benefit also jumps there. Those jumps are separate sign-change
    opportunities and are tested alongside the smooth roots — including at
    ``t_lo`` itself, which is only ever a segment's lower edge and so is
    reachable by no segment loop.

    A root that merely *touches* zero is not a crossing: the sign is tested on
    both sides rather than at the root, so a curve tangent to zero is not
    reported as changing sign.
    """
    if len(y_true) == 0 or t_hi <= t_lo:
        return []

    edges = sorted(
        {float(t_lo), float(t_hi)}
        | {float(a) for a in np.unique(y_proba) if t_lo < a < t_hi}
    )
    eps = 1e-9
    candidates: list[float] = []
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

    def nb_at(t: float) -> float:
        return float(_compute_net_benefits(y_true, y_proba, np.array([t]))[0])

    found: list[float] = []
    for t in candidates:
        before, after = nb_at(t - eps), nb_at(t + eps)
        if np.isnan(before) or np.isnan(after):
            continue
        if np.sign(before) * np.sign(after) >= 0:
            continue
        found.append(t)
    return sorted(set(found))


def _compute_percentile_cis(
    boot_results: dict[str, list[float]],
    metric_names: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute 2.5/97.5 percentile CIs from bootstrap distributions."""
    cis: dict[str, tuple[float, float]] = {}
    for m in metric_names:
        vals = np.array(boot_results[m])
        if np.all(np.isnan(vals)):
            cis[m] = (float("nan"), float("nan"))
        else:
            cis[m] = (
                float(np.nanpercentile(vals, 2.5)),
                float(np.nanpercentile(vals, 97.5)),
            )
    return cis


def _replicate_rng_for(
    entropy: int,
    stream: int,
    attr_id: int,
    replicate: int,
) -> np.random.Generator:
    """Generator for one addressable replicate.

    ``SeedSequence(entropy, spawn_key=...)`` is the construction
    :meth:`numpy.random.SeedSequence.spawn` uses, but *addressable*: any
    replicate can be regenerated on demand without materialising the ones
    before it. Results therefore depend only on
    ``(entropy, stream, attr_id, replicate)`` — not on call order, worker
    count, chunk size or scheduling — which is what makes the patient path
    both reproducible and safe to parallelise.

    Module-level rather than a method so worker processes can call it without
    pickling the audit object.
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=entropy, spawn_key=(stream, attr_id, replicate))
    )


def _metrics_for_kind(
    kind: str,
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    threshold: float,
) -> dict[str, float]:
    """Dispatch to the metric computation for one axis.

    A string rather than a callable so nothing has to be pickled across the
    process boundary — closures and bound methods are not picklable, and a
    ``functools.partial`` of a private function is fragile to refactor.
    """
    if kind == "discrimination":
        return _compute_subgroup_metrics(y_true, y_proba, threshold)
    return _compute_calibration_boot_metrics(y_true, y_proba)


def _replicate_chunk_subgroups(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    strata: tuple[npt.NDArray[np.intp], ...],
    offsets: npt.NDArray[np.intp],
    entropy: int,
    attr_id: int,
    b_start: int,
    b_stop: int,
    kind: str,
    metrics: list[str],
    threshold: float,
) -> npt.NDArray[np.floating[Any]]:
    """Compute ``(n_subgroups, n_metrics, chunk)`` for a block of replicates.

    One task evaluates *every* subgroup of one attribute for a contiguous
    block of replicates, so the shared-replicate invariant holds inside the
    worker: a replicate's index array is built once and all subgroups read
    their slice of it.
    """
    out = np.full((len(strata), len(metrics), b_stop - b_start), float("nan"), dtype=np.float64)
    for offset, b in enumerate(range(b_start, b_stop)):
        rng = _replicate_rng_for(entropy, _STREAM_ATTRIBUTE, attr_id, b)
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        for k in range(len(strata)):
            values = _metrics_for_kind(
                kind, y_true[idx[offsets[k] : offsets[k + 1]]],
                y_proba[idx[offsets[k] : offsets[k + 1]]], threshold
            )
            for j, m in enumerate(metrics):
                out[k, j, offset] = values[m]
    return out


def _replicate_chunk_overall(
    y_true: npt.NDArray[np.floating[Any]],
    y_proba: npt.NDArray[np.floating[Any]],
    entropy: int,
    b_start: int,
    b_stop: int,
    kind: str,
    metrics: list[str],
    threshold: float,
) -> npt.NDArray[np.floating[Any]]:
    """Compute ``(n_metrics, chunk)`` of whole-cohort replicate values."""
    n = len(y_true)
    out = np.full((len(metrics), b_stop - b_start), float("nan"), dtype=np.float64)
    for offset, b in enumerate(range(b_start, b_stop)):
        rng = _replicate_rng_for(entropy, _STREAM_OVERALL, 0, b)
        idx = rng.integers(0, n, n)
        values = _metrics_for_kind(kind, y_true[idx], y_proba[idx], threshold)
        for j, m in enumerate(metrics):
            out[j, offset] = values[m]
    return out


def _chunk_bounds(n_replicates: int, n_jobs: int | None) -> list[tuple[int, int]]:
    """Split ``range(n_replicates)`` into contiguous blocks.

    A pure function of ``(n_replicates, n_jobs)``, so the partition is
    inspectable and testable. Results must not depend on it — that is asserted
    by the determinism tests — but a deterministic partition keeps the work
    distribution reproducible too.
    """
    if n_jobs is None or n_jobs in (0, 1):
        return [(0, n_replicates)]
    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    n_chunks = max(1, min(n_replicates, 4 * workers))
    edges = np.linspace(0, n_replicates, n_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges) if b > a]


def _run_chunks(worker: Any, tasks: list[Any], n_jobs: int | None) -> list[Any]:
    """Run chunk tasks, in parallel when *n_jobs* asks for it.

    ``n_jobs=None`` bypasses joblib entirely — no process pool, no loky import,
    zero overhead on the library default path.
    """
    if n_jobs in (None, 0, 1) or len(tasks) == 1:
        return [worker(*t) for t in tasks]

    from joblib import Parallel, delayed, parallel_backend

    # Workers are numpy-bound; without this, N processes x M BLAS threads
    # oversubscribes the machine and runs slower than serial.
    with parallel_backend("loky", inner_max_num_threads=1):
        return list(Parallel(n_jobs=n_jobs)(delayed(worker)(*t) for t in tasks))


def _percentile_ci(values: npt.NDArray[np.floating[Any]]) -> tuple[float, float]:
    """2.5/97.5 percentile interval over the estimable replicates only.

    Replicates where the metric was undefined are ``nan`` and are excluded
    here. That makes the interval *conditional on estimability*: if 37% of
    replicates lacked an event, the surviving 63% all had one, and the
    resulting interval is narrower and shifted relative to the marginal one.
    It is the only defensible option short of refusing to report, but it is
    why ``proportion_nonestimable`` must be read alongside every interval.
    """
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return (float("nan"), float("nan"))
    return (
        float(np.nanpercentile(values, 2.5)),
        float(np.nanpercentile(values, 97.5)),
    )


def _range_over_subgroups(
    matrix: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Max-minus-min across subgroups, per replicate.

    Parameters
    ----------
    matrix : ndarray of shape (n_subgroups, n_replicates)
        Per-subgroup metric values from the **same** replicates.

    Returns
    -------
    ndarray of shape (n_replicates,)
        ``nan`` wherever fewer than two subgroups were estimable — a range
        over one value is not a gap.

    Notes
    -----
    For two subgroups this equals ``|A - B|`` elementwise, which is what makes
    the gap distribution derivable from the stored subgroup distributions.
    """
    n_estimable = np.count_nonzero(~np.isnan(matrix), axis=0)
    out = np.full(matrix.shape[1], float("nan"))
    usable = n_estimable >= 2
    if usable.any():
        out[usable] = np.nanmax(matrix[:, usable], axis=0) - np.nanmin(
            matrix[:, usable], axis=0
        )
    return out


def _validate_gap_ci(
    point: float,
    ci_low: float,
    ci_high: float,
) -> tuple[float, float]:
    """Suppress a bootstrap CI when the point estimate falls outside it.

    Gap statistics (max-pairwise range) are computed on the original data
    while the CI comes from bootstrap percentiles of resampled data.  For
    highly skewed distributions the two can diverge, producing a CI that
    does not contain the point estimate.  In that case the CI is unreliable
    and is suppressed (returned as NaN, NaN).
    """
    if np.isnan(point) or np.isnan(ci_low) or np.isnan(ci_high):
        return (ci_low, ci_high)
    if ci_low <= point <= ci_high:
        return (ci_low, ci_high)
    return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# FairnessAudit
# ---------------------------------------------------------------------------


class FairnessAudit:
    """Subgroup-stratified fairness audit for a binary clinical prediction model.

    No computation is done in ``__init__``; work happens lazily when an
    analysis method (e.g. ``discrimination()``) is called. Results are
    cached on the instance so repeated calls are free.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities in [0, 1].
    sensitive_features : dict[str | tuple[str, ...], array-like | None]
        Mapping from attribute name (or tuple of attribute names for
        intersectional analysis) to group labels, each of length
        ``n_samples``. Tuple keys define interaction terms — their
        constituent marginals must also be present as string keys.
        A ``None`` value for a tuple key means labels are auto-computed
        from the marginals.
    threshold : float
        Decision threshold in (0, 1) for converting probabilities to
        binary predictions.
    n_bootstrap : int, optional
        Number of bootstrap resamples for confidence intervals.
        Default is 2000.
    random_state : int | None, optional
        Seed for the bootstrap RNG. ``None`` means non-deterministic.
    ece_bins : int, optional
        Number of equal-frequency bins for ECE computation. Default is 10.
    intersectional_min_events : int, optional
        Minimum number of events in an interaction cell before empirical
        Bayes shrinkage is applied. Default is 50.
    dca_ylim : tuple[float | None, float | None] or None, optional
        Y-axis limits for decision curve plots in the report.
        ``None`` (default) auto-scales to ``[-0.01, max(NB) * 1.1]``.
    resample_scheme : {"case_control", "patient"}, keyword-only, optional
        How bootstrap replicates are drawn within each subgroup.

        - ``"case_control"`` resamples positives and
          negatives separately at fixed sizes, so every replicate carries the
          subgroup's observed event count. This estimates uncertainty
          *conditional on the observed subgroup prevalence*. It is defensible
          for rank-based metrics (AUROC, AUPRC) and **understates uncertainty
          for every prevalence-dependent metric** — calibration intercept and
          slope, Brier, PPV, NPV, prevalence, standardized net benefit. The
          code path is frozen so the published analysis stays reproducible.
        - ``"patient"`` (default) resamples patients within each subgroup stratum,
          preserving the subgroup's ``n`` while letting its event count vary.
          Every subgroup of an attribute shares one replicate index array, so
          gap statistics are reductions over the stored subgroup
          distributions rather than a second, independent resample, and the
          whole-cohort row is computed once and reused across attributes.

        See :meth:`_legacy_case_control_indices` for why the frozen path must
        not be refactored.

        Note that ``"case_control"`` structurally cannot produce a replicate
        with zero events: it pins the event count at its observed value, so a
        subgroup with one event has exactly one in every replicate. The
        estimability columns therefore always read 0% under that scheme.
    nonestimable_threshold : float, keyword-only, optional
        Proportion of non-estimable replicates above which a cell is flagged
        ``not_estimable``. Default 0.10. The flag is a *reporting* threshold,
        not a validity one: percentile CIs are computed over estimable
        replicates only, so an interval is conditional on estimability — and
        therefore narrower than the marginal interval — from the very first
        non-estimable replicate, not from 10%.
        A tuple ``(lo, hi)`` sets explicit limits; use ``None`` within
        the tuple for no bound on that side.
    event_floors : dict[str, int] | None, keyword-only, optional
        Minimum **event** counts required before a metric is reported
        inferentially, keyed by rule name (``"auroc"``,
        ``"calibration_intercept"``, ``"calibration_slope"``,
        ``"net_benefit"``). ``None`` (default) uses
        :data:`~isitfair.estimability.DEFAULT_EVENT_FLOORS`; ``{}`` disables
        the rule entirely, which is how the frozen published analyses pin
        their original behaviour.

        The floor is applied as a **conservative union** with the
        replicate-based *nonestimable_threshold* rule: a cell below its floor
        is suppressed even when every replicate estimated cleanly, because the
        floor binds on the observed event count regardless of how the
        replicates happened to fall. Net benefit has no replicate-based
        signal at all under the patient scheme, so the floor is the only rule
        available to it.

        Suppression **withholds intervals, never point estimates** — a
        suppressed point is still a real quantity that was computed. A gap row
        inherits the floor failure of any contributing subgroup, because a gap
        is exactly as estimable as its worst constituent.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shape, out-of-range values,
        length mismatch, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    >>> sex = np.array(["M", "F", "M", "F", "M", "F"])
    >>> audit = FairnessAudit(
    ...     y_true=y, y_proba=p,
    ...     sensitive_features={"sex": sex},
    ...     threshold=0.5, n_bootstrap=50, random_state=0,
    ... )
    >>> df = audit.discrimination()
    >>> df.columns.tolist()[:5]
    ['attribute', 'subgroup', 'shrunk', 'n', 'n_events']
    """

    def __init__(
        self,
        y_true: Any,
        y_proba: Any,
        sensitive_features: dict[str | tuple[str, ...], Any],
        threshold: float,
        n_bootstrap: int = 2000,
        random_state: int | None = None,
        ece_bins: int = 10,
        intersectional_min_events: int = 50,
        dca_ylim: tuple[float | None, float | None] | None = None,
        *,
        resample_scheme: str = "patient",
        nonestimable_threshold: float = 0.10,
        event_floors: dict[str, int] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        _check_index_alignment(
            {
                "y_true": y_true,
                "y_proba": y_proba,
                **{
                    f"sensitive_features[{k!r}]": v
                    for k, v in (sensitive_features or {}).items()
                },
            }
        )

        # --- coerce & validate y_true ---
        self._y_true = _to_1d_float_array(y_true, "y_true")
        n_missing = int(np.isnan(self._y_true).sum())
        if n_missing:
            msg = (
                f"`y_true` has {n_missing} missing value(s); drop those rows from "
                "every input before auditing"
            )
            raise ValueError(msg)
        unique_labels = set(np.unique(self._y_true))
        if not unique_labels.issubset({0.0, 1.0}):
            msg = f"`y_true` must contain only 0 and 1, got unique values {sorted(unique_labels)}"
            raise ValueError(msg)

        # --- coerce & validate y_proba ---
        self._y_proba = _to_1d_float_array(y_proba, "y_proba")
        n_missing = int(np.isnan(self._y_proba).sum())
        if n_missing:
            msg = (
                f"`y_proba` has {n_missing} missing value(s); drop those rows from "
                "every input before auditing"
            )
            raise ValueError(msg)
        n_outside = int(((self._y_proba < 0.0) | (self._y_proba > 1.0)).sum())
        if n_outside:
            msg = (
                f"`y_proba` values must be in [0, 1]; {n_outside} value(s) fall "
                f"outside (range {self._y_proba.min():.4g} to {self._y_proba.max():.4g}). "
                "Pass predicted probabilities, not logits or scores."
            )
            raise ValueError(msg)
        if set(np.unique(self._y_proba)).issubset({0.0, 1.0}):
            warnings.warn(
                "`y_proba` contains only 0 and 1. isitfair evaluates predicted "
                "probabilities; hard class labels make calibration and decision "
                "curves meaningless. Pass `model.predict_proba(X)[:, 1]`.",
                UserWarning,
                stacklevel=2,
            )

        n = len(self._y_true)
        if len(self._y_proba) != n:
            msg = (
                f"`y_true` and `y_proba` must have the same length, "
                f"got {n} and {len(self._y_proba)}"
            )
            raise ValueError(msg)

        # --- validate threshold ---
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"`threshold` must be in (0, 1), got {threshold}")
        self._threshold = float(threshold)

        # --- coerce & validate sensitive_features ---
        if not sensitive_features:
            raise ValueError("`sensitive_features` must be a non-empty dict")

        # Two-pass processing: marginals first, then interactions
        self._sensitive: dict[str, npt.NDArray[Any]] = {}
        self._interactions: dict[str, tuple[str, ...]] = {}
        # interaction name -> {cell label -> its marginal labels, in key order}.
        # Shrinkage looks the marginal up here rather than re-parsing the cell
        # label, which breaks on labels that contain the " / " separator and
        # on user-supplied interaction arrays.
        self._cell_constituents: dict[str, dict[str, tuple[str, ...]]] = {}

        # Pass 1: register string-keyed (marginal) attributes
        for key, attr_values in sensitive_features.items():
            if isinstance(key, str):
                arr = _to_1d_array(attr_values, f"sensitive_features['{key}']")
                if len(arr) != n:
                    msg = f"sensitive feature '{key}' has length {len(arr)}, expected {n}"
                    raise ValueError(msg)
                n_levels = len(np.unique(arr))
                if arr.dtype.kind in "iuf" and n_levels > 20:
                    warnings.warn(
                        f"sensitive feature '{key}' is numeric with {n_levels} distinct "
                        "values, so every value becomes its own subgroup. Bin "
                        "continuous variables first, e.g. `pd.cut(age, [0, 65, 80, 120])`.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._sensitive[key] = arr

        # Pass 2: resolve tuple-keyed (interaction) attributes
        for key, attr_values in sensitive_features.items():
            if isinstance(key, tuple):
                # Validate all constituents present as marginals
                for constituent in key:
                    if constituent not in self._sensitive:
                        msg = (
                            f"interaction key {key!r} references marginal "
                            f"'{constituent}' which is not present as a "
                            f"string key in sensitive_features"
                        )
                        raise ValueError(msg)
                display_name = _interaction_attr_name(key)
                if attr_values is None:
                    # Auto-compute from marginals
                    marginal_arrays = [self._sensitive[c] for c in key]
                    arr = _build_interaction_labels(marginal_arrays)
                else:
                    arr = _to_1d_array(
                        attr_values,
                        f"sensitive_features[{key!r}]",
                    )
                    if len(arr) != n:
                        msg = f"sensitive feature {key!r} has length {len(arr)}, expected {n}"
                        raise ValueError(msg)
                self._cell_constituents[display_name] = self._map_cells_to_marginals(
                    key, arr
                )
                self._sensitive[display_name] = arr
                self._interactions[display_name] = key

        # --- bootstrap config ---
        if n_bootstrap < 1:
            raise ValueError(f"`n_bootstrap` must be >= 1, got {n_bootstrap}")
        self._n_bootstrap = n_bootstrap
        self._random_state = random_state

        if resample_scheme not in _RESAMPLE_SCHEMES:
            msg = (
                f"`resample_scheme` must be one of {sorted(_RESAMPLE_SCHEMES)}, "
                f"got {resample_scheme!r}"
            )
            raise ValueError(msg)
        self._resample_scheme = resample_scheme

        if not 0.0 <= nonestimable_threshold <= 1.0:
            msg = (
                "`nonestimable_threshold` must be in [0, 1], "
                f"got {nonestimable_threshold}"
            )
            raise ValueError(msg)
        self._nonestimable_threshold = float(nonestimable_threshold)

        # Validated here rather than at first use, so a bad floor is reported
        # against the constructor call that supplied it.
        self._event_floors = resolve_event_floors(event_floors)

        # Materialise the entropy now rather than at first use: with
        # random_state=None every axis of a single audit must still agree on
        # the same replicates, which requires one fixed entropy value.
        self._boot_entropy = np.random.SeedSequence(random_state).entropy

        if n_jobs is not None and n_jobs == 0:
            msg = "`n_jobs` must be a non-zero int or None, got 0"
            raise ValueError(msg)
        self._n_jobs = n_jobs

        # --- ECE config ---
        if ece_bins < 2:
            raise ValueError(f"`ece_bins` must be >= 2, got {ece_bins}")
        self._ece_bins = ece_bins

        # --- intersectional config ---
        self._intersectional_min_events = intersectional_min_events

        # --- DCA plot config ---
        self._dca_ylim = dca_ylim

        # --- result cache ---
        self._cache: dict[str, pd.DataFrame] = {}

        # --- per-replicate bootstrap distributions ---
        # Keyed by axis, then by (attribute, subgroup, metric). Retained so gap
        # statistics can be derived from the same replicates as the subgroup
        # statistics, and so the arithmetic is independently checkable.
        self._boot_store: dict[str, dict[tuple[str, str, str], BootstrapCell]] = {}

        # Per-rung out-of-fold scores from recalibration_ladder(), so the
        # reliability diagrams and the table describe the same partition.
        self._ladder_scores: dict[str, dict[str, npt.NDArray[np.floating[Any]]]] = {}

        # Whole-cohort replicate values, computed once per axis and reused in
        # every attribute section so the overall row is identical everywhere.
        self._overall_boot_cache: dict[str, dict[str, npt.NDArray[np.floating[Any]]]] = {}

        # Stable integer per attribute, so a replicate's identity does not
        # depend on dict iteration order.
        self._attribute_ids = {name: i for i, name in enumerate(self._sensitive)}

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{name} ({len(np.unique(values))})" for name, values in self._sensitive.items()
        )
        return (
            f"FairnessAudit(n={len(self._y_true)}, events={int(self._y_true.sum())}, "
            f"threshold={self._threshold}, n_bootstrap={self._n_bootstrap}, "
            f"random_state={self._random_state}, attributes: {attrs})"
        )

    # ------------------------------------------------------------------
    # Bootstrap infrastructure (shared across axes)
    # ------------------------------------------------------------------

    def _stratified_bootstrap_indices(
        self,
        y_true_sub: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp] | None]:
        """Generate bootstrap index sets for the configured resampling scheme.

        Used by the curve axes and the decision curve, which resample one
        slice at a time. The scalar axes take the shared-index route in
        :meth:`_patient_axis_cis` instead, because only they have gap
        statistics that must reduce over the same replicates.
        """
        if self._resample_scheme == "case_control":
            return self._legacy_case_control_indices(y_true_sub, rng)
        return self._patient_slice_indices(y_true_sub, rng)

    def _patient_slice_indices(
        self,
        y_true_sub: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp] | None]:
        """Patient-level bootstrap indices for one slice.

        Resamples patients with replacement, preserving the slice's size while
        letting its event count vary. Unlike the frozen path this yields
        replicates in which the slice has no events at all; those are returned
        rather than suppressed, and the metric functions downstream mark them
        non-estimable.
        """
        n = len(y_true_sub)
        if n == 0:
            return [None] * self._n_bootstrap
        return [rng.integers(0, n, n) for _ in range(self._n_bootstrap)]

    def _legacy_case_control_indices(
        self,
        y_true_sub: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp] | None]:
        """Outcome-stratified bootstrap indices. **FROZEN — do not modify.**

        Positives and negatives are resampled separately so that each bootstrap
        sample preserves the event count of the subgroup (Naderalvojoud 2025
        convention).

        This is the scheme used for the originally submitted analysis. It is
        retained unchanged so those numbers stay exactly reproducible.

        Fixing the event count is the scheme's limitation: it estimates
        uncertainty *conditional on the observed subgroup prevalence*, which
        removes the dominant variance component for every prevalence-dependent
        metric. Prefer ``resample_scheme="patient"`` for new work.

        Any change here — including the order in which draws are taken from
        *rng* — breaks the reproduction guarantee. Monte Carlo noise alone moves
        CI endpoints by 5-20x more than a 3-decimal-place tolerance allows, so a
        "harmless" refactor is not detectable as harmless.
        """
        pos_idx = np.where(y_true_sub == 1.0)[0]
        neg_idx = np.where(y_true_sub == 0.0)[0]
        indices: list[npt.NDArray[np.intp] | None] = []
        for _ in range(self._n_bootstrap):
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                indices.append(None)
                continue
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            indices.append(np.concatenate([boot_pos, boot_neg]))
        return indices

    def _bootstrap_subgroup_metrics(
        self,
        y_true_sub: npt.NDArray[np.floating[Any]],
        y_proba_sub: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
        store_as: tuple[str, str, str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Run stratified bootstrap on a single subgroup, return CI dict.

        Returns mapping ``metric_name -> (ci_low, ci_high)`` for the
        percentile 2.5/97.5 interval.

        ``store_as`` is an optional ``(axis, attribute, subgroup)`` key; when
        given, the per-replicate arrays this function already builds are
        recorded in :attr:`_boot_store`. Recording is a pure side effect — it
        draws nothing from *rng* — so the frozen path stays byte-identical.
        """
        boot_indices = self._stratified_bootstrap_indices(y_true_sub, rng)
        boot_results: dict[str, list[float]] = {m: [] for m in _CI_METRICS}
        for idx in boot_indices:
            if idx is None:
                for m in _CI_METRICS:
                    boot_results[m].append(float("nan"))
                continue
            metrics = _compute_subgroup_metrics(y_true_sub[idx], y_proba_sub[idx], self._threshold)
            for m in _CI_METRICS:
                boot_results[m].append(metrics[m])
        if store_as is not None:
            self._record_replicates(store_as, boot_results)
        return _compute_percentile_cis(boot_results, _CI_METRICS)

    def _bootstrap_calibration_metrics(
        self,
        y_true_sub: npt.NDArray[np.floating[Any]],
        y_proba_sub: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
        store_as: tuple[str, str, str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Run stratified bootstrap for calibration CIs (intercept, slope, Brier).

        See :meth:`_bootstrap_subgroup_metrics` for ``store_as``.
        """
        boot_indices = self._stratified_bootstrap_indices(y_true_sub, rng)
        boot_results: dict[str, list[float]] = {m: [] for m in _CALIBRATION_CI_METRICS}
        for idx in boot_indices:
            if idx is None:
                for m in _CALIBRATION_CI_METRICS:
                    boot_results[m].append(float("nan"))
                continue
            metrics = _compute_calibration_boot_metrics(y_true_sub[idx], y_proba_sub[idx])
            for m in _CALIBRATION_CI_METRICS:
                boot_results[m].append(metrics[m])
        if store_as is not None:
            self._record_replicates(store_as, boot_results)
        return _compute_percentile_cis(boot_results, _CALIBRATION_CI_METRICS)

    # ------------------------------------------------------------------
    # Shared replicate indices (patient scheme)
    # ------------------------------------------------------------------

    def _replicate_rng(self, stream: int, attr_id: int, replicate: int) -> np.random.Generator:
        """Return the generator for one addressable replicate.

        ``SeedSequence(entropy, spawn_key=...)`` is the construction
        :meth:`numpy.random.SeedSequence.spawn` uses, but *addressable*: any
        replicate can be regenerated on demand without materialising the ones
        before it. Results therefore depend only on
        ``(entropy, stream, attr_id, replicate)`` — not on call order, worker
        count, chunk size or scheduling. That is what makes the patient path
        both reproducible and safe to parallelise.
        """
        seed_sequence = np.random.SeedSequence(
            entropy=self._boot_entropy,
            spawn_key=(stream, attr_id, replicate),
        )
        return np.random.default_rng(seed_sequence)

    def _patient_indices_for_replicate(
        self,
        strata: tuple[npt.NDArray[np.intp], ...],
        stream: int,
        attr_id: int,
        replicate: int,
    ) -> npt.NDArray[np.intp]:
        """Draw one replicate's indices, resampling patients within each stratum.

        Each subgroup keeps its observed size; only its event count varies.
        That is the fix: the calibration intercept solves
        ``sum(y) = sum(expit(alpha + eta))``, so pinning ``sum(y)`` — as the
        frozen scheme does — makes alpha nearly deterministic and collapses
        its interval.

        Strata are concatenated in a fixed order, so subgroup *k* always
        occupies the same slice of the returned array. No boolean masking is
        needed downstream, and a subgroup cannot vanish from a replicate the
        way it can when the whole cohort is resampled and then split.
        """
        rng = self._replicate_rng(stream, attr_id, replicate)
        return np.concatenate(
            [stratum[rng.integers(0, len(stratum), len(stratum))] for stratum in strata]
        )

    @staticmethod
    def _stratum_offsets(strata: tuple[npt.NDArray[np.intp], ...]) -> npt.NDArray[np.intp]:
        """Cumulative boundaries of each stratum within a replicate index array."""
        return np.cumsum([0, *[len(s) for s in strata]])

    def _patient_replicate_matrix(
        self,
        attr_name: str,
        groups: npt.NDArray[Any],
        attr_values: npt.NDArray[Any],
        metrics: list[str],
        kind: str,
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Compute a ``(n_subgroups, n_metrics, B)`` array from shared indices.

        Every subgroup of the attribute is evaluated on the *same* replicate,
        which is what lets gap statistics be derived from these arrays instead
        of from a second, independent resample. That equality is the structural
        guarantee the frozen path cannot offer.

        Parameters
        ----------
        kind : {"discrimination", "calibration"}
            Which metric set to compute. A string rather than a callable so
            worker processes need not unpickle one.
        """
        strata = tuple(np.where(attr_values == g)[0] for g in groups)
        offsets = self._stratum_offsets(strata)
        attr_id = self._attribute_ids[attr_name]

        tasks = [
            (
                self._y_true,
                self._y_proba,
                strata,
                offsets,
                self._boot_entropy,
                attr_id,
                lo,
                hi,
                kind,
                metrics,
                self._threshold,
            )
            for lo, hi in _chunk_bounds(self._n_bootstrap, self._n_jobs)
        ]
        parts = _run_chunks(_replicate_chunk_subgroups, tasks, self._n_jobs)
        stacked = np.concatenate(parts, axis=-1)
        return {m: stacked[:, j, :] for j, m in enumerate(metrics)}

    def _overall_replicates(
        self,
        metrics: list[str],
        kind: str,
        cache_key: str,
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Compute whole-cohort replicate values **once**, then reuse.

        The overall cohort should carry a single confidence interval. On the
        frozen path the overall bootstrap sits inside the per-attribute loop, so
        the same overall AUROC carries one interval per attribute. Computing it
        once and reusing the identical array makes them equal by construction
        rather than by luck.
        """
        cached = self._overall_boot_cache.get(cache_key)
        if cached is not None:
            return cached

        tasks = [
            (
                self._y_true,
                self._y_proba,
                self._boot_entropy,
                lo,
                hi,
                kind,
                metrics,
                self._threshold,
            )
            for lo, hi in _chunk_bounds(self._n_bootstrap, self._n_jobs)
        ]
        parts = _run_chunks(_replicate_chunk_overall, tasks, self._n_jobs)
        stacked = np.concatenate(parts, axis=-1)
        out = {m: stacked[j, :] for j, m in enumerate(metrics)}

        self._overall_boot_cache[cache_key] = out
        return out

    def _patient_axis_cis(
        self,
        axis: str,
        attr_name: str,
        groups: npt.NDArray[Any],
        attr_values: npt.NDArray[Any],
        metrics: list[str],
        kind: str,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Compute one attribute's CIs under the patient scheme, in one pass.

        Every subgroup and the gap statistics come from the *same* replicate
        index array, and the per-replicate values are stored so the gap is a
        pure reduction over them rather than a second resample.

        Returns
        -------
        dict
            ``{subgroup_label: {metric: (ci_low, ci_high)}}``, including
            ``"Overall"``.
        """
        labels = [str(g) for g in groups]
        matrices = self._patient_replicate_matrix(
            attr_name, groups, attr_values, metrics, kind
        )
        overall = self._overall_replicates(metrics, kind, axis)

        store = self._boot_store.setdefault(axis, {})
        cis: dict[str, dict[str, tuple[float, float]]] = {}

        for k, label in enumerate(labels):
            cis[label] = {}
            for m in metrics:
                values = matrices[m][k]
                store[(attr_name, label, m)] = BootstrapCell(attr_name, label, m, values)
                cis[label][m] = _percentile_ci(values)

        cis["Overall"] = {}
        for m in metrics:
            values = overall[m]
            store[(attr_name, "Overall", m)] = BootstrapCell(attr_name, "Overall", m, values)
            cis["Overall"][m] = _percentile_ci(values)

        return cis

    def _patient_gap_ci(
        self,
        axis: str,
        attr_name: str,
        groups: npt.NDArray[Any],
        metric: str,
        gap_name: str,
    ) -> tuple[float, float]:
        """Derive a gap CI from the already-stored subgroup replicates.

        This never resamples. The gap distribution is a reduction over the
        subgroup matrices for the same replicates, which is exactly the
        property the frozen path lacks and the reason its gap intervals could
        not be reconciled with their constituents.
        """
        store = self._boot_store.get(axis, {})
        rows = [
            store[(attr_name, str(g), metric)].values
            for g in groups
            if (attr_name, str(g), metric) in store
        ]
        if len(rows) < 2:
            return (float("nan"), float("nan"))
        gap_values = _range_over_subgroups(np.vstack(rows))
        store[(attr_name, "GAP", gap_name)] = BootstrapCell(
            attr_name, "GAP", gap_name, gap_values
        )
        return _percentile_ci(gap_values)

    def _discrimination_gap_cis(
        self,
        attr_name: str,
        groups: npt.NDArray[Any],
    ) -> dict[str, tuple[float, float]]:
        """Percentile CIs for the four threshold-specific parity gaps.

        Every gap is a reduction over the *stored* per-subgroup replicates, so
        it shares the replicate index array with the rates it is built from and
        the dependence between a gap and its constituents is preserved. Nothing
        is resampled here.

        - ``equal_opportunity_difference`` — range over sensitivity.
        - ``predictive_equality`` — range over FPR. A range is invariant under
          ``x -> 1 - x``, so this is the range over specificity.
        - ``equalized_odds_gap`` — the larger of the two, taken *per replicate*
          rather than as the max of the two intervals, which would be wider
          than the composite actually is.
        - ``statistical_parity`` — range over the flag rate, reconstructed from
          sensitivity, specificity and prevalence.

        Returns an empty mapping under the frozen ``case_control`` scheme,
        which stores no per-subgroup matrices to reduce over.
        """
        store = self._boot_store.get("discrimination", {})
        labels = [str(g) for g in groups]

        def stack(metric: str) -> npt.NDArray[np.floating[Any]] | None:
            rows = [
                store[(attr_name, lab, metric)].values
                for lab in labels
                if (attr_name, lab, metric) in store
            ]
            return np.vstack(rows) if len(rows) >= 2 else None

        sens, spec, prev = stack("sensitivity"), stack("specificity"), stack("prevalence")
        if sens is None or spec is None:
            return {}

        eo = _range_over_subgroups(sens)
        pe = _range_over_subgroups(spec)
        eq = np.fmax(eo, pe)
        eq[np.isnan(eo) | np.isnan(pe)] = np.nan

        distributions = {
            "equal_opportunity_difference": eo,
            "predictive_equality": pe,
            "equalized_odds_gap": eq,
        }
        if prev is not None:
            flag_rate = sens * prev + (1.0 - spec) * (1.0 - prev)
            distributions["statistical_parity"] = _range_over_subgroups(flag_rate)

        out: dict[str, tuple[float, float]] = {}
        for gap_name, values in distributions.items():
            store[(attr_name, "GAP", gap_name)] = BootstrapCell(
                attr_name, "GAP", gap_name, values
            )
            out[gap_name] = _percentile_ci(values)
        return out

    def _record_replicates(
        self,
        key: tuple[str, str, str],
        boot_results: dict[str, list[float]],
    ) -> None:
        """Record per-replicate arrays into :attr:`_boot_store`."""
        axis, attribute, subgroup = key
        store = self._boot_store.setdefault(axis, {})
        for metric, values in boot_results.items():
            store[(attribute, subgroup, metric)] = BootstrapCell(
                attribute=attribute,
                subgroup=subgroup,
                metric=metric,
                values=np.asarray(values, dtype=np.float64),
            )

    def _estimability_row(
        self,
        axis: str,
        attribute: str,
        subgroup: str,
        metrics: list[str],
    ) -> dict[str, Any]:
        """Summarise estimability across *metrics* for one row.

        Aggregation is deliberately conservative: ``n_estimable`` is the
        minimum across the row's CI metrics and ``not_estimable`` fires if
        *any* of them exceeds the threshold, because a row is only as
        trustworthy as its least estimable metric. The exact per-metric record
        is in :meth:`bootstrap_diagnostics`.
        """
        cells = [
            cell
            for m in metrics
            if (cell := self._boot_store.get(axis, {}).get((attribute, subgroup, m))) is not None
        ]
        if not cells:
            return {
                "n_replicates": self._n_bootstrap,
                "n_estimable": float("nan"),
                "proportion_nonestimable": float("nan"),
                "not_estimable": False,
            }
        n_estimable = min(c.n_estimable for c in cells)
        n_replicates = cells[0].n_replicates
        proportion = 1.0 - n_estimable / n_replicates if n_replicates else float("nan")
        return {
            "n_replicates": n_replicates,
            "n_estimable": n_estimable,
            "proportion_nonestimable": proportion,
            "not_estimable": any(c.not_estimable(self._nonestimable_threshold) for c in cells),
        }

    # ------------------------------------------------------------------
    # Event floors (see isitfair.estimability)
    # ------------------------------------------------------------------

    def _subgroup_counts(self) -> dict[tuple[str, str], tuple[int, int]]:
        """``(attribute, subgroup) -> (n, n_events)`` for every audited cell.

        Built from the sensitive-feature arrays rather than from a results
        frame, so it is available to axes that carry no ``n_events`` column of
        their own — ``decision_curve()`` is the one that matters, because net
        benefit has no replicate-based estimability signal and the event floor
        is therefore the only rule available to it.
        """
        cached: dict[tuple[str, str], tuple[int, int]] | None = getattr(
            self, "_counts_cache", None
        )
        if cached is not None:
            return cached
        counts: dict[tuple[str, str], tuple[int, int]] = {}
        overall = (len(self._y_true), int(self._y_true.sum()))
        for attr_name, values in self._sensitive.items():
            counts[(attr_name, "Overall")] = overall
            for level in pd.unique(values):
                mask = values == level
                counts[(attr_name, str(level))] = (
                    int(mask.sum()),
                    int(self._y_true[mask].sum()),
                )
        self._counts_cache = counts
        return counts

    def _apply_event_floors(self, df: pd.DataFrame, axis: str) -> pd.DataFrame:
        """Stamp *df* with the floor verdict and blank the intervals it suppresses.

        Three columns are appended: ``event_floor`` and ``below_floor`` for the
        axis's primary metric, and ``not_estimable_final`` — the conservative
        union of the event floor over *every* metric the row carries with the
        replicate rule already in ``not_estimable``.

        Interval columns are blanked per metric, not per row: a stratum with 98
        events keeps its calibration-intercept interval (floor 25) and loses
        its slope interval (floor 100), which is the distinction the row-level
        flag cannot express. **Point estimates are never blanked.**

        Gap rows are resolved in a second pass, because a gap inherits the
        floor failure of its constituent subgroups and those verdicts have to
        exist first.
        """
        primary, spec = _AXIS_FLOOR_SPEC[axis]
        n_rows = len(df)
        out = df.copy()

        floor_primary = self._event_floors.get(primary)
        event_floor = np.full(n_rows, float(floor_primary) if floor_primary else np.nan)
        below = np.zeros(n_rows, dtype=bool)
        final = (
            out["not_estimable"].eq(True).to_numpy(dtype=bool).copy()
            if "not_estimable" in out.columns
            else np.zeros(n_rows, dtype=bool)
        )
        if floor_primary is None:
            event_floor = np.full(n_rows, np.nan)

        subgroups = out["subgroup"].astype(str).to_numpy()
        attributes = out["attribute"].astype(str).to_numpy()
        is_gap = subgroups == "GAP"

        counts = self._subgroup_counts()
        events = np.array(
            [
                np.nan if g else float(counts.get((a, sub), (0, -1))[1])
                for a, sub, g in zip(attributes, subgroups, is_gap, strict=True)
            ]
        )

        # First pass: real subgroups, per metric.
        failed_by_axis: dict[tuple[str, str], set[str]] = {}
        for metric, bounds in spec:
            if metric not in out.columns:
                continue
            rule = FLOOR_FOR.get(metric)
            floor = self._event_floors.get(rule) if rule else None
            if floor is None:
                continue
            fails = (~is_gap) & (events >= 0) & (events < floor)
            if not fails.any():
                continue
            final |= fails
            if metric == primary:
                below |= fails
            for col in bounds:
                if col in out.columns:
                    out.loc[fails, col] = np.nan
            for a, sub in zip(attributes[fails], subgroups[fails], strict=True):
                if sub != "Overall":
                    failed_by_axis.setdefault((a, rule or metric), set()).add(sub)

        # Second pass: a gap is only as estimable as its worst constituent.
        if is_gap.any() and "gap_metric" in out.columns:
            gap_metrics = out["gap_metric"].astype("object").to_numpy()
            for i in np.flatnonzero(is_gap):
                source = GAP_SOURCE.get(str(gap_metrics[i]))
                if source is None:
                    continue
                floor = self._event_floors.get(source)
                if floor is None or not failed_by_axis.get((attributes[i], source)):
                    continue
                event_floor[i] = float(floor)
                below[i] = True
                final[i] = True
                mask = np.zeros(n_rows, dtype=bool)
                mask[i] = True
                for col in ("gap_ci_low", "gap_ci_high"):
                    if col in out.columns:
                        out.loc[mask, col] = np.nan

        out["event_floor"] = event_floor
        out["below_floor"] = below
        out["not_estimable_final"] = final
        return out

    def estimability(self) -> pd.DataFrame:
        """Per-metric reportability verdict for every audited cell.

        The authoritative estimability object, and the one downstream code
        should read rather than re-deriving a floor: a rule restated in five
        places becomes five different rules.

        One row per ``(attribute, subgroup, metric)``, carrying both rules and
        their union. Gap rows are filed under their own metric name
        (``max_intercept_gap``, not ``calibration_intercept``) and inherit the
        floor failure of any constituent subgroup.

        Returns
        -------
        pandas.DataFrame
            Columns ``attribute``, ``subgroup``, ``metric``, ``n``,
            ``n_events``, ``event_floor``, ``below_floor``, ``n_replicates``,
            ``n_estimable``, ``proportion_nonestimable``, ``not_estimable``
            (the replicate rule alone), ``not_estimable_final`` (the union)
            and ``rule`` — a human-readable statement of which rule fired.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.binomial(1, 0.2, 400).astype(float)
        >>> p = rng.uniform(0, 1, 400)
        >>> a = FairnessAudit(y, p, {"g": rng.choice(["x", "y"], 400)},
        ...                   threshold=0.3, n_bootstrap=20, random_state=0)
        >>> est = a.estimability()
        >>> sorted(est["metric"].unique())[:3]
        ['auprc', 'auroc', 'brier']
        """
        cached = self._cache.get("estimability")
        if cached is not None:
            return cached

        diagnostics = pd.concat(
            [self.bootstrap_diagnostics(axis) for axis in _BOOTSTRAP_AXES],
            ignore_index=True,
        )
        counts = self._subgroup_counts()

        # Net benefit carries no replicate distribution, so it appears in no
        # diagnostics frame. Its floor is the only rule it has, and omitting
        # the rows is how the floor came to bind on nothing at all.
        nb_rows = [
            {
                "attribute": attr,
                "subgroup": sub,
                "metric": "net_benefit",
                "n_replicates": float("nan"),
                "n_estimable": float("nan"),
                "proportion_nonestimable": float("nan"),
                "not_estimable": False,
            }
            for (attr, sub) in counts
        ]

        # The decision-curve cache key is threshold-dependent, so gap rows are
        # collected from every cached frame that carries a `gap_metric` column
        # rather than from a fixed list of axis names. Keys already present in
        # `diagnostics` are skipped: a gap with a stored replicate distribution
        # is described there with its real counts, and adding it again here
        # would duplicate the row with empty ones.
        seen: set[tuple[str, str]] = {
            (str(a), str(m))
            for a, sub, m in zip(
                diagnostics["attribute"],
                diagnostics["subgroup"],
                diagnostics["metric"],
                strict=True,
            )
            if str(sub) == "GAP"
        }
        gap_rows: list[dict[str, Any]] = []
        for frame in list(self._cache.values()):
            if not isinstance(frame, pd.DataFrame) or "gap_metric" not in frame.columns:
                continue
            for _, r in frame[frame["subgroup"].astype(str) == "GAP"].iterrows():
                key = (str(r["attribute"]), str(r["gap_metric"]))
                if key in seen or key[1] == "nan":
                    continue
                seen.add(key)
                gap_rows.append(
                    {
                        "attribute": key[0],
                        "subgroup": "GAP",
                        "metric": key[1],
                        "n_replicates": float("nan"),
                        "n_estimable": float("nan"),
                        "proportion_nonestimable": float("nan"),
                        "not_estimable": False,
                    }
                )

        rows = pd.concat(
            [diagnostics, pd.DataFrame(nb_rows), pd.DataFrame(gap_rows)],
            ignore_index=True,
        )

        records: list[dict[str, Any]] = []
        failed_by_axis: dict[tuple[str, str], set[str]] = {}
        for _, r in rows.iterrows():
            attr, sub, metric = (
                str(r["attribute"]),
                str(r["subgroup"]),
                str(r["metric"]),
            )
            n, events = counts.get((attr, sub), (-1, -1))
            rule = FLOOR_FOR.get(metric)
            floor = self._event_floors.get(rule) if rule else None
            fails = bool(
                floor is not None
                and sub != "GAP"
                and events >= 0
                and floor_fails(events, metric, self._event_floors)
            )
            boot = bool(r["not_estimable"])
            if fails and sub != "Overall" and rule:
                failed_by_axis.setdefault((attr, rule), set()).add(sub)
            records.append(
                {
                    "attribute": attr,
                    "subgroup": sub,
                    "metric": metric,
                    "n": float(n) if n >= 0 else float("nan"),
                    "n_events": float(events) if events >= 0 else float("nan"),
                    "event_floor": float(floor) if floor is not None else float("nan"),
                    "below_floor": fails,
                    "n_replicates": r["n_replicates"],
                    "n_estimable": r["n_estimable"],
                    "proportion_nonestimable": r["proportion_nonestimable"],
                    "not_estimable": boot,
                    "not_estimable_final": bool(fails or boot),
                    "rule": _floor_rule_text(fails, boot, self._nonestimable_threshold),
                }
            )

        for rec in records:
            if rec["subgroup"] != "GAP":
                continue
            source = GAP_SOURCE.get(str(rec["metric"]))
            if source is None:
                continue
            failed = sorted(failed_by_axis.get((rec["attribute"], source), set()))
            if not failed:
                continue
            floor = self._event_floors.get(source)
            rec["event_floor"] = float(floor) if floor is not None else float("nan")
            rec["below_floor"] = True
            rec["not_estimable_final"] = True
            rec["rule"] = (
                f"inherits the {source} event floor from its constituent "
                f"subgroups ({', '.join(failed)}), which fail it"
            )

        df = pd.DataFrame(
            records,
            columns=[
                "attribute",
                "subgroup",
                "metric",
                "n",
                "n_events",
                "event_floor",
                "below_floor",
                "n_replicates",
                "n_estimable",
                "proportion_nonestimable",
                "not_estimable",
                "not_estimable_final",
                "rule",
            ],
        ).sort_values(["attribute", "subgroup", "metric"], kind="stable").reset_index(drop=True)
        self._cache["estimability"] = df
        return df

    # ------------------------------------------------------------------
    # Intersectional shrinkage helpers
    # ------------------------------------------------------------------

    def _needs_shrinkage(
        self,
        attr_name: str,
        subgroup_label: str,
        n_events: int,
    ) -> bool:
        """True iff attr is an interaction AND n_events < threshold AND not Overall."""
        if subgroup_label == "Overall":
            return False
        if attr_name not in self._interactions:
            return False
        return n_events < self._intersectional_min_events

    def _map_cells_to_marginals(
        self,
        key: tuple[str, ...],
        cells: npt.NDArray[Any],
    ) -> dict[str, tuple[str, ...]]:
        """Map each interaction cell label to its constituents' marginal labels.

        Raises if a cell spans more than one combination of marginal labels,
        which happens when a user-supplied interaction array is not nested in
        its marginals (or when auto-built labels collide).
        """
        # Positional columns: 0 is the cell, 1.. the constituents in key order.
        frame = pd.DataFrame(
            list(zip(
                [str(c) for c in cells],
                *[[str(v) for v in self._sensitive[c]] for c in key],
            ))
        ).drop_duplicates()
        ambiguous = frame.loc[frame[0].duplicated(), 0].unique()
        if len(ambiguous):
            msg = (
                f"interaction {key!r}: cell label(s) {list(ambiguous[:3])} each "
                f"cover more than one combination of {list(key)}. Every cell of an "
                "interaction must correspond to exactly one combination of its "
                "marginal labels; pass `None` to have the labels built for you."
            )
            raise ValueError(msg)
        return {row[0]: tuple(row[1:]) for row in frame.itertuples(index=False)}

    def _get_shrinkage_params(
        self,
        attr_name: str,
        cell_label: str,
        marginal_metrics_cache: dict[str, dict[str, dict[str, float]]],
    ) -> tuple[dict[str, float], float]:
        """Return (marginal_rates, prior_strength) for a shrunk cell.

        Shrinkage target: rates from the first constituent's matching subgroup.
        Prior strength: mean of [total_events / n_groups for each constituent].
        """
        constituents = self._interactions[attr_name]

        # Marginal rates from first constituent
        first_attr = constituents[0]
        first_label = self._cell_constituents[attr_name][cell_label][0]
        first_metrics = marginal_metrics_cache[first_attr][first_label]
        marginal_rates = {m: first_metrics[m] for m in _RATE_METRICS}

        # Prior strength: mean of total_events / n_groups per constituent
        strengths: list[float] = []
        for c in constituents:
            cached = marginal_metrics_cache[c]
            total_events = sum(m["n_events"] for m in cached.values())
            n_groups = len(cached)
            strengths.append(total_events / n_groups if n_groups > 0 else 0.0)
        prior_strength = float(np.mean(strengths))

        return marginal_rates, prior_strength

    def _apply_discrimination_shrinkage(
        self,
        metrics: dict[str, float],
        y_true: npt.NDArray[np.floating[Any]],
        y_proba: npt.NDArray[np.floating[Any]],
        marginal_rates: dict[str, float],
        prior_strength: float,
    ) -> dict[str, float]:
        """Apply empirical Bayes shrinkage to rate metrics in a discrimination result.

        AUROC and AUPRC are left untouched; only rate metrics are shrunk.
        """
        n = len(y_true)
        n_events = int(y_true.sum())
        y_pred = (y_proba >= self._threshold).astype(int)

        if n_events == 0 or n_events == n:
            # Degenerate: shrink prevalence only, rest stay NaN
            metrics["prevalence"] = _shrink_rate(
                n_events,
                n,
                marginal_rates["prevalence"],
                prior_strength,
            )
            return metrics

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Map each rate metric to (successes, trials)
        rate_map: dict[str, tuple[int, int]] = {
            "prevalence": (n_events, n),
            "sensitivity": (int(tp), int(tp + fn)),
            "specificity": (int(tn), int(tn + fp)),
            "ppv": (int(tp), int(tp + fp)),
            "npv": (int(tn), int(tn + fn)),
        }

        for m in _RATE_METRICS:
            successes, trials = rate_map[m]
            if trials > 0:
                metrics[m] = _shrink_rate(
                    successes,
                    trials,
                    marginal_rates[m],
                    prior_strength,
                )
            # else leave as NaN

        return metrics

    # ------------------------------------------------------------------
    # Gap statistics — calibration and DCA
    # ------------------------------------------------------------------

    @staticmethod
    def _max_pairwise_range(vals: list[float]) -> float:
        """Max absolute pairwise difference among finite values."""
        finite = [v for v in vals if not np.isnan(v)]
        if len(finite) < 2:
            return float("nan")
        return max(finite) - min(finite)

    def _calibration_gap_point(
        self,
        attr_values: npt.NDArray[Any],
        groups: npt.NDArray[Any],
    ) -> dict[str, float]:
        """Compute max-pairwise intercept and slope gaps for one attribute."""
        intercepts: list[float] = []
        slopes: list[float] = []
        for group in groups:
            mask = attr_values == group
            y_t = self._y_true[mask]
            y_p = self._y_proba[mask]
            m = _compute_calibration_boot_metrics(y_t, y_p)
            intercepts.append(m["calibration_intercept"])
            slopes.append(m["calibration_slope"])
        return {
            "max_intercept_gap": self._max_pairwise_range(intercepts),
            "max_slope_gap": self._max_pairwise_range(slopes),
        }

    def _bootstrap_calibration_gaps(
        self,
        attr_values: npt.NDArray[Any],
        groups: npt.NDArray[Any],
        rng: np.random.Generator,
    ) -> dict[str, tuple[float, float]]:
        """Bootstrap CIs for calibration gaps (intercept, slope)."""
        boot_indices = self._stratified_bootstrap_indices(self._y_true, rng)
        results: dict[str, list[float]] = {
            "max_intercept_gap": [],
            "max_slope_gap": [],
        }
        for idx in boot_indices:
            if idx is None:
                results["max_intercept_gap"].append(float("nan"))
                results["max_slope_gap"].append(float("nan"))
                continue
            b_attr = attr_values[idx]
            b_yt = self._y_true[idx]
            b_yp = self._y_proba[idx]
            intercepts: list[float] = []
            slopes: list[float] = []
            for group in groups:
                mask = b_attr == group
                if mask.sum() < 3:
                    intercepts.append(float("nan"))
                    slopes.append(float("nan"))
                    continue
                m = _compute_calibration_boot_metrics(b_yt[mask], b_yp[mask])
                intercepts.append(m["calibration_intercept"])
                slopes.append(m["calibration_slope"])
            results["max_intercept_gap"].append(self._max_pairwise_range(intercepts))
            results["max_slope_gap"].append(self._max_pairwise_range(slopes))
        return _compute_percentile_cis(results, ["max_intercept_gap", "max_slope_gap"])

    def _dca_gap_point(
        self,
        attr_values: npt.NDArray[Any],
        groups: npt.NDArray[Any],
        thresholds: npt.NDArray[np.floating[Any]],
    ) -> float:
        """Max across thresholds of max-pairwise NB difference."""
        # Compute NB per subgroup per threshold
        nb_matrix: list[npt.NDArray[np.floating[Any]]] = []
        for group in groups:
            mask = attr_values == group
            y_t = self._y_true[mask]
            y_p = self._y_proba[mask]
            nb_matrix.append(_compute_net_benefits(y_t, y_p, thresholds))
        if len(nb_matrix) < 2:
            return float("nan")
        max_gap = 0.0
        for t_idx in range(len(thresholds)):
            vals = [nb[t_idx] for nb in nb_matrix]
            finite = [v for v in vals if not np.isnan(v)]
            if len(finite) >= 2:
                gap = max(finite) - min(finite)
                max_gap = max(max_gap, gap)
        return max_gap

    def _bootstrap_dca_gaps(
        self,
        attr_values: npt.NDArray[Any],
        groups: npt.NDArray[Any],
        thresholds: npt.NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Bootstrap CIs for max-net-benefit gap."""
        boot_indices = self._stratified_bootstrap_indices(self._y_true, rng)
        gap_vals: list[float] = []
        for idx in boot_indices:
            if idx is None:
                gap_vals.append(float("nan"))
                continue
            b_attr = attr_values[idx]
            b_yt = self._y_true[idx]
            b_yp = self._y_proba[idx]
            nb_matrix: list[npt.NDArray[np.floating[Any]]] = []
            for group in groups:
                mask = b_attr == group
                if mask.sum() < 2:
                    nb_matrix.append(np.full(len(thresholds), float("nan")))
                    continue
                nb_matrix.append(_compute_net_benefits(b_yt[mask], b_yp[mask], thresholds))
            max_gap = 0.0
            for t_idx in range(len(thresholds)):
                vals = [nb[t_idx] for nb in nb_matrix]
                finite = [v for v in vals if not np.isnan(v)]
                if len(finite) >= 2:
                    gap = max(finite) - min(finite)
                    max_gap = max(max_gap, gap)
            gap_vals.append(max_gap)
        arr = np.array(gap_vals)
        if np.all(np.isnan(arr)):
            return (float("nan"), float("nan"))
        return (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
        )

    # ------------------------------------------------------------------
    # Gap statistics — discrimination
    # ------------------------------------------------------------------

    def _compute_gaps(
        self,
        subgroup_metrics: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute fairness gap statistics across subgroups of one attribute.

        Definitions (following Hardt et al. 2016 and Chouldechova 2017):

        - equal_opportunity_difference: max - min sensitivity across subgroups
        - predictive_equality: max - min FPR (= 1 - specificity) across subgroups
        - equalized_odds_gap: max(equal_opportunity_diff, predictive_equality)
        - statistical_parity: max - min positive-prediction rate across subgroups

        All gaps are absolute (non-negative).
        """
        sensitivities = [m["sensitivity"] for m in subgroup_metrics.values()]
        specificities = [m["specificity"] for m in subgroup_metrics.values()]
        fprs = [1.0 - s for s in specificities]

        # Positive prediction rate = (TP + FP) / n
        pprs = []
        for m in subgroup_metrics.values():
            n = m["n"]
            if n > 0:
                n_events = m["n_events"]
                sens = m["sensitivity"]
                spec = m["specificity"]
                tp = sens * n_events
                fp = (1.0 - spec) * (n - n_events)
                pprs.append((tp + fp) / n)
            else:
                pprs.append(float("nan"))

        def _range(vals: list[float]) -> float:
            finite = [v for v in vals if not np.isnan(v)]
            if len(finite) < 2:
                return float("nan")
            return max(finite) - min(finite)

        eo_diff = _range(sensitivities)
        pe = _range(fprs)

        eq_odds = float("nan") if np.isnan(eo_diff) or np.isnan(pe) else max(eo_diff, pe)

        return {
            "equal_opportunity_difference": eo_diff,
            "predictive_equality": pe,
            "equalized_odds_gap": eq_odds,
            "statistical_parity": _range(pprs),
        }

    # ------------------------------------------------------------------
    # Public API: discrimination()
    # ------------------------------------------------------------------

    def discrimination(self) -> pd.DataFrame:
        """Compute subgroup-stratified discrimination metrics with bootstrap CIs.

        Returns a long-format DataFrame with one row per (attribute, subgroup)
        combination, one "Overall" row per attribute, and gap-statistic rows
        (subgroup = "GAP") per attribute.

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``n``, ``n_events``,
            ``prevalence``, ``auroc``, ``auroc_ci_low``, ``auroc_ci_high``,
            ``auprc``, ``auprc_ci_low``, ``auprc_ci_high``,
            ``sensitivity``, ``sensitivity_ci_low``, ``sensitivity_ci_high``,
            ``specificity``, ``specificity_ci_low``, ``specificity_ci_high``,
            ``ppv``, ``ppv_ci_low``, ``ppv_ci_high``,
            ``npv``, ``npv_ci_low``, ``npv_ci_high``.

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array([0, 0, 1, 1, 0, 1])
        >>> p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        >>> sex = np.array(["M", "F", "M", "F", "M", "F"])
        >>> audit = FairnessAudit(
        ...     y_true=y, y_proba=p,
        ...     sensitive_features={"sex": sex},
        ...     threshold=0.5, n_bootstrap=50, random_state=0,
        ... )
        >>> df = audit.discrimination()
        >>> sorted(df["subgroup"].unique())
        ['F', 'GAP', 'M', 'Overall']
        """
        if "discrimination" in self._cache:
            return self._cache["discrimination"]

        rng = np.random.default_rng(self._random_state)
        rows: list[dict[str, Any]] = []
        marginal_metrics_cache: dict[str, dict[str, dict[str, float]]] = {}

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions

            patient_cis = (
                self._patient_axis_cis(
                    "discrimination",
                    attr_name,
                    groups,
                    attr_values,
                    _DISCRIMINATION_BOOT_METRICS,
                    "discrimination",
                )
                if self._resample_scheme == "patient"
                else None
            )

            # --- Overall row for this attribute ---
            overall_metrics = _compute_subgroup_metrics(
                self._y_true, self._y_proba, self._threshold
            )
            if patient_cis is not None:
                overall_cis = patient_cis["Overall"]
            else:
                overall_cis = self._bootstrap_subgroup_metrics(
                    self._y_true,
                    self._y_proba,
                    rng,
                    store_as=("discrimination", attr_name, "Overall"),
                )
            row: dict[str, Any] = {
                "attribute": attr_name,
                "subgroup": "Overall",
                "shrunk": False,
                **overall_metrics,
            }
            for m in _CI_METRICS:
                row[f"{m}_ci_low"] = overall_cis[m][0]
                row[f"{m}_ci_high"] = overall_cis[m][1]
            row.update(
                self._estimability_row("discrimination", attr_name, "Overall", _CI_METRICS)
            )
            rows.append(row)

            # --- Per-subgroup rows ---
            subgroup_metrics: dict[str, dict[str, float]] = {}
            for group in groups:
                mask = attr_values == group
                y_t = self._y_true[mask]
                y_p = self._y_proba[mask]

                metrics = _compute_subgroup_metrics(y_t, y_p, self._threshold)
                # Always call bootstrap to keep RNG path deterministic
                if patient_cis is not None:
                    cis = patient_cis[str(group)]
                else:
                    cis = self._bootstrap_subgroup_metrics(
                        y_t,
                        y_p,
                        rng,
                        store_as=("discrimination", attr_name, str(group)),
                    )

                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    str(group),
                    n_events,
                )

                if shrunk:
                    marginal_rates, prior_strength = self._get_shrinkage_params(
                        attr_name,
                        str(group),
                        marginal_metrics_cache,
                    )
                    metrics = self._apply_discrimination_shrinkage(
                        metrics,
                        y_t,
                        y_p,
                        marginal_rates,
                        prior_strength,
                    )
                    # Set all CIs to NaN for shrunk cells
                    cis = {m: (float("nan"), float("nan")) for m in _CI_METRICS}

                subgroup_metrics[str(group)] = metrics
                row = {
                    "attribute": attr_name,
                    "subgroup": str(group),
                    "shrunk": shrunk,
                    **metrics,
                }
                for m in _CI_METRICS:
                    row[f"{m}_ci_low"] = cis[m][0]
                    row[f"{m}_ci_high"] = cis[m][1]
                row.update(
                    self._estimability_row(
                        "discrimination", attr_name, str(group), _CI_METRICS
                    )
                )
                rows.append(row)

            # Save marginal metrics in cache (for use by interactions)
            if not is_interaction:
                marginal_metrics_cache[attr_name] = subgroup_metrics

            # --- Gap rows ---
            gaps = self._compute_gaps(subgroup_metrics)
            gap_cis = (
                self._discrimination_gap_cis(attr_name, groups)
                if patient_cis is not None
                else {}
            )
            for gap_name, gap_value in gaps.items():
                gap_row: dict[str, Any] = {
                    "attribute": attr_name,
                    "subgroup": "GAP",
                    "shrunk": False,
                    "n": float("nan"),
                    "n_events": float("nan"),
                    "prevalence": float("nan"),
                }
                for m in _CI_METRICS:
                    gap_row[m] = float("nan")
                    gap_row[f"{m}_ci_low"] = float("nan")
                    gap_row[f"{m}_ci_high"] = float("nan")
                # Store the gap name and value in the appropriate metric column
                gap_row["gap_metric"] = gap_name
                gap_row["gap_value"] = gap_value
                lo, hi = gap_cis.get(gap_name, (float("nan"), float("nan")))
                gap_row["gap_ci_low"], gap_row["gap_ci_high"] = _validate_gap_ci(
                    gap_value, lo, hi
                )
                # The frozen case_control path stores no per-subgroup matrices
                # to reduce over, so its gaps stay interval-free and their
                # estimability is undefined rather than zero.
                gap_row.update(
                    self._estimability_row("discrimination", attr_name, "GAP", [gap_name])
                    if gap_cis
                    else _EMPTY_ESTIMABILITY
                )
                rows.append(gap_row)

        df = pd.DataFrame(rows)

        # Enforce stable column order
        base_cols = ["attribute", "subgroup", "shrunk", "n", "n_events", "prevalence"]
        metric_cols: list[str] = []
        for m in _CI_METRICS:
            metric_cols.extend([m, f"{m}_ci_low", f"{m}_ci_high"])
        extra_cols = ["gap_metric", "gap_value", "gap_ci_low", "gap_ci_high"]
        for col in extra_cols:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[base_cols + metric_cols + extra_cols + _ESTIMABILITY_COLUMNS]
        df = self._apply_event_floors(df, "discrimination")

        self._cache["discrimination"] = df
        return df

    # ------------------------------------------------------------------
    # Public API: calibration()
    # ------------------------------------------------------------------

    def calibration(self) -> pd.DataFrame:
        """Compute subgroup-stratified calibration metrics with bootstrap CIs.

        Returns a long-format DataFrame with one row per (attribute, subgroup)
        combination and one "Overall" row per attribute.

        Calibration intercept and slope follow the Steyerberg 2009 convention:
        logistic regression of ``y_true`` on ``logit(y_proba)``, with slope
        fixed to 1 (intercept) or free (slope). ECE uses equal-frequency
        (quantile-based) bins. ICI uses LOESS smoothing (Austin & Steyerberg
        2019).

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``shrunk``, ``n``,
            ``n_events``, ``calibration_intercept``, ``intercept_ci_low``,
            ``intercept_ci_high``, ``calibration_slope``, ``slope_ci_low``,
            ``slope_ci_high``, ``brier``, ``brier_ci_low``,
            ``brier_ci_high``, ``ece``, ``ici``, ``gap_metric``,
            ``gap_value``, ``gap_ci_low``, ``gap_ci_high``.

            Gap rows (``subgroup == "GAP"``) report ``max_intercept_gap``
            and ``max_slope_gap`` — the max absolute pairwise difference
            across subgroups — with bootstrap CIs.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.01, 0.99, n)
        >>> y = rng.binomial(1, p).astype(float)
        >>> g = np.array(["A", "B"] * (n // 2))
        >>> audit = FairnessAudit(
        ...     y_true=y, y_proba=p,
        ...     sensitive_features={"group": g},
        ...     threshold=0.5, n_bootstrap=50, random_state=0,
        ... )
        >>> df = audit.calibration()
        >>> {'calibration_intercept', 'calibration_slope', 'brier'} <= set(df.columns)
        True
        """
        if "calibration" in self._cache:
            return self._cache["calibration"]

        rng = np.random.default_rng(self._random_state)
        rows: list[dict[str, Any]] = []

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions

            # Under the patient scheme every subgroup of this attribute is
            # evaluated on one shared set of replicates, in a single pass
            # before the per-subgroup loop, so the gap statistics below can be
            # derived from the same replicates rather than resampled again.
            patient_cis = (
                self._patient_axis_cis(
                    "calibration",
                    attr_name,
                    groups,
                    attr_values,
                    _CALIBRATION_CI_METRICS,
                    "calibration",
                )
                if self._resample_scheme == "patient"
                else None
            )

            for subgroup_label, y_t, y_p in self._iter_overall_and_subgroups(
                attr_name, attr_values, groups
            ):
                point = _compute_calibration_boot_metrics(y_t, y_p)
                ece = _compute_ece(y_t, y_p, self._ece_bins)
                ici = _compute_ici(y_t, y_p)
                if patient_cis is not None:
                    cis = patient_cis[subgroup_label]
                else:
                    # Always call bootstrap for RNG determinism
                    cis = self._bootstrap_calibration_metrics(
                        y_t,
                        y_p,
                        rng,
                        store_as=("calibration", attr_name, subgroup_label),
                    )

                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    subgroup_label,
                    n_events,
                )
                if shrunk:
                    cis = {m: (float("nan"), float("nan")) for m in _CALIBRATION_CI_METRICS}

                row: dict[str, Any] = {
                    "attribute": attr_name,
                    "subgroup": subgroup_label,
                    "shrunk": shrunk,
                    "n": float(len(y_t)),
                    "n_events": float(n_events),
                    **point,
                    "ece": ece,
                    "ici": ici,
                }
                for m in _CALIBRATION_CI_METRICS:
                    low_col, high_col = _CALIBRATION_CI_COLUMNS[m]
                    row[low_col] = cis[m][0]
                    row[high_col] = cis[m][1]
                row.update(
                    self._estimability_row(
                        "calibration", attr_name, subgroup_label, _CALIBRATION_CI_METRICS
                    )
                )
                rows.append(row)

            # --- Calibration gap rows ---
            gap_point = self._calibration_gap_point(attr_values, groups)
            if patient_cis is None:
                # Frozen path: a *separate* whole-cohort resample, which is why
                # its gap intervals cannot be reconciled with the per-subgroup
                # ones. Retained deliberately; see the class docstring.
                gap_cis = self._bootstrap_calibration_gaps(attr_values, groups, rng)
            else:
                gap_cis = {
                    gap_name: self._patient_gap_ci(
                        "calibration", attr_name, groups, metric, gap_name
                    )
                    for gap_name, metric in _CALIBRATION_GAP_SOURCE.items()
                }
            for gap_name in ("max_intercept_gap", "max_slope_gap"):
                gv = gap_point[gap_name]
                if patient_cis is None:
                    g_lo, g_hi = _validate_gap_ci(
                        gv, gap_cis[gap_name][0], gap_cis[gap_name][1]
                    )
                else:
                    # No suppression needed: the interval and the point estimate
                    # now come from the same replicates, so the point estimate
                    # cannot fall outside for resampling reasons.
                    g_lo, g_hi = gap_cis[gap_name]
                gap_row: dict[str, Any] = {
                    "attribute": attr_name,
                    "subgroup": "GAP",
                    "shrunk": False,
                    "n": float("nan"),
                    "n_events": float("nan"),
                    "calibration_intercept": float("nan"),
                    "intercept_ci_low": float("nan"),
                    "intercept_ci_high": float("nan"),
                    "calibration_slope": float("nan"),
                    "slope_ci_low": float("nan"),
                    "slope_ci_high": float("nan"),
                    "brier": float("nan"),
                    "brier_ci_low": float("nan"),
                    "brier_ci_high": float("nan"),
                    "ece": float("nan"),
                    "ici": float("nan"),
                    "gap_metric": gap_name,
                    "gap_value": gv,
                    "gap_ci_low": g_lo,
                    "gap_ci_high": g_hi,
                    **_EMPTY_ESTIMABILITY,
                }
                rows.append(gap_row)

        df = pd.DataFrame(rows)

        # Enforce stable column order
        col_order = [
            "attribute",
            "subgroup",
            "shrunk",
            "n",
            "n_events",
            "calibration_intercept",
            "intercept_ci_low",
            "intercept_ci_high",
            "calibration_slope",
            "slope_ci_low",
            "slope_ci_high",
            "brier",
            "brier_ci_low",
            "brier_ci_high",
            "ece",
            "ici",
            "gap_metric",
            "gap_value",
            "gap_ci_low",
            "gap_ci_high",
            *_ESTIMABILITY_COLUMNS,
        ]
        df = df[col_order]
        df = self._apply_event_floors(df, "calibration")

        self._cache["calibration"] = df
        return df

    # ------------------------------------------------------------------
    # Public API: calibration_curves()
    # ------------------------------------------------------------------

    def calibration_curves(self) -> pd.DataFrame:
        """Compute LOESS-smoothed calibration curves with bootstrap CIs.

        For each (attribute, subgroup) including "Overall", a LOESS curve
        is fitted to ``(y_proba, y_true)`` and evaluated at 100 uniformly
        spaced points spanning the range of predicted probabilities.
        Bootstrap CIs are the 2.5/97.5 percentiles of the smoothed curve
        across resamples.

        The LOESS smoother uses ``statsmodels.nonparametric.smoothers_lowess``
        with ``frac=2/3`` (default bandwidth) and ``it=0`` (no robustifying
        iterations, essential for binary outcomes).

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``predicted_prob``,
            ``observed_prob``, ``ci_low``, ``ci_high``.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.01, 0.99, n)
        >>> y = rng.binomial(1, p).astype(float)
        >>> g = np.array(["A", "B"] * (n // 2))
        >>> audit = FairnessAudit(
        ...     y_true=y, y_proba=p,
        ...     sensitive_features={"group": g},
        ...     threshold=0.5, n_bootstrap=50, random_state=0,
        ... )
        >>> curves = audit.calibration_curves()
        >>> curves.shape[1]
        7
        """
        if "calibration_curves" in self._cache:
            return self._cache["calibration_curves"]

        rng = np.random.default_rng(self._random_state)
        p_min = max(0.0, float(self._y_proba.min()))
        p_max = min(1.0, float(self._y_proba.max()))
        eval_points = np.linspace(p_min, p_max, 100)
        rows: list[dict[str, Any]] = []

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions

            for subgroup_label, y_t, y_p in self._iter_overall_and_subgroups(
                attr_name, attr_values, groups
            ):
                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    subgroup_label,
                    n_events,
                )

                # Point estimate
                point_curve = _loess_curve_at_grid(y_t, y_p, eval_points)

                # Always call bootstrap for RNG determinism
                boot_indices = self._stratified_bootstrap_indices(y_t, rng)
                boot_curves: list[npt.NDArray[np.floating[Any]]] = []
                for idx in boot_indices:
                    if idx is None:
                        boot_curves.append(np.full(len(eval_points), float("nan")))
                        continue
                    boot_curves.append(_loess_curve_at_grid(y_t[idx], y_p[idx], eval_points))
                boot_matrix = np.array(boot_curves)

                for i, ep in enumerate(eval_points):
                    if shrunk:
                        ci_low = float("nan")
                        ci_high = float("nan")
                    else:
                        col_vals = boot_matrix[:, i]
                        if np.all(np.isnan(col_vals)):
                            ci_low = float("nan")
                            ci_high = float("nan")
                        else:
                            ci_low = float(np.nanpercentile(col_vals, 2.5))
                            ci_high = float(np.nanpercentile(col_vals, 97.5))
                    rows.append(
                        {
                            "attribute": attr_name,
                            "subgroup": subgroup_label,
                            "shrunk": shrunk,
                            "predicted_prob": float(ep),
                            "observed_prob": float(point_curve[i]),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
                    )

        df = pd.DataFrame(
            rows,
            columns=[
                "attribute",
                "subgroup",
                "shrunk",
                "predicted_prob",
                "observed_prob",
                "ci_low",
                "ci_high",
            ],
        )
        self._cache["calibration_curves"] = df
        return df

    # ------------------------------------------------------------------
    # Public API: roc_curves()
    # ------------------------------------------------------------------

    def roc_curves(self) -> pd.DataFrame:
        """Compute subgroup-stratified ROC curves with bootstrap CIs.

        For each (attribute, subgroup) including "Overall", the true-positive
        rate is evaluated at 100 evenly spaced false-positive rates spanning
        ``[0, 1]`` (vertical averaging). Bootstrap CIs are the 2.5/97.5
        percentiles of the true-positive rate across resamples.

        The grid is fixed rather than data-dependent (unlike
        :meth:`calibration_curves`, whose grid spans the observed prediction
        range) because a ROC curve lives on the unit square by construction.
        A fixed grid also makes curves comparable across attributes and
        between a baseline and a mitigated audit.

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``shrunk``, ``fpr``,
            ``tpr``, ``ci_low``, ``ci_high``.

            No gap rows: AUROC gap statistics belong to
            :meth:`discrimination`.

        Notes
        -----
        Bands are **pointwise** 95% percentile intervals, not simultaneous;
        they do not support the claim that the whole true curve lies inside
        the band.

        Under the default ``resample_scheme="patient"`` the band is the
        unconditional sampling distribution: patients are resampled within the
        subgroup, so its event count varies. Under ``"case_control"`` the class
        margins are held fixed and the band is conditional on them, which
        understates uncertainty for prevalence-dependent quantities.

        The trapezoid area under the 100-point grid curve is an approximation
        and differs from ``discrimination()["auroc"]`` in the third or fourth
        decimal. Report AUROC from :meth:`discrimination`; never integrate
        these curves.

        Subgroups containing a single outcome class yield an all-NaN block,
        matching :func:`isitfair.subgroup_metrics`.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.01, 0.99, n)
        >>> y = rng.binomial(1, p).astype(float)
        >>> g = np.array(["A", "B"] * (n // 2))
        >>> audit = FairnessAudit(
        ...     y_true=y, y_proba=p,
        ...     sensitive_features={"group": g},
        ...     threshold=0.5, n_bootstrap=50, random_state=0,
        ... )
        >>> curves = audit.roc_curves()
        >>> curves.shape[1]
        7
        """
        if "roc_curves" in self._cache:
            return self._cache["roc_curves"]

        rng = np.random.default_rng(self._random_state)
        fpr_grid = np.linspace(0.0, 1.0, 100)
        rows: list[dict[str, Any]] = []

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions

            for subgroup_label, y_t, y_p in self._iter_overall_and_subgroups(
                attr_name, attr_values, groups
            ):
                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    subgroup_label,
                    n_events,
                )

                point_curve = _roc_tpr_at_grid(y_t, y_p, fpr_grid)

                # Always call bootstrap for RNG determinism
                boot_indices = self._stratified_bootstrap_indices(y_t, rng)
                boot_curves: list[npt.NDArray[np.floating[Any]]] = []
                for idx in boot_indices:
                    if idx is None:
                        boot_curves.append(np.full(len(fpr_grid), float("nan")))
                        continue
                    boot_curves.append(_roc_tpr_at_grid(y_t[idx], y_p[idx], fpr_grid))
                boot_matrix = np.array(boot_curves)

                for i, f in enumerate(fpr_grid):
                    if shrunk:
                        ci_low = float("nan")
                        ci_high = float("nan")
                    else:
                        col_vals = boot_matrix[:, i]
                        if np.all(np.isnan(col_vals)):
                            ci_low = float("nan")
                            ci_high = float("nan")
                        else:
                            ci_low = float(np.nanpercentile(col_vals, 2.5))
                            ci_high = float(np.nanpercentile(col_vals, 97.5))
                    rows.append(
                        {
                            "attribute": attr_name,
                            "subgroup": subgroup_label,
                            "shrunk": shrunk,
                            "fpr": float(f),
                            "tpr": float(point_curve[i]),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
                    )

        df = pd.DataFrame(
            rows,
            columns=[
                "attribute",
                "subgroup",
                "shrunk",
                "fpr",
                "tpr",
                "ci_low",
                "ci_high",
            ],
        )
        self._cache["roc_curves"] = df
        return df

    # ------------------------------------------------------------------
    # Public API: bootstrap_distributions() / bootstrap_diagnostics()
    # ------------------------------------------------------------------

    def bootstrap_distributions(
        self,
        axis: str = "discrimination",
        *,
        attribute: str | None = None,
        subgroup: str | None = None,
        metric: str | None = None,
        copy: bool = False,
    ) -> dict[tuple[str, str, str], npt.NDArray[np.floating[Any]]]:
        """Return the stored per-replicate bootstrap distributions.

        Exposed so gap arithmetic can be checked independently: a gap
        statistic must be derivable from the same replicates as the subgroup
        statistics it compares, and this is the evidence for that.

        Parameters
        ----------
        axis : {"discrimination", "calibration"}, default "discrimination"
            Which analysis axis to return. Computed on demand if not yet run.
        attribute, subgroup, metric : str or None, optional
            Exact-match filters. ``subgroup="Overall"`` selects the
            whole-cohort distribution.
        copy : bool, default False
            By default the arrays are read-only views into internal storage.
            Pass ``True`` for writable copies.

        Returns
        -------
        dict[tuple[str, str, str], ndarray]
            Keyed by ``(attribute, subgroup, metric)``. Every value has length
            ``n_bootstrap`` and holds ``nan`` at non-estimable replicates —
            replicates are never dropped.

        Raises
        ------
        ValueError
            If *axis* is unknown, or if the filters match nothing. A typo in a
            subgroup label returning an empty dict is how broken analyses ship,
            so it raises and lists the available values instead.

        Notes
        -----
        The curve axes (:meth:`calibration_curves`, :meth:`roc_curves`) are not
        stored. They are 100 points wide, would add ~100x the memory, and have
        no gap statistic that would need them.

        Examples
        --------
        >>> dists = audit.bootstrap_distributions("calibration")  # doctest: +SKIP
        >>> dists[("sex", "female", "calibration_intercept")].shape  # doctest: +SKIP
        (2000,)
        """
        if axis not in _BOOTSTRAP_AXES:
            msg = f"`axis` must be one of {sorted(_BOOTSTRAP_AXES)}, got {axis!r}"
            raise ValueError(msg)

        if axis not in self._boot_store:
            getattr(self, axis)()
        store = self._boot_store.get(axis, {})

        selected = {
            key: cell.values
            for key, cell in store.items()
            if (attribute is None or key[0] == attribute)
            and (subgroup is None or key[1] == subgroup)
            and (metric is None or key[2] == metric)
        }

        if not selected:
            available = {
                "attribute": sorted({k[0] for k in store}),
                "subgroup": sorted({k[1] for k in store}),
                "metric": sorted({k[2] for k in store}),
            }
            msg = (
                f"No bootstrap distributions match "
                f"attribute={attribute!r}, subgroup={subgroup!r}, metric={metric!r} "
                f"on axis {axis!r}. Available: {available}"
            )
            raise ValueError(msg)

        if copy:
            return {k: v.copy() for k, v in selected.items()}
        views = {}
        for k, v in selected.items():
            view = v.view()
            view.flags.writeable = False
            views[k] = view
        return views

    def bootstrap_diagnostics(self, axis: str = "discrimination") -> pd.DataFrame:
        """Per-cell replicate accounting for one axis.

        The authoritative record required to interpret a confidence interval:
        percentile CIs are computed over estimable replicates only, so an
        interval built from 63% of replicates is conditional on estimability
        and is narrower than the marginal one. This frame says how often that
        happened, per metric.

        Parameters
        ----------
        axis : {"discrimination", "calibration"}, default "discrimination"
            Which analysis axis to summarise. Computed on demand if not run.

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``metric``,
            ``n_replicates``, ``n_estimable``, ``proportion_nonestimable``,
            ``not_estimable``. One row per ``(attribute, subgroup, metric)``.
        """
        if axis not in _BOOTSTRAP_AXES:
            msg = f"`axis` must be one of {sorted(_BOOTSTRAP_AXES)}, got {axis!r}"
            raise ValueError(msg)

        if axis not in self._boot_store:
            getattr(self, axis)()

        rows = [
            {
                "attribute": cell.attribute,
                "subgroup": cell.subgroup,
                "metric": cell.metric,
                "n_replicates": cell.n_replicates,
                "n_estimable": cell.n_estimable,
                "proportion_nonestimable": cell.proportion_nonestimable,
                "not_estimable": cell.not_estimable(self._nonestimable_threshold),
            }
            for cell in self._boot_store.get(axis, {}).values()
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "attribute",
                "subgroup",
                "metric",
                "n_replicates",
                "n_estimable",
                "proportion_nonestimable",
                "not_estimable",
            ],
        )

    # ------------------------------------------------------------------
    # Public API: hosmer_lemeshow()
    # ------------------------------------------------------------------

    def hosmer_lemeshow(
        self,
        *,
        detail: bool = False,
        n_bins: int = 10,
        df_mode: str = "validation",
    ) -> pd.DataFrame:
        """Compute subgroup-stratified Hosmer–Lemeshow goodness-of-fit.

        Wraps :func:`isitfair.hosmer_lemeshow` per subgroup. Kramer &
        Zimmerman (2007) argue that the statistic must be read alongside the
        number of observations, the per-decile observed-versus-expected table,
        and adjunct calibration measures — never on its own — because it
        scales with sample size. This method returns the first three; the
        adjunct measures are in :meth:`calibration`.

        Parameters
        ----------
        detail : bool, default False
            When ``False``, return one summary row per (attribute, subgroup).
            When ``True``, return the per-decile table instead.
        n_bins : int, default 10
            Number of risk groups per subgroup. Deliberately independent of
            the ``ece_bins`` constructor parameter.
        df_mode : {"validation", "development"}, default "validation"
            Degrees-of-freedom convention; see :func:`isitfair.hosmer_lemeshow`.

        Returns
        -------
        pd.DataFrame
            With ``detail=False``: ``attribute``, ``subgroup``, ``shrunk``,
            ``n``, ``n_events``, ``hl_statistic``, ``hl_df``, ``hl_p_value``,
            ``n_bins_used``, ``bins_reduced``, ``decile_slope``,
            ``decile_intercept``.

            Gap rows (``subgroup == "GAP"``) report ``max_hl_gap`` — the
            maximum absolute pairwise difference in the statistic across
            subgroups. It carries no confidence interval; see Notes.

            With ``detail=True``: ``attribute``, ``subgroup``, ``decile``,
            ``n``, ``observed``, ``expected``, ``observed_rate``,
            ``expected_rate``, ``difference``. There is no ``shrunk`` column;
            join to the summary frame on ``(attribute, subgroup)``.

        Warnings
        --------
        Hosmer–Lemeshow statistics are **not comparable across subgroups of
        different size**. A large subgroup can be flagged for a clinically
        irrelevant deviation while a small subgroup with worse fit is not.
        Read every value beside its ``n`` and its decile table, and do not
        rank subgroups by ``hl_p_value``. The ``max_hl_gap`` row is provided
        for consistency with the other axes and inherits this caveat in full.

        Notes
        -----
        No bootstrap confidence intervals are produced. The C statistic is a
        test statistic, not an estimand: the percentile bootstrap resamples
        from the alternative rather than the null, and under the null its
        expectation equals the degrees of freedom regardless of sample size
        while under the alternative it grows with sample size. An interval
        would therefore largely re-encode ``n``, which is already its own
        column.

        Subgroups flagged ``shrunk`` report their statistic unsuppressed —
        shrinkage in this package suppresses bootstrap CIs, and there are none
        here — but such cells are too small to interpret and usually fall into
        the fewer-than-three-bins NaN branch anyway.

        Running this across every subgroup of every attribute is a family of
        tests. No multiplicity adjustment is applied; use
        :func:`isitfair.fdr_correct` if one is wanted, and adjust within an
        attribute rather than pooling "Overall" rows with subgroup rows.

        This method consumes no random state.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.01, 0.99, n)
        >>> y = rng.binomial(1, p).astype(float)
        >>> g = np.array(["A", "B"] * (n // 2))
        >>> audit = FairnessAudit(
        ...     y_true=y, y_proba=p,
        ...     sensitive_features={"group": g},
        ...     threshold=0.5, n_bootstrap=50, random_state=0,
        ... )
        >>> hl = audit.hosmer_lemeshow()
        >>> hl.columns.tolist()[:5]
        ['attribute', 'subgroup', 'shrunk', 'n', 'n_events']
        """
        cache_key = f"hosmer_lemeshow_{n_bins}_{df_mode}_{int(detail)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        summary_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions
            subgroup_statistics: list[float] = []

            for subgroup_label, y_t, y_p in self._iter_overall_and_subgroups(
                attr_name, attr_values, groups
            ):
                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    subgroup_label,
                    n_events,
                )
                result = _hosmer_lemeshow(y_t, y_p, n_bins=n_bins, df_mode=df_mode)

                summary_rows.append(
                    {
                        "attribute": attr_name,
                        "subgroup": subgroup_label,
                        "shrunk": shrunk,
                        "n": float(result.n),
                        "n_events": float(result.n_events),
                        "hl_statistic": result.statistic,
                        "hl_df": result.df,
                        "hl_p_value": result.p_value,
                        "n_bins_used": float(result.n_bins_used),
                        "bins_reduced": result.bins_reduced,
                        "decile_slope": result.decile_slope,
                        "decile_intercept": result.decile_intercept,
                        # Reported here as well as on `calibration()`, because
                        # the statistic must be read *alongside* complementary
                        # measures rather than in place of them -- and a caller
                        # who has to join two frames to do that is a caller who
                        # will publish the statistic on its own.
                        "ece": _compute_ece(y_t, y_p, self._ece_bins),
                        "ici": _compute_ici(y_t, y_p),
                    }
                )

                if subgroup_label != "Overall":
                    subgroup_statistics.append(result.statistic)

                for row in result.table.to_dict("records"):
                    detail_row: dict[str, Any] = {
                        "attribute": attr_name,
                        "subgroup": subgroup_label,
                    }
                    detail_row.update({str(k): v for k, v in row.items()})
                    detail_rows.append(detail_row)

            # --- Gap row ---
            summary_rows.append(
                {
                    "attribute": attr_name,
                    "subgroup": "GAP",
                    "shrunk": False,
                    "n": float("nan"),
                    "n_events": float("nan"),
                    "hl_statistic": float("nan"),
                    "hl_df": float("nan"),
                    "hl_p_value": float("nan"),
                    "n_bins_used": float("nan"),
                    "bins_reduced": False,
                    "decile_slope": float("nan"),
                    "decile_intercept": float("nan"),
                    "ece": float("nan"),
                    "ici": float("nan"),
                    "gap_metric": "max_hl_gap",
                    "gap_value": self._max_pairwise_range(subgroup_statistics),
                }
            )

        if detail:
            df = pd.DataFrame(
                detail_rows,
                columns=[
                    "attribute",
                    "subgroup",
                    "decile",
                    "n",
                    "observed",
                    "expected",
                    "observed_rate",
                    "expected_rate",
                    "difference",
                ],
            )
            if not df.empty:
                df = df.astype({"decile": "int64"})
        else:
            df = pd.DataFrame(summary_rows)
            for col in ("gap_metric", "gap_value"):
                if col not in df.columns:
                    df[col] = None if col == "gap_metric" else float("nan")
            df = df[
                [
                    "attribute",
                    "subgroup",
                    "shrunk",
                    "n",
                    "n_events",
                    "hl_statistic",
                    "hl_df",
                    "hl_p_value",
                    "n_bins_used",
                    "bins_reduced",
                    "decile_slope",
                    "decile_intercept",
                    "ece",
                    "ici",
                    "gap_metric",
                    "gap_value",
                ]
            ]

        self._cache[cache_key] = df
        return df

    # ------------------------------------------------------------------
    # Public API: decision_curve()
    # ------------------------------------------------------------------

    def decision_curve(
        self,
        thresholds: Any = None,
        *,
        zero_crossing: bool = False,
    ) -> pd.DataFrame:
        """Compute subgroup-stratified decision curves wrapping ``dcurves``.

        Net benefit is computed at each threshold via ``dcurves.dca``.
        Bootstrap CIs use the same TP/N - FP/N * pt/(1-pt) formula for
        efficiency. Standardized net benefit follows Naderalvojoud 2025:
        NB / subgroup prevalence.

        Parameters
        ----------
        thresholds : array-like or None, optional
            Threshold probabilities at which to evaluate. Defaults to 20
            points linearly spaced in [0.02, 0.40].

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``shrunk``,
            ``threshold``, ``net_benefit``, ``net_benefit_ci_low``,
            ``net_benefit_ci_high``, ``standardized_net_benefit``,
            ``treat_all_nb``, ``treat_none_nb``, ``gap_metric``,
            ``gap_value``, ``gap_ci_low``, ``gap_ci_high``.

            Gap rows (``subgroup == "GAP"``, ``threshold == NaN``)
            report ``max_net_benefit_gap`` — the max across thresholds
            of the max pairwise subgroup difference in net benefit —
            with bootstrap CIs.
        """
        if thresholds is None:
            thresholds_arr: npt.NDArray[np.floating[Any]] = np.linspace(0.02, 0.40, 20)
        else:
            thresholds_arr = np.asarray(thresholds, dtype=np.float64)

        # The cache key must encode the thresholds: two calls with different
        # grids are different results, and a bare "decision_curve" key would
        # silently return the first call's frame for every later call.
        cache_key = (
            f"decision_curve_{thresholds_arr.tobytes().hex()}_{int(zero_crossing)}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        rng = np.random.default_rng(self._random_state)
        rows: list[dict[str, Any]] = []

        for attr_name, attr_values in self._sensitive.items():
            groups = np.unique(attr_values)
            is_interaction = attr_name in self._interactions

            for subgroup_label, y_t, y_p in self._iter_overall_and_subgroups(
                attr_name, attr_values, groups
            ):
                n_events = int(y_t.sum())
                shrunk = is_interaction and self._needs_shrinkage(
                    attr_name,
                    subgroup_label,
                    n_events,
                )
                prevalence = float(y_t.mean())

                # --- Point estimate via dcurves ---
                # Skip dcurves if subgroup has zero events or zero non-events
                # (dcurves divides by zero in this case).
                if n_events == 0 or n_events == len(y_t):
                    model_df = pd.DataFrame(
                        {
                            "net_benefit": [float("nan")] * len(thresholds_arr),
                            "threshold": thresholds_arr,
                        }
                    )
                    all_df = model_df.copy()
                else:
                    dca_input = pd.DataFrame(
                        {
                            "outcome": y_t.astype(int),
                            "model": y_p,
                        }
                    )
                    dca_result: pd.DataFrame = dca(
                        data=dca_input,
                        outcome="outcome",
                        modelnames=["model"],
                        thresholds=thresholds_arr.tolist(),
                    )
                    model_df = (
                        dca_result[dca_result["model"] == "model"]
                        .sort_values("threshold")
                        .reset_index(drop=True)
                    )
                    all_df = (
                        dca_result[dca_result["model"] == "all"]
                        .sort_values("threshold")
                        .reset_index(drop=True)
                    )

                # --- Bootstrap CIs on net benefit ---
                boot_indices = self._stratified_bootstrap_indices(y_t, rng)
                boot_nbs: list[npt.NDArray[np.floating[Any]]] = []
                # Standardized net benefit is NB divided by prevalence, and
                # under the patient scheme prevalence varies between
                # replicates. Dividing the NB interval by the *observed*
                # prevalence would hold it fixed and understate the interval
                # exactly the way the frozen resampler does, so the ratio is
                # formed per replicate and the percentile taken afterwards.
                boot_prev: list[float] = []
                for idx in boot_indices:
                    if idx is None:
                        boot_nbs.append(
                            np.full(len(thresholds_arr), float("nan"), dtype=np.float64)
                        )
                        boot_prev.append(float("nan"))
                        continue
                    boot_nbs.append(_compute_net_benefits(y_t[idx], y_p[idx], thresholds_arr))
                    boot_prev.append(float(y_t[idx].mean()))
                boot_matrix = np.array(boot_nbs)
                prev_vec = np.array(boot_prev, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    snb_matrix = np.where(
                        prev_vec[:, None] > 0, boot_matrix / prev_vec[:, None], np.nan
                    )

                crossings = (
                    _zero_crossings(
                        y_t, y_p, float(thresholds_arr[0]), float(thresholds_arr[-1])
                    )
                    if zero_crossing
                    else []
                )
                first_crossing = float(crossings[0]) if crossings else float("nan")

                for i, pt in enumerate(thresholds_arr):
                    pt_f = float(pt)
                    nb = (
                        float(model_df["net_benefit"].iloc[i])
                        if i < len(model_df)
                        else float("nan")
                    )
                    treat_all = (
                        float(all_df["net_benefit"].iloc[i]) if i < len(all_df) else float("nan")
                    )

                    # Bootstrap CIs
                    if shrunk:
                        ci_low = float("nan")
                        ci_high = float("nan")
                    else:
                        col_vals = boot_matrix[:, i]
                        if np.all(np.isnan(col_vals)):
                            ci_low = float("nan")
                            ci_high = float("nan")
                        else:
                            ci_low = float(np.nanpercentile(col_vals, 2.5))
                            ci_high = float(np.nanpercentile(col_vals, 97.5))

                    # Standardized net benefit (Naderalvojoud 2025)
                    snb = nb / prevalence if prevalence > 0 else float("nan")
                    if shrunk:
                        snb_low = snb_high = float("nan")
                    else:
                        snb_vals = snb_matrix[:, i]
                        if np.all(np.isnan(snb_vals)):
                            snb_low = snb_high = float("nan")
                        else:
                            snb_low = float(np.nanpercentile(snb_vals, 2.5))
                            snb_high = float(np.nanpercentile(snb_vals, 97.5))

                    row_out: dict[str, Any] = {
                        "attribute": attr_name,
                        "subgroup": subgroup_label,
                        "shrunk": shrunk,
                        "threshold": pt_f,
                        "net_benefit": nb,
                        "net_benefit_ci_low": ci_low,
                        "net_benefit_ci_high": ci_high,
                        "standardized_net_benefit": snb,
                        "standardized_net_benefit_ci_low": snb_low,
                        "standardized_net_benefit_ci_high": snb_high,
                        "treat_all_nb": treat_all,
                        "treat_none_nb": 0.0,
                    }
                    if zero_crossing:
                        row_out["zero_crossing"] = first_crossing
                    rows.append(row_out)

            # --- DCA gap row ---
            gap_val = self._dca_gap_point(attr_values, groups, thresholds_arr)
            gap_ci = self._bootstrap_dca_gaps(attr_values, groups, thresholds_arr, rng)
            g_lo, g_hi = _validate_gap_ci(gap_val, gap_ci[0], gap_ci[1])
            rows.append(
                {
                    "attribute": attr_name,
                    "subgroup": "GAP",
                    "shrunk": False,
                    "threshold": float("nan"),
                    "net_benefit": float("nan"),
                    "net_benefit_ci_low": float("nan"),
                    "net_benefit_ci_high": float("nan"),
                    "standardized_net_benefit": float("nan"),
                    "standardized_net_benefit_ci_low": float("nan"),
                    "standardized_net_benefit_ci_high": float("nan"),
                    "treat_all_nb": float("nan"),
                    "treat_none_nb": float("nan"),
                    "gap_metric": "max_net_benefit_gap",
                    "gap_value": gap_val,
                    "gap_ci_low": g_lo,
                    "gap_ci_high": g_hi,
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "attribute",
                "subgroup",
                "shrunk",
                "threshold",
                "net_benefit",
                "net_benefit_ci_low",
                "net_benefit_ci_high",
                "standardized_net_benefit",
                "standardized_net_benefit_ci_low",
                "standardized_net_benefit_ci_high",
                "treat_all_nb",
                "treat_none_nb",
                "gap_metric",
                "gap_value",
                "gap_ci_low",
                "gap_ci_high",
                *(["zero_crossing"] if zero_crossing else []),
            ],
        )
        df = self._apply_event_floors(df, "decision_curve")
        self._cache[cache_key] = df
        return df

    # ------------------------------------------------------------------
    # Public API: recalibration_ladder()
    # ------------------------------------------------------------------

    def _fit_ladder(
        self,
        by: str,
        method: str,
        folds: npt.NDArray[np.intp],
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Cross-fit the two grouped rungs, returning one score vector per rung.

        Both grouped rungs come out of a *single* ``fit_transform`` per fold.
        :class:`~isitfair.GroupRecalibration` loops over sensitive attributes
        independently, so passing a constant attribute alongside the real one
        yields the cohort-wide transform and the group-specific transforms from
        the same call — which makes it impossible for the two rungs to disagree
        about the fold partition, and that disagreement is the whole reason a
        ladder can be misread.
        """
        from isitfair.mitigation import GroupRecalibration

        y, p = self._y_true, self._y_proba
        groups = self._sensitive[by]
        out: dict[str, npt.NDArray[np.floating[Any]]] = {
            "common": np.full(len(y), np.nan, dtype=np.float64),
            "group_specific": np.full(len(y), np.nan, dtype=np.float64),
        }

        for k in sorted(set(folds.tolist())):
            test = folds == k
            train = ~test
            for stratum in sorted({str(g) for g in groups}):
                in_train = train & (groups.astype(str) == stratum)
                if int(y[in_train].sum()) == 0:
                    msg = (
                        f"fold {k}, stratum {stratum!r}: zero events in the "
                        "training union. Isotonic would silently return a "
                        "constant and Platt would raise; neither may be "
                        "papered over. Use fewer folds, or drop the stratum."
                    )
                    raise ValueError(msg)

            const_train = np.full(int(train.sum()), "all")
            const_test = np.full(int(test.sum()), "all")
            result = GroupRecalibration(method=method).fit_transform(
                y[train],
                p[train],
                p[test],
                {"__cohort__": const_train, by: groups[train]},
                {"__cohort__": const_test, by: groups[test]},
            )
            out["common"][test] = result["__cohort__"]
            out["group_specific"][test] = result[by]

        for name, vec in out.items():
            if np.any(np.isnan(vec)):
                msg = f"{method}/{name}: {int(np.isnan(vec).sum())} rows never scored"
                raise ValueError(msg)
        return out

    def recalibration_ladder(
        self,
        *,
        by: str,
        method: str = "isotonic",
        folds: Any = None,
        cv: int = 5,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Compare no recalibration, common cohort-wide, and group-specific.

        The three rungs are the reportable object, not the transform. Climbing
        them in order is what separates a *shift* that affects every patient
        from a *gradient* that differs between subgroups: correcting the shift
        first is what tells you whether group-specific correction is needed at
        all, and skipping that step attributes a global problem to a subgroup.

        Both fitted rungs are **cross-fitted**, so no observation is scored by a
        calibrator fitted on itself.

        Parameters
        ----------
        by : str
            Name of the sensitive attribute the group-specific rung may see.
            Must be one of this audit's marginal attributes.
        method : {"isotonic", "platt", "beta"}, default "isotonic"
            Calibrator family, passed to
            :class:`~isitfair.GroupRecalibration`.
        folds : array-like of int, or None
            Fold assignment, one entry per observation. **Pass the model's own
            cross-validation folds when it had any.** A ladder cross-fitted on
            fresh folds is a different analysis: the calibrator would see rows
            the model was tested on, and the leakage profile no longer matches
            the one the predictions were generated under. ``None`` generates a
            stratified *cv*-fold split from *random_state*.
        cv : int, default 5
            Number of folds when *folds* is None.
        random_state : int or None
            Seed for the generated split. ``None`` (default) uses the audit's
            own *random_state*, so a seeded audit gives a reproducible ladder.
            Ignored when *folds* is given, which is the point: supplying folds
            removes the randomness entirely.

        Returns
        -------
        pandas.DataFrame
            One row per ``(rung, subgroup)`` — rung in ``"none"``,
            ``"common"``, ``"group_specific"`` — with ``n``, ``n_events``,
            ``calibration_intercept`` (+ CI), ``calibration_slope`` (+ CI),
            ``brier`` (+ CI), ``ece``, ``ici``, ``net_benefit`` at this audit's
            threshold, and the event-floor columns. A ``GAP`` row per rung
            carries ``max_intercept_gap``, so the residual gradient after each
            rung can be read directly.

        Raises
        ------
        ValueError
            If *by* is not a marginal attribute, if *folds* has the wrong
            length, or if any (fold, stratum) training union has zero events.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 800
        >>> g = rng.choice(["a", "b"], n)
        >>> y = rng.binomial(1, 0.3, n).astype(float)
        >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
        >>> a = FairnessAudit(y, p, {"g": g}, threshold=0.3,
        ...                   n_bootstrap=20, random_state=0)
        >>> lad = a.recalibration_ladder(by="g", random_state=0)
        >>> sorted(lad["rung"].unique())
        ['common', 'group_specific', 'none']
        """
        if by not in self._sensitive or by in self._interactions:
            msg = (
                f"`by` must be one of this audit's marginal attributes "
                f"{sorted(set(self._sensitive) - set(self._interactions))}, got {by!r}"
            )
            raise ValueError(msg)

        if random_state is None:
            random_state = self._random_state
        cache_key = f"ladder_{by}_{method}_{cv}_{random_state}_{folds is not None}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        y = self._y_true
        if folds is None:
            from sklearn.model_selection import StratifiedKFold

            fold_arr: npt.NDArray[np.intp] = np.empty(len(y), dtype=np.intp)
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
            for k, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
                fold_arr[test_idx] = k
        else:
            fold_arr = np.asarray(folds, dtype=np.intp)
            if len(fold_arr) != len(y):
                msg = f"`folds` has length {len(fold_arr)}, expected {len(y)}"
                raise ValueError(msg)

        scores = {"none": self._y_proba, **self._fit_ladder(by, method, fold_arr)}
        self._ladder_scores[cache_key] = scores

        groups = self._sensitive[by]
        thresholds = np.array([self._threshold])
        rows: list[dict[str, Any]] = []
        for rung in ("none", "common", "group_specific"):
            p_rung = scores[rung]
            rng = np.random.default_rng(self._random_state)
            intercepts: list[float] = []
            for label, mask in [("Overall", np.ones(len(y), dtype=bool))] + [
                (str(g), groups.astype(str) == str(g))
                for g in np.unique(groups.astype(str))
            ]:
                y_t, y_p = y[mask], p_rung[mask]
                point = _compute_calibration_boot_metrics(y_t, y_p)
                cis = self._bootstrap_calibration_metrics(y_t, y_p, rng)
                n_events = int(y_t.sum())
                if label != "Overall":
                    intercepts.append(point["calibration_intercept"])
                row: dict[str, Any] = {
                    "rung": rung,
                    "attribute": by,
                    "subgroup": label,
                    "n": float(len(y_t)),
                    "n_events": float(n_events),
                    "ece": _compute_ece(y_t, y_p, self._ece_bins),
                    "ici": _compute_ici(y_t, y_p),
                    "net_benefit": float(
                        _compute_net_benefits(y_t, y_p, thresholds)[0]
                    ),
                }
                row.update(point)
                for m, prefix in (
                    ("calibration_intercept", "intercept"),
                    ("calibration_slope", "slope"),
                    ("brier", "brier"),
                ):
                    lo, hi = cis[m]
                    below = n_events >= 0 and floor_fails(n_events, m, self._event_floors)
                    row[f"{prefix}_ci_low"] = float("nan") if below else lo
                    row[f"{prefix}_ci_high"] = float("nan") if below else hi
                floor = self._event_floors.get("calibration_intercept")
                row["event_floor"] = float(floor) if floor is not None else float("nan")
                row["below_floor"] = bool(
                    floor is not None
                    and floor_fails(n_events, "calibration_intercept", self._event_floors)
                )
                row["not_estimable_final"] = bool(
                    row["below_floor"]
                    or floor_fails(n_events, "calibration_slope", self._event_floors)
                )
                rows.append(row)

            finite = [v for v in intercepts if not np.isnan(v)]
            rows.append(
                {
                    "rung": rung,
                    "attribute": by,
                    "subgroup": "GAP",
                    "n": float("nan"),
                    "n_events": float("nan"),
                    "gap_metric": "max_intercept_gap",
                    "gap_value": (
                        max(finite) - min(finite) if len(finite) >= 2 else float("nan")
                    ),
                }
            )

        df = pd.DataFrame(rows)
        ordered = [
            "rung", "attribute", "subgroup", "n", "n_events",
            "calibration_intercept", "intercept_ci_low", "intercept_ci_high",
            "calibration_slope", "slope_ci_low", "slope_ci_high",
            "brier", "brier_ci_low", "brier_ci_high",
            "ece", "ici", "net_benefit",
            "gap_metric", "gap_value",
            "event_floor", "below_floor", "not_estimable_final",
        ]
        for col in ordered:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[ordered]
        self._cache[cache_key] = df
        return df

    def recalibration_ladder_scores(
        self,
        *,
        by: str,
        method: str = "isotonic",
        folds: Any = None,
        cv: int = 5,
        random_state: int | None = None,
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """The out-of-fold score vector for each rung of the ladder.

        Same arguments as :meth:`recalibration_ladder`, which must produce the
        same partition. Returned so that reliability diagrams across the rungs
        can be drawn from the scores the table describes, rather than from a
        second, differently-partitioned fit.
        """
        if random_state is None:
            random_state = self._random_state
        cache_key = f"ladder_{by}_{method}_{cv}_{random_state}_{folds is not None}"
        if cache_key not in self._ladder_scores:
            self.recalibration_ladder(
                by=by, method=method, folds=folds, cv=cv, random_state=random_state
            )
        return self._ladder_scores[cache_key]

    # ------------------------------------------------------------------
    # Public API: run_all()
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, pd.DataFrame]:
        """Run all three analysis axes and return results.

        Runs every axis once. Results are cached, so calling ``run_all()``
        followed by individual methods incurs no extra cost.

        The curve axes and the estimability verdict are included because a
        method advertised as running everything should not quietly omit two of
        the three figure-bearing axes, nor the table that says which of its own
        numbers may be interpreted.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``"discrimination"``, ``"roc_curves"``, ``"calibration"``,
            ``"calibration_curves"``, ``"hosmer_lemeshow"``,
            ``"decision_curve"`` and ``"estimability"``.
        """
        return {
            "discrimination": self.discrimination(),
            "roc_curves": self.roc_curves(),
            "calibration": self.calibration(),
            "calibration_curves": self.calibration_curves(),
            "hosmer_lemeshow": self.hosmer_lemeshow(),
            "decision_curve": self.decision_curve(),
            "estimability": self.estimability(),
        }

    # ------------------------------------------------------------------
    # Public API: report()
    # ------------------------------------------------------------------

    def report(
        self,
        out: Any,
        title: str | None = None,
        metadata: dict[str, str] | None = None,
        mitigation_audits: dict[str, FairnessAudit] | None = None,
        table_one_data: Any = None,
        table_one_kwargs: dict[str, Any] | None = None,
        pdf: bool = False,
    ) -> Any:
        """Generate an HTML fairness report.

        Delegates to :func:`isitfair.report.generate_report`. The report
        includes all three analysis axes (discrimination, calibration,
        clinical utility) with embedded plots, tables, and gap statistics.
        Publication-ready 300-DPI PNGs of every plot are also written to a
        ``"<report-stem>_figures/"`` directory alongside the HTML file.

        Parameters
        ----------
        out : str or Path
            Output file path for the HTML report.
        title : str or None, optional
            Report title. Defaults to ``"Fairness Audit Report"``.
        metadata : dict[str, str] or None, optional
            Key-value pairs to display in the header.
        mitigation_audits : dict[str, FairnessAudit] or None, optional
            Mitigated audits for before/after comparison (as returned by
            :meth:`mitigate`).
        table_one_data : pandas.DataFrame, TableOneResult, or None, optional
            If given, a Table 1 section is rendered between the audit
            summary and the discrimination section. Pass a DataFrame to
            let the report call :func:`isitfair.table_one` for you, or a
            pre-built :class:`~isitfair.TableOneResult`.
        table_one_kwargs : dict, optional
            Forwarded to :func:`isitfair.table_one` when ``table_one_data``
            is a DataFrame. ``group_col`` defaults to the audit's first
            sensitive attribute.
        pdf : bool, default False
            Also write a print-typeset PDF beside the HTML (same stem,
            ``.pdf``); see :func:`isitfair.report_to_pdf`. Needs
            ``pip install isitfair[pdf]``.

        Returns
        -------
        Path
            The path to the generated HTML report file.

        Examples
        --------
        >>> audit.report("report.html", title="My Model Audit")  # doctest: +SKIP
        PosixPath('report.html')
        >>> audit.report("report.html", pdf=True)   # also writes report.pdf  # doctest: +SKIP
        PosixPath('report.html')
        """
        from isitfair.report import generate_report

        return generate_report(
            audit=self,
            out=out,
            title=title,
            metadata=metadata,
            mitigation_audits=mitigation_audits,
            table_one_data=table_one_data,
            table_one_kwargs=table_one_kwargs,
            pdf=pdf,
        )

    # ------------------------------------------------------------------
    # Public API: mitigate()
    # ------------------------------------------------------------------

    def mitigate(
        self,
        method: Any,
        calibration_data: tuple[Any, Any, dict[str, Any]],
    ) -> dict[str, FairnessAudit]:
        """Apply a mitigation method and return new audits with mitigated probabilities.

        This is a convenience method that runs a :class:`~isitfair.mitigation.Mitigation`
        instance on the audit's own test data and returns a *new*
        ``FairnessAudit`` per sensitive attribute, each with the same
        ``y_true`` but mitigated ``y_proba``.

        Parameters
        ----------
        method : Mitigation
            A mitigation object (e.g.
            ``GroupRecalibration(method="isotonic")``).
        calibration_data : tuple[array-like, array-like, dict[str, array-like]]
            A 3-tuple ``(y_true_calib, y_proba_calib, sensitive_calib)``.

        Returns
        -------
        dict[str, FairnessAudit]
            One audit per attribute name. Each has the same ``y_true``,
            ``threshold``, ``n_bootstrap``, ``random_state``, and
            ``sensitive_features`` as the original, but ``y_proba`` is
            replaced with the mitigated probabilities for that attribute.

        Examples
        --------
        >>> from isitfair.mitigation import GroupRecalibration
        >>> recalibrated = audit.mitigate(  # doctest: +SKIP
        ...     method=GroupRecalibration(method="isotonic"),
        ...     calibration_data=(y_calib, p_calib, {"sex": sex_calib}),
        ... )
        >>> recalibrated["sex"].calibration()  # compare to audit.calibration()  # doctest: +SKIP
        """
        from isitfair.mitigation import Mitigation, Reweighing

        if isinstance(method, Reweighing):
            msg = (
                "Reweighing cannot be used with `mitigate()` because it "
                "requires model retraining. Use `Reweighing.compute_weights()` "
                "to get sample weights, retrain your model with those weights, "
                "then construct a new FairnessAudit on the retrained predictions."
            )
            raise NotImplementedError(msg)

        if not isinstance(method, Mitigation):
            msg = "`method` must be a Mitigation instance"
            raise TypeError(msg)

        y_calib, p_calib, sens_calib = calibration_data

        # Build sensitive_test from marginal attributes only (no interactions)
        sensitive_test: dict[str, npt.NDArray[Any]] = {
            name: arr for name, arr in self._sensitive.items() if name not in self._interactions
        }

        mitigated = method.fit_transform(
            y_true_calib=y_calib,
            y_proba_calib=p_calib,
            y_proba_test=self._y_proba,
            sensitive_calib=sens_calib,
            sensitive_test=sensitive_test,
        )

        # Reconstruct sensitive_features with original key types
        sf: dict[str | tuple[str, ...], Any] = {}
        for name, arr in self._sensitive.items():
            if name in self._interactions:
                sf[self._interactions[name]] = arr
            else:
                sf[name] = arr

        # Build a new FairnessAudit per attribute
        result: dict[str, FairnessAudit] = {}
        for attr_name, new_proba in mitigated.items():
            result[attr_name] = FairnessAudit(
                y_true=self._y_true,
                y_proba=new_proba,
                sensitive_features=sf,
                threshold=self._threshold,
                n_bootstrap=self._n_bootstrap,
                random_state=self._random_state,
                ece_bins=self._ece_bins,
                intersectional_min_events=self._intersectional_min_events,
                dca_ylim=self._dca_ylim,
                resample_scheme=self._resample_scheme,
                nonestimable_threshold=self._nonestimable_threshold,
                event_floors=self._event_floors,
                n_jobs=self._n_jobs,
            )
        return result

    # ------------------------------------------------------------------
    # Public API: cell_diagnostics()
    # ------------------------------------------------------------------

    def cell_diagnostics(self) -> pd.DataFrame:
        """Return a diagnostic DataFrame for interaction cells.

        One row per interaction cell with shrinkage metadata.

        Returns
        -------
        pd.DataFrame
            Columns: ``attribute``, ``subgroup``, ``n``, ``n_events``,
            ``shrunk``, ``marginal_rate``, ``cell_rate``,
            ``shrinkage_factor``.
            Empty (correct columns) when no interactions are defined.
        """
        col_names = [
            "attribute",
            "subgroup",
            "n",
            "n_events",
            "shrunk",
            "marginal_rate",
            "cell_rate",
            "shrinkage_factor",
        ]
        if "cell_diagnostics" in self._cache:
            return self._cache["cell_diagnostics"]

        if not self._interactions:
            df = pd.DataFrame(columns=col_names)
            self._cache["cell_diagnostics"] = df
            return df

        # Build marginal metrics cache (prevalence only)
        marginal_prev: dict[str, dict[str, float]] = {}
        for attr_name, attr_values in self._sensitive.items():
            if attr_name in self._interactions:
                continue
            groups = np.unique(attr_values)
            marginal_prev[attr_name] = {}
            for group in groups:
                mask = attr_values == group
                y_t = self._y_true[mask]
                n_events = int(y_t.sum())
                n_groups = len(groups)
                marginal_prev[attr_name][str(group)] = float(y_t.mean())
            marginal_prev[attr_name]["_n_groups"] = float(n_groups)
            marginal_prev[attr_name]["_total_events"] = float(
                sum(int(self._y_true[attr_values == g].sum()) for g in groups)
            )

        rows: list[dict[str, Any]] = []
        for attr_name in self._interactions:
            attr_values = self._sensitive[attr_name]
            constituents = self._interactions[attr_name]
            groups = np.unique(attr_values)

            # Compute prior_strength
            strengths: list[float] = []
            for c in constituents:
                te = marginal_prev[c]["_total_events"]
                ng = marginal_prev[c]["_n_groups"]
                strengths.append(te / ng if ng > 0 else 0.0)
            prior_strength = float(np.mean(strengths))

            for group in groups:
                mask = attr_values == group
                n = int(mask.sum())
                n_events = int(self._y_true[mask].sum())
                cell_rate = n_events / n if n > 0 else float("nan")
                shrunk = self._needs_shrinkage(attr_name, str(group), n_events)

                # Marginal rate from first constituent
                first_attr = constituents[0]
                first_label = self._cell_constituents[attr_name][str(group)][0]
                m_rate = marginal_prev[first_attr].get(first_label, float("nan"))

                # Shrinkage factor: prior_strength / (prior_strength + n)
                shrinkage_factor = prior_strength / (prior_strength + n) if n > 0 else float("nan")

                rows.append(
                    {
                        "attribute": attr_name,
                        "subgroup": str(group),
                        "n": n,
                        "n_events": n_events,
                        "shrunk": shrunk,
                        "marginal_rate": m_rate,
                        "cell_rate": cell_rate,
                        "shrinkage_factor": shrinkage_factor,
                    }
                )

        df = pd.DataFrame(rows, columns=col_names)
        self._cache["cell_diagnostics"] = df
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_overall_and_subgroups(
        self,
        attr_name: str,
        attr_values: npt.NDArray[Any],
        groups: npt.NDArray[Any],
    ) -> list[tuple[str, npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
        """Yield (label, y_true_slice, y_proba_slice) for Overall + each subgroup."""
        result: list[tuple[str, npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]] = [
            ("Overall", self._y_true, self._y_proba)
        ]
        for group in groups:
            mask = attr_values == group
            result.append((str(group), self._y_true[mask], self._y_proba[mask]))
        return result
