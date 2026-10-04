"""Per-subgroup classification metrics and calibration goodness-of-fit.

The :func:`subgroup_metrics` function is the primitive used internally by
:class:`isitfair.FairnessAudit` to evaluate a single subgroup slice. It is
exposed publicly so users running ad-hoc analyses (outside the full audit
pipeline) can compute the same metric set with the same NaN-handling
semantics — for example, comparing per-cluster performance from
out-of-fold predictions.

Metrics returned: ``n``, ``n_events``, ``prevalence``, ``auroc``, ``auprc``,
``sensitivity``, ``specificity``, ``ppv``, ``npv``. Each is ``nan`` exactly
when it is mathematically undefined for that slice, judged per metric — a
slice with no events leaves AUROC, AUPRC and sensitivity undefined but not
specificity or NPV.

:func:`hosmer_lemeshow` computes the Hosmer–Lemeshow C statistic together
with the supplementary information that Kramer & Zimmerman (2007) argue must
accompany it: the number of observations, the full per-decile table of
observed versus expected events, and a regression of the observed on the
expected decile means. The statistic on its own is sample-size dependent and
is not comparable across subgroups of differing size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import chi2, linregress
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

_HL_TABLE_COLUMNS = [
    "decile",
    "n",
    "observed",
    "expected",
    "observed_rate",
    "expected_rate",
    "difference",
]

_HL_DF_MODES = ("validation", "development")


def subgroup_metrics(
    y_true: Any,
    y_proba: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination metrics for one subgroup slice.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcome labels (0 or 1).
    y_proba : array-like of shape (n,)
        Predicted probabilities in ``[0, 1]``.
    threshold : float, default 0.5
        Decision threshold for converting probabilities to predicted classes
        for the confusion-matrix metrics (sensitivity, specificity, PPV, NPV).
        AUROC and AUPRC do not depend on this threshold.

    Returns
    -------
    dict[str, float]
        Keys: ``n``, ``n_events``, ``prevalence``, ``auroc``, ``auprc``,
        ``sensitivity``, ``specificity``, ``ppv``, ``npv``.

        Each metric is ``nan`` exactly when it is mathematically undefined,
        decided per metric rather than per slice:

        - ``auroc``, ``auprc`` — need both classes present.
        - ``sensitivity`` — needs at least one event.
        - ``specificity`` — needs at least one non-event. It is **defined** on
          a slice with no events, and is reported.
        - ``ppv`` — needs at least one positive prediction.
        - ``npv`` — needs at least one negative prediction.

        ``prevalence`` is defined whenever ``n > 0``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 100)
    >>> p = rng.random(100)
    >>> m = subgroup_metrics(y, p, threshold=0.5)
    >>> sorted(m)
    ['auprc', 'auroc', 'n', 'n_events', 'npv', 'ppv', 'prevalence', 'sensitivity', 'specificity']
    """
    y_true_arr: npt.NDArray[np.floating[Any]] = np.asarray(y_true, dtype=np.float64)
    y_proba_arr: npt.NDArray[np.floating[Any]] = np.asarray(y_proba, dtype=np.float64)

    n = len(y_true_arr)
    n_events = int(y_true_arr.sum())
    prevalence = float(y_true_arr.mean()) if n > 0 else float("nan")

    # Estimability is decided per metric, not per slice. A slice with no
    # events leaves AUROC, AUPRC and sensitivity undefined, but specificity
    # (TN / (TN + FP)) is perfectly well defined, as are NPV and — given any
    # positive prediction — PPV. Treating the whole slice as unusable
    # discarded information and, once these counts became reportable,
    # overstated how often each metric could not be computed.
    both_classes = 0 < n_events < n

    if n == 0:
        auroc = auprc = sensitivity = specificity = ppv = npv = float("nan")
    else:
        auroc = float(roc_auc_score(y_true_arr, y_proba_arr)) if both_classes else float("nan")
        auprc = (
            float(average_precision_score(y_true_arr, y_proba_arr))
            if both_classes
            else float("nan")
        )
        y_pred = (y_proba_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")

    return {
        "n": float(n),
        "n_events": float(n_events),
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }


# ---------------------------------------------------------------------------
# Hosmer–Lemeshow goodness-of-fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HosmerLemeshowResult:
    """Result of a Hosmer–Lemeshow goodness-of-fit test.

    ``eq=False`` is deliberate: the synthesised ``__eq__`` of a plain frozen
    dataclass compares a tuple of all fields, which raises
    ``ValueError: The truth value of a DataFrame is ambiguous`` because of the
    ``table`` field.

    Attributes
    ----------
    statistic : float
        The Hosmer–Lemeshow C statistic. ``nan`` when undefined (see
        :func:`hosmer_lemeshow`).
    df : float
        Degrees of freedom of the reference chi-square distribution.
        ``nan`` when fewer than three bins contribute.
    p_value : float
        Upper-tail probability of ``statistic`` under chi-square with ``df``
        degrees of freedom. ``nan`` when ``df`` is ``nan``.
    n : int
        Number of observations.
    n_events : int
        Number of observed events.
    n_bins_used : int
        Number of bins that contributed to the statistic. May be smaller than
        the requested ``n_bins`` when predicted probabilities are tied, when
        the slice is small, or when a bin is degenerate.
    bins_reduced : bool
        ``True`` when ``n_bins_used < n_bins``.
    decile_slope : float
        Slope of the least-squares regression of observed on expected decile
        rates. Perfect calibration gives 1.
    decile_intercept : float
        Intercept of the same regression. Perfect calibration gives 0.
    table : pandas.DataFrame
        Per-bin table with columns ``decile``, ``n``, ``observed``,
        ``expected``, ``observed_rate``, ``expected_rate``, ``difference``.
    """

    statistic: float
    df: float
    p_value: float
    n: int
    n_events: int
    n_bins_used: int
    bins_reduced: bool
    decile_slope: float
    decile_intercept: float
    table: pd.DataFrame


def _tie_snapped_bins(
    y_proba_sorted: npt.NDArray[np.floating[Any]],
    order: npt.NDArray[np.intp],
    n_bins: int,
) -> list[npt.NDArray[np.intp]]:
    """Split *order* into approximately equal-sized bins without splitting ties.

    Equal-frequency cut positions are computed first, then any cut that falls
    inside a run of identical predicted probabilities is moved to the nearer
    end of that run. This keeps bins as close to equal-count as the tie
    structure allows while making bin membership independent of the input row
    order — a naive ``np.array_split(np.argsort(p), g)`` assigns tied
    observations by their position in the argsort, so permuting the rows
    changes the statistic.
    """
    n = len(y_proba_sorted)
    interior = np.cumsum([len(s) for s in np.array_split(np.arange(n), min(n_bins, n))])[:-1]

    snapped: list[int] = []
    for cut in interior:
        e = int(cut)
        if e <= 0 or e >= n:
            continue
        value = y_proba_sorted[e]
        if y_proba_sorted[e - 1] != value:
            # The cut already sits on a value change; nothing to snap.
            snapped.append(e)
            continue
        start = int(np.searchsorted(y_proba_sorted, value, side="left"))
        stop = int(np.searchsorted(y_proba_sorted, value, side="right"))
        # Move to whichever end of the tie run is nearer; forward on a tie.
        snapped.append(start if (e - start) < (stop - e) else stop)

    edges = sorted({e for e in snapped if 0 < e < n})
    return [b for b in np.split(order, edges) if len(b) > 0]


def hosmer_lemeshow(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
    df_mode: str = "validation",
) -> HosmerLemeshowResult:
    """Hosmer–Lemeshow C statistic with Kramer's supplementary information.

    The C statistic groups observations into approximately equal-sized bins of
    predicted risk and compares observed with expected event counts:

    .. math::

        C = \\sum_j \\frac{(O_j - E_j)^2}{n_j \\bar{p}_j (1 - \\bar{p}_j)}

    where :math:`O_j` is the observed event count in bin *j*, :math:`E_j` the
    sum of predicted probabilities, :math:`n_j` the bin size and
    :math:`\\bar{p}_j = E_j / n_j`.

    Kramer & Zimmerman (2007) argue that this statistic must never be reported
    alone, because it scales with sample size: a clinically irrelevant
    deviation becomes significant in a large cohort, and a large deviation can
    go undetected in a small one. They recommend reporting the number of
    observations, the full per-bin observed-versus-expected table, and a
    regression of the observed on the expected bin means alongside it. This
    function returns all of those.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcome labels (0 or 1).
    y_proba : array-like of shape (n,)
        Predicted probabilities in ``[0, 1]``.
    n_bins : int, default 10
        Number of risk groups. Ten (deciles) is the convention, and is what
        Kramer's C statistic is defined over. Must be at least 2.
    df_mode : {"validation", "development"}, default "validation"
        Degrees of freedom of the reference chi-square distribution.

        - ``"validation"`` uses ``df = g``, correct when the predictions come
          from a model fitted elsewhere. This is what :mod:`isitfair` always
          evaluates, so it is the default.
        - ``"development"`` uses ``df = g - 2``, the classical Hosmer–Lemeshow
          convention, correct when the model's intercept and slope were
          estimated on the very data being tested.

        See Notes for the simulation behind this default.

    Returns
    -------
    HosmerLemeshowResult
        Statistic, degrees of freedom, p-value, bin counts, the decile-mean
        regression coefficients, and the per-bin table.

    Raises
    ------
    ValueError
        If ``n_bins < 2``, if ``df_mode`` is not one of the accepted values,
        if the inputs have different lengths, if ``y_true`` holds values other
        than 0 and 1, or if ``y_proba`` falls outside ``[0, 1]``.

    Notes
    -----
    **Binning and ties.** Bins are equal-frequency, with cut points snapped to
    tie boundaries so that identical predicted probabilities always land in the
    same bin. Without this, bin membership depends on the input row order, and
    for a model producing many tied scores (any tree ensemble) permuting the
    rows alone moves the p-value materially. Snapping also matches R's
    ``ResourceSelection::hoslem.test``, which cuts on value-based quantiles.
    Because ties are never split, ``n_bins_used`` can be smaller than
    ``n_bins``; ``bins_reduced`` flags this rather than failing silently.

    **Degrees of freedom.** ``df = g - 2`` is the penalty for estimating an
    intercept and a slope on the data under test, which imposes two linear
    constraints on the residuals. This package evaluates supplied predictions
    and never fits the model, so no degrees of freedom are spent. Simulation on
    well-calibrated data with ``g = 10`` confirms it: the statistic's mean and
    variance track chi-square with 10 df (mean 9.8–10.3, variance 20–25), and
    the type-I error rate at the nominal 5% level is 4.6–6.5% under ``df = g``
    against 10.6–14.2% under ``df = g - 2``. Both conventions are available and
    ``df`` is always reported, so the choice is never hidden.

    **Degenerate bins.** A bin in which every prediction is 0 (or every
    prediction is 1) has zero variance. If it contains no events (or only
    events) it contributes nothing and is excluded from ``n_bins_used``. If it
    contains an event the model gave probability zero, the statistic genuinely
    diverges and ``nan`` is returned — the denominator is deliberately not
    clamped, since that would manufacture a finite p-value from an infinite
    one. Very small but non-zero bin probabilities produce legitimately large
    contributions; this instability is one of Kramer's own arguments for
    reading the statistic alongside adjunct measures.

    **One-class slices.** The statistic, degrees of freedom and p-value are
    ``nan``, matching the convention in :func:`subgroup_metrics`. The per-bin
    table is still returned, because observed and expected counts per bin are
    well defined and are exactly the supplementary information Kramer asks for.

    **The decile regression is unweighted** least squares, as Kramer describes
    it. Bin rates are heteroscedastic, so weighted least squares would be more
    efficient, but the published recommendation is implemented verbatim so that
    a reader reproduces these numbers.

    This function is fully deterministic and consumes no random state.

    References
    ----------
    Kramer AA, Zimmerman JE. Assessing the calibration of mortality benchmarks
    in critical care: the Hosmer-Lemeshow test revisited. *Critical Care
    Medicine* 2007;35:2052–6.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.95, 500)
    >>> y = rng.binomial(1, p).astype(float)
    >>> res = hosmer_lemeshow(y, p, n_bins=10)
    >>> res.n_bins_used
    10
    >>> res.table.columns.tolist()
    ['decile', 'n', 'observed', 'expected', 'observed_rate', 'expected_rate', 'difference']
    >>> bool(res.p_value > 0.05)
    True
    """
    if n_bins < 2:
        msg = f"`n_bins` must be at least 2, got {n_bins}"
        raise ValueError(msg)
    if df_mode not in _HL_DF_MODES:
        msg = f"`df_mode` must be one of {_HL_DF_MODES}, got {df_mode!r}"
        raise ValueError(msg)

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_proba, dtype=np.float64)
    if y.shape != p.shape:
        msg = f"`y_true` and `y_proba` must have the same shape, got {y.shape} and {p.shape}"
        raise ValueError(msg)
    if y.size and not np.all((y == 0.0) | (y == 1.0)):
        msg = "`y_true` must contain only 0 and 1"
        raise ValueError(msg)
    if p.size and (np.nanmin(p) < 0.0 or np.nanmax(p) > 1.0):
        msg = "`y_proba` must lie in [0, 1]"
        raise ValueError(msg)

    n = int(y.size)
    n_events = int(y.sum()) if n else 0

    if n == 0:
        return HosmerLemeshowResult(
            statistic=float("nan"),
            df=float("nan"),
            p_value=float("nan"),
            n=0,
            n_events=0,
            n_bins_used=0,
            bins_reduced=True,
            decile_slope=float("nan"),
            decile_intercept=float("nan"),
            table=pd.DataFrame(columns=_HL_TABLE_COLUMNS),
        )

    # `kind="stable"` keeps the row-to-bin assignment reproducible; the tie
    # snapping below makes bin *membership* permutation-invariant regardless.
    order = np.argsort(p, kind="stable")
    bins = _tie_snapped_bins(p[order], order, n_bins)

    rows: list[dict[str, Any]] = []
    contributions: list[float] = []
    contributing_expected: list[float] = []
    contributing_observed: list[float] = []
    diverges = False

    for i, idx in enumerate(bins, start=1):
        n_j = len(idx)
        observed = float(y[idx].sum())
        expected = float(p[idx].sum())
        rows.append(
            {
                "decile": i,
                "n": float(n_j),
                "observed": observed,
                "expected": expected,
                "observed_rate": observed / n_j,
                "expected_rate": expected / n_j,
                "difference": observed - expected,
            }
        )

        denominator = expected * (n_j - expected) / n_j
        if denominator > 0.0:
            contributions.append((observed - expected) ** 2 / denominator)
            contributing_expected.append(expected / n_j)
            contributing_observed.append(observed / n_j)
        elif observed == expected:
            # All-zero (or all-one) predictions matched by the outcome: this
            # bin contributes nothing and does not count toward n_bins_used.
            continue
        else:
            # The model assigned probability zero to an event that occurred
            # (or probability one to a non-event): the statistic diverges.
            diverges = True

    table = pd.DataFrame(rows, columns=_HL_TABLE_COLUMNS)
    if not table.empty:
        table = table.astype({"decile": "int64"})

    n_bins_used = len(contributions)
    bins_reduced = n_bins_used < n_bins
    one_class = n_events == 0 or n_events == n

    statistic = float("nan") if diverges or one_class else float(np.sum(contributions))

    df_value = float(n_bins_used) if df_mode == "validation" else float(n_bins_used - 2)
    if n_bins_used < 3 or df_value <= 0 or np.isnan(statistic):
        df_out = float("nan")
        p_value = float("nan")
    else:
        df_out = df_value
        p_value = float(chi2.sf(statistic, df_value))

    slope, intercept = _decile_regression(
        np.asarray(contributing_expected, dtype=np.float64),
        np.asarray(contributing_observed, dtype=np.float64),
        one_class=one_class,
    )

    return HosmerLemeshowResult(
        statistic=statistic,
        df=df_out,
        p_value=p_value,
        n=n,
        n_events=n_events,
        n_bins_used=n_bins_used,
        bins_reduced=bins_reduced,
        decile_slope=slope,
        decile_intercept=intercept,
        table=table,
    )


def _decile_regression(
    expected_rate: npt.NDArray[np.floating[Any]],
    observed_rate: npt.NDArray[np.floating[Any]],
    *,
    one_class: bool,
) -> tuple[float, float]:
    """Unweighted least-squares fit of observed on expected bin rates.

    Kramer & Zimmerman (2007) recommend this as an adjunct to the statistic:
    a well-calibrated model gives a slope indistinguishable from 1 and an
    intercept indistinguishable from 0. Returns ``(nan, nan)`` when the fit
    would carry no information.
    """
    if one_class or len(expected_rate) < 3:
        return float("nan"), float("nan")
    if np.any(np.isnan(expected_rate)) or np.any(np.isnan(observed_rate)):
        return float("nan"), float("nan")
    if float(np.ptp(expected_rate)) == 0.0:
        return float("nan"), float("nan")

    fit = linregress(expected_rate, observed_rate)
    return float(fit.slope), float(fit.intercept)
