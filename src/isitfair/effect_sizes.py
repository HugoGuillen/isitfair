"""Effect-size statistics for two- and multi-group comparisons.

Effect sizes complement p-values: a tiny p-value with a negligible effect size is
often clinically uninteresting, and a large effect with a non-significant p-value
may still be worth investigating. The fairness audit reports both.

The functions in this module accept array-likes (lists, NumPy arrays, pandas
Series) and return plain floats. They are pure, deterministic, and have no
side effects.

The :func:`effect_size_label` helper applies Cohen's conventional cutoffs
(0.2 / 0.5 / 0.8) to ``d`` and ``h`` statistics. Different cutoffs apply to
Cramér's V and rank-biserial r — pass ``kind`` to use the appropriate ones.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

ArrayLike = Any
EffectKind = Literal["d", "h", "v", "r", "eta2"]


def _as_float_array(x: ArrayLike, name: str) -> npt.NDArray[np.floating[Any]]:
    """Coerce input to a 1-D float array, dropping NaN values."""
    arr: npt.NDArray[np.floating[Any]] = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    finite: npt.NDArray[np.floating[Any]] = arr[~np.isnan(arr)]
    return finite


def cohens_d(group_a: ArrayLike, group_b: ArrayLike) -> float:
    """Cohen's d for the difference in means of two independent samples.

    Computed with the pooled standard deviation using ``ddof=1`` (unbiased
    sample variance). NaN values are dropped before computation.

    Parameters
    ----------
    group_a, group_b : array-like
        Two independent samples of continuous values.

    Returns
    -------
    float
        ``(mean_a - mean_b) / s_pooled``. Returns ``nan`` if either group has
        fewer than 2 finite observations; returns ``0.0`` if the pooled
        standard deviation is exactly zero.

    Examples
    --------
    >>> round(cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]), 3)
    -1.265
    """
    a = _as_float_array(group_a, "group_a")
    b = _as_float_array(group_b, "group_b")
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    s_pool = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)))
    if s_pool == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / s_pool)


def cohens_h(p_a: float, p_b: float) -> float:
    """Cohen's h for the difference between two proportions.

    Uses the arcsine-square-root transform, which stabilises variance and
    makes the statistic comparable across the full ``[0, 1]`` range
    (unlike a raw difference, which compresses near 0 and 1).

    Parameters
    ----------
    p_a, p_b : float
        Two proportions in ``[0, 1]``.

    Returns
    -------
    float
        ``2 * arcsin(sqrt(p_a)) - 2 * arcsin(sqrt(p_b))``.

    Raises
    ------
    ValueError
        If either proportion is outside ``[0, 1]`` or is NaN.

    Examples
    --------
    >>> round(cohens_h(0.20, 0.10), 3)
    0.284
    """
    if not (0.0 <= p_a <= 1.0) or not (0.0 <= p_b <= 1.0):
        raise ValueError(f"proportions must be in [0, 1]; got p_a={p_a}, p_b={p_b}")
    return float(2.0 * np.arcsin(np.sqrt(p_a)) - 2.0 * np.arcsin(np.sqrt(p_b)))


def cramers_v(a: ArrayLike, b: ArrayLike) -> float:
    """Cramér's V for association between two categorical variables.

    Cramér's V ranges from 0 (no association) to 1 (perfect association)
    and is the chi-square statistic normalised by sample size and the
    smaller marginal degree of freedom. It is symmetric in its arguments
    and works for any number of categories.

    Parameters
    ----------
    a, b : array-like
        Two categorical arrays of the same length. Values are compared by
        equality; pass strings, ints, or mixed types — anything hashable.

    Returns
    -------
    float
        Cramér's V in ``[0, 1]``. Returns ``0.0`` when either variable has
        only one observed category (degree of freedom is zero).

    Examples
    --------
    >>> cramers_v(["a", "a", "b", "b", "b"], ["x", "x", "y", "y", "y"])
    1.0
    """
    arr_a = pd.Series(a)
    arr_b = pd.Series(b)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"length mismatch: {len(arr_a)} vs {len(arr_b)}")
    ct = pd.crosstab(arr_a, arr_b)
    n = int(ct.values.sum())
    k = min(ct.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    chi2_stat = float(stats.chi2_contingency(ct.values, correction=False)[0])
    return float(np.sqrt(chi2_stat / (n * k)))


def rank_biserial_r(group_a: ArrayLike, group_b: ArrayLike) -> float:
    """Rank-biserial correlation r for two independent samples.

    Derived from the Mann–Whitney U statistic:
    ``r = 2 * U / (n_a * n_b) - 1``, where ``U`` is the U statistic from
    SciPy's two-sided ``mannwhitneyu`` for ``(group_a, group_b)``. Ranges
    from -1 to +1: **positive means ``group_a`` tends to be larger**,
    matching the sign convention of :func:`cohens_d`. Some software reports
    the inverted convention ``1 - 2U/(n_a n_b)``; flip the sign when
    comparing with it.

    Parameters
    ----------
    group_a, group_b : array-like
        Two independent samples of ordinal or continuous values.

    Returns
    -------
    float
        Rank-biserial r in ``[-1, 1]``. Returns ``nan`` if either group is
        empty after NaN removal.

    Examples
    --------
    >>> rank_biserial_r([5, 6, 7, 8], [1, 2, 3, 4])
    1.0
    """
    a = _as_float_array(group_a, "group_a")
    b = _as_float_array(group_b, "group_b")
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * float(u_stat) / (n_a * n_b) - 1.0)


_CUTOFFS: dict[EffectKind, tuple[float, float, float]] = {
    # Cohen (1988) conventions for standardised mean / proportion differences.
    "d": (0.2, 0.5, 0.8),
    "h": (0.2, 0.5, 0.8),
    # Cohen (1988) Section 7.3 for Cramér's V at df=1; the most common case.
    "v": (0.1, 0.3, 0.5),
    # Rank-biserial r interpreted on the same scale as Pearson r.
    "r": (0.1, 0.3, 0.5),
    # Cohen (1988) for variance explained: eta^2 is on a squared scale, so the
    # cutoffs are the squares of the r conventions rather than the same numbers.
    "eta2": (0.01, 0.06, 0.14),
}


def eta_squared(values: ArrayLike, groups: ArrayLike) -> float:
    """One-way η²: the share of variance in *values* explained by *groups*.

    The effect size matched to a continuous variable compared across an
    attribute with **more than two** levels, where Cohen's d does not apply.

    Parameters
    ----------
    values : array-like
        Continuous measurements.
    groups : array-like
        Group label per observation, same length as *values*.

    Returns
    -------
    float
        η² in [0, 1]; 0 when the values are constant.

    Raises
    ------
    ValueError
        If the two inputs have different lengths.

    Examples
    --------
    >>> round(eta_squared([1.0, 1.1, 5.0, 5.2], ["a", "a", "b", "b"]), 3)
    0.998
    """
    vals = np.asarray(values, dtype=float)
    grp = np.asarray(groups)
    if len(vals) != len(grp):
        msg = f"`values` and `groups` must have the same length, got {len(vals)} and {len(grp)}"
        raise ValueError(msg)
    grand_mean = float(np.mean(vals))
    ss_total = float(np.sum((vals - grand_mean) ** 2))
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for lvl in pd.unique(grp):
        mask = grp == lvl
        if mask.sum() == 0:
            continue
        ss_between += int(mask.sum()) * (float(np.mean(vals[mask])) - grand_mean) ** 2
    return float(ss_between / ss_total)


def effect_size_label(value: float, kind: EffectKind = "d") -> str:
    """Map an effect-size magnitude to a qualitative label.

    Parameters
    ----------
    value : float
        Effect-size estimate; sign is ignored.
    kind : {'d', 'h', 'v', 'r', 'eta2'}, default 'd'
        Which statistic ``value`` represents. Cutoffs:

        - ``'d'`` (Cohen's d) and ``'h'`` (Cohen's h): 0.2 / 0.5 / 0.8.
        - ``'v'`` (Cramér's V) and ``'r'`` (rank-biserial r):
          0.1 / 0.3 / 0.5.
        - ``'eta2'`` (η²): 0.01 / 0.06 / 0.14, the conventional
          small/medium/large cutoffs for a variance-explained statistic.

    Returns
    -------
    str
        One of ``'negligible'``, ``'small'``, ``'medium'``, ``'large'``.
        Returns ``'undefined'`` if ``value`` is NaN.

    Examples
    --------
    >>> effect_size_label(0.6, kind="d")
    'medium'
    >>> effect_size_label(0.15, kind="v")
    'small'
    """
    if np.isnan(value):
        return "undefined"
    if kind not in _CUTOFFS:
        raise ValueError(f"unknown kind: {kind!r}; expected one of {list(_CUTOFFS)}")
    small, medium, large = _CUTOFFS[kind]
    mag = abs(float(value))
    if mag < small:
        return "negligible"
    if mag < medium:
        return "small"
    if mag < large:
        return "medium"
    return "large"
