"""Inference utilities: multiple testing correction, interaction tests, bootstrap.

This module collects the small statistical primitives that fairness audits
need but that are not part of the audit pipeline proper:

- :func:`fdr_correct` — Benjamini–Hochberg false discovery rate.
- :func:`interaction_test` — likelihood-ratio test for a sensitive ×
  covariate interaction on a binary outcome (with a public low-level
  :func:`lrt`).
- :func:`bootstrap_metric_difference` — bootstrap CI and p-value for the
  between-group difference of a metric (paired by default).
- :func:`odds_ratio_2x2` — crude OR with Wald 95 % CI.

The :class:`MetricDifferenceResult`, :class:`InteractionResult`, and
:class:`OddsRatioResult` dataclasses make return values discoverable from
the REPL and friendly to type-checkers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ArrayLike = Any
MetricCallable = Callable[[npt.NDArray[Any], npt.NDArray[Any]], float]

# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def fdr_correct(
    pvals: ArrayLike,
    *,
    method: str = "fdr_bh",
) -> npt.NDArray[np.floating[Any]]:
    """Adjust p-values for multiple testing.

    Thin wrapper over :func:`statsmodels.stats.multitest.multipletests` that
    returns only the q-values, with NaN inputs preserved (statsmodels would
    raise on those).

    Parameters
    ----------
    pvals : array-like of float
        Raw p-values in ``[0, 1]``. NaN values are propagated.
    method : str, default ``'fdr_bh'``
        Any method accepted by ``statsmodels.stats.multitest.multipletests``.
        The default is Benjamini–Hochberg (1995), which controls the
        expected false discovery rate under independence and positive
        regression dependence. Use ``'fdr_by'`` for the Benjamini–Yekutieli
        variant valid under arbitrary dependence.

    Returns
    -------
    ndarray of float
        Adjusted p-values (q-values), same length as ``pvals``.

    Examples
    --------
    >>> fdr_correct([0.001, 0.01, 0.04, 0.5]).round(4).tolist()
    [0.004, 0.02, 0.0533, 0.5]
    """
    pvals_arr = np.asarray(pvals, dtype=float).ravel()
    mask = ~np.isnan(pvals_arr)
    out = np.full(pvals_arr.shape, np.nan, dtype=float)
    if mask.sum() == 0:
        return out
    _, q, _, _ = multipletests(pvals_arr[mask], method=method)
    out[mask] = q
    return out


# ---------------------------------------------------------------------------
# Likelihood-ratio test
# ---------------------------------------------------------------------------


def lrt(
    x_base: ArrayLike,
    x_full: ArrayLike,
    y: ArrayLike,
    *,
    c_penalty: float = 1e5,
    max_iter: int = 2000,
) -> tuple[float, float, int]:
    """Likelihood-ratio test between two nested logistic-regression models.

    The full model must include all columns of the base model as a prefix
    (the LRT degree of freedom is ``x_full.shape[1] - x_base.shape[1]``).
    Both models are fit with scikit-learn's :class:`LogisticRegression` with
    a very weak L2 penalty (``C=1e5``) to approximate unpenalised MLE while
    keeping fitting stable when the design matrix is ill-conditioned.

    Parameters
    ----------
    x_base, x_full : array-like of shape (n, p_base) and (n, p_full)
        Design matrices for the nested and full models. ``p_full > p_base``.
    y : array-like of shape (n,)
        Binary outcome (0 / 1).
    c_penalty : float, default 1e5
        Inverse regularisation strength passed to ``LogisticRegression``.
        Larger values approximate unpenalised MLE more closely; values below
        ~10 introduce material shrinkage and bias the LRT statistic.
    max_iter : int, default 2000
        Optimiser iteration cap.

    Returns
    -------
    p_value : float
        Right-tail probability under ``chi2(df)``.
    statistic : float
        ``2 * (loglik_full - loglik_base)``, clipped at 0 to avoid tiny
        negative values from numerical noise.
    df : int
        Degrees of freedom of the test (``x_full.shape[1] - x_base.shape[1]``).

    Raises
    ------
    ValueError
        If ``x_full`` does not have more columns than ``x_base``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x1 = rng.standard_normal(n)
    >>> x2 = rng.standard_normal(n)
    >>> y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1)))).astype(int)
    >>> p, _, df = lrt(x1.reshape(-1, 1), np.c_[x1, x2], y)
    >>> df
    1
    """
    xb = np.asarray(x_base, dtype=float)
    xf = np.asarray(x_full, dtype=float)
    y_arr = np.asarray(y, dtype=int).ravel()

    if xb.ndim == 1:
        xb = xb.reshape(-1, 1)
    if xf.ndim == 1:
        xf = xf.reshape(-1, 1)

    df = int(xf.shape[1] - xb.shape[1])
    if df <= 0:
        raise ValueError(
            f"x_full must have more columns than x_base; got x_base.shape[1]={xb.shape[1]}, "
            f"x_full.shape[1]={xf.shape[1]}"
        )

    def _loglik(x: npt.NDArray[Any]) -> float:
        clf = LogisticRegression(max_iter=max_iter, C=c_penalty, solver="lbfgs")
        clf.fit(x, y_arr)
        p = np.clip(clf.predict_proba(x)[:, 1], 1e-10, 1 - 1e-10)
        return float(np.sum(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))

    stat = max(2.0 * (_loglik(xf) - _loglik(xb)), 0.0)
    p_value = float(stats.chi2.sf(stat, df=df))
    return p_value, float(stat), df


@dataclass(frozen=True)
class InteractionResult:
    """Result of a sensitive × covariate interaction test on a binary outcome.

    Attributes
    ----------
    p_value : float
        Likelihood-ratio test p-value.
    statistic : float
        Likelihood-ratio statistic.
    df : int
        Degrees of freedom (number of interaction terms tested).
    delta_r2 : float
        Cox–Snell pseudo-R² attributable to the interaction terms:
        ``1 - exp(-statistic / n)``. Useful as a magnitude companion to the
        p-value.
    n : int
        Sample size used in the test (rows with non-missing values).
    """

    p_value: float
    statistic: float
    df: int
    delta_r2: float
    n: int


def interaction_test(
    data: pd.DataFrame,
    *,
    outcome: str,
    sensitive: str,
    covariate: str,
    adjust_for: tuple[str, ...] = (),
    sensitive_levels: tuple[Any, ...] | None = None,
    covariate_levels: tuple[Any, ...] | None = None,
) -> InteractionResult:
    """Test whether a sensitive attribute interacts with a covariate on a binary outcome.

    Fits two nested logistic-regression models — one with main effects only,
    one with the full set of interaction terms — and runs a likelihood-ratio
    test. Categorical sensitive attributes and categorical covariates are
    one-hot encoded with the first level dropped as the reference; the
    interaction is the Cartesian product of the remaining one-hots.

    For a binary sensitive × binary covariate test the interaction term is the
    product ``sex * comorbidity`` of the two 0/1 indicators. For multi-level
    factors the test has ``df = (n_sens_levels - 1) * (n_cov_levels - 1)``.

    Parameters
    ----------
    data : DataFrame
        Long-format data with one row per observation.
    outcome : str
        Column name of the binary outcome (0/1).
    sensitive : str
        Column name of the sensitive attribute.
    covariate : str
        Column name of the covariate whose interaction with ``sensitive``
        is being tested.
    adjust_for : tuple of str, default ``()``
        Additional covariates included as linear adjustments in both
        models (not interacted).
    sensitive_levels, covariate_levels : tuple, optional
        Override the level order. The first level is used as the reference
        in dummy coding. If ``None``, the sorted unique values are used.

    Returns
    -------
    InteractionResult
        Fields: ``p_value``, ``statistic``, ``df``, ``delta_r2``, ``n``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     'y': rng.integers(0, 2, 500),
    ...     'sex': rng.integers(0, 2, 500),
    ...     'comorb': rng.integers(0, 2, 500),
    ...     'age': rng.normal(60, 10, 500),
    ... })
    >>> res = interaction_test(df, outcome='y', sensitive='sex',
    ...                        covariate='comorb', adjust_for=('age',))
    >>> res.df
    1
    """
    cols = [outcome, sensitive, covariate, *adjust_for]
    sub = data[cols].dropna().copy()
    n = len(sub)
    if n == 0:
        raise ValueError("no non-missing rows in `data` for the requested columns")

    sens_levels = (
        tuple(sensitive_levels)
        if sensitive_levels is not None
        else tuple(sorted(sub[sensitive].unique()))
    )
    cov_levels = (
        tuple(covariate_levels)
        if covariate_levels is not None
        else tuple(sorted(sub[covariate].unique()))
    )

    if len(sens_levels) < 2 or len(cov_levels) < 2:
        raise ValueError(
            f"both `sensitive` and `covariate` must have ≥2 levels; "
            f"got {len(sens_levels)} for {sensitive!r}, {len(cov_levels)} for {covariate!r}"
        )

    # Dummy code (drop first level as reference).
    sens_dummies = pd.get_dummies(
        pd.Categorical(sub[sensitive], categories=list(sens_levels)),
        drop_first=True,
        dtype=float,
    )
    sens_dummies.columns = pd.Index([f"{sensitive}=={lvl}" for lvl in sens_levels[1:]])
    cov_dummies = pd.get_dummies(
        pd.Categorical(sub[covariate], categories=list(cov_levels)),
        drop_first=True,
        dtype=float,
    )
    cov_dummies.columns = pd.Index([f"{covariate}=={lvl}" for lvl in cov_levels[1:]])

    adjust_block = (
        sub[list(adjust_for)].astype(float).reset_index(drop=True)
        if adjust_for
        else pd.DataFrame(index=range(n))
    )
    base_block = pd.concat(
        [
            adjust_block,
            sens_dummies.reset_index(drop=True),
            cov_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    # Interaction terms: outer product of the two dummy blocks.
    inter_cols = {}
    for s_col in sens_dummies.columns:
        for c_col in cov_dummies.columns:
            inter_cols[f"{s_col} × {c_col}"] = (
                sens_dummies[s_col].to_numpy() * cov_dummies[c_col].to_numpy()
            )
    inter_block = pd.DataFrame(inter_cols)

    full_block = pd.concat([base_block, inter_block.reset_index(drop=True)], axis=1)
    y_arr = sub[outcome].astype(int).to_numpy()

    p_val, stat, df = lrt(base_block.to_numpy(), full_block.to_numpy(), y_arr)
    delta_r2 = float(1.0 - np.exp(-stat / n))
    return InteractionResult(p_value=p_val, statistic=stat, df=df, delta_r2=delta_r2, n=n)


# ---------------------------------------------------------------------------
# Bootstrap metric difference
# ---------------------------------------------------------------------------


def _resolve_metric(metric: str | MetricCallable) -> MetricCallable:
    if callable(metric):
        return metric
    if metric == "auc":

        def _auc(y_true: npt.NDArray[Any], y_score: npt.NDArray[Any]) -> float:
            return float(roc_auc_score(y_true, y_score))

        return _auc
    raise ValueError(
        f"unknown metric {metric!r}; pass 'auc' or a callable (y_true, y_score) -> float"
    )


@dataclass(frozen=True)
class MetricDifferenceResult:
    """Result of a bootstrap test for the between-group difference of a metric.

    For a two-group comparison the difference is ``metric_a - metric_b``,
    where ``a`` and ``b`` follow the order of ``group_levels``. For a
    multi-group comparison with a reference, the result is the maximum
    absolute pairwise gap; per-group point estimates and pairwise
    differences against the reference are available in ``pairwise_deltas``.

    Attributes
    ----------
    metric_observed : dict
        Point estimates of the metric per group on the original data.
    delta : float
        Observed difference (two-group) or maximum pairwise ``|gap|``
        (multi-group).
    ci_low, ci_high : float
        Percentile bootstrap CI of ``delta`` at the requested level.
    p_value : float
        Two-sided percentile bootstrap p-value with the ``(B + 1)``
        continuity correction. Tests the null ``delta == 0``.
    n_boot : int
        Number of *estimable* bootstrap resamples. Equal to ``n_estimable``;
        retained under this name for backward compatibility.
    n_replicates : int
        Number of replicates drawn, including non-estimable ones.
    n_estimable : int
        Replicates in which the difference was defined.
    proportion_nonestimable : float
        Fraction of replicates in which it was not. A replicate fails when a
        group's resample contains no events, so a high value means the
        interval is conditional on a configuration that may not be typical --
        read it alongside the CI, never on its own.
    paired : bool
        Whether the bootstrap was paired across groups (resampling rows
        and then re-stratifying) or independent.
    group_levels : tuple
        The group values, in the order used for the comparison.
    pairwise_deltas : dict
        For multi-group calls, the observed pairwise differences against
        the reference. Empty for two-group calls.
    boot_deltas : ndarray
        The bootstrap distribution of ``delta``.
    """

    metric_observed: dict[Any, float]
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    n_replicates: int
    n_estimable: int
    proportion_nonestimable: float
    paired: bool
    group_levels: tuple[Any, ...]
    pairwise_deltas: dict[Any, float] = field(default_factory=dict)
    boot_deltas: npt.NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([], dtype=float)
    )


def bootstrap_metric_difference(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group: ArrayLike,
    *,
    metric: str | MetricCallable = "auc",
    n_boot: int = 1000,
    random_state: int | np.random.Generator | None = None,
    paired: bool = True,
    ci: float = 0.95,
    reference_group: Any = None,
    group_levels: tuple[Any, ...] | None = None,
) -> MetricDifferenceResult:
    """Bootstrap CI and p-value for the between-group difference of a metric.

    For two groups, computes ``metric(group_a) - metric(group_b)``. For more
    than two groups, computes the maximum absolute pairwise difference
    against ``reference_group`` (or the first level if not given) and
    bootstraps that summary.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcome labels.
    y_score : array-like of shape (n,)
        Predicted scores or probabilities.
    group : array-like of shape (n,)
        Group membership.
    metric : 'auc' or callable, default ``'auc'``
        Metric to compute per group. A custom callable receives
        ``(y_true, y_score)`` slices and must return a float.
    n_boot : int, default 1000
        Number of bootstrap resamples.
    random_state : int, Generator, or None
        Seed for reproducibility.
    paired : bool, default True
        If True, each bootstrap iteration resamples patients from the
        pooled dataset and re-stratifies by ``group``; the same resampled
        cohort flows through every group's metric. This is the appropriate
        scheme for fairness audits where the same source population is
        being audited. If False, each group is resampled independently
        (the "two independent samples" scheme).
    ci : float, default 0.95
        Confidence-interval level for ``ci_low`` / ``ci_high``.
    reference_group, group_levels :
        Used for multi-group (k > 2) comparisons. ``reference_group`` is
        the level subtracted from every other group; ``group_levels``
        overrides the level order (sorted unique by default).

    Returns
    -------
    MetricDifferenceResult

    Raises
    ------
    ValueError
        If fewer than two groups are present, or if the inputs have different
        lengths.

    Notes
    -----
    Non-estimable replicates -- those where a group's resample contained no
    events, leaving the metric undefined -- are retained as ``nan`` and
    counted, never silently dropped. The interval and the p-value are computed
    over the estimable replicates only, which makes them conditional on
    estimability; ``proportion_nonestimable`` reports how often that mattered.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> g = rng.integers(0, 2, n)
    >>> y = rng.integers(0, 2, n)
    >>> p = rng.random(n)
    >>> res = bootstrap_metric_difference(y, p, g, n_boot=200, random_state=0)
    >>> res.n_boot > 0
    True
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    group_arr = np.asarray(group)

    if not (len(y_true_arr) == len(y_score_arr) == len(group_arr)):
        raise ValueError("y_true, y_score, and group must have the same length")

    metric_fn = _resolve_metric(metric)
    rng = np.random.default_rng(random_state)

    if group_levels is None:
        group_levels = tuple(np.sort(pd.unique(group_arr)))
    if len(group_levels) < 2:
        raise ValueError(f"need ≥2 groups; got {group_levels!r}")

    indices_by_group: dict[Any, npt.NDArray[np.intp]] = {
        lvl: np.where(group_arr == lvl)[0] for lvl in group_levels
    }

    def _per_group(idx_map: dict[Any, npt.NDArray[np.intp]]) -> dict[Any, float]:
        out: dict[Any, float] = {}
        for lvl, idx in idx_map.items():
            if len(idx) == 0:
                out[lvl] = float("nan")
                continue
            try:
                out[lvl] = float(metric_fn(y_true_arr[idx], y_score_arr[idx]))
            except Exception:
                out[lvl] = float("nan")
        return out

    def _summary(per_group: dict[Any, float]) -> tuple[float, dict[Any, float]]:
        if len(group_levels) == 2:
            a, b = group_levels
            return per_group[a] - per_group[b], {}
        ref = reference_group if reference_group is not None else group_levels[0]
        pairwise = {lvl: per_group[lvl] - per_group[ref] for lvl in group_levels if lvl != ref}
        finite = [v for v in pairwise.values() if not np.isnan(v)]
        if not finite:
            return float("nan"), pairwise
        return float(max(finite, key=abs)), pairwise

    metric_observed = _per_group(indices_by_group)
    delta, pairwise_obs = _summary(metric_observed)

    boot_deltas: list[float] = []
    for _ in range(n_boot):
        if paired:
            # Resample row indices; partition by group_arr to keep stratification.
            sample = rng.integers(0, len(y_true_arr), len(y_true_arr))
            resampled_groups = group_arr[sample]
            idx_map = {lvl: sample[resampled_groups == lvl] for lvl in group_levels}
        else:
            idx_map = {
                lvl: rng.choice(
                    indices_by_group[lvl], size=len(indices_by_group[lvl]), replace=True
                )
                for lvl in group_levels
            }
        per_group_b = _per_group(idx_map)
        delta_b, _ = _summary(per_group_b)
        # NaN replicates are RETAINED, not skipped. Dropping them silently
        # shrinks the denominator and biases both the interval and the p-value
        # toward the null: a replicate fails when a group's resample has no
        # events, which is exactly the configuration that produces extreme
        # gaps. The count is reported instead.
        boot_deltas.append(delta_b)

    boot_arr = np.asarray(boot_deltas, dtype=float)
    n_replicates = len(boot_arr)
    n_succ = int(np.count_nonzero(~np.isnan(boot_arr)))
    proportion_nonestimable = (
        1.0 - n_succ / n_replicates if n_replicates else float("nan")
    )
    if n_succ == 0:
        return MetricDifferenceResult(
            metric_observed=metric_observed,
            delta=delta,
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=float("nan"),
            n_boot=0,
            n_replicates=n_replicates,
            n_estimable=0,
            proportion_nonestimable=proportion_nonestimable,
            paired=paired,
            group_levels=group_levels,
            pairwise_deltas=pairwise_obs,
            boot_deltas=boot_arr,
        )

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.nanpercentile(boot_arr, 100.0 * alpha))
    ci_high = float(np.nanpercentile(boot_arr, 100.0 * (1.0 - alpha)))

    # Two-sided p with (B + 1) continuity correction; tests delta == 0.
    # Computed over estimable replicates only, so the interval is conditional
    # on estimability -- read proportion_nonestimable alongside it.
    n_ge = int(np.sum(boot_arr >= 0))
    n_le = int(np.sum(boot_arr <= 0))
    p_upper = (n_ge + 1) / (n_succ + 1)
    p_lower = (n_le + 1) / (n_succ + 1)
    p_value = float(min(1.0, 2.0 * min(p_upper, p_lower)))

    return MetricDifferenceResult(
        metric_observed=metric_observed,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_boot=n_succ,
        n_replicates=n_replicates,
        n_estimable=n_succ,
        proportion_nonestimable=proportion_nonestimable,
        paired=paired,
        group_levels=group_levels,
        pairwise_deltas=pairwise_obs,
        boot_deltas=boot_arr,
    )


# ---------------------------------------------------------------------------
# Odds ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OddsRatioResult:
    """Crude 2 × 2 odds ratio with Wald 95 % CI.

    Attributes
    ----------
    odds_ratio : float
        ``(a * d) / (b * c)`` where the 2 × 2 table is::

                            outcome=0   outcome=1
            exposed=1            b            a
            exposed=0            d            c

    ci_low, ci_high : float
        Wald confidence-interval bounds on the natural scale.
    p_value : float
        Two-sided Wald p-value (``2 * (1 - Φ(|log OR / SE|))``).
    se_log_or : float
        Standard error of ``log(OR)``.
    n_exposed : int
        ``a + b``.
    n_unexposed : int
        ``c + d``.
    rate_exposed : float
        ``a / (a + b)``.
    rate_unexposed : float
        ``c / (c + d)``.
    continuity_applied : bool
        Whether a Haldane–Anscombe 0.5 correction was added to every cell.
    """

    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    se_log_or: float
    n_exposed: int
    n_unexposed: int
    rate_exposed: float
    rate_unexposed: float
    continuity_applied: bool


def odds_ratio_2x2(
    exposure: ArrayLike,
    outcome: ArrayLike,
    *,
    continuity: Literal["none", "haldane"] = "haldane",
    ci_level: float = 0.95,
) -> OddsRatioResult:
    """Crude 2 × 2 odds ratio with Wald CI for a binary exposure on a binary outcome.

    Parameters
    ----------
    exposure : array-like of shape (n,)
        Binary exposure (0 or 1).
    outcome : array-like of shape (n,)
        Binary outcome (0 or 1).
    continuity : ``'none'`` or ``'haldane'``, default ``'haldane'``
        If ``'haldane'``, adds 0.5 to every cell of the 2 × 2 table when any
        cell is zero (Haldane–Anscombe correction), keeping the OR finite.
        If ``'none'``, returns an OR of ``nan`` when any cell is zero.
    ci_level : float, default 0.95
        Confidence level for the Wald CI.

    Returns
    -------
    OddsRatioResult
        Use ``.odds_ratio``, ``.ci_low``, ``.ci_high``, ``.p_value``.

    Examples
    --------
    >>> res = odds_ratio_2x2([1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1])
    >>> isinstance(res.odds_ratio, float)
    True
    """
    exp_arr = np.asarray(exposure, dtype=int).ravel()
    out_arr = np.asarray(outcome, dtype=int).ravel()
    if len(exp_arr) != len(out_arr):
        raise ValueError(
            f"exposure and outcome must have the same length; got {len(exp_arr)}, {len(out_arr)}"
        )

    a = int(np.sum((exp_arr == 1) & (out_arr == 1)))
    b = int(np.sum((exp_arr == 1) & (out_arr == 0)))
    c = int(np.sum((exp_arr == 0) & (out_arr == 1)))
    d = int(np.sum((exp_arr == 0) & (out_arr == 0)))

    any_zero = min(a, b, c, d) == 0
    continuity_applied = False
    af, bf, cf, df_ = float(a), float(b), float(c), float(d)
    if any_zero and continuity == "haldane":
        af += 0.5
        bf += 0.5
        cf += 0.5
        df_ += 0.5
        continuity_applied = True
    elif any_zero:
        return OddsRatioResult(
            odds_ratio=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=float("nan"),
            se_log_or=float("nan"),
            n_exposed=a + b,
            n_unexposed=c + d,
            rate_exposed=float("nan") if (a + b) == 0 else a / (a + b),
            rate_unexposed=float("nan") if (c + d) == 0 else c / (c + d),
            continuity_applied=False,
        )

    or_v = (af * df_) / (bf * cf)
    log_or = float(np.log(or_v))
    se = float(np.sqrt(1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df_))
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(log_or / se))))

    return OddsRatioResult(
        odds_ratio=float(or_v),
        ci_low=float(np.exp(log_or - z * se)),
        ci_high=float(np.exp(log_or + z * se)),
        p_value=p_val,
        se_log_or=se,
        n_exposed=a + b,
        n_unexposed=c + d,
        rate_exposed=a / (a + b) if (a + b) > 0 else float("nan"),
        rate_unexposed=c / (c + d) if (c + d) > 0 else float("nan"),
        continuity_applied=continuity_applied,
    )
