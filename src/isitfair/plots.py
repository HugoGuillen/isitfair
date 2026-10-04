"""Plotting helpers for fairness audits and their supporting statistics.

Two families live here. The **statistics** plots — volcano, lollipop, forest —
take an effect-size frame from the inference module. The **audit** plots —
:func:`subgroup_forest_plot`, :func:`roc_plot`, :func:`reliability_plot`,
:func:`decision_curve_plot`, :func:`recalibration_ladder_plot` — take the
long-format frames :class:`~isitfair.FairnessAudit` returns, so every figure the
HTML report draws can also be drawn on its own.

All of them accept long-format DataFrames (or any source the user provides with
the right columns) and return a :class:`matplotlib.axes.Axes`. They never call
``plt.show()`` or ``plt.savefig()`` — composition and persistence are the
caller's job.

Two conventions the audit plots carry deliberately, because they are what makes
a subgroup figure honest:

* A stratum that fails its event floor is drawn **dashed with a hollow legend
  marker and labelled** *not estimable*, rather than dropped. A reader can then
  see what was audited and found unsupportable, instead of what was quietly
  omitted. Pass ``mark_not_estimable=False`` to disable.
* A calibration curve is **clipped to its own stratum's observed predicted
  range**, so interpolation beyond the observed data is never rendered as a
  fitted result.

The volcano plot is two-group only (it relies on a directional effect size).
Forest and lollipop plots support an arbitrary number of features.

The optional ``adjustText`` package, if installed, repels overlapping labels
on the volcano plot. Without it the function falls back to a static label
placement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

try:  # pragma: no cover - optional dependency
    from adjustText import adjust_text as _adjust_text

    _HAS_ADJUST_TEXT = True
except ImportError:  # pragma: no cover
    _HAS_ADJUST_TEXT = False


def _resolve_colors(
    effects: npt.NDArray[Any],
    positive_color: str | None,
    negative_color: str | None,
) -> list[str]:
    pos = positive_color or "#E87A7A"
    neg = negative_color or "#5B9BD5"
    return [pos if e > 0 else neg for e in effects]


def volcano_plot(
    results: pd.DataFrame,
    *,
    effect_col: str = "effect_size",
    q_col: str = "q_fdr",
    label_col: str = "variable",
    label_formatter: Callable[[str], str] | None = None,
    q_threshold: float = 0.05,
    effect_thresholds: tuple[float, ...] = (0.2, 0.5),
    label_min_effect: float = 0.15,
    label_only_significant: bool = True,
    positive_label: str = "Higher in group A",
    negative_label: str = "Higher in group B",
    positive_color: str | None = None,
    negative_color: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Effect-size vs -log10(q) scatter, directional by sign.

    Two-group only: requires that ``effect_col`` carries a *signed* effect
    size (e.g. Cohen's d or Cohen's h with a fixed reference). Points are
    coloured by sign, and a horizontal line marks the FDR threshold.

    Parameters
    ----------
    results : DataFrame
        Long-format results, one row per variable. Must contain
        ``effect_col``, ``q_col``, and ``label_col``.
    effect_col : str, default ``'effect_size'``
        Column carrying the signed effect size.
    q_col : str, default ``'q_fdr'``
        Column carrying FDR-adjusted p-values.
    label_col : str, default ``'variable'``
        Column carrying the per-point label.
    label_formatter : callable, optional
        Receives the raw label, returns the display label. Use to strip
        prefixes like ``com_elix_``.
    q_threshold : float, default 0.05
        Significance threshold; points below this q-value are emphasised.
    effect_thresholds : tuple of float, default ``(0.2, 0.5)``
        Vertical reference lines for "small" / "medium" effect-size bands.
    label_min_effect : float, default 0.15
        Minimum ``|effect|`` for a point to receive a text label.
    label_only_significant : bool, default True
        If True, only points with ``q < q_threshold`` are labelled.
    positive_label, negative_label : str
        Legend entries for positive- and negative-sign effects.
    positive_color, negative_color : str, optional
        Hex colours; defaults are coral (#E87A7A) and steel-blue (#5B9BD5).
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
        Target axes; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If any of the required columns are absent.
    """
    for c in (effect_col, q_col, label_col):
        if c not in results.columns:
            raise KeyError(f"`results` must contain column {c!r}; got {list(results.columns)}")

    df = results[[effect_col, q_col, label_col]].dropna(subset=[effect_col, q_col]).copy()
    df["neg_log_q"] = -np.log10(df[q_col].clip(lower=1e-300))
    if label_formatter is not None:
        df["_label"] = df[label_col].astype(str).apply(label_formatter)
    else:
        df["_label"] = df[label_col].astype(str)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    sig_mask = df[q_col] < q_threshold
    colors = _resolve_colors(df[effect_col].to_numpy(), positive_color, negative_color)

    ax.scatter(
        df.loc[~sig_mask, effect_col],
        df.loc[~sig_mask, "neg_log_q"],
        c=[c for c, m in zip(colors, sig_mask, strict=False) if not m],
        alpha=0.3,
        s=20,
    )
    ax.scatter(
        df.loc[sig_mask, effect_col],
        df.loc[sig_mask, "neg_log_q"],
        c=[c for c, m in zip(colors, sig_mask, strict=False) if m],
        alpha=0.85,
        s=40,
        edgecolor=".2",
        lw=0.5,
    )

    label_mask = df[effect_col].abs() > label_min_effect
    if label_only_significant:
        label_mask &= sig_mask
    texts = [
        ax.text(row[effect_col], row["neg_log_q"], row["_label"], fontsize=7, alpha=0.9)
        for _, row in df[label_mask].iterrows()
    ]
    if _HAS_ADJUST_TEXT and texts:  # pragma: no branch
        _adjust_text(
            texts,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5, "alpha": 0.7},
            ax=ax,
        )

    ax.axhline(-np.log10(q_threshold), color="gray", ls="--", lw=0.8, alpha=0.5)
    xlim = ax.get_xlim()
    ax.text(
        xlim[1],
        -np.log10(q_threshold),
        f" q={q_threshold}",
        fontsize=7,
        va="center",
        color="gray",
    )
    for thresh in effect_thresholds:
        ax.axvline(thresh, color="gray", ls=":", lw=0.6, alpha=0.4)
        ax.axvline(-thresh, color="gray", ls=":", lw=0.6, alpha=0.4)

    ax.set_xlabel(f"{effect_col} (signed)")
    ax.set_ylabel("-log10(q)")
    if title is not None:
        ax.set_title(title)
    pos = positive_color or "#E87A7A"
    neg = negative_color or "#5B9BD5"
    ax.legend(
        handles=[
            Patch(color=pos, label=positive_label),
            Patch(color=neg, label=negative_label),
        ],
        loc="upper center",
        fontsize=8,
    )
    return ax


def lollipop_plot(
    results: pd.DataFrame,
    *,
    effect_col: str = "effect_size",
    label_col: str = "variable",
    q_col: str | None = "q_fdr",
    q_threshold: float = 0.05,
    label_formatter: Callable[[str], str] | None = None,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
    positive_color: str | None = None,
    negative_color: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Horizontal lollipop chart of signed effect sizes.

    One row per feature, sorted by effect size; line + point with
    direction-coded colour. Non-significant points (if ``q_col`` is
    provided) are rendered with reduced alpha.

    Parameters
    ----------
    results : DataFrame
        Long-format results. Must contain ``effect_col`` and ``label_col``.
        If ``q_col`` is given it is used to dim non-significant points.
    Other parameters
        See :func:`volcano_plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    for c in (effect_col, label_col):
        if c not in results.columns:
            raise KeyError(f"`results` must contain column {c!r}; got {list(results.columns)}")

    df = (
        results[[effect_col, label_col] + ([q_col] if q_col else [])]
        .dropna(subset=[effect_col])
        .copy()
    )
    df["_label"] = (
        df[label_col].astype(str).apply(label_formatter) if label_formatter else df[label_col]
    )
    df = df.sort_values(effect_col).reset_index(drop=True)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.3 + 1)), dpi=150)

    sig_mask = (df[q_col] < q_threshold) if q_col else pd.Series(True, index=df.index)
    pos = positive_color or "#E87A7A"
    neg = negative_color or "#5B9BD5"

    for raw_i, row in df.iterrows():
        i = int(raw_i)  # type: ignore[call-overload]
        c = pos if row[effect_col] > 0 else neg
        is_sig = bool(sig_mask.iloc[i])
        alpha_line = 0.7 if is_sig else 0.25
        size = 40 if is_sig else 14
        alpha_pt = 1.0 if is_sig else 0.4
        ax.plot([0, row[effect_col]], [i, i], color=c, lw=1.2, alpha=alpha_line)
        ax.scatter(row[effect_col], i, color=c, s=size, zorder=3, alpha=alpha_pt)

    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["_label"].tolist(), fontsize=7)
    ax.set_xlabel(effect_col)
    if title is not None:
        ax.set_title(title)
    ax.legend(
        handles=[
            Patch(color=pos, label=positive_label),
            Patch(color=neg, label=negative_label),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return ax


def forest_plot(
    results: pd.DataFrame,
    *,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str = "variable",
    group_col: str | None = None,
    q_col: str | None = None,
    q_threshold: float = 0.05,
    log_scale: bool = True,
    null_value: float = 1.0,
    label_formatter: Callable[[str], str] | None = None,
    group_colors: dict[Any, str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Forest plot of point estimates with confidence intervals.

    Each row is a feature; estimates are drawn as points with horizontal CI
    whiskers. If ``group_col`` is given (e.g. sex), estimates for each
    group are drawn on offset y-positions for the same feature.

    Parameters
    ----------
    results : DataFrame
        Long-format. Must contain ``estimate_col``, ``ci_low_col``,
        ``ci_high_col``, ``label_col``. ``group_col`` is optional; if given,
        the same feature label appears once and groups are offset.
    estimate_col : str
        Column carrying the point estimate (e.g. odds ratio).
    ci_low_col, ci_high_col : str
        Columns carrying CI bounds on the same scale as ``estimate_col``.
    log_scale : bool, default True
        If True, plot ``log(estimate)`` and ``log(CI)``. Appropriate for
        ratio statistics like ORs.
    null_value : float, default 1.0
        Reference null line (e.g. 1.0 for ORs). Drawn as a vertical dashed
        line after the log transform if ``log_scale=True``.
    group_colors : dict, optional
        Maps group values to hex colours.
    q_col : str, optional
        If given, ``q < q_threshold`` rows are annotated with an asterisk.

    Returns
    -------
    matplotlib.axes.Axes
    """
    for c in (estimate_col, ci_low_col, ci_high_col, label_col):
        if c not in results.columns:
            raise KeyError(f"`results` must contain column {c!r}; got {list(results.columns)}")

    df = results.dropna(subset=[estimate_col, ci_low_col, ci_high_col]).copy()
    if label_formatter is not None:
        df["_label"] = df[label_col].astype(str).apply(label_formatter)
    else:
        df["_label"] = df[label_col].astype(str)

    if log_scale:
        df["_est"] = np.log(df[estimate_col])
        df["_lo"] = np.log(df[ci_low_col])
        df["_hi"] = np.log(df[ci_high_col])
        null = float(np.log(null_value))
    else:
        df["_est"] = df[estimate_col]
        df["_lo"] = df[ci_low_col]
        df["_hi"] = df[ci_high_col]
        null = null_value

    labels = list(df["_label"].drop_duplicates())
    n_feat = len(labels)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, max(4, n_feat * 0.42 + 1)), dpi=150)

    if group_col is not None:
        groups = list(df[group_col].drop_duplicates())
        offset = 0.22
        for j, grp in enumerate(groups):
            sub = df[df[group_col] == grp]
            color = (group_colors or {}).get(grp, f"C{j}")
            for _, row in sub.iterrows():
                if row["_label"] not in labels:
                    continue
                y_pos = labels.index(row["_label"]) + (j - (len(groups) - 1) / 2.0) * offset
                ax.scatter(row["_est"], y_pos, color=color, s=28, zorder=3, label=grp)
                ax.plot([row["_lo"], row["_hi"]], [y_pos, y_pos], color=color, lw=1.6, alpha=0.7)
                if q_col and row.get(q_col, 1) < q_threshold:
                    ax.text(row["_hi"] + 0.05, y_pos, "*", color=color, fontsize=10, va="center")
        # Deduplicate legend labels (one entry per group).
        handles, hl_labels = ax.get_legend_handles_labels()
        seen: dict[str, Any] = {}
        for h, name in zip(handles, hl_labels, strict=False):
            seen.setdefault(name, h)
        ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)
    else:
        for i, (_, row) in enumerate(df.iterrows()):
            ax.scatter(row["_est"], i, color="C0", s=28, zorder=3)
            ax.plot([row["_lo"], row["_hi"]], [i, i], color="C0", lw=1.6, alpha=0.7)
            if q_col and row.get(q_col, 1) < q_threshold:
                ax.text(row["_hi"] + 0.05, i, "*", color="C0", fontsize=10, va="center")

    ax.axvline(null, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"log({estimate_col})" if log_scale else estimate_col)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Audit figures
#
# Moved here from `isitfair.report` so that every figure the HTML report draws
# can also be drawn on its own. The report keeps thin wrappers that build a
# figure, call these, and encode the result; the drawing logic lives once.
# ---------------------------------------------------------------------------

#: Muted, print-friendly, colour-vision-safe palette.
_COLOURS = [
    "#4e79a7",  # blue
    "#f28e2b",  # orange
    "#e15759",  # red
    "#76b7b2",  # teal
    "#59a14f",  # green
    "#edc948",  # gold
    "#b07aa1",  # purple
    "#ff9da7",  # pink
    "#9c755f",  # brown
    "#bab0ac",  # grey
]

_NOT_ESTIMABLE_SUFFIX = " — not estimable"


def _require(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    """Fail up front, naming the column, rather than deep inside matplotlib."""
    for col in columns:
        if col not in frame.columns:
            msg = f"`{what}` must contain column {col!r}; got {list(frame.columns)}"
            raise KeyError(msg)


def _flagged_subgroups(
    metrics: pd.DataFrame | None,
    attribute: str,
    *,
    enabled: bool,
) -> set[str]:
    """Subgroups of *attribute* that failed the event floor, if we can tell.

    Reads ``not_estimable_final`` — the conservative union of the event floor
    and the bootstrap replicate rule. Returns an empty set when the caller did
    not supply a metric frame, so the plots degrade to plain curves rather than
    silently claiming everything is estimable.
    """
    if not enabled or metrics is None or "not_estimable_final" not in metrics.columns:
        return set()
    rows = metrics[
        (metrics["attribute"].astype(str) == attribute)
        & metrics["not_estimable_final"].eq(True).to_numpy(dtype=bool)
    ]
    return {str(s) for s in rows["subgroup"] if str(s) not in {"Overall", "GAP"}}


def _label_for(group: str, flagged: set[str], suffix: str = "") -> str:
    """Legend label, annotated when the stratum is not estimable."""
    base = f"{group}{suffix}"
    return f"{base}{_NOT_ESTIMABLE_SUFFIX}" if group in flagged else base


def _calibration_axis_limits(
    sub: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute (xlim, ylim) for a calibration plot from the data range.

    Axes are scaled to the data rather than forced to [0, 1] so that
    low-prevalence calibration curves stay legible.
    """
    all_pred = np.asarray(sub["predicted_prob"].values, dtype=float)
    all_obs = np.asarray(sub["observed_prob"].values, dtype=float)
    all_ci_high = np.asarray(sub["ci_high"].values, dtype=float)

    valid_obs = np.asarray(all_obs[~np.isnan(all_obs)])
    valid_ci = np.asarray(all_ci_high[~np.isnan(all_ci_high)])
    y_vals = np.concatenate([valid_obs, valid_ci]) if len(valid_ci) else valid_obs

    x_min = float(np.nanmin(all_pred))
    x_max = float(np.nanmax(all_pred))
    y_max_data = float(np.nanmax(y_vals)) if len(y_vals) else x_max

    # Pad by 5% of the range, ensure minimum visible range
    x_range = max(x_max - x_min, 0.05)
    x_lo = max(0.0, x_min - 0.02 * x_range)
    x_hi = min(1.0, x_max + 0.05 * x_range)
    y_hi = min(1.0, max(y_max_data, x_max) * 1.1 + 0.02)
    return (x_lo, x_hi), (0.0, y_hi)


def _subgroup_colour_map(*frames: pd.DataFrame) -> dict[str, str]:
    """Assign a stable colour to every subgroup appearing in *frames*."""
    labels: list[str] = []
    for frame in frames:
        for label in frame["subgroup"].unique():
            if label != "Overall" and str(label) not in labels:
                labels.append(str(label))
    return {label: _COLOURS[i % len(_COLOURS)] for i, label in enumerate(labels)}


def _draw_calibration_axes(
    ax: Any,
    sub: pd.DataFrame,
    *,
    colour_map: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    legend: bool = True,
    flagged: set[str] | None = None,
) -> None:
    """Draw one calibration panel onto *ax*.

    Shared by the single-panel calibration figure and the before/after
    mitigation comparison, so the two cannot drift apart. Colours come from
    *colour_map* keyed by subgroup label rather than by position, so a
    subgroup keeps its colour even when a panel is missing a group.

    Each subgroup is plotted from its own rows only, so a curve never extends
    past the predicted range that stratum actually spans.
    """
    flagged = flagged or set()
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim

    subgroups = [s for s in sub["subgroup"].unique() if s != "Overall"]
    overall = sub[sub["subgroup"] == "Overall"].sort_values("predicted_prob")

    # Diagonal reference (perfect calibration) — only within data range
    diag_max = min(x_hi, y_hi)
    ax.plot(
        [x_lo, diag_max],
        [x_lo, diag_max],
        "--",
        color="#999999",
        linewidth=0.8,
        label="Perfect",
    )

    # Overall reference curve in black (behind subgroup curves)
    if not overall.empty:
        ax.plot(
            overall["predicted_prob"],
            overall["observed_prob"],
            color="#333333",
            linewidth=2.0,
            alpha=0.5,
            label="Overall",
            zorder=1,
        )
        overall_ci_low = overall["ci_low"].values.astype(float)
        overall_ci_high = overall["ci_high"].values.astype(float)
        has_ci = ~(np.isnan(overall_ci_low) | np.isnan(overall_ci_high))
        if has_ci.any():
            ax.fill_between(
                overall["predicted_prob"].values[has_ci],
                overall_ci_low[has_ci],
                overall_ci_high[has_ci],
                alpha=0.08,
                color="#333333",
                zorder=1,
            )

    # Subgroup curves
    for group in subgroups:
        g_data = sub[sub["subgroup"] == group].sort_values("predicted_prob")
        colour = colour_map[str(group)]
        is_flagged = str(group) in flagged
        ax.plot(
            g_data["predicted_prob"],
            g_data["observed_prob"],
            color=colour,
            linewidth=1.5,
            linestyle="--" if is_flagged else "-",
            label=_label_for(str(group), flagged),
            zorder=2,
        )
        if is_flagged:
            # A band drawn round an interval we decline to report would be the
            # same claim in a different ink.
            continue
        ci_low = g_data["ci_low"].values.astype(float)
        ci_high = g_data["ci_high"].values.astype(float)
        has_ci = ~(np.isnan(ci_low) | np.isnan(ci_high))
        if has_ci.any():
            ax.fill_between(
                g_data["predicted_prob"].values[has_ci],
                ci_low[has_ci],
                ci_high[has_ci],
                alpha=0.15,
                color=colour,
                zorder=2,
            )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if legend:
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def subgroup_forest_plot(
    discrimination: pd.DataFrame,
    *,
    attribute: str,
    metric: str = "auroc",
    ax: Any = None,
    title: str | None = None,
    mark_not_estimable: bool = True,
) -> Any:
    """Forest plot of one discrimination metric across an attribute's subgroups.

    Parameters
    ----------
    discrimination : pandas.DataFrame
        Output of :meth:`~isitfair.FairnessAudit.discrimination`.
    attribute : str
        Which sensitive attribute to draw.
    metric : str, default ``"auroc"``
        Any metric carrying ``<metric>_ci_low`` / ``<metric>_ci_high`` columns.
    ax : matplotlib.axes.Axes, optional
        Draw onto this axes; a new figure is created when omitted.
    title : str, optional
        Defaults to ``"Discrimination — {attribute}"``.
    mark_not_estimable : bool, default True
        Draw strata failing their event floor with a hollow marker and a
        labelled legend entry, rather than dropping them.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If *attribute* has no subgroup rows.

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> y = rng.binomial(1, 0.3, 400).astype(float)
    >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
    >>> a = FairnessAudit(y, p, {"g": rng.choice(["x", "y"], 400)},
    ...                   threshold=0.3, n_bootstrap=20, random_state=0)
    >>> ax = subgroup_forest_plot(a.discrimination(), attribute="g")
    >>> ax.get_xlabel()
    'AUROC'
    """
    _require(
        discrimination,
        ("attribute", "subgroup", metric, f"{metric}_ci_low", f"{metric}_ci_high"),
        "discrimination",
    )
    sub = discrimination[
        (discrimination["attribute"].astype(str) == attribute)
        & (~discrimination["subgroup"].isin(["Overall", "GAP"]))
    ].copy()
    if sub.empty:
        msg = f"no subgroup rows for attribute {attribute!r} in `discrimination`"
        raise ValueError(msg)
    overall = discrimination[
        (discrimination["attribute"].astype(str) == attribute)
        & (discrimination["subgroup"] == "Overall")
    ]
    flagged = _flagged_subgroups(discrimination, attribute, enabled=mark_not_estimable)

    if ax is None:
        _fig, ax = plt.subplots(
            figsize=(5, max(2.5, 0.5 * len(sub) + 1)), dpi=150
        )

    labels = [str(s) for s in sub["subgroup"].to_numpy()]
    values = sub[metric].to_numpy(dtype=float)
    ci_low = sub[f"{metric}_ci_low"].to_numpy(dtype=float)
    ci_high = sub[f"{metric}_ci_high"].to_numpy(dtype=float)
    y_pos = np.arange(len(sub))

    for i, label in enumerate(labels):
        colour = _COLOURS[i % len(_COLOURS)]
        if not np.isnan(ci_low[i]) and not np.isnan(ci_high[i]):
            ax.plot(
                [ci_low[i], ci_high[i]],
                [y_pos[i], y_pos[i]],
                color=colour,
                linewidth=1.5,
                solid_capstyle="round",
            )
        is_flagged = label in flagged
        ax.plot(
            values[i],
            y_pos[i],
            "o",
            color=colour,
            markerfacecolor="none" if is_flagged else colour,
            markersize=6,
            zorder=3,
        )

    if not overall.empty:
        overall_val = float(overall[metric].iloc[0])
        if not np.isnan(overall_val):
            ax.axvline(overall_val, color="#999999", linestyle="--", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_label_for(label, flagged) for label in labels])
    ax.set_xlabel(metric.upper() if metric in {"auroc", "auprc"} else metric)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def roc_plot(
    roc_curves: pd.DataFrame,
    *,
    attribute: str,
    discrimination: pd.DataFrame | None = None,
    ax: Any = None,
    title: str | None = None,
    mark_not_estimable: bool = True,
) -> Any:
    """Per-subgroup ROC curves with bootstrap bands.

    Axes are pinned to the unit square with an equal aspect ratio: a rescaled
    ROC whose chance line is not at 45 degrees misleads.

    Parameters
    ----------
    roc_curves : pandas.DataFrame
        Output of :meth:`~isitfair.FairnessAudit.roc_curves`.
    attribute : str
        Which sensitive attribute to draw.
    discrimination : pandas.DataFrame, optional
        Output of :meth:`~isitfair.FairnessAudit.discrimination`. Supplies the
        AUROC printed beside each curve — taken from there rather than
        integrated from the plotted points, so the number beside the curve is
        the one reported everywhere else — and the event-floor flags.
    ax, title, mark_not_estimable
        As :func:`subgroup_forest_plot`. *title* defaults to ``"ROC — {attribute}"``.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> y = rng.binomial(1, 0.3, 400).astype(float)
    >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
    >>> a = FairnessAudit(y, p, {"g": rng.choice(["x", "y"], 400)},
    ...                   threshold=0.3, n_bootstrap=20, random_state=0)
    >>> ax = roc_plot(a.roc_curves(), attribute="g", discrimination=a.discrimination())
    >>> ax.get_aspect()
    1.0
    """
    _require(roc_curves, ("attribute", "subgroup", "fpr", "tpr"), "roc_curves")
    sub = roc_curves[roc_curves["attribute"].astype(str) == attribute].copy()
    if sub.empty:
        msg = f"no rows for attribute {attribute!r} in `roc_curves`"
        raise ValueError(msg)
    flagged = _flagged_subgroups(discrimination, attribute, enabled=mark_not_estimable)

    def auroc_suffix(group: str) -> str:
        if discrimination is None or "auroc" not in discrimination.columns:
            return ""
        match = discrimination[
            (discrimination["attribute"].astype(str) == attribute)
            & (discrimination["subgroup"].astype(str) == group)
        ]
        if match.empty or np.isnan(float(match["auroc"].iloc[0])):
            return " (AUROC —)"
        return f" (AUROC {float(match['auroc'].iloc[0]):.3f})"

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8, label="Chance")

    overall = sub[sub["subgroup"] == "Overall"].sort_values("fpr")
    if not overall.empty and not overall["tpr"].isna().all():
        ax.plot(
            overall["fpr"],
            overall["tpr"],
            color="#333333",
            linewidth=2.0,
            alpha=0.5,
            label=f"Overall{auroc_suffix('Overall')}",
            zorder=1,
        )
        if {"ci_low", "ci_high"} <= set(overall.columns):
            lo = overall["ci_low"].to_numpy(dtype=float)
            hi = overall["ci_high"].to_numpy(dtype=float)
            has_ci = ~(np.isnan(lo) | np.isnan(hi))
            if has_ci.any():
                ax.fill_between(
                    overall["fpr"].to_numpy()[has_ci],
                    lo[has_ci],
                    hi[has_ci],
                    alpha=0.08,
                    color="#333333",
                    zorder=1,
                )

    subgroups = [s for s in sub["subgroup"].unique() if s != "Overall"]
    for i, group in enumerate(subgroups):
        g_data = sub[sub["subgroup"] == group].sort_values("fpr")
        if g_data["tpr"].isna().all():
            # A single-class subgroup has no ROC curve; an empty legend entry
            # would be worse than no entry.
            continue
        colour = _COLOURS[i % len(_COLOURS)]
        is_flagged = str(group) in flagged
        ax.plot(
            g_data["fpr"],
            g_data["tpr"],
            color=colour,
            linewidth=1.5,
            linestyle="--" if is_flagged else "-",
            label=_label_for(str(group), flagged, auroc_suffix(str(group))),
            zorder=2,
        )
        if is_flagged or not {"ci_low", "ci_high"} <= set(g_data.columns):
            continue
        lo = g_data["ci_low"].to_numpy(dtype=float)
        hi = g_data["ci_high"].to_numpy(dtype=float)
        has_ci = ~(np.isnan(lo) | np.isnan(hi))
        if has_ci.any():
            ax.fill_between(
                g_data["fpr"].to_numpy()[has_ci],
                lo[has_ci],
                hi[has_ci],
                alpha=0.15,
                color=colour,
                zorder=2,
            )

    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def reliability_plot(
    calibration_curves: pd.DataFrame,
    *,
    attribute: str,
    calibration: pd.DataFrame | None = None,
    ax: Any = None,
    title: str | None = None,
    mark_not_estimable: bool = True,
    legend: bool = True,
) -> Any:
    """LOESS reliability diagram with bootstrap bands, per subgroup.

    Each subgroup is drawn from its own rows, so a curve never extends past the
    predicted range that stratum actually spans — extrapolation is not rendered
    as a fitted result.

    Parameters
    ----------
    calibration_curves : pandas.DataFrame
        Output of :meth:`~isitfair.FairnessAudit.calibration_curves`.
    attribute : str
        Which sensitive attribute to draw.
    calibration : pandas.DataFrame, optional
        Output of :meth:`~isitfair.FairnessAudit.calibration`, supplying the
        event-floor flags. Without it no stratum can be marked, because the
        curve frame carries no estimability verdict of its own.
    ax, title, mark_not_estimable
        As :func:`subgroup_forest_plot`.
    legend : bool, default True
        Draw the legend.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> y = rng.binomial(1, 0.3, 400).astype(float)
    >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
    >>> a = FairnessAudit(y, p, {"g": rng.choice(["x", "y"], 400)},
    ...                   threshold=0.3, n_bootstrap=20, random_state=0)
    >>> ax = reliability_plot(a.calibration_curves(), attribute="g")
    >>> ax.get_xlabel()
    'Predicted probability'
    """
    _require(
        calibration_curves,
        ("attribute", "subgroup", "predicted_prob", "observed_prob", "ci_low", "ci_high"),
        "calibration_curves",
    )
    sub = calibration_curves[
        calibration_curves["attribute"].astype(str) == attribute
    ].copy()
    if sub.empty:
        msg = f"no rows for attribute {attribute!r} in `calibration_curves`"
        raise ValueError(msg)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    xlim, ylim = _calibration_axis_limits(sub)
    _draw_calibration_axes(
        ax,
        sub,
        colour_map=_subgroup_colour_map(sub),
        xlim=xlim,
        ylim=ylim,
        title=title if title is not None else "",
        legend=legend,
        flagged=_flagged_subgroups(calibration, attribute, enabled=mark_not_estimable),
    )
    return ax


def decision_curve_plot(
    decision_curve: pd.DataFrame,
    *,
    attribute: str,
    ax: Any = None,
    title: str | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    mark_not_estimable: bool = True,
) -> Any:
    """Decision curves with a **per-subgroup** treat-all reference.

    Treating everyone in a subgroup has a net benefit that depends on that
    subgroup's own prevalence, so a single treat-all line drawn from the overall
    row compares each subgroup against a strategy nobody would apply to it, and
    crosses zero at the cohort prevalence rather than the subgroup's. The frame
    carries the right value per subgroup in ``treat_all_nb``.

    Parameters
    ----------
    decision_curve : pandas.DataFrame
        Output of :meth:`~isitfair.FairnessAudit.decision_curve`.
    attribute : str
        Which sensitive attribute to draw.
    ax, title, mark_not_estimable
        As :func:`subgroup_forest_plot`.
    ylim : tuple, optional
        Explicit y-limits; ``None`` within the tuple means no bound on that
        side. Omitted, the axis clips to ``[-0.01, max(NB) * 1.1]`` so that
        low-prevalence curves stay visible while the treat-all reference is
        allowed to run off the bottom.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> y = rng.binomial(1, 0.3, 400).astype(float)
    >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
    >>> a = FairnessAudit(y, p, {"g": rng.choice(["x", "y"], 400)},
    ...                   threshold=0.3, n_bootstrap=20, random_state=0)
    >>> ax = decision_curve_plot(a.decision_curve(), attribute="g")
    >>> ax.get_ylabel()
    'Net benefit'
    """
    _require(
        decision_curve,
        ("attribute", "subgroup", "threshold", "net_benefit", "treat_all_nb"),
        "decision_curve",
    )
    sub = decision_curve[
        (decision_curve["attribute"].astype(str) == attribute)
        & (decision_curve["subgroup"] != "GAP")
    ].copy()
    if sub.empty:
        msg = f"no rows for attribute {attribute!r} in `decision_curve`"
        raise ValueError(msg)
    flagged = _flagged_subgroups(decision_curve, attribute, enabled=mark_not_estimable)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # Treat-none really is shared: treating nobody has net benefit zero whatever
    # the subgroup's prevalence.
    ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8, label="Treat none")

    subgroups = [s for s in sub["subgroup"].unique() if s != "Overall"]
    for i, group in enumerate(subgroups):
        g_data = sub[sub["subgroup"] == group].sort_values("threshold")
        colour = _COLOURS[i % len(_COLOURS)]
        is_flagged = str(group) in flagged
        ax.plot(
            g_data["threshold"],
            g_data["treat_all_nb"],
            "--",
            color=colour,
            linewidth=0.8,
            alpha=0.7,
            label=f"Treat all ({group})",
        )
        ax.plot(
            g_data["threshold"],
            g_data["net_benefit"],
            color=colour,
            linewidth=1.5,
            linestyle=":" if is_flagged else "-",
            label=_label_for(str(group), flagged),
        )
        if is_flagged or "net_benefit_ci_low" not in g_data.columns:
            continue
        lo = g_data["net_benefit_ci_low"].to_numpy(dtype=float)
        hi = g_data["net_benefit_ci_high"].to_numpy(dtype=float)
        has_ci = ~(np.isnan(lo) | np.isnan(hi))
        if has_ci.any():
            ax.fill_between(
                g_data["threshold"].to_numpy()[has_ci],
                lo[has_ci],
                hi[has_ci],
                alpha=0.15,
                color=colour,
            )

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")

    if ylim is not None:
        y_lo, y_hi = ylim
        if y_lo is not None:
            ax.set_ylim(bottom=y_lo)
        if y_hi is not None:
            ax.set_ylim(top=y_hi)
    else:
        nb_vals = sub[~sub["subgroup"].isin(["Overall"])]["net_benefit"].to_numpy(dtype=float)
        nb_finite = nb_vals[~np.isnan(nb_vals)]
        if len(nb_finite) > 0:
            ax.set_ylim(-0.01, max(0.01, float(np.max(nb_finite)) * 1.1))

    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def recalibration_ladder_plot(
    ladder: pd.DataFrame,
    *,
    metric: str = "calibration_intercept",
    ax: Any = None,
    title: str | None = None,
    mark_not_estimable: bool = True,
) -> Any:
    """One metric across the three rungs of a recalibration ladder.

    Reads the frame from
    :meth:`~isitfair.FairnessAudit.recalibration_ladder`. The point of the
    figure is the *shape*: a cohort-wide shift collapses between rung 0 and
    rung 1, while a genuine between-stratum gradient survives it — which is
    what tells you whether group-specific correction is doing anything a
    common one could not.

    Parameters
    ----------
    ladder : pandas.DataFrame
        Output of :meth:`~isitfair.FairnessAudit.recalibration_ladder`.
    metric : str, default ``"calibration_intercept"``
        Any per-subgroup column of the ladder frame.
    ax, title, mark_not_estimable
        As :func:`subgroup_forest_plot`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> g = rng.choice(["a", "b"], n)
    >>> y = rng.binomial(1, 0.3, n).astype(float)
    >>> p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    >>> a = FairnessAudit(y, p, {"g": g}, threshold=0.3,
    ...                   n_bootstrap=20, random_state=0)
    >>> ax = recalibration_ladder_plot(a.recalibration_ladder(by="g", random_state=0))
    >>> ax.get_xlabel()
    'Recalibration rung'
    """
    _require(ladder, ("rung", "subgroup", metric), "ladder")
    rungs = [r for r in ("none", "common", "group_specific") if r in set(ladder["rung"])]
    if not rungs:
        msg = f"`ladder` carries no recognised rung; got {sorted(set(ladder['rung']))}"
        raise ValueError(msg)

    sub = ladder[~ladder["subgroup"].isin(["GAP"])].copy()
    subgroups = [s for s in sub["subgroup"].unique() if s != "Overall"]
    flagged = (
        {
            str(s)
            for s in sub[sub["not_estimable_final"].eq(True).to_numpy(dtype=bool)]["subgroup"]
            if str(s) != "Overall"
        }
        if mark_not_estimable and "not_estimable_final" in sub.columns
        else set()
    )

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    x = np.arange(len(rungs))
    if metric == "calibration_intercept":
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=0.8, label="Unbiased")
    elif metric == "calibration_slope":
        ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, label="Ideal")

    overall = sub[sub["subgroup"] == "Overall"].set_index("rung")
    if not overall.empty:
        ax.plot(
            x,
            [float(pd.to_numeric(overall.loc[r, metric])) for r in rungs],
            color="#333333",
            linewidth=2.0,
            alpha=0.5,
            marker="o",
            label="Overall",
            zorder=1,
        )

    for i, group in enumerate(subgroups):
        g_data = sub[sub["subgroup"] == group].set_index("rung")
        if not set(rungs) <= set(g_data.index):
            continue
        is_flagged = str(group) in flagged
        colour = _COLOURS[i % len(_COLOURS)]
        ax.plot(
            x,
            [float(pd.to_numeric(g_data.loc[r, metric])) for r in rungs],
            color=colour,
            linewidth=1.5,
            linestyle="--" if is_flagged else "-",
            marker="o",
            markerfacecolor="none" if is_flagged else colour,
            label=_label_for(str(group), flagged),
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "-") for r in rungs])
    ax.set_xlabel("Recalibration rung")
    ax.set_ylabel(metric.replace("_", " "))
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
