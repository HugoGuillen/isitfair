"""HTML report generator for FairnessAudit results.

Renders a self-contained HTML report with embedded SVG plots and styled
tables. No JavaScript dependencies — the report is static, print-friendly,
and accessible.

Plots are generated via matplotlib and embedded as base64-encoded SVG.
Templates use Jinja2.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isitfair.effect_sizes import cohens_h, effect_size_label
from isitfair.plots import (
    _calibration_axis_limits,
    _draw_calibration_axes,
    _flagged_subgroups,
    _subgroup_colour_map,
    decision_curve_plot,
    reliability_plot,
    roc_plot,
    subgroup_forest_plot,
)
from isitfair.table_one import TableOneResult, table_one

# Report figures are rendered head-less, so a GUI backend that cannot open a
# display must not be allowed to fail the import. But do NOT clobber a backend
# the caller deliberately selected: `%matplotlib inline` runs before the import
# in a notebook cell, and unconditionally switching here silently disabled
# inline plotting for the rest of the session -- every figure in a notebook
# that imports isitfair came back as "FigureCanvasAgg is non-interactive".
# Saving a figure works under any backend, so leaving an inline one alone
# costs report generation nothing.
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

# Map a discrimination gap_metric name to the per-subgroup rate column on which
# Cohen's h is meaningful. `equalized_odds_gap` is composite (max of two) and
# `statistical_parity` uses a column that isn't carried in `disc_df`, so both
# are omitted — they render as "—" in the gap table.
_GAP_METRIC_TO_RATE: dict[str, str] = {
    "equal_opportunity_difference": "sensitivity",
    "predictive_equality": "specificity",
}

_TEMPLATE_DIR = Path(__file__).parent / "templates"



# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _fig_to_base64_svg(
    fig: matplotlib.figure.Figure,
    png_path: Path | None = None,
) -> str:
    """Render a matplotlib figure to a base64-encoded SVG data URI.

    Uses a fixed hashsalt and suppresses the date metadata to ensure
    deterministic output across identical inputs.

    If *png_path* is given, the figure is additionally written to that path
    as a publication-ready 300-DPI PNG before being closed. The PNG's
    ``Software`` metadata tag is suppressed so identical inputs produce
    byte-identical files.
    """
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": None},
        )
    buf = io.BytesIO()
    old_hashsalt = matplotlib.rcParams.get("svg.hashsalt", None)
    matplotlib.rcParams["svg.hashsalt"] = "isitfair"
    try:
        fig.savefig(
            buf,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None},
        )
    finally:
        if old_hashsalt is None:
            matplotlib.rcParams["svg.hashsalt"] = ""
        else:
            matplotlib.rcParams["svg.hashsalt"] = old_hashsalt
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _sanitize_filename(name: str) -> str:
    """Turn an attribute name into a safe filename stem.

    Non-alphanumeric runs (including the ``×`` used for interaction
    attributes and surrounding whitespace) collapse to single underscores.
    """
    out = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return out or "attribute"


# ---------------------------------------------------------------------------
# Forest plot — discrimination metrics
# ---------------------------------------------------------------------------


def _forest_plot_discrimination(
    disc_df: pd.DataFrame,
    attr_name: str,
    png_path: Path | None = None,
) -> str:
    """Two-panel AUROC/AUPRC forest plot for one attribute.

    The drawing lives in :func:`isitfair.plots.subgroup_forest_plot`, which
    draws one metric onto one axes; this wrapper lays out the pair and encodes
    the result. Returns a base64 SVG data URI, and writes a 300-DPI PNG to
    *png_path* when given.
    """
    sub = disc_df[
        (disc_df["attribute"] == attr_name) & (~disc_df["subgroup"].isin(["Overall", "GAP"]))
    ]
    if sub.empty:
        return ""

    metrics = ("auroc", "auprc")
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5 * len(metrics), max(2.5, 0.5 * len(sub) + 1)),
        squeeze=False,
    )
    for col_idx, metric in enumerate(metrics):
        subgroup_forest_plot(
            disc_df,
            attribute=attr_name,
            metric=metric,
            ax=axes[0, col_idx],
        )
    fig.suptitle(f"Discrimination — {attr_name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64_svg(fig, png_path)


# ---------------------------------------------------------------------------
# Calibration curve plot
# ---------------------------------------------------------------------------


def _intercept_verdicts(audit: Any) -> pd.DataFrame:
    """Per-subgroup calibration-intercept verdicts, for marking calibration curves.

    A calibration curve loses its band where the calibration-intercept interval
    is withheld: the same verdict that blanks that interval in the calibration
    table. The row-level ``not_estimable_final`` of ``calibration()`` is the
    union with the stricter slope floor, and would strip the band from a stratum
    whose intercept interval the table still prints. Read from
    ``estimability()``, the single authoritative verdict; no floor is restated.
    """
    est: pd.DataFrame = audit.estimability()
    return est[est["metric"] == "calibration_intercept"]


def _calibration_curve_plot(
    curves_df: pd.DataFrame,
    attr_name: str,
    png_path: Path | None = None,
    cal_df: pd.DataFrame | None = None,
) -> str:
    """Single-panel calibration figure for one attribute.

    Delegates the drawing to :func:`isitfair.plots.reliability_plot`. Pass
    *cal_df* -- a frame with ``attribute``, ``subgroup`` and
    ``not_estimable_final`` columns, normally :func:`_intercept_verdicts` -- to
    mark strata whose interval is withheld. Without it no stratum is marked.

    Returns a base64 SVG data URI. If *png_path* is given, also writes a
    300-DPI PNG there.
    """
    if curves_df[curves_df["attribute"] == attr_name].empty:
        return ""
    fig, ax = plt.subplots(figsize=(5, 5))
    reliability_plot(
        curves_df,
        attribute=attr_name,
        calibration=cal_df,
        ax=ax,
        title=f"Calibration — {attr_name}",
    )
    fig.tight_layout()
    return _fig_to_base64_svg(fig, png_path)


def _calibration_comparison_plot(
    before_curves: pd.DataFrame,
    after_curves: pd.DataFrame,
    attr_name: str,
    png_path: Path | None = None,
    before_flagged: set[str] | None = None,
    after_flagged: set[str] | None = None,
) -> str:
    """Reliability diagrams before and after mitigation, side by side.

    *before_flagged* and *after_flagged* name the strata each panel draws
    dashed and without a band, each from its own audit's verdict.

    Both panels share axis limits computed from the union of the two frames
    and a single subgroup-to-colour map. Independent autoscaling would let a
    recalibration look like an improvement purely through rescaling, and
    positional colours would recolour a subgroup whenever one panel is
    missing a group.

    Returns a base64 SVG data URI. If *png_path* is given, also writes a
    300-DPI PNG there.
    """
    before = before_curves[before_curves["attribute"] == attr_name].copy()
    after = after_curves[after_curves["attribute"] == attr_name].copy()
    if before.empty or after.empty:
        return ""

    combined = pd.concat([before, after], ignore_index=True)
    xlim, ylim = _calibration_axis_limits(combined)
    colour_map = _subgroup_colour_map(before, after)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, panel, title, show_legend, flagged in (
        (axes[0], before, "Before mitigation", True, before_flagged),
        (axes[1], after, "After mitigation", False, after_flagged),
    ):
        _draw_calibration_axes(
            ax,
            panel,
            colour_map=colour_map,
            xlim=xlim,
            ylim=ylim,
            title=title,
            legend=show_legend,
            flagged=flagged,
        )

    fig.suptitle(f"Calibration before and after — {attr_name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64_svg(fig, png_path)


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------


def _roc_curve_plot(
    roc_df: pd.DataFrame,
    disc_df: pd.DataFrame,
    attr_name: str,
    png_path: Path | None = None,
) -> str:
    """Per-subgroup ROC figure for one attribute.

    Delegates the drawing to :func:`isitfair.plots.roc_plot`; *disc_df* supplies
    the AUROC printed beside each curve and the event-floor flags.

    Returns a base64 SVG data URI. If *png_path* is given, also writes a
    300-DPI PNG there.
    """
    if roc_df[roc_df["attribute"] == attr_name].empty:
        return ""
    fig, ax = plt.subplots(figsize=(5, 5))
    roc_plot(
        roc_df,
        attribute=attr_name,
        discrimination=disc_df,
        ax=ax,
        title=f"ROC — {attr_name}",
    )
    fig.tight_layout()
    return _fig_to_base64_svg(fig, png_path)


# ---------------------------------------------------------------------------
# Decision curve plot
# ---------------------------------------------------------------------------


def _decision_curve_plot(
    dc_df: pd.DataFrame,
    attr_name: str,
    dca_ylim: tuple[float | None, float | None] | None = None,
    png_path: Path | None = None,
) -> str:
    """Decision-curve figure for one attribute.

    Delegates the drawing to :func:`isitfair.plots.decision_curve_plot`, which
    draws the treat-all reference **per subgroup**.

    Returns a base64 SVG data URI. If *png_path* is given, also writes a
    300-DPI PNG there.
    """
    sub = dc_df[(dc_df["attribute"] == attr_name) & (dc_df["subgroup"] != "GAP")]
    if sub.empty:
        return ""
    fig, ax = plt.subplots(figsize=(6, 4))
    decision_curve_plot(
        dc_df,
        attribute=attr_name,
        ax=ax,
        title=f"Decision curve — {attr_name}",
        ylim=dca_ylim,
    )
    fig.tight_layout()
    return _fig_to_base64_svg(fig, png_path)


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------


def _format_ci(val: float, ci_low: float, ci_high: float, decimals: int = 3) -> str:
    """Format a value with its CI as 'val (ci_low–ci_high)'."""
    if np.isnan(val):
        return "—"
    v = f"{val:.{decimals}f}"
    if np.isnan(ci_low) or np.isnan(ci_high):
        return v
    return f"{v} ({ci_low:.{decimals}f}–{ci_high:.{decimals}f})"


def _two_group_rates(
    disc_df: pd.DataFrame, attribute: str, metric: str
) -> tuple[float, float] | None:
    """Return the two per-subgroup rates for a two-group attribute, or None.

    Used to compute Cohen's h on rate-metric gaps in the discrimination table.
    Returns None if the attribute has anything other than two real subgroups,
    or if either rate is missing.
    """
    sub = disc_df[
        (disc_df["attribute"] == attribute) & (~disc_df["subgroup"].isin(["Overall", "GAP"]))
    ]
    if len(sub) != 2:
        return None
    vals = sub[metric].to_numpy(dtype=float)
    if np.isnan(vals).any():
        return None
    return float(vals[0]), float(vals[1])


def _rate_gap_effect_size(disc_df: pd.DataFrame, attribute: str, metric: str) -> str:
    """Cohen's h formatted as 'h=±X.XX (magnitude)' or '—' if not applicable.

    ``metric`` is the gap_metric label produced by ``audit.discrimination()``
    (e.g. ``equal_opportunity_difference``). We translate it to the underlying
    per-subgroup rate column (``sensitivity`` / ``specificity``) and compute
    Cohen's h on the two-group rates. Composite or non-rate gap metrics
    return "—".
    """
    rate_col = _GAP_METRIC_TO_RATE.get(metric)
    if rate_col is None:
        return "—"
    rates = _two_group_rates(disc_df, attribute, rate_col)
    if rates is None:
        return "—"
    p_a, p_b = rates
    if not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0):
        return "—"
    h = cohens_h(p_a, p_b)
    return f"h={h:+.2f} ({effect_size_label(h, kind='h')})"


def _discrimination_summary_table(disc_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build summary table rows for the discrimination section."""
    rows = []
    for _, r in disc_df.iterrows():
        if r["subgroup"] == "GAP":
            continue
        row: dict[str, Any] = {
            "attribute": r["attribute"],
            "subgroup": r["subgroup"],
            "shrunk": bool(r["shrunk"]),
            "n": int(r["n"]) if not np.isnan(r["n"]) else "—",
            "n_events": int(r["n_events"]) if not np.isnan(r["n_events"]) else "—",
            "prevalence": f"{r['prevalence']:.3f}" if not np.isnan(r["prevalence"]) else "—",
            "auroc": _format_ci(r["auroc"], r["auroc_ci_low"], r["auroc_ci_high"]),
            "auprc": _format_ci(r["auprc"], r["auprc_ci_low"], r["auprc_ci_high"]),
            "sensitivity": _format_ci(
                r["sensitivity"], r["sensitivity_ci_low"], r["sensitivity_ci_high"]
            ),
            "specificity": _format_ci(
                r["specificity"], r["specificity_ci_low"], r["specificity_ci_high"]
            ),
        }
        rows.append(row)
    return rows


def _calibration_summary_table(cal_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build summary table rows for the calibration section (excludes GAP rows)."""
    rows = []
    for _, r in cal_df[cal_df["subgroup"] != "GAP"].iterrows():
        row: dict[str, Any] = {
            "attribute": r["attribute"],
            "subgroup": r["subgroup"],
            "shrunk": bool(r["shrunk"]),
            "n": int(r["n"]),
            "intercept": _format_ci(
                r["calibration_intercept"], r["intercept_ci_low"], r["intercept_ci_high"]
            ),
            "slope": _format_ci(r["calibration_slope"], r["slope_ci_low"], r["slope_ci_high"]),
            "brier": _format_ci(r["brier"], r["brier_ci_low"], r["brier_ci_high"]),
            "ece": f"{r['ece']:.4f}" if not np.isnan(r["ece"]) else "—",
            "ici": f"{r['ici']:.4f}" if not np.isnan(r["ici"]) else "—",
        }
        rows.append(row)
    return rows


def _hosmer_lemeshow_table(hl_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the Hosmer-Lemeshow summary rows (excludes GAP rows).

    ``n`` is deliberately adjacent to the statistic: Kramer & Zimmerman
    (2007) argue that the two must always be read together, because the
    statistic scales with subgroup size.
    """
    rows = []
    for _, r in hl_df[hl_df["subgroup"] != "GAP"].iterrows():
        statistic = r["hl_statistic"]
        p_value = r["hl_p_value"]
        df_value = r["hl_df"]
        slope = r["decile_slope"]
        intercept = r["decile_intercept"]
        rows.append(
            {
                "attribute": r["attribute"],
                "subgroup": r["subgroup"],
                "shrunk": bool(r["shrunk"]),
                "n": int(r["n"]),
                "n_events": int(r["n_events"]),
                "statistic": "—" if np.isnan(statistic) else f"{statistic:.2f}",
                "df": "—" if np.isnan(df_value) else f"{df_value:.0f}",
                "p_value": "—" if np.isnan(p_value) else f"{p_value:.4f}",
                "n_bins_used": "—" if np.isnan(r["n_bins_used"]) else f"{r['n_bins_used']:.0f}",
                "bins_reduced": bool(r["bins_reduced"]),
                "decile_slope": "—" if np.isnan(slope) else f"{slope:.3f}",
                "decile_intercept": "—" if np.isnan(intercept) else f"{intercept:+.3f}",
            }
        )
    return rows


def _hosmer_lemeshow_deciles(hl_detail_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group the per-decile table into one block per (attribute, subgroup).

    This is Kramer's requirement (ii): the statistic must be accompanied by
    the full table of observed versus expected events per decile.
    """
    blocks: list[dict[str, Any]] = []
    for (attribute, subgroup), block in hl_detail_df.groupby(
        ["attribute", "subgroup"], sort=False
    ):
        blocks.append(
            {
                "attribute": attribute,
                "subgroup": subgroup,
                "rows": [
                    {
                        "decile": int(r["decile"]),
                        "n": int(r["n"]),
                        "observed": f"{r['observed']:.0f}",
                        "expected": f"{r['expected']:.1f}",
                        "observed_rate": f"{r['observed_rate']:.4f}",
                        "expected_rate": f"{r['expected_rate']:.4f}",
                        "difference": f"{r['difference']:+.1f}",
                    }
                    for _, r in block.iterrows()
                ],
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# TRIPOD+AI checklist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TripodItem:
    """A single TRIPOD+AI checklist item relevant to fairness auditing."""

    item: str
    description: str
    category: str  # "addressed", "conditional", "user_responsibility"
    section: str
    condition: str  # key into condition_map for conditional items; "" otherwise
    todo: bool  # True if item number needs manual verification


# Fairness-relevant items from Collins et al. (2024), BMJ 385:e078378 (Table 2).
# Verified against PMC11019967.
_TRIPOD_AI_ITEMS: list[_TripodItem] = [
    _TripodItem(
        item="3c",
        description="Describe any known health inequalities between sociodemographic groups",
        category="user_responsibility",
        section="User must provide context in manuscript introduction",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="5a",
        description="Describe data sources, rationale, and representativeness",
        category="user_responsibility",
        section="User must describe data representativeness in manuscript methods",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="7",
        description=(
            "Describe data pre-processing, including whether similar across sociodemographic groups"
        ),
        category="user_responsibility",
        section="User must verify preprocessing does not vary by group",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="8a",
        description="Outcome assessment consistency across sociodemographic groups",
        category="user_responsibility",
        section="User must verify outcome labels are measured consistently",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="8b",
        description="Assessor demographics for outcome assessment",
        category="user_responsibility",
        section="User must report assessor demographics",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="9c",
        description="Assessor demographics for predictor measurement",
        category="user_responsibility",
        section="User must report assessor demographics",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="12e",
        description="All measures used to evaluate model performance",
        category="addressed",
        section="Discrimination, Calibration, and Clinical utility sections",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="12f",
        description=(
            "Describe any model updating (eg recalibration) for sociodemographic groups or settings"
        ),
        category="conditional",
        section="Mitigation section",
        condition="has_mitigation",
        todo=False,
    ),
    _TripodItem(
        item="14",
        description="Fairness approaches and rationale",
        category="addressed",
        section="Audit summary and gap statistics throughout",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="20b",
        description="Report demographic characteristics and group differences",
        category="addressed",
        section="Discrimination section (subgroup sizes, prevalences)",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="20c",
        description="Compare predictor distributions including demographics",
        category="user_responsibility",
        section="User must report predictor distributions per group in manuscript",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="23a",
        description=(
            "Report performance estimates with CIs, including for key sociodemographic subgroups"
        ),
        category="addressed",
        section="Discrimination, Calibration, and Clinical utility sections",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="25",
        description="Overall interpretation including fairness issues",
        category="user_responsibility",
        section="User must interpret results in manuscript discussion",
        condition="",
        todo=False,
    ),
    _TripodItem(
        item="26",
        description="Limitations including non-representative sample, bias effects",
        category="user_responsibility",
        section="User must discuss limitations in manuscript",
        condition="",
        todo=False,
    ),
]


def _build_tripod_checklist(
    *,
    has_mitigation: bool,
    has_interactions: bool,
) -> list[dict[str, str]]:
    """Build TRIPOD+AI checklist rows for the report template."""
    condition_map: dict[str, bool] = {
        "has_mitigation": has_mitigation,
        "has_interactions": has_interactions,
    }

    rows: list[dict[str, str]] = []
    for entry in _TRIPOD_AI_ITEMS:
        if entry.category == "addressed":
            status = "Addressed"
            detail = entry.section
        elif entry.category == "conditional":
            active = condition_map.get(entry.condition, False)
            if active:
                status = "Addressed"
                detail = entry.section
            else:
                status = "Not included"
                if entry.condition == "has_mitigation":
                    detail = "No mitigation requested"
                else:
                    detail = f"Condition not met: {entry.condition}"
        else:
            status = "User responsibility"
            detail = entry.section

        rows.append(
            {
                "item": entry.item,
                "description": entry.description,
                "status": status,
                "detail": detail,
                "todo": "yes" if entry.todo else "no",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_report(
    audit: Any,
    out: str | Path,
    title: str | None = None,
    metadata: dict[str, str] | None = None,
    mitigation_audits: dict[str, Any] | None = None,
    table_one_data: pd.DataFrame | TableOneResult | None = None,
    table_one_kwargs: dict[str, Any] | None = None,
    pdf: bool = False,
) -> Path:
    """Generate an HTML fairness report from a FairnessAudit.

    Parameters
    ----------
    audit : FairnessAudit
        The audit to report on. Must have been constructed already.
    out : str or Path
        Output file path for the HTML report.
    title : str or None, optional
        Report title. Defaults to "Fairness Audit Report".
    metadata : dict[str, str] or None, optional
        Key-value pairs to display in the report header (e.g. model name,
        dataset, date).
    mitigation_audits : dict[str, FairnessAudit] or None, optional
        If provided, includes a before/after mitigation comparison section.
        Keys are attribute names, values are mitigated FairnessAudit instances
        (as returned by ``FairnessAudit.mitigate()``).
    table_one_data : pandas.DataFrame, TableOneResult, or None, optional
        If provided, a Table 1 section is inserted between the audit summary
        and the discrimination section.

        - Pass a ``DataFrame`` to let the report build Table 1 for you via
          :func:`isitfair.table_one`; pass ``table_one_kwargs`` to control
          its arguments (``group_col`` defaults to the first sensitive
          attribute of the audit).
        - Pass a pre-built ``TableOneResult`` to use it as-is.
    table_one_kwargs : dict, optional
        Forwarded to :func:`isitfair.table_one` when ``table_one_data`` is a
        DataFrame. Ignored otherwise.
    pdf : bool, default False
        Also write a print-typeset PDF beside the HTML (same stem, ``.pdf``)
        via :func:`report_to_pdf`. Needs the ``pdf`` extra
        (``pip install isitfair[pdf]``).

    Returns
    -------
    Path
        The path to the generated HTML report file.

    Raises
    ------
    ImportError
        If *pdf* is true and WeasyPrint is not installed. The HTML has already
        been written by then.

    Notes
    -----
    The HTML report is self-contained — plots are embedded as SVG. In
    addition, a publication-ready 300-DPI PNG of every plot is written to a
    ``"<report-stem>_figures/"`` directory created alongside the report
    (e.g. ``audit_report.html`` → ``audit_report_figures/``). Filenames are
    ``discrimination_<attr>.png``, ``roc_<attr>.png``,
    ``calibration_<attr>.png``, and ``decision_curve_<attr>.png``, one per
    sensitive attribute, plus ``mitigation_calibration_<attr>.png`` for each
    entry in *mitigation_audits*.

    When *mitigation_audits* is given, each mitigated audit's
    ``calibration_curves()`` is computed for **every** attribute it carries,
    not only the mitigated one, because ``FairnessAudit.mitigate()`` copies
    the full ``sensitive_features`` dict onto each returned audit. With k
    mitigated attributes that is roughly k times the base calibration-curve
    cost. Results are cached per audit, so nothing is computed twice.
    """
    out_path = Path(out)
    if title is None:
        title = "Fairness Audit Report"

    # --- Compute all analysis axes ---
    disc_df = audit.discrimination()
    roc_df = audit.roc_curves()
    cal_df = audit.calibration()
    curves_df = audit.calibration_curves()
    hl_df = audit.hosmer_lemeshow()
    hl_detail_df = audit.hosmer_lemeshow(detail=True)
    dc_df = audit.decision_curve()
    cell_diag_df = audit.cell_diagnostics()

    # --- Attributes list ---
    attributes = list(dict.fromkeys(disc_df["attribute"].values))

    # --- Figures directory (publication-ready 300-DPI PNGs) ---
    # Written alongside the HTML report in "<report-stem>_figures/". The HTML
    # itself stays self-contained (plots are embedded as SVG); these PNGs are
    # the print-resolution copies for manuscripts and slides.
    figures_dir = out_path.parent / f"{out_path.stem}_figures"

    # --- Generate plots per attribute ---
    forest_plots: dict[str, str] = {}
    roc_plots: dict[str, str] = {}
    cal_curve_plots: dict[str, str] = {}
    dc_plots: dict[str, str] = {}

    dca_ylim = audit._dca_ylim
    intercept_verdicts = _intercept_verdicts(audit)

    for attr in attributes:
        stem = _sanitize_filename(attr)
        forest_plots[attr] = _forest_plot_discrimination(
            disc_df, attr, png_path=figures_dir / f"discrimination_{stem}.png"
        )
        roc_plots[attr] = _roc_curve_plot(
            roc_df, disc_df, attr, png_path=figures_dir / f"roc_{stem}.png"
        )
        cal_curve_plots[attr] = _calibration_curve_plot(
            curves_df,
            attr,
            png_path=figures_dir / f"calibration_{stem}.png",
            cal_df=intercept_verdicts,
        )
        dc_plots[attr] = _decision_curve_plot(
            dc_df, attr, dca_ylim=dca_ylim, png_path=figures_dir / f"decision_curve_{stem}.png"
        )

    # --- Summary tables ---
    disc_table = _discrimination_summary_table(disc_df)
    cal_table = _calibration_summary_table(cal_df)
    hl_table = _hosmer_lemeshow_table(hl_df)
    hl_decile_blocks = _hosmer_lemeshow_deciles(hl_detail_df)
    hl_n_tests = len(hl_table)

    # --- Hosmer-Lemeshow gap statistics ---
    hl_gap_table: list[dict[str, Any]] = []
    for _, r in hl_df[hl_df["subgroup"] == "GAP"].iterrows():
        hl_gap_table.append(
            {
                "attribute": r["attribute"],
                "metric": r["gap_metric"],
                "value": "—" if np.isnan(r["gap_value"]) else f"{r['gap_value']:.2f}",
            }
        )

    # --- Gap statistics (discrimination) ---
    gap_rows = disc_df[disc_df["subgroup"] == "GAP"].copy()
    gap_table: list[dict[str, Any]] = []
    for _, r in gap_rows.iterrows():
        gap_table.append(
            {
                "attribute": r["attribute"],
                "metric": r["gap_metric"],
                "value": f"{r['gap_value']:.4f}" if not np.isnan(r["gap_value"]) else "—",
                "effect_size": _rate_gap_effect_size(disc_df, r["attribute"], r["gap_metric"]),
            }
        )

    # --- Gap statistics (calibration) ---
    cal_gap_rows = cal_df[cal_df["subgroup"] == "GAP"].copy()
    cal_gap_table: list[dict[str, Any]] = []
    for _, r in cal_gap_rows.iterrows():
        val = r["gap_value"]
        ci_lo = r["gap_ci_low"]
        ci_hi = r["gap_ci_high"]
        cal_gap_table.append(
            {
                "attribute": r["attribute"],
                "metric": r["gap_metric"],
                "value": _format_ci(val, ci_lo, ci_hi, decimals=4),
            }
        )

    # --- Gap statistics (DCA) ---
    dc_gap_rows = dc_df[dc_df["subgroup"] == "GAP"].copy()
    dc_gap_table: list[dict[str, Any]] = []
    for _, r in dc_gap_rows.iterrows():
        val = r["gap_value"]
        ci_lo = r["gap_ci_low"]
        ci_hi = r["gap_ci_high"]
        dc_gap_table.append(
            {
                "attribute": r["attribute"],
                "metric": r["gap_metric"],
                "value": _format_ci(val, ci_lo, ci_hi, decimals=4),
            }
        )

    # --- Mitigation section ---
    mitigation_data: list[dict[str, Any]] = []
    if mitigation_audits:
        for attr_name, mit_audit in mitigation_audits.items():
            before_disc = disc_df[
                (disc_df["attribute"] == attr_name)
                & (~disc_df["subgroup"].isin(["Overall", "GAP"]))
            ]
            mit_disc = mit_audit.discrimination()
            after_disc = mit_disc[
                (mit_disc["attribute"] == attr_name)
                & (~mit_disc["subgroup"].isin(["Overall", "GAP"]))
            ]
            before_rows = []
            for _, r in before_disc.iterrows():
                before_rows.append(
                    {
                        "subgroup": r["subgroup"],
                        "auroc": _format_ci(r["auroc"], r["auroc_ci_low"], r["auroc_ci_high"]),
                        "sensitivity": _format_ci(
                            r["sensitivity"],
                            r["sensitivity_ci_low"],
                            r["sensitivity_ci_high"],
                        ),
                    }
                )
            after_rows = []
            for _, r in after_disc.iterrows():
                after_rows.append(
                    {
                        "subgroup": r["subgroup"],
                        "auroc": _format_ci(r["auroc"], r["auroc_ci_low"], r["auroc_ci_high"]),
                        "sensitivity": _format_ci(
                            r["sensitivity"],
                            r["sensitivity_ci_low"],
                            r["sensitivity_ci_high"],
                        ),
                    }
                )
            # Calibration is what recalibration actually targets, so it gets
            # the same before/after treatment as discrimination.
            mit_cal = mit_audit.calibration()
            before_cal = [
                r
                for r in _calibration_summary_table(cal_df)
                if r["attribute"] == attr_name and r["subgroup"] != "Overall"
            ]
            after_cal = [
                r
                for r in _calibration_summary_table(mit_cal)
                if r["attribute"] == attr_name and r["subgroup"] != "Overall"
            ]

            def _gap_rows(frame: pd.DataFrame, attr: str = attr_name) -> list[dict[str, Any]]:
                out = []
                sel = frame[(frame["attribute"] == attr) & (frame["subgroup"] == "GAP")]
                for _, r in sel.iterrows():
                    out.append(
                        {
                            "metric": r["gap_metric"],
                            "value": _format_ci(
                                r["gap_value"], r["gap_ci_low"], r["gap_ci_high"], decimals=4
                            ),
                        }
                    )
                return out

            mit_stem = _sanitize_filename(attr_name)
            cal_plot = _calibration_comparison_plot(
                curves_df,
                mit_audit.calibration_curves(),
                attr_name,
                png_path=figures_dir / f"mitigation_calibration_{mit_stem}.png",
                before_flagged=_flagged_subgroups(
                    intercept_verdicts, attr_name, enabled=True
                ),
                after_flagged=_flagged_subgroups(
                    _intercept_verdicts(mit_audit), attr_name, enabled=True
                ),
            )

            mitigation_data.append(
                {
                    "attribute": attr_name,
                    "before": before_rows,
                    "after": after_rows,
                    "before_cal": before_cal,
                    "after_cal": after_cal,
                    "gap_before": _gap_rows(cal_df),
                    "gap_after": _gap_rows(mit_cal),
                    "cal_plot": cal_plot,
                }
            )

    # --- Cell diagnostics ---
    has_interactions = len(cell_diag_df) > 0
    cell_diag_table: list[dict[str, Any]] = []
    if has_interactions:
        for _, r in cell_diag_df.iterrows():
            cell_diag_table.append(
                {
                    "attribute": r["attribute"],
                    "subgroup": r["subgroup"],
                    "n": int(r["n"]),
                    "n_events": int(r["n_events"]),
                    "shrunk": bool(r["shrunk"]),
                    "marginal_rate": f"{r['marginal_rate']:.4f}",
                    "cell_rate": f"{r['cell_rate']:.4f}",
                    "shrinkage_factor": f"{r['shrinkage_factor']:.4f}",
                }
            )

    # --- TRIPOD+AI checklist ---
    tripod_checklist = _build_tripod_checklist(
        has_mitigation=bool(mitigation_audits),
        has_interactions=has_interactions,
    )

    # --- Table 1 (optional) ---
    has_table_one = table_one_data is not None
    table_one_html = ""
    table_one_effect_rows: list[dict[str, Any]] = []
    table_one_summary = ""
    if isinstance(table_one_data, TableOneResult):
        t1_result: TableOneResult | None = table_one_data
    elif isinstance(table_one_data, pd.DataFrame):
        kw = dict(table_one_kwargs or {})
        kw.setdefault("group_col", attributes[0])
        t1_result = table_one(table_one_data, **kw)
    else:
        t1_result = None

    if t1_result is not None:
        table_one_html = t1_result.to_html()
        eff_sorted = t1_result.effect_sizes.sort_values("q_fdr", na_position="last")
        for _, r in eff_sorted.iterrows():
            es = r["effect_size"]
            es_str = "—" if pd.isna(es) else f"{float(es):+.3f}"
            q = r["q_fdr"]
            q_str = "—" if pd.isna(q) else f"{float(q):.4f}"
            p = r["p_value"]
            p_str = "—" if pd.isna(p) else f"{float(p):.4g}"
            table_one_effect_rows.append(
                {
                    "variable": r["variable"],
                    "kind": r["kind"],
                    "p_value": p_str,
                    "q_fdr": q_str,
                    "effect_size": es_str,
                    "effect_kind": r["effect_kind"],
                    "magnitude": r["magnitude"],
                    "n": int(r["n"]),
                }
            )
        eff = t1_result.effect_sizes.dropna(subset=["q_fdr"])
        n_sig = int((eff["q_fdr"] < 0.05).sum())
        n_total = len(eff)
        table_one_summary = (
            f"{n_sig} of {n_total} variables differ significantly across "
            f"{t1_result.group_col!r} at FDR q<0.05 (Benjamini–Hochberg)."
        )

    # --- Audit metadata ---
    audit_info = {
        "n": len(audit._y_true),
        "n_events": int(audit._y_true.sum()),
        "prevalence": f"{float(audit._y_true.mean()):.3f}",
        "threshold": f"{audit._threshold:.3f}",
        "n_bootstrap": audit._n_bootstrap,
        "n_attributes": len(attributes),
        # The report used to describe the frozen case-control scheme
        # unconditionally, so every patient-scheme report misdescribed its own
        # intervals. The template branches on these two keys instead.
        "resample_scheme": audit._resample_scheme,
        "resample_label": (
            "outcome-stratified"
            if audit._resample_scheme == "case_control"
            else "subgroup-stratified patient-level"
        ),
    }

    # --- Render template ---
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        title=title,
        metadata=metadata or {},
        audit_info=audit_info,
        attributes=attributes,
        disc_table=disc_table,
        cal_table=cal_table,
        hl_table=hl_table,
        hl_decile_blocks=hl_decile_blocks,
        hl_gap_table=hl_gap_table,
        hl_n_tests=hl_n_tests,
        gap_table=gap_table,
        cal_gap_table=cal_gap_table,
        dc_gap_table=dc_gap_table,
        forest_plots=forest_plots,
        roc_plots=roc_plots,
        cal_curve_plots=cal_curve_plots,
        dc_plots=dc_plots,
        has_mitigation=bool(mitigation_audits),
        mitigation_data=mitigation_data,
        has_interactions=has_interactions,
        cell_diag_table=cell_diag_table,
        tripod_checklist=tripod_checklist,
        has_table_one=has_table_one,
        table_one_html=table_one_html,
        table_one_effect_rows=table_one_effect_rows,
        table_one_summary=table_one_summary,
    )

    out_path.write_text(html, encoding="utf-8")
    if pdf:
        report_to_pdf(out_path)
    return out_path


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

# Print typesetting, injected as the document's last <style> so it wins over the
# template's screen styles (WeasyPrint's `stylesheets=` are user-origin and
# lose to author styles). Serif body text, sans-serif tables ruled above and
# below the header and at the foot, header rows repeated on every page a table
# spans, rows never split, one page per top-level section, figures two to a
# row (two-panel figures take the full width) and the per-decile tables three.
# Nothing is recomputed.
_PDF_CSS = """
@page {
    size: A4 landscape;
    margin: 14mm 12mm 15mm 12mm;
    @top-right { content: string(report-title); font: 7pt Georgia, serif; color: #888; }
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font: 8pt Georgia, serif; color: #666;
    }
}
@page :first { @top-right { content: none; } }
body {
    font-family: Georgia, 'Times New Roman', serif; color: #111;
    font-size: 8.6pt; line-height: 1.42;
    max-width: none; margin: 0; padding: 0;
}
h1 { font-size: 15pt; font-weight: normal; color: #111; margin: 0 0 8pt 0;
     string-set: report-title content(); }
h2 { font-size: 12pt; font-weight: normal; color: #111;
     border-bottom: .8pt solid #222; padding-bottom: 2pt; margin: 0 0 8pt 0;
     break-before: page; break-after: avoid; }
#summary h2 { break-before: auto; margin-top: 10pt; }
h3 { font-size: 9.5pt; font-weight: 600; color: #222; margin: 12pt 0 4pt 0;
     break-after: avoid; }
p, .muted { font-size: 8.2pt; line-height: 1.45; color: #333; text-align: justify; }
/* figures are lines of one anonymous block; the default orphans: 2 would push a
   whole run of them past a page that has room for one row */
.section { orphans: 1; widows: 1; }
.section p { orphans: 2; widows: 2; }
.report-header { border-bottom: .8pt solid #222; padding-bottom: 6pt; margin-bottom: 8pt; }
.metadata-grid { display: grid; grid-template-columns: 34mm 1fr; gap: 2pt 8pt;
                 font-size: 8.2pt; margin-top: 6pt; }
.metadata-grid dt { font-weight: 600; color: #333; }
.metadata-grid dd { margin: 0; }
.audit-info { background: none; border: none; border-top: .35pt solid #bbb;
              border-bottom: .35pt solid #bbb; border-radius: 0; padding: 4pt 0; }
.audit-info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2pt 12pt;
                   font-size: 8.2pt; }
.audit-info-grid .label { color: #333; }
table, .tableone-wrap table, .tripod-checklist table, details table {
    border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0;
    font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 6.8pt;
    page-break-inside: auto; break-inside: auto;
}
.tableone-wrap { overflow: visible; }
thead { display: table-header-group; }
th, thead th {
    text-align: left; vertical-align: bottom; font-weight: 600;
    background: #f2f2f0; border: none;
    border-top: .8pt solid #222; border-bottom: .8pt solid #222; padding: 3pt 4pt;
}
td, tbody td { padding: 2.4pt 4pt; border: none; border-bottom: .35pt solid #dedede;
               vertical-align: top; }
tbody tr:last-child td { border-bottom: .8pt solid #222; }
tr { break-inside: avoid; }
tbody tr:hover { background: none; }
.shrunk-row { color: #777; }
details { display: inline-block; width: 32.6%; vertical-align: top;
          margin: 0 .5% 8pt 0; border-left: none; padding-left: 0; break-inside: avoid; }
details summary { font-size: 7.6pt; font-weight: 600; list-style: none; }
details table { font-size: 6.2pt; margin: 2pt 0 0 0; }
.plot-container { display: inline-block; width: 49.4%; vertical-align: top;
                  text-align: center; margin: 4pt 0 8pt 0; break-inside: avoid; }
.plot-container img { width: 100%; height: auto; max-height: 125mm; object-fit: contain; }
.plot-container.plot-wide { display: block; width: 100%; }
.plot-container.plot-wide img { max-height: 170mm; }
.report-footer { font-size: 7.5pt; color: #666; border-top: .8pt solid #222; }
.methodology { font-size: 8pt; }
"""


# Two-panel figures (forest plots, before/after reliability diagrams) are
# unreadable at half width; the PDF gives them the full page width.
_WIDE_PLOT = re.compile(
    r'(<div class="plot-container)(">\s*<img [^>]*alt="(?:Forest plot|Calibration before and after))'
)


def report_to_pdf(html: str | Path, out: str | Path | None = None) -> Path:
    """Typeset an isitfair HTML report as a PDF.

    Renders the report through WeasyPrint with a print stylesheet layered over
    its own: A4 landscape, serif text, ruled sans-serif tables whose header
    row repeats on every page, rows never split across pages, one page per
    section, and page numbers. Nothing is recomputed and no value changes --
    the PDF is the HTML, set for print -- so an existing report can be
    converted without re-running its audit.

    The output is deterministic: the PDF identifier is derived from the HTML's
    bytes, no creation date is written, and the embedded font subsets carry a
    fixed timestamp (``SOURCE_DATE_EPOCH``, set only for the duration of the
    call), so the same report gives the same PDF bytes.

    Parameters
    ----------
    html : str or Path
        Path to a report written by :func:`generate_report`.
    out : str, Path or None, optional
        Output path. Defaults to *html* with a ``.pdf`` suffix.

    Returns
    -------
    Path
        The path to the written PDF.

    Raises
    ------
    ImportError
        If WeasyPrint is not installed (``pip install isitfair[pdf]``).
    FileNotFoundError
        If *html* does not exist.

    Examples
    --------
    >>> audit.report("report.html")                     # doctest: +SKIP
    >>> report_to_pdf("report.html")                    # doctest: +SKIP
    PosixPath('report.pdf')
    """
    try:
        import weasyprint
    except ImportError as exc:
        raise ImportError(
            "PDF export needs WeasyPrint. Install it with `pip install isitfair[pdf]`."
        ) from exc

    html_path = Path(html)
    if not html_path.is_file():
        raise FileNotFoundError(f"no report at {html_path}")
    out_path = Path(out) if out is not None else html_path.with_suffix(".pdf")
    source = html_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32].encode("ascii")
    source = _WIDE_PLOT.sub(r"\1 plot-wide\2", source)
    style = f"<style>{_PDF_CSS}</style>"
    if "</head>" in source:
        source = source.replace("</head>", style + "</head>", 1)
    else:
        source = style + source
    # fontTools stamps every font subset with the current time unless
    # SOURCE_DATE_EPOCH is set; pin it for this call only, then restore it.
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        weasyprint.HTML(
            string=source, base_url=str(html_path.resolve().parent), media_type="print"
        ).write_pdf(out_path, pdf_identifier=digest)
    finally:
        if previous is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous
    return out_path
