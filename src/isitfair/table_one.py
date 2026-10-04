"""Table 1: cohort characterisation with effect sizes and BH-FDR.

This module wraps the `tableone <https://pypi.org/project/tableone/>`_ package
to do the heavy lifting (categorical detection, p-value computation,
multi-group support, missing-data handling, HTML/LaTeX rendering) and adds
the two things the fairness audit needs that ``tableone`` does not provide
natively:

1. **Benjamini–Hochberg FDR** adjustment of the per-variable p-values.
   ``tableone``'s ``pval_adjust`` parameter supports Bonferroni / Šidák /
   Holm–Šidák / Simes–Hochberg / Hommel but not ``fdr_bh``, which is the
   default in clinical fairness reporting.
2. **Per-variable effect sizes**: Cohen's d (continuous, two groups),
   rank-biserial r (continuous, two groups), Cohen's h (binary, two groups),
   Cramér's V (categorical, any number of groups), and η² (continuous,
   k > 2) — with conventional magnitude labels.

The main entry point is :func:`table_one`, which returns a
:class:`TableOneResult` holding both the rendered tableone object and the
effect-size overlay DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from tableone import TableOne

from isitfair.effect_sizes import (
    EffectKind,
    cohens_d,
    cohens_h,
    cramers_v,
    effect_size_label,
    eta_squared,
)
from isitfair.inference import fdr_correct

ContinuousSummary = Literal["median_iqr", "mean_sd"]


def _eta_squared(values: npt.NDArray[np.floating[Any]], groups: npt.NDArray[Any]) -> float:
    """One-way η² for continuous values across k groups.

    Kept as a private alias so this module's call sites read unchanged; the
    implementation is :func:`isitfair.eta_squared`.
    """
    return eta_squared(values, groups)


@dataclass(frozen=True)
class TableOneResult:
    """Container for the rendered Table 1 and its effect-size overlay.

    Attributes
    ----------
    table : pandas.DataFrame
        The rendered tableone DataFrame (multi-index rows and columns),
        augmented with the Benjamini–Hochberg adjusted ``q_fdr`` column
        alongside the raw p-value column from tableone. This is what
        ``.to_html()`` / ``.to_latex()`` / ``.to_csv()`` render.
    effect_sizes : pandas.DataFrame
        One row per tested variable with columns: ``variable``, ``kind``,
        (``binary``, ``continuous``, ``categorical``), ``p_value``,
        ``q_fdr``, ``effect_size``, ``effect_kind``, ``magnitude``,
        ``n``. The ``effect_kind`` column says which statistic the
        ``effect_size`` corresponds to (``'d'``, ``'h'``, ``'v'``, ``'r'``,
        or ``'eta2'``).
    underlying : tableone.TableOne
        The original TableOne object (without the ``q_fdr`` column),
        for users who want the un-augmented tableone outputs.
    group_col : str
        Name of the grouping column.
    group_levels : tuple
        The group values, in the order tableone used.
    """

    table: pd.DataFrame
    effect_sizes: pd.DataFrame
    underlying: TableOne
    group_col: str
    group_levels: tuple[Any, ...]

    def to_html(self, **kwargs: Any) -> str:
        """Render the Table 1 with the BH-adjusted ``q_fdr`` column to HTML."""
        return str(self.table.to_html(**kwargs))

    def to_latex(self, **kwargs: Any) -> str:
        """Render the Table 1 with the BH-adjusted ``q_fdr`` column to LaTeX."""
        return str(self.table.to_latex(**kwargs))

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Write the Table 1 with the BH-adjusted ``q_fdr`` column to CSV.

        Defaults to ``encoding='utf-8-sig'`` so Excel-on-Windows correctly
        decodes intersection labels containing the ``×`` (U+00D7) character;
        override by passing ``encoding=`` explicitly.
        """
        kwargs.setdefault("encoding", "utf-8-sig")
        self.table.to_csv(path, **kwargs)


def _classify_columns(
    data: pd.DataFrame,
    columns: list[str],
    continuous_cols: tuple[str, ...] | None,
    binary_cols: tuple[str, ...] | None,
    categorical_cols: tuple[str, ...] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve which columns are continuous / binary / categorical.

    Auto-detection: a column is binary if its non-null values form a subset
    of ``{0, 1}``; otherwise categorical if dtype is object/bool/category or
    has ≤10 unique values; otherwise continuous.
    """
    if continuous_cols is None and binary_cols is None and categorical_cols is None:
        cont, binr, cat = [], [], []
        for col in columns:
            vals = data[col].dropna()
            if vals.isin([0, 1]).all() and vals.nunique() <= 2:
                binr.append(col)
            elif data[col].dtype.kind in {"O", "b"} or vals.nunique() <= 10:
                cat.append(col)
            else:
                cont.append(col)
        return cont, binr, cat

    cont = list(continuous_cols or ())
    binr = list(binary_cols or ())
    cat = list(categorical_cols or ())
    declared = set(cont) | set(binr) | set(cat)
    for col in columns:
        if col in declared:
            continue
        vals = data[col].dropna()
        if vals.isin([0, 1]).all() and vals.nunique() <= 2:
            binr.append(col)
        elif data[col].dtype.kind in {"O", "b"} or vals.nunique() <= 10:
            cat.append(col)
        else:
            cont.append(col)
    return cont, binr, cat


def _compute_effect_size(
    col: str,
    kind: Literal["continuous", "binary", "categorical"],
    data: pd.DataFrame,
    group_col: str,
    group_levels: tuple[Any, ...],
) -> tuple[float, EffectKind | Literal["eta2"], str, int]:
    """Compute the appropriate effect size for one variable.

    Returns
    -------
    effect_size : float
    effect_kind : str
        One of ``'d'``, ``'h'``, ``'v'``, ``'r'``, or ``'eta2'``.
    magnitude : str
    n : int
        Sample size used for the effect-size computation.
    """
    sub = data[[col, group_col]].dropna()
    n = len(sub)
    if n == 0:
        return float("nan"), "d", "undefined", 0

    if kind == "continuous":
        if len(group_levels) == 2:
            a = sub.loc[sub[group_col] == group_levels[0], col].to_numpy(dtype=float)
            b = sub.loc[sub[group_col] == group_levels[1], col].to_numpy(dtype=float)
            d = cohens_d(a, b)
            return d, "d", effect_size_label(d, kind="d"), n
        eta2 = _eta_squared(sub[col].to_numpy(dtype=float), sub[group_col].to_numpy())
        return eta2, "eta2", effect_size_label(np.sqrt(eta2), kind="r"), n

    if kind == "binary":
        if len(group_levels) == 2:
            p_a = float(sub.loc[sub[group_col] == group_levels[0], col].mean())
            p_b = float(sub.loc[sub[group_col] == group_levels[1], col].mean())
            h = cohens_h(p_a, p_b)
            return h, "h", effect_size_label(h, kind="h"), n
        v = cramers_v(sub[col].to_numpy(), sub[group_col].to_numpy())
        return v, "v", effect_size_label(v, kind="v"), n

    # categorical
    v = cramers_v(sub[col].to_numpy(), sub[group_col].to_numpy())
    return v, "v", effect_size_label(v, kind="v"), n


def table_one(
    data: pd.DataFrame,
    *,
    group_col: str,
    columns: tuple[str, ...] | None = None,
    continuous_cols: tuple[str, ...] | None = None,
    binary_cols: tuple[str, ...] | None = None,
    categorical_cols: tuple[str, ...] | None = None,
    nonnormal: tuple[str, ...] | None = None,
    continuous_summary: ContinuousSummary = "median_iqr",
    group_labels: dict[Any, str] | None = None,
    fisher_threshold: int = 5,
    fdr_method: str = "fdr_bh",
    rename: dict[str, str] | None = None,
) -> TableOneResult:
    """Cohort characterisation table with effect sizes and BH-FDR.

    Parameters
    ----------
    data : DataFrame
        Long-format data with one row per observation.
    group_col : str
        Column to stratify by. Two or more levels supported.
    columns : tuple of str, optional
        Variables to include. If None, all columns other than ``group_col``.
    continuous_cols, binary_cols, categorical_cols : tuple of str, optional
        Override auto-detection of variable types. Any column not assigned
        is detected from data type and cardinality.
    nonnormal : tuple of str, optional
        Continuous variables that should be summarised as ``median [IQR]``
        regardless of ``continuous_summary``. If None and
        ``continuous_summary='median_iqr'``, all continuous columns are
        treated as non-normal.
    continuous_summary : ``'median_iqr'`` or ``'mean_sd'``, default
        ``'median_iqr'``
        Default summary style for continuous columns. Per-column overrides
        go via ``nonnormal``.
    group_labels : dict, optional
        Display labels for ``group_col`` values, e.g. ``{0: 'Female',
        1: 'Male'}``. The column itself is renamed via ``rename`` if you
        want a friendlier group header.
    fisher_threshold : int, default 5
        Forwarded to tableone; cells below this expected count switch from
        χ² to Fisher's exact.
    fdr_method : str, default ``'fdr_bh'``
        Any method accepted by
        :func:`statsmodels.stats.multitest.multipletests`.
    rename : dict, optional
        Map of column names to display names, forwarded to tableone.

    Returns
    -------
    TableOneResult

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     'sex': rng.integers(0, 2, 200),
    ...     'age': rng.normal(60, 10, 200),
    ...     'comorb': rng.integers(0, 2, 200),
    ... })
    >>> res = table_one(df, group_col='sex')
    >>> 'q_fdr' in res.effect_sizes.columns
    True
    """
    if columns is None:
        columns = tuple(c for c in data.columns if c != group_col)

    cont, binr, cat = _classify_columns(
        data, list(columns), continuous_cols, binary_cols, categorical_cols
    )

    group_levels_internal: tuple[Any, ...] = tuple(sorted(data[group_col].dropna().unique()))
    data_display = data.copy()
    if group_labels is not None:
        data_display[group_col] = data_display[group_col].map(lambda v: group_labels.get(v, v))
        group_levels = tuple(group_labels.get(v, v) for v in group_levels_internal)
    else:
        group_levels = group_levels_internal

    nonnormal_resolved: list[str]
    if nonnormal is None:
        nonnormal_resolved = cont if continuous_summary == "median_iqr" else []
    else:
        nonnormal_resolved = list(nonnormal)

    # Tableone: build, run, extract p-values.
    t1 = TableOne(
        data_display,
        columns=list(columns),
        categorical=binr + cat,
        groupby=group_col,
        nonnormal=nonnormal_resolved,
        pval=True,
        pval_adjust=None,
        htest_name=False,
        rename=rename,
        missing=False,
    )

    p_col = next(
        (c for c in t1.tableone.columns if isinstance(c, tuple) and c[-1] == "P-Value"),
        None,
    )
    # tableone renders variable labels via ``rename``; invert it so the
    # p-values and q-FDR extracted below join back on the original column
    # names rather than the display names.
    inv_rename = {v: k for k, v in rename.items()} if rename else {}

    raw_pvals: dict[str, float] = {}
    if p_col is not None:
        for idx, raw in t1.tableone[p_col].items():
            var_label = idx[0] if isinstance(idx, tuple) else idx
            try:
                p_str = str(raw).strip()
                if p_str in {"", "nan"}:
                    continue
                pv = float(p_str.lstrip("<"))
            except ValueError:
                continue
            # Strip the summary suffix (", median [Q1,Q3]" / ", n (%)") and
            # undo any display rename, back to the original column name.
            base = var_label.split(",")[0]
            base = inv_rename.get(base, base)
            raw_pvals.setdefault(base, pv)

    # Effect sizes per variable.
    fisher_threshold_unused = fisher_threshold  # tableone handles its own switch
    del fisher_threshold_unused

    rows = []
    for col in cont:
        if col not in data.columns:
            continue
        es, ek, mag, n = _compute_effect_size(
            col, "continuous", data, group_col, group_levels_internal
        )
        rows.append(
            {
                "variable": col,
                "kind": "continuous",
                "p_value": raw_pvals.get(col, float("nan")),
                "effect_size": es,
                "effect_kind": ek,
                "magnitude": mag,
                "n": n,
            }
        )
    for col in binr:
        if col not in data.columns:
            continue
        es, ek, mag, n = _compute_effect_size(col, "binary", data, group_col, group_levels_internal)
        rows.append(
            {
                "variable": col,
                "kind": "binary",
                "p_value": raw_pvals.get(col, float("nan")),
                "effect_size": es,
                "effect_kind": ek,
                "magnitude": mag,
                "n": n,
            }
        )
    for col in cat:
        if col not in data.columns:
            continue
        es, ek, mag, n = _compute_effect_size(
            col, "categorical", data, group_col, group_levels_internal
        )
        rows.append(
            {
                "variable": col,
                "kind": "categorical",
                "p_value": raw_pvals.get(col, float("nan")),
                "effect_size": es,
                "effect_kind": ek,
                "magnitude": mag,
                "n": n,
            }
        )

    eff = pd.DataFrame(rows)
    if len(eff):
        eff["q_fdr"] = fdr_correct(eff["p_value"].to_numpy(), method=fdr_method)
    else:
        eff["q_fdr"] = pd.Series(dtype=float)

    eff = eff[
        ["variable", "kind", "p_value", "q_fdr", "effect_size", "effect_kind", "magnitude", "n"]
    ]

    # Bolt the q_fdr column onto the rendered tableone table. Match tableone's
    # convention of placing the per-variable p-value on the first row of each
    # variable group only, so q_fdr lines up with P-Value instead of repeating
    # the same value on every category row.
    table = t1.tableone.copy()
    if p_col is not None:
        q_map = dict(zip(eff["variable"], eff["q_fdr"], strict=False))
        q_col = (p_col[0], "q_fdr") if isinstance(p_col, tuple) else "q_fdr"
        new_col = []
        for idx, p_raw in zip(table.index, table[p_col], strict=True):
            p_str = str(p_raw).strip()
            if p_str in {"", "nan"}:
                new_col.append("")
                continue
            var_label = idx[0] if isinstance(idx, tuple) else idx
            base = var_label.split(",")[0]
            base = inv_rename.get(base, base)
            q = q_map.get(base, float("nan"))
            new_col.append("" if np.isnan(q) else f"{q:.3g}")
        table[q_col] = new_col

    return TableOneResult(
        table=table,
        effect_sizes=eff,
        underlying=t1,
        group_col=group_col,
        group_levels=group_levels,
    )
