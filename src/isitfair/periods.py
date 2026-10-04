"""Compare the same audit across two or more periods under one rule.

The package does not refit models — that is a training concern, and models are
user input. But *comparing* two already-constructed audits is evaluation, and
doing it by hand is where a temporal analysis usually goes wrong: the periods
end up judged under different estimability rules, or aligned on subgroups one
of them does not contain.

:func:`compare_periods` takes audits that are already built and puts them side
by side on one index, applying the union of every period's floor verdict so a
row is interpreted only where **both** periods support it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["compare_periods"]

_AXIS_METRICS: dict[str, tuple[str, ...]] = {
    "discrimination": ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"),
    "calibration": ("calibration_intercept", "calibration_slope", "brier", "ece", "ici"),
}

_CI_SUFFIX = {
    "calibration_intercept": "intercept",
    "calibration_slope": "slope",
}


def _ci_columns(metric: str) -> tuple[str, str]:
    """The interval column names for *metric* as the audit frames spell them."""
    stem = _CI_SUFFIX.get(metric, metric)
    return f"{stem}_ci_low", f"{stem}_ci_high"


def compare_periods(
    audits: dict[str, Any],
    *,
    axis: str = "calibration",
    attribute: str | None = None,
) -> pd.DataFrame:
    """Put one audit axis side by side across periods.

    Parameters
    ----------
    audits : dict[str, FairnessAudit]
        Period label to audit, e.g.
        ``{"2018-2020": dev_audit, "2021-2022": holdout_audit}``. Insertion
        order is preserved, so the development period should come first.
    axis : {"discrimination", "calibration"}, default "calibration"
        Which axis to compare.
    attribute : str or None
        Restrict to one sensitive attribute. ``None`` compares all attributes
        the audits share.

    Returns
    -------
    pandas.DataFrame
        One row per ``(attribute, subgroup, metric)``, with a column block per
        period (``<period>``, ``<period>_ci_low``, ``<period>_ci_high``,
        ``<period>_n_events``), plus:

        - ``delta`` — last period minus first. A difference, not a test: the
          two periods are different patients, so this is not a paired contrast
          and carries no interval.
        - ``not_estimable`` — the **union** across periods. A row is
          interpretable only where every period supports it, because a
          comparison is exactly as estimable as its weaker side.
        - ``periods_estimable`` — how many periods individually cleared their
          floors, so a union failure can be attributed.

    Raises
    ------
    ValueError
        If fewer than two audits are given, if *axis* is not comparable, or if
        the audits disagree about the decision threshold — a threshold-specific
        metric compared across two thresholds is not a period effect.

    Notes
    -----
    Refitting is the caller's job and deliberately out of scope. Everything the
    model pipeline learns — preprocessing, feature selection, any class
    rebalancing — must be fitted on the development period alone, or the
    holdout is not held out.

    A period difference is not evidence of model drift on its own. Coding
    practice, documentation completeness and outcome ascertainment also change
    over time; compare case mix before attributing a difference to the model.

    Examples
    --------
    >>> import numpy as np
    >>> from isitfair import FairnessAudit
    >>> rng = np.random.default_rng(0)
    >>> def mk(seed):
    ...     r = np.random.default_rng(seed)
    ...     y = r.binomial(1, 0.3, 600).astype(float)
    ...     p = np.clip(0.3 + 0.3 * y + r.normal(0, 0.2, 600), 0.01, 0.99)
    ...     return FairnessAudit(y, p, {"g": r.choice(["a", "b"], 600)},
    ...                          threshold=0.3, n_bootstrap=20, random_state=0)
    >>> out = compare_periods({"dev": mk(1), "holdout": mk(2)})
    >>> "delta" in out.columns
    True
    """
    if len(audits) < 2:
        msg = f"`audits` needs at least two periods, got {len(audits)}"
        raise ValueError(msg)
    if axis not in _AXIS_METRICS:
        msg = f"`axis` must be one of {sorted(_AXIS_METRICS)}, got {axis!r}"
        raise ValueError(msg)

    thresholds = {a._threshold for a in audits.values()}
    if len(thresholds) > 1:
        msg = (
            f"audits disagree about the decision threshold ({sorted(thresholds)}). "
            "Sensitivity, specificity and net benefit are threshold-specific, so "
            "comparing them across two thresholds measures the threshold, not the "
            "period."
        )
        raise ValueError(msg)

    labels = list(audits)
    frames = {label: getattr(a, axis)() for label, a in audits.items()}

    if attribute is not None:
        available = sorted({str(x) for f in frames.values() for x in f["attribute"]})
        if attribute not in available:
            msg = f"`attribute` {attribute!r} is in none of the audits; available: {available}"
            raise ValueError(msg)

    # The per-metric verdict, not the row-level one. A stratum with 98 events
    # keeps its calibration intercept (floor 25) and loses its slope (floor
    # 100); reading the row flag would suppress the intercept comparison
    # because a different metric on the same row failed.
    verdicts: dict[str, dict[tuple[str, str, str], bool]] = {}
    for label, audit in audits.items():
        est = audit.estimability()
        verdicts[label] = {
            (str(r["attribute"]), str(r["subgroup"]), str(r["metric"])): bool(
                r["not_estimable_final"]
            )
            for _, r in est.iterrows()
        }

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label in labels:
        frame = frames[label]
        rows = frame[frame["subgroup"] != "GAP"]
        if attribute is not None:
            rows = rows[rows["attribute"] == attribute]
        for _, r in rows.iterrows():
            for metric in _AXIS_METRICS[axis]:
                if metric not in rows.columns:
                    continue
                key = (str(r["attribute"]), str(r["subgroup"]), metric)
                rec = records.setdefault(
                    key,
                    {
                        "attribute": key[0],
                        "subgroup": key[1],
                        "metric": key[2],
                        "periods_estimable": 0,
                        "not_estimable": False,
                    },
                )
                lo_col, hi_col = _ci_columns(metric)
                rec[label] = r[metric]
                rec[f"{label}_ci_low"] = r.get(lo_col, float("nan"))
                rec[f"{label}_ci_high"] = r.get(hi_col, float("nan"))
                rec[f"{label}_n_events"] = r.get("n_events", float("nan"))
                # Metrics with no floor rule of their own (ECE, ICI) inherit
                # the row's verdict, which is the conservative reading.
                flagged = verdicts[label].get(
                    key, bool(r.get("not_estimable_final", r.get("not_estimable", False)))
                )
                rec["not_estimable"] = rec["not_estimable"] or flagged
                rec["periods_estimable"] += 0 if flagged else 1

    out = pd.DataFrame(list(records.values()))
    if out.empty:
        return out

    # Rows present in only some periods are still shown, with the missing
    # period blank: dropping them would hide a subgroup that disappeared, which
    # is itself a finding about the cohort.
    for label in labels:
        if label not in out.columns:
            out[label] = float("nan")
    missing_any = out[labels].isna().any(axis=1)
    out.loc[missing_any, "not_estimable"] = True

    out["delta"] = out[labels[-1]] - out[labels[0]]

    ordered = ["attribute", "subgroup", "metric"]
    for label in labels:
        ordered += [label, f"{label}_ci_low", f"{label}_ci_high", f"{label}_n_events"]
    ordered += ["delta", "periods_estimable", "not_estimable"]
    ordered = [c for c in ordered if c in out.columns]
    return (
        out[ordered]
        .sort_values(["attribute", "subgroup", "metric"], kind="stable")
        .reset_index(drop=True)
    )
