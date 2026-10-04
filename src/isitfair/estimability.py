"""Event floors and the conservative-union estimability rule.

A subgroup metric is reportable only when the subgroup carries enough
**events** to support it, and only when the bootstrap actually managed to
estimate it. Those are two different rules and they are applied as a
conservative union:

.. code-block:: text

    not reportable = (events < the metric's floor)
                     OR (> nonestimable_threshold of replicates non-estimable)

A cell below its floor is suppressed *even when every replicate estimated
cleanly*, because the floor binds on the observed event count regardless of
how the replicates happened to fall. The replicate rule alone cannot see this:
net benefit, for instance, has no replicate-based estimability signal at all
under the patient scheme, so the event floor is the only rule available to it.

**Suppression withholds intervals, never point estimates.** A suppressed point
is still a real quantity that was computed; deleting it hides what was audited
rather than what was found unsupportable. Report it, label it not estimable,
and do not interpret it inferentially.

The defaults in :data:`DEFAULT_EVENT_FLOORS` are the floors pre-registered for
the perioperative audit this package was built for. They are conservative and
their reasoning is in :data:`EVENT_FLOOR_RATIONALE`; a user with a different
prevalence regime should set their own and record why. Passing an empty dict
disables the floors entirely, which is how the frozen published analyses pin
their original behaviour.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = [
    "DEFAULT_EVENT_FLOORS",
    "EVENT_FLOOR_RATIONALE",
    "FLOOR_FOR",
    "GAP_SOURCE",
    "floor_fails",
    "resolve_event_floors",
    "resolve_floor",
]


#: Minimum events required before a metric is reported inferentially.
DEFAULT_EVENT_FLOORS: dict[str, int] = {
    "auroc": 25,
    "calibration_intercept": 25,
    "calibration_slope": 100,
    "net_benefit": 50,
}

#: Why each floor sits where it does. Quoted in reports so that a reader can
#: judge the rule rather than take the number on trust.
EVENT_FLOOR_RATIONALE: dict[str, str] = {
    "auroc": (
        "25 events. The c-statistic is estimable whenever both classes are "
        "present, but its standard error is approximately 0.5/sqrt(E) in a "
        "low-prevalence regime; 25 events holds the 95% interval to about "
        "+/-0.20, the widest interval that can still separate a strong "
        "c-statistic from chance."
    ),
    "calibration_intercept": (
        "25 events. The standard error of calibration-in-the-large is "
        "approximately 1/sqrt(E); 25 events gives +/-0.39 on the log-odds "
        "scale, small relative to the between-subgroup gradients an audit is "
        "looking for."
    ),
    "calibration_slope": (
        "100 events. Riley et al. (Stat Med 2021) require roughly 100 events "
        "and 100 non-events for a calibration slope precise enough to be "
        "interpreted; the slope is the most sample-hungry of the four because "
        "it estimates a gradient across the risk range rather than a level."
    ),
    "net_benefit": (
        "50 events. Net benefit depends on the true positives among those "
        "flagged, a subset of the events. Below roughly 50 events the flagged "
        "count falls into single digits and the estimate is driven by "
        "individual patients."
    ),
}

#: Which floor governs which bootstrapped metric. One mapping, so that the
#: audit frames, :meth:`FairnessAudit.estimability` and any downstream
#: consumer cannot disagree about which rule applies to what.
FLOOR_FOR: dict[str, str] = {
    "auroc": "auroc",
    "auprc": "auroc",
    "sensitivity": "auroc",
    "specificity": "auroc",
    "ppv": "auroc",
    "npv": "auroc",
    "calibration_intercept": "calibration_intercept",
    "calibration_slope": "calibration_slope",
    "brier": "calibration_intercept",
    "net_benefit": "net_benefit",
    "standardized_net_benefit": "net_benefit",
}

#: Which per-subgroup metric each gap statistic is a reduction over. A gap has
#: no event count of its own, so it inherits the floor failure of any
#: contributing subgroup. Kept separate from :data:`FLOOR_FOR` so that a first
#: pass over subgroup metrics never accidentally resolves a gap.
GAP_SOURCE: dict[str, str] = {
    "max_intercept_gap": "calibration_intercept",
    "max_slope_gap": "calibration_slope",
    "max_net_benefit_gap": "net_benefit",
    "max_auroc_gap": "auroc",
    "equal_opportunity_difference": "auroc",
    "predictive_equality": "auroc",
    "equalized_odds_gap": "auroc",
    "statistical_parity": "auroc",
}


def resolve_event_floors(event_floors: dict[str, int] | None) -> dict[str, int]:
    """Normalise the ``event_floors`` argument into a validated mapping.

    ``None`` means the pre-registered defaults; an empty dict disables the
    floors. Any other mapping is used as given, after checking that its keys
    are floor rules this module knows how to apply.

    Parameters
    ----------
    event_floors : dict of str to int, or None
        Floors keyed by rule name (``"auroc"``, ``"calibration_intercept"``,
        ``"calibration_slope"``, ``"net_benefit"``).

    Returns
    -------
    dict of str to int
        The resolved floors. An empty dict means no floor is applied.

    Raises
    ------
    ValueError
        If a key is not a known floor rule, or a value is not a
        non-negative integer.

    Examples
    --------
    >>> resolve_event_floors(None)["calibration_slope"]
    100
    >>> resolve_event_floors({})
    {}
    >>> resolve_event_floors({"auroc": 10})
    {'auroc': 10}
    """
    if event_floors is None:
        return dict(DEFAULT_EVENT_FLOORS)
    known = set(DEFAULT_EVENT_FLOORS)
    unknown = sorted(set(event_floors) - known)
    if unknown:
        msg = (
            f"unknown event-floor rule(s) {unknown}; valid rules are "
            f"{sorted(known)}. Floors are keyed by rule, not by metric — "
            "see isitfair.estimability.FLOOR_FOR for the metric-to-rule map."
        )
        raise ValueError(msg)
    resolved: dict[str, int] = {}
    for rule, value in event_floors.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"event floor for {rule!r} must be a non-negative int, got {value!r}"
            raise ValueError(msg)
        resolved[rule] = int(value)
    return resolved


def resolve_floor(metric: str, floors: dict[str, int] | None = None) -> int | None:
    """The event floor governing *metric*, or ``None`` if it has none.

    Resolves a subgroup metric through :data:`FLOOR_FOR` and a gap metric
    through :data:`GAP_SOURCE`, so callers do not have to know which kind of
    name they are holding.

    Parameters
    ----------
    metric : str
        A subgroup metric (``"auroc"``, ``"calibration_slope"``, ...) or a gap
        metric (``"max_intercept_gap"``, ...).
    floors : dict of str to int, or None
        Floors as accepted by :func:`resolve_event_floors`.

    Returns
    -------
    int or None
        The floor, or ``None`` when floors are disabled or the metric has no
        rule (for example ``"ece"``, which carries no interval).

    Examples
    --------
    >>> resolve_floor("brier")
    25
    >>> resolve_floor("max_slope_gap")
    100
    >>> resolve_floor("ece") is None
    True
    """
    resolved = resolve_event_floors(floors)
    rule = FLOOR_FOR.get(metric) or GAP_SOURCE.get(metric)
    if rule is None:
        return None
    return resolved.get(rule)


def floor_fails(events: Any, metric: str, floors: dict[str, int] | None = None) -> bool:
    """Does *events* fall below the event floor governing *metric*?

    Parameters
    ----------
    events : int or float
        The subgroup's observed event count. A missing value raises rather
        than answering — see Raises.
    metric : str
        A subgroup metric or a gap metric; resolved through
        :data:`FLOOR_FOR` then :data:`GAP_SOURCE`.
    floors : dict of str to int, or None
        Floors as accepted by :func:`resolve_event_floors`. An empty dict
        disables the rule, so this always returns ``False``.

    Returns
    -------
    bool
        ``True`` when the metric is below its floor and must not be reported
        inferentially.

    Raises
    ------
    KeyError
        If *metric* has no floor rule at all.
    ValueError
        If *events* is missing. A ``GAP`` row has no event count of its own,
        and answering ``False`` for an unknown count is how a gap silently
        escapes the very floor its constituents fail. Ask about a gap by its
        own metric name and resolve it from its constituent subgroups.

    Examples
    --------
    >>> floor_fails(6, "calibration_intercept")
    True
    >>> floor_fails(98, "calibration_slope")
    True
    >>> floor_fails(98, "calibration_intercept")
    False
    >>> floor_fails(6, "calibration_intercept", floors={})
    False
    """
    resolved = resolve_event_floors(floors)
    rule = FLOOR_FOR.get(metric) or GAP_SOURCE.get(metric)
    if rule is None:
        msg = f"no event floor is defined for metric {metric!r}"
        raise KeyError(msg)
    if events is None or pd.isna(events):
        msg = (
            f"cannot apply the {rule} floor to metric {metric!r}: the event "
            "count is missing. A GAP row has none of its own — ask about it by "
            "its gap metric, and resolve it from its constituent subgroups."
        )
        raise ValueError(msg)
    floor = resolved.get(rule)
    if floor is None:
        return False
    return bool(int(events) < floor)
