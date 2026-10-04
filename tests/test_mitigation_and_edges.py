"""Mitigation, the recalibration ladder, and sparse subgroups."""

import numpy as np
import pandas as pd
import pytest

from isitfair import FairnessAudit, GroupRecalibration, Reweighing, WassersteinPostprocessing

FLOORS = {"auroc": 5, "calibration_intercept": 5, "calibration_slope": 10, "net_benefit": 5}


@pytest.mark.parametrize("method", [GroupRecalibration(), WassersteinPostprocessing()])
def test_mitigate_returns_one_audit_per_attribute(cohort, kw, method):
    sf = {"sex": cohort["sex"], "race": cohort["race"]}
    audit = FairnessAudit(cohort["y"], cohort["p"], sf, event_floors=FLOORS, **kw)
    mitigated = audit.mitigate(method, calibration_data=(cohort["y"], cohort["p"], sf))
    assert set(mitigated) == {"sex", "race"}
    for new in mitigated.values():
        assert new._event_floors == FLOORS  # custom floors carry over
        assert not new.calibration().empty


def test_reweighing_is_refused_by_mitigate(cohort, kw):
    audit = FairnessAudit(cohort["y"], cohort["p"], {"sex": cohort["sex"]}, **kw)
    with pytest.raises(NotImplementedError, match=r"retrain"):
        audit.mitigate(Reweighing(), calibration_data=(cohort["y"], cohort["p"], {"sex": cohort["sex"]}))


def test_recalibration_ladder_follows_the_audit_seed(cohort, kw):
    sf = {"sex": cohort["sex"]}
    one = FairnessAudit(cohort["y"], cohort["p"], sf, **kw).recalibration_ladder(by="sex")
    two = FairnessAudit(cohort["y"], cohort["p"], sf, **kw).recalibration_ladder(by="sex")
    pd.testing.assert_frame_equal(one, two)
    assert set(one["rung"]) == {"none", "common", "group_specific"}


def test_single_class_and_tiny_subgroups_do_not_raise(cohort, kw):
    y, p = cohort["y"].copy(), cohort["p"]
    group = np.array(["big"] * len(y), dtype=object)
    group[:3] = "tiny"  # three patients
    no_events = np.flatnonzero(y == 0)[3:40]
    group[no_events] = "no_events"  # one outcome class only
    audit = FairnessAudit(y, p, {"g": group}, **kw)
    disc = audit.discrimination().set_index("subgroup")
    assert np.isnan(disc.loc["no_events", "auroc"])
    est = audit.estimability()
    # Every floored metric loses its interval; prevalence has no floor by design.
    floored = est[(est["subgroup"] == "no_events") & (est["metric"] != "prevalence")]
    assert floored["not_estimable_final"].all()
    assert not audit.calibration().empty
    assert not audit.decision_curve().empty
