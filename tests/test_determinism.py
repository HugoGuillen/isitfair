"""The determinism guarantees the README and user guide make."""

import pandas as pd

from isitfair import FairnessAudit

AXES = ("discrimination", "calibration", "decision_curve", "estimability")


def _frames(audit):
    return {axis: getattr(audit, axis)() for axis in AXES}


def test_same_seed_same_output(cohort, kw):
    sf = {"sex": cohort["sex"], "race": cohort["race"]}
    a = _frames(FairnessAudit(cohort["y"], cohort["p"], sf, **kw))
    b = _frames(FairnessAudit(cohort["y"], cohort["p"], sf, **kw))
    for axis in AXES:
        pd.testing.assert_frame_equal(a[axis], b[axis])


def test_results_do_not_depend_on_n_jobs(cohort, kw):
    sf = {"sex": cohort["sex"]}
    serial = FairnessAudit(cohort["y"], cohort["p"], sf, **kw)
    parallel = FairnessAudit(cohort["y"], cohort["p"], sf, n_jobs=2, **kw)
    for axis in ("discrimination", "calibration"):
        pd.testing.assert_frame_equal(getattr(serial, axis)(), getattr(parallel, axis)())


def test_adding_an_intersection_leaves_marginals_unchanged(cohort, kw):
    marg = {"sex": cohort["sex"], "race": cohort["race"]}
    base = FairnessAudit(cohort["y"], cohort["p"], marg, **kw)
    plus = FairnessAudit(cohort["y"], cohort["p"], {**marg, ("sex", "race"): None}, **kw)
    for axis in ("discrimination", "calibration"):
        before = getattr(base, axis)()
        after = getattr(plus, axis)()
        after = after[after["attribute"].isin(marg)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after)
