"""Intersectional cells and empirical-Bayes shrinkage."""

import numpy as np
import pytest

from isitfair import FairnessAudit


@pytest.mark.parametrize("key", [("sex", "race"), ("race", "sex")])
def test_labels_containing_the_separator(cohort, kw, key):
    # "Asian / Pacific Islander" contains the " / " used to join cell labels.
    audit = FairnessAudit(
        cohort["y"], cohort["p"], {"sex": cohort["sex"], "race": cohort["race"], key: None}, **kw
    )
    disc = audit.discrimination()
    diag = audit.cell_diagnostics()
    assert len(diag) == 6
    assert diag["shrunk"].any()  # the small cells exercise the shrinkage path
    assert not diag["marginal_rate"].isna().any()
    assert (disc["attribute"] == " × ".join(key)).any()


def test_user_supplied_interaction_labels(cohort, kw):
    custom = np.array([f"{s}-{r}" for s, r in zip(cohort["sex"], cohort["race"])])
    audit = FairnessAudit(
        cohort["y"],
        cohort["p"],
        {"sex": cohort["sex"], "race": cohort["race"], ("sex", "race"): custom},
        **kw,
    )
    audit.discrimination()
    diag = audit.cell_diagnostics()
    assert set(diag["subgroup"]) == set(custom)
    assert not diag["marginal_rate"].isna().any()


def test_interaction_labels_not_nested_in_marginals(cohort, kw):
    wrong = np.where(cohort["race"] == "White", "white", "other")  # ignores sex
    with pytest.raises(ValueError, match=r"more than one combination"):
        FairnessAudit(
            cohort["y"],
            cohort["p"],
            {"sex": cohort["sex"], "race": cohort["race"], ("sex", "race"): wrong},
            **kw,
        )


def test_interaction_requires_marginals(cohort, kw):
    with pytest.raises(ValueError, match=r"not present as a\s+string key"):
        FairnessAudit(cohort["y"], cohort["p"], {"sex": cohort["sex"], ("sex", "race"): None}, **kw)
