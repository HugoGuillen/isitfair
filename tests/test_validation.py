"""Inputs a user's own data typically gets wrong are rejected with a usable message."""

import numpy as np
import pandas as pd
import pytest

from isitfair import FairnessAudit, compare_periods


def test_missing_label_in_string_attribute(cohort, kw):
    sex = pd.Series(cohort["sex"], dtype=object)
    sex[:20] = np.nan
    with pytest.raises(ValueError, match=r"has 20 missing value\(s\).*fillna"):
        FairnessAudit(cohort["y"], cohort["p"], {"sex": sex}, **kw)


def test_missing_label_in_numeric_attribute(cohort, kw):
    grp = np.where(cohort["sex"] == "F", 1.0, 2.0)
    grp[:50] = np.nan
    with pytest.raises(ValueError, match=r"has 50 missing value\(s\)"):
        FairnessAudit(cohort["y"], cohort["p"], {"grp": grp}, **kw)


def test_stringified_missing_label_warns(cohort, kw):
    sex = pd.Series(cohort["sex"], dtype=object)
    sex[:20] = np.nan
    with pytest.warns(UserWarning, match=r"look like missing"):
        FairnessAudit(cohort["y"], cohort["p"], {"sex": sex.astype(str)}, **kw)


def test_missing_probability(cohort, kw):
    p = cohort["p"].copy()
    p[:5] = np.nan
    with pytest.raises(ValueError, match=r"`y_proba` has 5 missing"):
        FairnessAudit(cohort["y"], p, {"sex": cohort["sex"]}, **kw)


def test_missing_outcome(cohort, kw):
    y = cohort["y"].copy()
    y[:3] = np.nan
    with pytest.raises(ValueError, match=r"`y_true` has 3 missing"):
        FairnessAudit(y, cohort["p"], {"sex": cohort["sex"]}, **kw)


def test_probabilities_out_of_range(cohort, kw):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\].*logits"):
        FairnessAudit(cohort["y"], cohort["p"] * 3 - 1, {"sex": cohort["sex"]}, **kw)


def test_two_column_predict_proba(cohort, kw):
    p2 = np.c_[1 - cohort["p"], cohort["p"]]
    with pytest.raises(ValueError, match=r"predict_proba\(X\)\[:, 1\]"):
        FairnessAudit(cohort["y"], p2, {"sex": cohort["sex"]}, **kw)


def test_string_outcome(cohort, kw):
    y = np.where(cohort["y"] == 1, "yes", "no")
    with pytest.raises(ValueError, match=r"Recode the outcome to 0/1"):
        FairnessAudit(y, cohort["p"], {"sex": cohort["sex"]}, **kw)


def test_non_binary_outcome(cohort, kw):
    with pytest.raises(ValueError, match=r"only 0 and 1"):
        FairnessAudit(cohort["y"] + 1, cohort["p"], {"sex": cohort["sex"]}, **kw)


def test_hard_labels_as_probabilities_warn(cohort, kw):
    labels = (cohort["p"] > 0.2).astype(float)
    with pytest.warns(UserWarning, match=r"contains only 0 and 1"):
        FairnessAudit(cohort["y"], labels, {"sex": cohort["sex"]}, **kw)


def test_unbinned_continuous_attribute_warns(cohort, kw):
    with pytest.warns(UserWarning, match=r"Bin continuous variables"):
        FairnessAudit(cohort["y"], cohort["p"], {"age": cohort["age"]}, **kw)


def test_binned_attribute_from_pd_cut(cohort, kw):
    age_group = pd.cut(cohort["age"], [0, 65, 80, 120])
    audit = FairnessAudit(cohort["y"], cohort["p"], {"age_group": age_group}, **kw)
    groups = set(audit.discrimination()["subgroup"]) - {"Overall", "GAP"}
    assert len(groups) == 3


def test_shuffled_pandas_index_rejected(cohort, kw):
    df = pd.DataFrame({"y": cohort["y"], "p": cohort["p"], "sex": cohort["sex"]})
    shuffled = df.sample(frac=1, random_state=3)
    with pytest.raises(ValueError, match=r"aligns inputs by\s+position"):
        FairnessAudit(shuffled["y"], shuffled["p"].sort_index(), {"sex": shuffled["sex"]}, **kw)


def test_length_mismatch(cohort, kw):
    with pytest.raises(ValueError, match=r"same length"):
        FairnessAudit(cohort["y"], cohort["p"][:-1], {"sex": cohort["sex"]}, **kw)


def test_compare_periods_unknown_attribute(cohort, kw):
    audit = FairnessAudit(cohort["y"], cohort["p"], {"sex": cohort["sex"]}, **kw)
    with pytest.raises(ValueError, match=r"available: \['sex'\]"):
        compare_periods({"a": audit, "b": audit}, attribute="sexx")
