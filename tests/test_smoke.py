"""The documented entry points run end to end."""

import numpy as np

from isitfair import FairnessAudit


def test_getting_started_example(tmp_path):
    # docs/getting_started.md, "Minimal example", verbatim apart from the path.
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.binomial(1, 0.15, n).astype(float)
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, n), 0.01, 0.99)
    sex = rng.choice(["female", "male"], n)

    audit = FairnessAudit(
        y_true=y_true,
        y_proba=y_proba,
        sensitive_features={"sex": sex},
        threshold=0.15,
        n_bootstrap=50,
        random_state=42,
    )
    results = audit.run_all()
    for axis in ("discrimination", "calibration", "decision_curve", "estimability"):
        assert not results[axis].empty

    out = audit.report(tmp_path / "fairness_report.html", title="My Model Audit")
    assert out.exists()
    assert "<html" in out.read_text(encoding="utf8").lower()
    figures = sorted(p.name for p in (tmp_path / "fairness_report_figures").glob("*.png"))
    assert figures == [
        "calibration_sex.png",
        "decision_curve_sex.png",
        "discrimination_sex.png",
        "roc_sex.png",
    ]


def test_dataframe_columns_as_inputs(cohort, kw):
    # The common path from a user's own data: columns of one DataFrame.
    import pandas as pd

    df = pd.DataFrame({"outcome": cohort["y"], "risk": cohort["p"], "sex": cohort["sex"]})
    df.index = df.index + 1000  # a non-default index must not matter
    audit = FairnessAudit(df["outcome"], df["risk"], {"sex": df["sex"]}, **kw)
    disc = audit.discrimination()
    assert set(disc["subgroup"]) == {"Overall", "F", "M", "GAP"}


def test_repr_is_informative(cohort, kw):
    audit = FairnessAudit(cohort["y"], cohort["p"], {"sex": cohort["sex"]}, **kw)
    text = repr(audit)
    assert text.startswith("FairnessAudit(n=1500")
    assert "sex (2)" in text
    assert "0x" not in text
