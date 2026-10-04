"""Audit your own model's predictions: a template to copy and edit.

You need one table (CSV) with one row per patient, holding:

- the observed binary outcome (0/1),
- your model's predicted probability of that outcome, made on patients the
  model was NOT trained on (a held-out test set or out-of-fold predictions),
- the patient attributes you want to audit across (sex, age, ...).

Edit the CONFIGURATION block below to match your column names, then run:

    python audit_your_own_data.py path/to/your_predictions.csv

Run without a path to see it work on a demonstration file built from the
public SUPPORT2 cohort shipped in this repository (needs the repository
checkout; the demo predictions come from examples/support2_data.py).

The script writes an HTML report plus a folder of 300-DPI figures. See
docs/user_guide/your_own_data.md for what each step does and why.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from isitfair import FairnessAudit

# --- CONFIGURATION: edit to match your data --------------------------------

OUTCOME = "hospdead"  # column with the observed outcome, coded 0/1
RISK = "y_proba"  # column with the predicted probability, between 0 and 1

# Audit axes: {name shown in the report: column in your file}.
ATTRIBUTES = {"sex": "sex", "race": "race", "age_group": "age"}

# Continuous columns must be binned into groups first: {column: bin edges}.
BINS = {"age": [0, 45, 65, 80, 120]}

# Label given to a missing attribute value (missing labels are not allowed).
MISSING_LABEL = "unknown"

# Two-way intersections to audit, by axis name. Use () for none.
INTERSECTIONS = [("sex", "age_group")]

THRESHOLD = 0.30  # the risk above which a patient would be acted on
N_BOOTSTRAP = 2000  # use 200 for a quick first pass
RANDOM_STATE = 42  # fixed seed: same data + same seed = identical output

# ---------------------------------------------------------------------------


def make_demo_csv(folder: Path) -> Path:
    """Write the SUPPORT2 test partition, with predictions, as a raw CSV."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from support2_data import fit_reference_model

    frame = fit_reference_model(prefer_cache=True).frame_test
    path = folder / "support2_demo_predictions.csv"
    frame[["hospdead", "y_proba", "sex", "race", "age"]].to_csv(path, index=False)
    return path


def load(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read the CSV and turn it into the three array inputs of FairnessAudit."""
    df = pd.read_csv(path)
    needed = [OUTCOME, RISK, *ATTRIBUTES.values()]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(f"columns {absent} not in {path.name}; available: {list(df.columns)}")

    # Rows without an outcome or a prediction cannot be audited.
    keep = df[OUTCOME].notna() & df[RISK].notna()
    if (~keep).any():
        print(f"dropping {(~keep).sum()} row(s) with a missing outcome or prediction")
    df = df[keep].reset_index(drop=True)

    attributes = {}
    for name, column in ATTRIBUTES.items():
        values = df[column]
        if column in BINS:
            values = pd.cut(values, BINS[column], right=False)  # outside the edges -> missing
        # "string", not str: str would turn a missing value into the text "nan".
        values = values.astype("string")
        n_missing = int(values.isna().sum())
        if n_missing:
            print(f"{name}: {n_missing} missing value(s) labelled {MISSING_LABEL!r}")
        attributes[name] = values.fillna(MISSING_LABEL).to_numpy(dtype=object)

    return df[OUTCOME].to_numpy(dtype=float), df[RISK].to_numpy(dtype=float), attributes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, help="your predictions file")
    parser.add_argument("--out", type=Path, default=Path("audit_report.html"))
    parser.add_argument("--fast", action="store_true", help="200 bootstrap replicates")
    args = parser.parse_args()

    if args.csv is None:
        print("no file given: running on the SUPPORT2 demonstration file")
        args.csv = make_demo_csv(Path(tempfile.mkdtemp()))
    y, p, attributes = load(args.csv)
    print(f"N = {len(y)}, events = {int(y.sum())}, prevalence = {y.mean():.3f}")

    # Each subgroup's event count decides what can carry an interval.
    for name, values in attributes.items():
        counts = pd.DataFrame({"g": values, "y": y}).groupby("g")["y"].agg(["size", "sum"])
        text = ", ".join(f"{g} n={int(r['size'])}/{int(r['sum'])} events" for g, r in counts.iterrows())
        print(f"  {name}: {text}")

    sensitive = dict(attributes)
    for pair in INTERSECTIONS:
        sensitive[tuple(pair)] = None

    audit = FairnessAudit(
        y_true=y,
        y_proba=p,
        sensitive_features=sensitive,
        threshold=THRESHOLD,
        n_bootstrap=200 if args.fast else N_BOOTSTRAP,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print(audit)

    disc = audit.discrimination()
    cal = audit.calibration()
    rows = disc["subgroup"] != "GAP"
    print(disc.loc[rows, ["attribute", "subgroup", "n", "n_events", "auroc", "auroc_ci_low", "auroc_ci_high"]].round(3).to_string(index=False))
    rows = cal["subgroup"] != "GAP"
    print(cal.loc[rows, ["attribute", "subgroup", "calibration_intercept", "calibration_slope", "ici"]].round(3).to_string(index=False))

    suppressed = audit.estimability().query("not_estimable_final")
    print(f"{len(suppressed)} subgroup-metric cells are reported without an interval (too few events)")

    out = audit.report(args.out, title=f"Fairness audit: {args.csv.name}")
    print(f"report written to {out.resolve()}")


if __name__ == "__main__":
    main()
