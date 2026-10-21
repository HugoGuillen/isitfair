"""Download six public datasets and build the prediction tables the notebooks audit.

Usage, from the repository root:

    python -I extra_examples/datasets.py [NAME ...]      # default: all six

For every dataset this

1. downloads the file from its original host (UCI through ``ucimlrepo``;
   COMPAS from ProPublica's GitHub) into ``extra_examples/data/raw/<name>/``,
   once, and parses that stored file;
2. fits a logistic regression (one-hot encoding plus
   ``LogisticRegression(max_iter=2000)``) on a stratified 70/30 split
   (``random_state=42``);
3. writes ``extra_examples/data/predictions/<name>.csv`` for the 30% test half,
   with columns ``y_true, y_proba, <attributes...>``, and a ``<name>.json``
   with the threshold and outcome definition.

As in ``examples/support2_data.py``, sex and race are audit axes and are held
out of the features; age stays in as a predictor and is binned only for the
audit. A missing attribute label is kept missing (empty in the CSV), so the
notebooks show how isitfair handles it.

The data are not redistributed with isitfair; check each source's terms
before reusing them.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
PRED = HERE / "data" / "predictions"
SEED = 42
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)
UCI_API = "https://archive.ics.uci.edu/api/dataset?id={}"


# --------------------------------------------------------------------------- fetch


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "isitfair-extra-examples"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_uci(name: str, uci_id: int) -> pd.DataFrame:
    """Download a UCI dataset once, keep the file, return the parsed frame."""
    from ucimlrepo import fetch_ucirepo

    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / "data.csv"
    if not data_file.exists():
        fetched = fetch_ucirepo(id=uci_id)
        url = fetched.metadata["data_url"]
        data_file.write_bytes(_get(url))
        (out / "DOWNLOAD.json").write_text(
            json.dumps({"data_url": url, "api_url": UCI_API.format(uci_id), "uci_id": uci_id,
                        "downloaded": dt.datetime.now().astimezone().isoformat(timespec="seconds")},
                       indent=2)
        )
    return pd.read_csv(data_file, low_memory=False)


def fetch_compas() -> pd.DataFrame:
    out = RAW / "compas"
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / "compas-scores-two-years.csv"
    if not data_file.exists():
        data_file.write_bytes(_get(COMPAS_URL))
        (out / "DOWNLOAD.json").write_text(
            json.dumps({"data_url": COMPAS_URL,
                        "downloaded": dt.datetime.now().astimezone().isoformat(timespec="seconds")},
                       indent=2)
        )
    return pd.read_csv(data_file, low_memory=False)


# --------------------------------------------------------------------------- recode


def age_bins(age: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(age, bins=edges, labels=labels, right=False).astype(object)


def prep_diabetes130() -> dict:
    df = fetch_uci("diabetes130", 296)
    y = (df["readmitted"] == "<30").astype(int)
    attrs = pd.DataFrame(
        {
            # '?' is UCI's missing marker; kept as written so the audit sees it.
            "race": df["race"].astype(object),
            "gender": df["gender"].astype(object),
            "age": df["age"].astype(object),  # already 10-year bins, '[0-10)' ...
        }
    )
    drop = ["readmitted", "race", "gender", "encounter_id", "patient_nbr",
            "weight", "payer_code", "medical_specialty"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return dict(X=X, y=y, attrs=attrs, threshold=0.15,
                outcome="readmitted == '<30' (readmission within 30 days)")


def prep_adult() -> dict:
    df = fetch_uci("adult", 2)
    inc = df["income"].astype(str).str.strip().str.rstrip(".")
    y = (inc == ">50K").astype(int)
    attrs = pd.DataFrame(
        {
            "sex": df["sex"].astype(object),
            "race": df["race"].astype(object),
            "age_group": age_bins(df["age"], [0, 30, 45, 60, 200], ["17-29", "30-44", "45-59", "60+"]),
        }
    )
    X = df.drop(columns=["income", "sex", "race"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.30,
                outcome="income > 50K (the test file's labels carry a trailing period)")


def prep_compas() -> dict:
    df = fetch_compas()
    # ProPublica's own analysis filter (compas-analysis/Compas Analysis.ipynb).
    keep = (
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    )
    df = df.loc[keep].reset_index(drop=True)
    y = df["two_year_recid"].astype(int)
    attrs = pd.DataFrame(
        {"race": df["race"].astype(object), "sex": df["sex"].astype(object),
         "age_cat": df["age_cat"].astype(object)}
    )
    X = df[["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count",
            "c_charge_degree"]].copy()
    return dict(X=X, y=y, attrs=attrs, threshold=0.50,
                outcome="two_year_recid == 1 (the COMPAS score itself is not a feature)")


def prep_bank() -> dict:
    df = fetch_uci("bank", 222)
    y = (df["y"] == "yes").astype(int)
    attrs = pd.DataFrame(
        {
            "age_group": age_bins(df["age"], [0, 30, 40, 50, 60, 200], ["18-29", "30-39", "40-49", "50-59", "60+"]),
            # 'unknown' arrives as NaN through UCI's CSV; recoded explicitly.
            "job": df["job"].fillna("unknown").astype(object),
            "marital": df["marital"].astype(object),
        }
    )
    # duration is only known after the call, a documented leak.
    X = df.drop(columns=["y", "duration"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.20,
                outcome="y == 'yes' (term deposit subscribed)")


def prep_german() -> dict:
    df = fetch_uci("german", 144)
    y = (df["class"] == 2).astype(int)  # 1 good, 2 bad
    status = df["Attribute9"]
    sex = status.map({"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"})
    attrs = pd.DataFrame(
        {"sex": sex.astype(object),
         "age_group": age_bins(df["Attribute13"], [0, 25, 35, 50, 200], ["19-24", "25-34", "35-49", "50+"])}
    )
    X = df.drop(columns=["class", "Attribute9"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.30,
                outcome="class == 2 (bad credit risk)")


def prep_heart_cleveland() -> dict:
    df = fetch_uci("heart_cleveland", 45)
    y = (df["num"] > 0).astype(int)
    attrs = pd.DataFrame(
        {"sex": df["sex"].map({1: "male", 0: "female"}).astype(object),
         "age_group": age_bins(df["age"], [0, 45, 55, 65, 200], ["<45", "45-54", "55-64", "65+"])}
    )
    X = df.drop(columns=["num", "sex"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.50,
                outcome="num > 0 (any angiographic disease)")


DATASETS = {
    "diabetes130": prep_diabetes130,
    "adult": prep_adult,
    "compas": prep_compas,
    "bank": prep_bank,
    "german": prep_german,
    "heart_cleveland": prep_heart_cleveland,
}
ATTRS = {
    "diabetes130": ["race", "gender", "age"],
    "adult": ["sex", "race", "age_group"],
    "compas": ["race", "sex", "age_cat"],
    "bank": ["age_group", "job", "marital"],
    "german": ["sex", "age_group"],
    "heart_cleveland": ["sex", "age_group"],
}


# --------------------------------------------------------------------------- model


def _model(X: pd.DataFrame) -> Pipeline:
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return Pipeline(
        [
            ("prep", ColumnTransformer(
                [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                 ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                     min_frequency=50, sparse_output=False)), cat)])),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def build(name: str) -> None:
    spec = DATASETS[name]()
    X, y, attrs = spec["X"], spec["y"].to_numpy(), spec["attrs"]
    for c in X.columns:  # object columns with stray whitespace / '?' as missing
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    tr, te = train_test_split(np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y)
    model = _model(X).fit(X.iloc[tr], y[tr])
    out = pd.DataFrame({"y_true": y[te], "y_proba": model.predict_proba(X.iloc[te])[:, 1]})
    for c in attrs.columns:
        out[c] = attrs[c].to_numpy()[te]
    PRED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PRED / f"{name}.csv", index=False)
    info = {"name": name, "threshold": spec["threshold"], "outcome": spec["outcome"],
            "attributes": ATTRS[name], "n_test": len(te), "prevalence_test": round(float(y[te].mean()), 4)}
    (PRED / f"{name}.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: n_test={len(te)} prevalence={info['prevalence_test']} threshold={spec['threshold']}",
          flush=True)


def load(name: str) -> pd.DataFrame:
    """Read a prediction table; attribute columns stay as object with NaN for missing."""
    return pd.read_csv(PRED / f"{name}.csv", dtype={a: object for a in ATTRS[name]})


def meta(name: str) -> dict:
    return json.loads((PRED / f"{name}.json").read_text())


if __name__ == "__main__":
    for nm in sys.argv[1:] or DATASETS:
        build(nm)
