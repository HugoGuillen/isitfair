"""Dataset registry for the isitfair hardening study.

Usage (from the repository root, isolated so nothing in raw/ can shadow a module):

    python -I hardening/datasets/registry.py [NAME ...]      # default: all

For every dataset this

1. fetches from the original host (UCI through ``ucimlrepo``; COMPAS from
   ProPublica's GitHub raw CSV) and keeps the download untouched under
   ``raw/<name>/`` (the original file from the host's ``data_url`` plus the UCI
   API metadata), then parses *that* file;
2. fits the fixed audited models on a stratified 70/30 split
   (``random_state=42``): a ``Pipeline`` of one-hot encoding plus
   ``LogisticRegression(max_iter=2000)``, and separately
   ``HistGradientBoostingClassifier``;
3. writes ``predictions/<name>.csv`` for the 30% test half only, with columns
   ``y_true, y_proba_lr, y_proba_gbm, <attributes...>``, and
   ``predictions/<name>.json`` with the facts CARDS.md is written from.

Mirroring ``examples/support2_data.py``: sex and race are audit axes and are
held out of the feature matrix; age stays in as a clinical/actuarial predictor
and is binned only for the audit. Attribute labels are written exactly as
recoded below; a missing label is kept missing (empty in the CSV) so the
notebooks exercise isitfair's own input validation.

``support2`` is not refit here: its predictions are the committed reference
model's (``examples/support2_data.py``, seed 99, 60/20/20 split), so they match
the MANIFEST. The GBM column for support2 is fitted on the same training rows.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PRED = HERE / "predictions"
REPO = HERE.parents[1]
SEED = 42
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)
UCI_API = "https://archive.ics.uci.edu/api/dataset?id={}"


# --------------------------------------------------------------------------- fetch


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "isitfair-hardening"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_uci(name: str, uci_id: int) -> pd.DataFrame:
    """Download a UCI dataset once, keep the bytes, return the parsed frame.

    ``ucimlrepo.fetch_ucirepo`` resolves the dataset's ``data_url`` (and splits
    features/targets); the file at that URL is what is stored untouched, and
    the frame is parsed from the stored file with ucimlrepo's own reader
    settings so the stored bytes are what the study used.
    """
    from ucimlrepo import fetch_ucirepo

    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / "data.csv"
    meta_file = out / "uci_api.json"
    if not data_file.exists():
        fetched = fetch_ucirepo(id=uci_id)
        url = fetched.metadata["data_url"]
        data_file.write_bytes(_get(url))
        meta_file.write_bytes(_get(UCI_API.format(uci_id)))
        fetched.variables.to_csv(out / "ucimlrepo_variables.csv", index=False)
        (out / "DOWNLOAD.json").write_text(
            json.dumps({"data_url": url, "api_url": UCI_API.format(uci_id),
                        "via": "ucimlrepo.fetch_ucirepo", "uci_id": uci_id,
                        "downloaded": dt.datetime.now().astimezone().isoformat(timespec="seconds")},
                       indent=2)
        )
    # ucimlrepo reads data_url with pandas defaults; do the same on the stored file.
    return pd.read_csv(data_file, low_memory=False)


def fetch_compas() -> pd.DataFrame:
    out = RAW / "compas"
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / "compas-scores-two-years.csv"
    if not data_file.exists():
        data_file.write_bytes(_get(COMPAS_URL))
        (out / "DOWNLOAD.json").write_text(
            json.dumps({"data_url": COMPAS_URL, "via": "urllib",
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
                outcome="readmitted == '<30' (readmission within 30 days)",
                features_note="all columns except race, gender, ids, and >39%-missing weight, "
                              "payer_code, medical_specialty; age kept as a categorical feature; "
                              "diag_1..3 one-hot with min_frequency=50")


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
                outcome="income in {'>50K', '>50K.'} (the test file's labels carry a trailing period)",
                features_note="all columns except income, sex, race; age kept numeric")


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
                outcome="two_year_recid == 1",
                features_note="age, priors_count, juv_fel/misd/other_count, c_charge_degree "
                              "(the COMPAS decile score itself is not a feature); rows filtered "
                              "with ProPublica's days_b_screening_arrest/is_recid/charge/score filter")


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
    X = df.drop(columns=["y", "duration"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.20,
                outcome="y == 'yes' (term deposit subscribed)",
                features_note="all columns except y and duration (duration is only known after the "
                              "call, a documented leak); job and marital stay in as features")


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
                outcome="class == 2 (bad credit risk)",
                features_note="all columns except class and Attribute9 (personal status and sex)")


def prep_heart_cleveland() -> dict:
    df = fetch_uci("heart_cleveland", 45)
    y = (df["num"] > 0).astype(int)
    attrs = pd.DataFrame(
        {"sex": df["sex"].map({1: "male", 0: "female"}).astype(object),
         "age_group": age_bins(df["age"], [0, 45, 55, 65, 200], ["<45", "45-54", "55-64", "65+"])}
    )
    X = df.drop(columns=["num", "sex"])
    return dict(X=X, y=y, attrs=attrs, threshold=0.50,
                outcome="num > 0 (any angiographic disease)",
                features_note="all columns except num and sex; ca/thal missing values median/mode-imputed")


# --------------------------------------------------------------------------- models


def _models(X: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    lr = Pipeline(
        [
            ("prep", ColumnTransformer(
                [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                 ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                     min_frequency=50, sparse_output=False)), cat)])),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    gbm = Pipeline(
        [
            ("prep", ColumnTransformer(
                [("num", "passthrough", num),
                 ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                     min_frequency=50, sparse_output=False)), cat)])),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )
    return lr, gbm


def build_real(name: str, spec: dict) -> dict:
    X, y, attrs = spec["X"], spec["y"].to_numpy(), spec["attrs"]
    for c in X.columns:  # object columns with stray whitespace / '?' as missing
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    lr, gbm = _models(X)
    lr.fit(X.iloc[tr], y[tr])
    gbm.fit(X.iloc[tr], y[tr])
    out = pd.DataFrame(
        {"y_true": y[te], "y_proba_lr": lr.predict_proba(X.iloc[te])[:, 1],
         "y_proba_gbm": gbm.predict_proba(X.iloc[te])[:, 1]}
    )
    for c in attrs.columns:
        out[c] = attrs[c].to_numpy()[te]
    return dict(frame=out, n_total=len(y), n_train=len(tr), n_test=len(te),
                threshold=spec["threshold"], outcome=spec["outcome"],
                features_note=spec["features_note"], n_features=X.shape[1])


def build_support2() -> dict:
    sys.path.insert(0, str(REPO / "examples"))
    from support2_data import (build_model_matrix, fit_reference_model, load_support2,
                               split_cohort, SEED as S2_SEED, THRESHOLD)

    cohort = fit_reference_model(prefer_cache=True)
    df = load_support2(prefer_cache=True)
    X = build_model_matrix(df)
    y = df["hospdead"].to_numpy().astype(float)
    tr, ca, te = split_cohort(X, y, seed=S2_SEED)
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(X.iloc[tr], y[tr])
    out = pd.DataFrame({"y_true": cohort.y_test, "y_proba_lr": cohort.p_test,
                        "y_proba_gbm": gbm.predict_proba(X.iloc[te])[:, 1]})
    for k, v in cohort.attrs_test.items():
        out[k] = v
    calib = pd.DataFrame({"y_true": cohort.y_calib, "y_proba_lr": cohort.p_calib,
                          "y_proba_gbm": gbm.predict_proba(X.iloc[ca])[:, 1]})
    for k, v in cohort.attrs_calib.items():
        calib[k] = v
    calib.to_csv(PRED / "support2_calib.csv", index=False)
    return dict(frame=out, n_total=len(y), n_train=len(tr), n_test=len(te), threshold=THRESHOLD,
                outcome="hospdead == 1 (in-hospital death)",
                features_note="examples/support2_data.py reference model (LR, seed 99, 60/20/20); "
                              "the 20% calibration partition is written to support2_calib.csv",
                n_features=X.shape[1])


# --------------------------------------------------------------------------- oracles


def make_oracle_calib(n: int = 20_000, seed: int = SEED) -> pd.DataFrame:
    """Two groups; y ~ Bernoulli(expit(a_g + b_g * logit(p))).

    Planted (a, b) = (0, 1) in A and (-0.8, 0.7) in B. ``p`` is the audited
    prediction (both prediction columns hold it).
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(np.array(["A", "B"]), size=n)
    p = expit(rng.normal(-1.0, 1.2, n))
    a = np.where(group == "A", 0.0, -0.8)
    b = np.where(group == "A", 1.0, 0.7)
    y = rng.binomial(1, expit(a + b * logit(p)))
    return pd.DataFrame({"y_true": y, "y_proba_lr": p, "y_proba_gbm": p, "group": group})


def make_oracle_sparse(n: int = 5_000, seed: int = SEED) -> pd.DataFrame:
    """Six levels; level L5 has exactly 3 events and L6 has 0."""
    rng = np.random.default_rng(seed)
    sizes = {"L1": 1800, "L2": 1200, "L3": 900, "L4": 600, "L5": 300, "L6": 200}
    level = np.repeat(np.array(list(sizes)), list(sizes.values()))
    p = expit(rng.normal(-2.2, 1.0, n))
    y = rng.binomial(1, p)
    for lab, k in (("L5", 3), ("L6", 0)):
        m = np.flatnonzero(level == lab)
        y[m] = 0
        y[rng.choice(m, size=k, replace=False, p=p[m] / p[m].sum())] = 1 if k else 0
    order = rng.permutation(n)
    return pd.DataFrame({"y_true": y[order], "y_proba_lr": p[order], "y_proba_gbm": p[order],
                         "level": level[order]})


# --------------------------------------------------------------------------- main

REAL = {
    "diabetes130": prep_diabetes130,
    "adult": prep_adult,
    "compas": prep_compas,
    "bank": prep_bank,
    "german": prep_german,
    "heart_cleveland": prep_heart_cleveland,
}
ATTRS = {
    "support2": ["sex", "age_group", "race_ethnicity"],
    "diabetes130": ["race", "gender", "age"],
    "adult": ["sex", "race", "age_group"],
    "compas": ["race", "sex", "age_cat"],
    "bank": ["age_group", "job", "marital"],
    "german": ["sex", "age_group"],
    "heart_cleveland": ["sex", "age_group"],
    "oracle_calib": ["group"],
    "oracle_sparse": ["level"],
}
ALL = ["support2", *REAL, "oracle_calib", "oracle_sparse"]


def summarize(name: str, info: dict) -> dict:
    f = info["frame"]
    per = {}
    for a in ATTRS[name]:
        g = f.groupby(f[a].fillna("<missing>").astype(str))["y_true"]
        per[a] = {str(k): {"n": int(v.size), "events": int(v.sum()), "prevalence": round(float(v.mean()), 4)}
                  for k, v in g}
    return {k: v for k, v in info.items() if k != "frame"} | {
        "name": name, "attributes": ATTRS[name], "prevalence_test": round(float(f["y_true"].mean()), 4),
        "events_test": int(f["y_true"].sum()), "per_subgroup": per,
        "built": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def build(name: str) -> dict:
    PRED.mkdir(parents=True, exist_ok=True)
    if name == "support2":
        info = build_support2()
    elif name in REAL:
        info = build_real(name, REAL[name]())
    elif name == "oracle_calib":
        f = make_oracle_calib()
        info = dict(frame=f, n_total=len(f), n_train=0, n_test=len(f), threshold=0.30,
                    outcome="synthetic, y ~ Bernoulli(expit(a_g + b_g logit p))",
                    features_note="no model; p is the prediction", n_features=0)
    elif name == "oracle_sparse":
        f = make_oracle_sparse()
        info = dict(frame=f, n_total=len(f), n_train=0, n_test=len(f), threshold=0.15,
                    outcome="synthetic, y ~ Bernoulli(p) with L5 forced to 3 events and L6 to 0",
                    features_note="no model; p is the prediction", n_features=0)
    else:
        raise KeyError(name)
    info["frame"].to_csv(PRED / f"{name}.csv", index=False)
    summary = summarize(name, info)
    (PRED / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: n_test={summary['n_test']} prevalence={summary['prevalence_test']} "
          f"threshold={summary['threshold']}", flush=True)
    return summary


def load(name: str) -> pd.DataFrame:
    """Read a prediction table; attribute columns stay as object with NaN for missing."""
    return pd.read_csv(PRED / f"{name}.csv", keep_default_na=True, dtype={a: object for a in ATTRS[name]})


def meta(name: str) -> dict:
    return json.loads((PRED / f"{name}.json").read_text())


if __name__ == "__main__":
    for nm in sys.argv[1:] or ALL:
        build(nm)
