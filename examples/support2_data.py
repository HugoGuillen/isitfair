"""SUPPORT2 cohort, model matrix, splits and reference model.

Every runnable artifact in this repository -- the example script and all three
notebooks -- builds its cohort through this module, so they all audit the
*same* held-out test partition under the same seed. Each function
covers one stage, so the example script can call them in sequence with the
pipeline visible step by step, while the notebooks call
:func:`fit_reference_model`, which chains them in one line.

The data is the SUPPORT2 cohort (Study to Understand Prognoses and Preferences
for Outcomes and Risks of Treatment): 9,105 critically ill adults from five US
medical centres, distributed by the UCI Machine Learning Repository as dataset
880 under CC BY 4.0.

    Knaus WA, Harrell FE, Lynn J, et al. The SUPPORT prognostic model:
    objective estimates of survival for seriously ill hospitalized adults.
    Ann Intern Med 1995;122:191-203.

Notes
-----
Race and sex serve as audit axes here, so the model matrix is built without
them.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = [
    "AGE_BINS",
    "MODEL_FEATURES",
    "SEED",
    "THRESHOLD",
    "Cohort",
    "build_model_matrix",
    "fit_model",
    "fit_reference_model",
    "load_support2",
    "make_sensitive_attributes",
    "n_bootstrap",
    "repo_root",
    "results_dir",
    "split_cohort",
]

# --- Prespecified analysis constants -----------------------------------------
# Fixed before any result was looked at, per the audit-design checklist.

SEED = 99
THRESHOLD = 0.30  # operating threshold for the rate metrics and net benefit
AGE_BINS = ("18-44", "45-64", "65-79", "80+")

#: Features available at ICU admission with low missingness. Race and sex are
#: held out as audit axes, so they do not appear here.
MODEL_FEATURES = [
    "age",
    "num.co",  # number of comorbidities
    "diabetes",
    "dementia",
    "meanbp",  # mean blood pressure
    "hrt",  # heart rate
    "resp",  # respiratory rate
    "temp",  # temperature
    "crea",  # creatinine
    "sod",  # sodium
    "scoma",  # Glasgow coma score component
    "sps",  # SUPPORT physiology score
    "aps",  # APACHE III physiology score
]


# --- Repository layout --------------------------------------------------------


def repo_root() -> Path:
    """Locate the repository root from anywhere inside it.

    Walks upward looking for ``pyproject.toml`` so the same code runs whether
    it was launched from the repository root, from ``examples/``, or from
    ``notebooks/``.

    Returns
    -------
    Path
        The directory containing ``pyproject.toml``.

    Raises
    ------
    RuntimeError
        If no ancestor directory contains ``pyproject.toml``.

    Examples
    --------
    >>> repo_root().name  # doctest: +SKIP
    'isitfair'
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate the repository root (no pyproject.toml above me)")


def results_dir(*parts: str) -> Path:
    """Return ``<repo root>/results/<parts...>``, creating it if needed.

    Parameters
    ----------
    *parts : str
        Path components below ``results/``.

    Returns
    -------
    Path
        The (existing) directory.

    Examples
    --------
    >>> results_dir("tables").name  # doctest: +SKIP
    'tables'
    """
    path = repo_root() / "results"
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def n_bootstrap(default: int = 2000, fast: int = 200) -> int:
    """Resolve the bootstrap replicate count from ``ISITFAIR_FAST``.

    The committed artifacts are built at ``default``. Setting
    ``ISITFAIR_FAST=1`` drops to ``fast`` for a quick pass while you are
    editing; quote intervals from a full-replicate run.

    Parameters
    ----------
    default : int, default 2000
        Replicates when ``ISITFAIR_FAST`` is unset.
    fast : int, default 200
        Replicates when ``ISITFAIR_FAST=1``.

    Returns
    -------
    int
        The replicate count to pass as ``n_bootstrap``.

    Examples
    --------
    >>> n_bootstrap() in (200, 2000)
    True
    """
    import os

    return fast if os.environ.get("ISITFAIR_FAST") == "1" else default


# --- Data ---------------------------------------------------------------------


def load_support2(*, prefer_cache: bool = False) -> pd.DataFrame:
    """Load the SUPPORT2 cohort as a single DataFrame.

    Tries the UCI repository first (``ucimlrepo.fetch_ucirepo(id=880)``), which
    is what the manuscript describes, and falls back to the copy bundled at
    ``examples/data/support2.csv.gz`` when the network or ``ucimlrepo`` is
    unavailable. The bundled copy is byte-for-byte the concatenation of the UCI
    feature and target frames and is redistributed under CC BY 4.0.

    Parameters
    ----------
    prefer_cache : bool, default False
        Skip the network entirely and read the bundled copy. Useful for
        deterministic reruns.

    Returns
    -------
    pandas.DataFrame
        9,105 rows x 45 columns (42 features + ``death``, ``hospdead``,
        ``sfdm2``).

    Examples
    --------
    >>> df = load_support2(prefer_cache=True)
    >>> df.shape
    (9105, 45)
    """
    cache = repo_root() / "examples" / "data" / "support2.csv.gz"

    if not prefer_cache:
        try:
            from ucimlrepo import fetch_ucirepo

            fetched = fetch_ucirepo(id=880)
            return pd.concat([fetched.data.features, fetched.data.targets], axis=1)
        except Exception as exc:  # network down, UCI moved, ucimlrepo absent
            if not cache.exists():
                raise RuntimeError(
                    f"UCI fetch failed ({exc}) and no bundled copy at {cache}"
                ) from exc
            print(f"UCI fetch unavailable ({type(exc).__name__}); using bundled copy.")

    with gzip.open(cache, "rt") as handle:
        return pd.read_csv(handle, low_memory=False)


def make_sensitive_attributes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the three audit axes from the raw cohort.

    Parameters
    ----------
    df : pandas.DataFrame
        The frame returned by :func:`load_support2`.

    Returns
    -------
    dict of str to numpy.ndarray
        ``sex``, ``age_group`` (binned at 45/65/80) and ``race_ethnicity``
        (missing filled with ``"other"``; 0.5% of rows).

    Examples
    --------
    >>> attrs = make_sensitive_attributes(load_support2(prefer_cache=True))
    >>> sorted(attrs)
    ['age_group', 'race_ethnicity', 'sex']
    """
    age = df["age"].to_numpy()
    age_group = np.where(
        age < 45,
        AGE_BINS[0],
        np.where(age < 65, AGE_BINS[1], np.where(age < 80, AGE_BINS[2], AGE_BINS[3])),
    )
    return {
        "sex": df["sex"].to_numpy(),
        "age_group": age_group,
        "race_ethnicity": df["race"].fillna("other").to_numpy(),
    }


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model design matrix.

    Takes :data:`MODEL_FEATURES`, adds dummy-coded disease class (``dzclass``)
    and cancer status (``ca``), and median-imputes what remains missing
    (creatinine is the worst offender at 0.7%).

    Parameters
    ----------
    df : pandas.DataFrame
        The frame returned by :func:`load_support2`.

    Returns
    -------
    pandas.DataFrame
        Numeric design matrix, complete, built from the clinical features only.

    Examples
    --------
    >>> X = build_model_matrix(load_support2(prefer_cache=True))
    >>> bool(X.isnull().any().any())
    False
    """
    X = df[MODEL_FEATURES].copy()
    dz = pd.get_dummies(df["dzclass"], prefix="dz", drop_first=True)
    ca = pd.get_dummies(df["ca"], prefix="ca", drop_first=True)
    X = pd.concat([X, dz, ca], axis=1).astype(float)
    return X.fillna(X.median())


@dataclass(frozen=True)
class Cohort:
    """Everything downstream of the split, in one place.

    Attributes
    ----------
    y_test, y_calib : numpy.ndarray
        Binary in-hospital mortality (``hospdead``) for each partition.
    p_test, p_calib : numpy.ndarray
        Predicted probabilities from the reference model.
    attrs_test, attrs_calib : dict of str to numpy.ndarray
        The three audit axes, aligned to the partition.
    frame_test : pandas.DataFrame
        The raw cohort rows of the test partition, with ``y`` and ``y_proba``
        columns appended. This is what the Table 1, effect-size and
        interaction analyses read.
    model, scaler : sklearn estimators
        The fitted reference model and its feature scaler.
    n_train : int
        Size of the training partition.
    """

    y_test: np.ndarray
    p_test: np.ndarray
    attrs_test: dict[str, np.ndarray]
    y_calib: np.ndarray
    p_calib: np.ndarray
    attrs_calib: dict[str, np.ndarray]
    frame_test: pd.DataFrame
    model: Any
    scaler: Any
    n_train: int


def split_cohort(
    X: pd.DataFrame, y: np.ndarray, *, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 train / calibration / test split.

    The calibration partition exists so that recalibration is fitted on data
    the audit never sees -- the leak-free arrangement any mitigation comparison
    requires.

    Parameters
    ----------
    X : pandas.DataFrame
        Design matrix (used only for its length).
    y : numpy.ndarray
        Binary outcome, used to stratify.
    seed : int, default :data:`SEED`
        Split seed.

    Returns
    -------
    tuple of numpy.ndarray
        Positional indices for train, calibration and test.

    Examples
    --------
    >>> df = load_support2(prefer_cache=True)
    >>> tr, ca, te = split_cohort(build_model_matrix(df), df["hospdead"].to_numpy())
    >>> len(tr), len(ca), len(te)
    (5463, 1821, 1821)
    """
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
    idx_calib, idx_test = train_test_split(
        idx_rest, test_size=0.50, random_state=seed, stratify=y[idx_rest]
    )
    return idx_train, idx_calib, idx_test


def fit_model(
    X: pd.DataFrame, y: np.ndarray, idx_train: np.ndarray, *, seed: int = SEED
) -> tuple[Any, Any]:
    """Fit the reference logistic regression on the training partition.

    A transparent baseline whose predictions are the audit's input. Any
    classifier that produces probabilities -- gradient boosting, a neural
    network behind a sklearn-style wrapper -- is audited in exactly the same
    way, so the workflow here transfers unchanged to your own model.

    Parameters
    ----------
    X : pandas.DataFrame
        Full design matrix.
    y : numpy.ndarray
        Full outcome vector.
    idx_train : numpy.ndarray
        Training-partition indices.
    seed : int, default :data:`SEED`
        Passed to ``LogisticRegression``.

    Returns
    -------
    tuple
        ``(model, scaler)``, both fitted on the training partition only.

    Examples
    --------
    >>> df = load_support2(prefer_cache=True)
    >>> X, y = build_model_matrix(df), df["hospdead"].to_numpy()
    >>> model, scaler = fit_model(X, y, split_cohort(X, y)[0])
    >>> model.coef_.shape[0]
    1
    """
    scaler = StandardScaler().fit(X.iloc[idx_train])
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    model.fit(scaler.transform(X.iloc[idx_train]), y[idx_train])
    return model, scaler


def fit_reference_model(*, seed: int = SEED, prefer_cache: bool = False) -> Cohort:
    """Chain the whole pipeline and return the audited :class:`Cohort`.

    Convenience wrapper over :func:`load_support2`,
    :func:`make_sensitive_attributes`, :func:`build_model_matrix`,
    :func:`split_cohort` and :func:`fit_model`. The example script calls those
    one at a time so each stage is visible; the notebooks call this.

    Parameters
    ----------
    seed : int, default :data:`SEED`
        Seed for the split and the classifier.
    prefer_cache : bool, default False
        Forwarded to :func:`load_support2`.

    Returns
    -------
    Cohort
        Test and calibration partitions with predictions and audit axes.

    Examples
    --------
    >>> cohort = fit_reference_model(prefer_cache=True)
    >>> len(cohort.y_test), int(cohort.y_test.sum())
    (1821, 472)
    """
    df = load_support2(prefer_cache=prefer_cache)
    y = df["hospdead"].to_numpy().astype(float)
    attrs = make_sensitive_attributes(df)
    X = build_model_matrix(df)

    idx_train, idx_calib, idx_test = split_cohort(X, y, seed=seed)
    model, scaler = fit_model(X, y, idx_train, seed=seed)

    def predict(idx: np.ndarray) -> np.ndarray:
        return model.predict_proba(scaler.transform(X.iloc[idx]))[:, 1]

    p_test = predict(idx_test)
    frame_test = df.iloc[idx_test].copy().reset_index(drop=True)
    frame_test["y"] = y[idx_test]
    frame_test["y_proba"] = p_test
    for name, values in attrs.items():
        frame_test[name] = values[idx_test]

    return Cohort(
        y_test=y[idx_test],
        p_test=p_test,
        attrs_test={k: v[idx_test] for k, v in attrs.items()},
        y_calib=y[idx_calib],
        p_calib=predict(idx_calib),
        attrs_calib={k: v[idx_calib] for k, v in attrs.items()},
        frame_test=frame_test,
        model=model,
        scaler=scaler,
        n_train=len(idx_train),
    )
