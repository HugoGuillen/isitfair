"""Phase 5: compare isitfair point estimates with independent implementations.

Usage: python hardening/phase5/oracles.py OUT_JSON [DATASET ...]

For every dataset and both model columns (y_proba_lr, y_proba_gbm), at
random_state=42, every (attribute, subgroup) row of discrimination(),
calibration(), decision_curve() and hosmer_lemeshow() is recomputed with:

- sklearn.metrics (AUROC, AUPRC, confusion-matrix rates at p >= threshold, Brier)
- fairlearn MetricFrame difference(method="between_groups") (gaps)
- statsmodels GLM(Binomial) (calibration intercept with offset, slope)
- from-scratch ECE (10 equal-frequency bins) and Hosmer-Lemeshow (chi2.sf, df=10)
- the net-benefit formula directly, and dcurves.dca directly

and the maximum absolute difference is recorded per metric, along with rows
where exactly one side is NaN. Interaction cells that isitfair shrinks toward
their marginal are excluded from the rate comparisons (by design they are not
the raw cell rate) and counted separately.
"""

from __future__ import annotations

import faulthandler
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dcurves import dca
from fairlearn.metrics import (MetricFrame, false_positive_rate, selection_rate,
                               true_positive_rate)
from scipy.special import logit
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hardening" / "datasets"))
import registry  # noqa: E402

from isitfair import FairnessAudit  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def rates(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y.astype(int), (p >= t).astype(int), labels=[0, 1]).ravel()
    return {"sensitivity": safe_div(tp, tp + fn), "specificity": safe_div(tn, tn + fp),
            "ppv": safe_div(tp, tp + fp), "npv": safe_div(tn, tn + fn)}


def glm_intercept(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() in (0, len(y)) or len(y) < 3:
        return float("nan")
    m = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit(p))
    return float(m.fit(tol=1e-12, maxiter=200).params[0])


def glm_slope(y: np.ndarray, p: np.ndarray) -> float:
    lp = logit(p)
    if y.sum() in (0, len(y)) or len(y) < 3 or np.ptp(lp) < 1e-8:
        return float("nan")
    m = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial())
    return float(m.fit(tol=1e-12, maxiter=200).params[1])


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    """From scratch: sort by p (stable), cut into 10 near-equal-count bins."""
    order = np.argsort(p, kind="stable")
    total = 0.0
    for b in np.array_split(order, min(10, len(p))):
        if len(b):
            total += len(b) * abs(p[b].mean() - y[b].mean())
    return total / len(p)


def hl10(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """From scratch: 10 equal-count groups of stable-sorted p; df = 10."""
    order = np.argsort(p, kind="stable")
    stat = 0.0
    for b in np.array_split(order, 10):
        o, e, n = y[b].sum(), p[b].sum(), len(b)
        stat += (o - e) ** 2 / (e * (1 - e / n))
    return float(stat), float(chi2.sf(stat, 10))


def nb_formula(y: np.ndarray, p: np.ndarray, t: float) -> float:
    n = len(y)
    flag = p >= t
    tp, fp = np.sum(flag & (y == 1)), np.sum(flag & (y == 0))
    return tp / n - fp / n * t / (1 - t)


class Diff:
    """Accumulates max |a - b| and one-sided NaNs per metric."""

    def __init__(self) -> None:
        self.d: dict[str, dict] = {}

    def add(self, metric: str, ours: float, ref: float, where: str) -> None:
        e = self.d.setdefault(metric, {"max_abs_diff": 0.0, "n": 0, "nan_mismatch": 0,
                                       "worst": None, "nan_examples": []})
        a, b = float(ours), float(ref)
        if np.isnan(a) and np.isnan(b):
            return
        if np.isnan(a) != np.isnan(b):
            e["nan_mismatch"] += 1
            if len(e["nan_examples"]) < 4:
                e["nan_examples"].append(f"{where}: isitfair={a:.6g} oracle={b:.6g}")
            return
        e["n"] += 1
        diff = abs(a - b)
        if diff > e["max_abs_diff"]:
            e["max_abs_diff"] = diff
            e["worst"] = f"{where}: isitfair={a!r} oracle={b!r}"


def check(name: str, column: str) -> dict:
    df = registry.load(name)
    attrs = registry.ATTRS[name]
    for a in attrs:
        df[a] = df[a].fillna("Unknown")
    y = df["y_true"].to_numpy(float)
    p = df[column].to_numpy(float)
    t = registry.meta(name)["threshold"]
    marg = {a: df[a].to_numpy(object) for a in attrs}
    sf = dict(marg)
    inter = tuple(attrs[:2]) if len(attrs) >= 2 else None
    if inter:
        sf[inter] = None
    audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features=sf, threshold=t,
                          n_bootstrap=20, random_state=SEED)
    t0 = time.perf_counter()
    disc, cal = audit.discrimination(), audit.calibration()
    dc, hl = audit.decision_curve(), audit.hosmer_lemeshow()
    secs = time.perf_counter() - t0
    labels = {**marg}
    if inter:
        inter_name = [k for k in disc.attribute.unique() if k not in attrs][0]
        labels[inter_name] = audit._sensitive[inter_name]

    D = Diff()
    shrunk_skipped = 0
    for attr, lab in labels.items():
        groups = ["Overall", *sorted(set(map(str, lab)))]
        for g in groups:
            m = np.ones(len(y), bool) if g == "Overall" else (lab.astype(str) == g)
            yt, pt = y[m], p[m]
            where = f"{attr}={g}"
            both = 0 < yt.sum() < len(yt)
            drow = disc[(disc.attribute == attr) & (disc.subgroup == g)].iloc[0]
            D.add("auroc", drow.auroc, roc_auc_score(yt, pt) if both else np.nan, where)
            D.add("auprc", drow.auprc, average_precision_score(yt, pt) if both else np.nan, where)
            if bool(drow.shrunk):
                shrunk_skipped += 1
            else:
                for k, v in rates(yt, pt, t).items():
                    D.add(k, drow[k], v, where)
            crow = cal[(cal.attribute == attr) & (cal.subgroup == g)].iloc[0]
            D.add("calibration_intercept", crow.calibration_intercept, glm_intercept(yt, pt), where)
            D.add("calibration_slope", crow.calibration_slope, glm_slope(yt, pt), where)
            D.add("brier", crow.brier, brier_score_loss(yt, pt), where)
            D.add("ece", crow.ece, ece10(yt, pt), where)
            hrow = hl[(hl.attribute == attr) & (hl.subgroup == g)].iloc[0]
            if both and len(yt) >= 10:
                with np.errstate(divide="ignore", invalid="ignore"):
                    st, pv = hl10(yt, pt)
                D.add("hl_statistic", hrow.hl_statistic, st, where)
                D.add("hl_p_value", hrow.hl_p_value, pv, where)
                D.add("hl_df", hrow.hl_df, 10.0, where)
            drows = dc[(dc.attribute == attr) & (dc.subgroup == g)].sort_values("threshold")
            if both:
                ref = dca(data=pd.DataFrame({"outcome": yt.astype(int), "model": pt}),
                          outcome="outcome", modelnames=["model"],
                          thresholds=drows.threshold.tolist())
                ref = ref[ref.model == "model"].sort_values("threshold")
                for (_, r), refnb in zip(drows.iterrows(), ref.net_benefit):
                    D.add("net_benefit_vs_dcurves", r.net_benefit, refnb, f"{where} t={r.threshold:.3f}")
                    D.add("net_benefit_vs_formula", r.net_benefit,
                          nb_formula(yt, pt, r.threshold), f"{where} t={r.threshold:.3f}")
        # Gaps (fairlearn)
        yp = (p >= t).astype(int)
        mf = MetricFrame(metrics={"tpr": true_positive_rate, "fpr": false_positive_rate,
                                  "sel": selection_rate},
                         y_true=y.astype(int), y_pred=yp, sensitive_features=lab.astype(str))
        dif = mf.difference(method="between_groups")
        gap = disc[(disc.attribute == attr) & (disc.subgroup == "GAP")].set_index("gap_metric").gap_value
        # Intersection gaps are formed from empirically shrunk cell rates by
        # design, so they are reported apart from the marginal ones.
        sfx = "" if attr in attrs else "_intersection"
        D.add("gap_equal_opportunity" + sfx, gap["equal_opportunity_difference"], dif["tpr"], attr)
        D.add("gap_predictive_equality" + sfx, gap["predictive_equality"], dif["fpr"], attr)
        D.add("gap_equalized_odds" + sfx, gap["equalized_odds_gap"], max(dif["tpr"], dif["fpr"]), attr)
        D.add("gap_statistical_parity" + sfx, gap["statistical_parity"], dif["sel"], attr)
    return {"dataset": name, "column": column, "n": len(y), "threshold": t,
            "seconds_isitfair": round(secs, 1), "shrunk_rows_skipped_for_rates": shrunk_skipped,
            "metrics": D.d}


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    out = Path(sys.argv[1])
    names = sys.argv[2:] or registry.ALL
    results = json.loads(out.read_text()) if out.exists() else []
    done = {(r["dataset"], r["column"]) for r in results}
    for name in names:
        for col in ("y_proba_lr", "y_proba_gbm"):
            if (name, col) in done:
                continue
            t0 = time.perf_counter()
            results.append(check(name, col))
            print(f"{name} {col}: {time.perf_counter() - t0:.1f}s", flush=True)
            out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
