"""Phase 5 follow-up: the differences in oracles_final.json that exceed 1e-9 on
ECE, Hosmer-Lemeshow and the equal-opportunity / equalized-odds gaps, checked
against from-scratch oracles that use isitfair's documented definitions:

- ECE and HL: 10 equal-count bins whose cuts are moved to the nearer end of a
  run of tied predictions (forward on an exact tie), instead of splitting ties
  by row position. Written here without importing isitfair.metrics.
- Gaps: TPR is undefined for a subgroup with no events and FPR for one with no
  non-events; those subgroups are left out of the range (fairlearn reports 0).

Usage: python hardening/phase5/oracle_definitions.py OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hardening" / "datasets"))
import registry  # noqa: E402

from isitfair import FairnessAudit  # noqa: E402

RUNS = [("adult", "y_proba_gbm"), ("compas", "y_proba_lr"), ("compas", "y_proba_gbm"),
        ("bank", "y_proba_gbm"), ("diabetes130", "y_proba_lr"), ("diabetes130", "y_proba_gbm"),
        ("oracle_sparse", "y_proba_lr"), ("oracle_sparse", "y_proba_gbm")]


def snapped_bins(p: np.ndarray, g: int = 10) -> list[np.ndarray]:
    order = np.argsort(p, kind="stable")
    ps = p[order]
    n = len(p)
    sizes = [n // g + (1 if i < n % g else 0) for i in range(min(g, n))]
    cuts = np.cumsum(sizes)[:-1]
    edges = set()
    for c in cuts:
        if 0 < c < n and ps[c - 1] == ps[c]:
            lo = c
            while lo > 0 and ps[lo - 1] == ps[c]:
                lo -= 1
            hi = c
            while hi < n and ps[hi] == ps[c]:
                hi += 1
            c = lo if (c - lo) < (hi - c) else hi
        if 0 < c < n:
            edges.add(int(c))
    return [b for b in np.split(order, sorted(edges)) if len(b)]


def ece(y, p):
    return sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in snapped_bins(p)) / len(p)


def hl_stat(y, p):
    s = 0.0
    for b in snapped_bins(p):
        o, e, n = y[b].sum(), p[b].sum(), len(b)
        s += (o - e) ** 2 / (e * (1 - e / n))
    return s


def gaps(y, p, lab, t):
    flag = p >= t
    tpr, fpr = [], []
    for g in sorted(set(lab)):
        m = lab == g
        if y[m].sum() > 0:
            tpr.append(flag[m & (y == 1)].mean())
        if (1 - y[m]).sum() > 0:
            fpr.append(flag[m & (y == 0)].mean())
    eo = max(tpr) - min(tpr)
    pe = max(fpr) - min(fpr)
    return eo, max(eo, pe)


def main() -> None:
    out = []
    for name, col in RUNS:
        df = registry.load(name)
        attrs = registry.ATTRS[name]
        for a in attrs:
            df[a] = df[a].fillna("Unknown")
        y, p = df["y_true"].to_numpy(float), df[col].to_numpy(float)
        t = registry.meta(name)["threshold"]
        sf = {a: df[a].to_numpy(object) for a in attrs}
        audit = FairnessAudit(y_true=y, y_proba=p, sensitive_features=sf, threshold=t,
                              n_bootstrap=20, random_state=42)
        cal, hl, disc = audit.calibration(), audit.hosmer_lemeshow(), audit.discrimination()
        worst = {"ece": 0.0, "hl_statistic": 0.0, "hl_p_value": 0.0, "gap_equal_opportunity": 0.0,
                 "gap_equalized_odds": 0.0}
        for a in attrs:
            lab = sf[a].astype(str)
            for g in ["Overall", *sorted(set(lab))]:
                m = np.ones(len(y), bool) if g == "Overall" else lab == g
                yt, pt = y[m], p[m]
                c = cal[(cal.attribute == a) & (cal.subgroup == g)].iloc[0]
                worst["ece"] = max(worst["ece"], abs(c.ece - ece(yt, pt)))
                h = hl[(hl.attribute == a) & (hl.subgroup == g)].iloc[0]
                if 0 < yt.sum() < len(yt) and len(yt) >= 10 and np.isfinite(h.hl_statistic):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s = hl_stat(yt, pt)
                    worst["hl_statistic"] = max(worst["hl_statistic"], abs(h.hl_statistic - s))
                    worst["hl_p_value"] = max(worst["hl_p_value"], abs(h.hl_p_value - chi2.sf(s, h.hl_df)))
            gap = disc[(disc.attribute == a) & (disc.subgroup == "GAP")].set_index("gap_metric").gap_value
            eo, eodds = gaps(y, p, lab, t)
            worst["gap_equal_opportunity"] = max(worst["gap_equal_opportunity"],
                                                 abs(gap["equal_opportunity_difference"] - eo))
            worst["gap_equalized_odds"] = max(worst["gap_equalized_odds"], abs(gap["equalized_odds_gap"] - eodds))
        print(name, col, {k: f"{v:.3g}" for k, v in worst.items()}, flush=True)
        out.append({"dataset": name, "column": col, "max_abs_diff": worst})
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
