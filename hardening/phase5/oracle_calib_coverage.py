"""Phase 5: oracle_calib recovery and bootstrap CI coverage.

Usage: python hardening/phase5/oracle_calib_coverage.py OUT_JSON [N_SIMS] [N_JOBS]

Data (registry.make_oracle_calib): logit(p) ~ N(-1, 1.2^2), groups A/B,
y ~ Bernoulli(expit(a_g + b_g logit p)), planted (a, b) = (0, 1) in A and
(-0.8, 0.7) in B.

What isitfair estimates:
- calibration_slope: b from the joint fit logit P(y) = a + b logit p, so the
  target is the planted b.
- calibration_intercept: calibration-in-the-large, the alpha in
  logit P(y) = alpha + logit p with the slope FIXED at 1 (Steyerberg). Its
  population value alpha* solves E[expit(a + b L)] = E[expit(alpha* + L)], so
  alpha* = a only when b = 1. For group A alpha* = 0; for B it is computed below
  by Gauss-Hermite quadrature. Coverage is reported against alpha* (the
  estimand) and, for transparency, against the planted a.
"""

from __future__ import annotations

import faulthandler
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.special import expit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hardening" / "datasets"))
import registry  # noqa: E402

from isitfair import FairnessAudit  # noqa: E402

PLANTED = {"A": (0.0, 1.0), "B": (-0.8, 0.7)}
MU, SD = -1.0, 1.2


def alpha_star(a: float, b: float) -> float:
    x, w = np.polynomial.hermite_e.hermegauss(200)  # probabilists' Hermite
    L = MU + SD * x
    w = w / w.sum()
    target = np.sum(w * expit(a + b * L))
    return brentq(lambda al: np.sum(w * expit(al + L)) - target, -10, 10, xtol=1e-14)


TARGETS = {g: {"slope": b, "intercept_estimand": alpha_star(a, b), "intercept_planted_a": a}
           for g, (a, b) in PLANTED.items()}


def one(sim: int, n_boot: int, data_seed: int | None = None) -> dict:
    df = registry.make_oracle_calib(seed=10_000 + sim if data_seed is None else data_seed)
    audit = FairnessAudit(y_true=df.y_true.to_numpy(float), y_proba=df.y_proba_lr.to_numpy(float),
                          sensitive_features={"group": df.group.to_numpy(object)},
                          threshold=0.3, n_bootstrap=n_boot, random_state=sim)
    cal = audit.calibration()
    out = {"sim": sim}
    for g in ("A", "B"):
        r = cal[(cal.subgroup == g)].iloc[0]
        out[g] = {k: float(r[k]) for k in ("calibration_intercept", "intercept_ci_low", "intercept_ci_high",
                                           "calibration_slope", "slope_ci_low", "slope_ci_high")}
    return out


def covers(lo: float, hi: float, v: float) -> bool:
    return bool(lo <= v <= hi)


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    out = Path(sys.argv[1])
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    res: dict = {"targets": TARGETS}

    t0 = time.perf_counter()
    single = one(42, 2000, data_seed=42)  # the registry's oracle_calib table, B = 2000
    res["registry_dataset_B2000"] = single
    res["registry_dataset_contains"] = {
        g: {"slope": covers(single[g]["slope_ci_low"], single[g]["slope_ci_high"], TARGETS[g]["slope"]),
            "intercept_estimand": covers(single[g]["intercept_ci_low"], single[g]["intercept_ci_high"],
                                         TARGETS[g]["intercept_estimand"]),
            "intercept_planted_a": covers(single[g]["intercept_ci_low"], single[g]["intercept_ci_high"],
                                          TARGETS[g]["intercept_planted_a"])}
        for g in ("A", "B")}
    print("single B=2000:", res["registry_dataset_contains"], f"{time.perf_counter() - t0:.0f}s", flush=True)
    out.write_text(json.dumps(res, indent=2))

    t0 = time.perf_counter()
    sims = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(s, 200) for s in range(n_sims))
    res["n_sims"], res["n_bootstrap"], res["seconds"] = n_sims, 200, round(time.perf_counter() - t0, 1)
    cov = {}
    for g in ("A", "B"):
        tg = TARGETS[g]
        cov[g] = {
            "slope": np.mean([covers(s[g]["slope_ci_low"], s[g]["slope_ci_high"], tg["slope"]) for s in sims]),
            "intercept_estimand": np.mean([covers(s[g]["intercept_ci_low"], s[g]["intercept_ci_high"],
                                                  tg["intercept_estimand"]) for s in sims]),
            "intercept_planted_a": np.mean([covers(s[g]["intercept_ci_low"], s[g]["intercept_ci_high"],
                                                   tg["intercept_planted_a"]) for s in sims]),
            "mean_slope": np.mean([s[g]["calibration_slope"] for s in sims]),
            "mean_intercept": np.mean([s[g]["calibration_intercept"] for s in sims]),
        }
    res["coverage"] = {g: {k: float(v) for k, v in d.items()} for g, d in cov.items()}
    res["sims"] = sims
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["coverage"], indent=1), flush=True)


if __name__ == "__main__":
    main()
