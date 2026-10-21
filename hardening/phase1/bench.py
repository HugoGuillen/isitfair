"""Phase 1: per-stage wall time of FairnessAudit on synthetic data.

Usage: python hardening/phase1/bench.py N OUT_JSON [--B 20] [--extrapolate STAGE ...]

Synthetic input, numpy.random.default_rng(42): prevalence ~0.10, a binary
attribute, an 8-level attribute and their intersection. Stages are timed with
time.perf_counter() on one audit, in the order the task lists them, so later
stages reuse earlier caches exactly as a user's session would.

A stage named in --extrapolate is not run at B; instead it is run on a second
audit built with n_bootstrap=1 and the B-replicate time is projected as
t(B) = t(1) * (B + 1) / 2 (point estimate plus B replicates of equal size). The
JSON records which stages were extrapolated.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import tempfile
import time
from pathlib import Path

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

from isitfair import FairnessAudit  # noqa: E402

STAGES = [
    "construct",
    "discrimination",
    "calibration",
    "calibration_curves",
    "decision_curve",
    "estimability",
    "hosmer_lemeshow",
    "recalibration_ladder",
    "report",
]
EXTRAS = ["roc_curves", "decision_curve_zero_crossing"]


def make_data(n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(42)
    sex = rng.choice(np.array(["F", "M"]), size=n)
    level_p = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03])
    grp = rng.choice(np.array([f"g{i}" for i in range(8)]), size=n, p=level_p)
    lp = rng.normal(-2.6, 1.0, n)
    p = expit(lp)
    y = rng.binomial(1, p).astype(float)
    attrs = {"sex": sex, "group8": grp, ("sex", "group8"): None}
    return y, p, attrs


def call(audit: FairnessAudit, stage: str, out_dir: Path) -> None:
    if stage == "discrimination":
        audit.discrimination()
    elif stage == "calibration":
        audit.calibration()
    elif stage == "calibration_curves":
        audit.calibration_curves()
    elif stage == "decision_curve":
        audit.decision_curve()
    elif stage == "estimability":
        audit.estimability()
    elif stage == "hosmer_lemeshow":
        audit.hosmer_lemeshow()
    elif stage == "recalibration_ladder":
        audit.recalibration_ladder(by="group8")
    elif stage == "report":
        audit.report(out_dir / "bench_report.html", pdf=False)
    elif stage == "roc_curves":
        audit.roc_curves()
    elif stage == "decision_curve_zero_crossing":
        audit.decision_curve(zero_crossing=True)
    else:
        raise ValueError(stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--extrapolate", nargs="*", default=[])
    ap.add_argument("--extras", action="store_true")
    args = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)

    y, p, attrs = make_data(args.n)
    print(f"n={args.n} events={int(y.sum())} prevalence={y.mean():.4f} B={args.B}", flush=True)
    result: dict = {"n": args.n, "B": args.B, "prevalence": float(y.mean()), "seconds": {},
                    "extrapolated": {}}
    out_dir = Path(tempfile.mkdtemp(prefix=f"bench{args.n}_"))
    t_all = time.perf_counter()

    def kw(b: int) -> dict:
        return dict(y_true=y, y_proba=p, sensitive_features=attrs, threshold=0.2,
                    n_bootstrap=b, random_state=42)

    t0 = time.perf_counter()
    audit = FairnessAudit(**kw(args.B))
    result["seconds"]["construct"] = time.perf_counter() - t0
    print(f"  construct {result['seconds']['construct']:.3f}s", flush=True)

    shadow = None
    for stage in STAGES[1:] + (EXTRAS if args.extras else []):
        if stage == "report" and stage in args.extrapolate:
            # report() recomputes calibration_curves at full B on the main audit.
            # Measure its own increment on a warmed B=1 audit, then add the
            # projected calibration_curves cost.
            if shadow is None:
                shadow = FairnessAudit(**kw(1))
            for warm in ("discrimination", "roc_curves", "calibration", "calibration_curves",
                         "decision_curve", "estimability", "hosmer_lemeshow"):
                call(shadow, warm, out_dir)
            t0 = time.perf_counter()
            call(shadow, "report", out_dir)
            inc = time.perf_counter() - t0
            cc = result["seconds"].get("calibration_curves", 0.0)
            result["extrapolated"]["report"] = {"increment_B1_warm": inc, "plus_calibration_curves": cc,
                                                "projected": inc + cc}
            result["seconds"]["report"] = inc + cc
            print(f"  report EXTRAPOLATED increment {inc:.1f}s + calibration_curves {cc:.1f}s", flush=True)
            continue
        if stage in args.extrapolate:
            if shadow is None:
                shadow = FairnessAudit(**kw(1))
            t0 = time.perf_counter()
            call(shadow, stage, out_dir)
            t1 = time.perf_counter() - t0
            proj = t1 * (args.B + 1) / 2
            result["extrapolated"][stage] = {"t_B1": t1, "projected": proj}
            result["seconds"][stage] = proj
            print(f"  {stage} EXTRAPOLATED from B=1 run {t1:.2f}s -> {proj:.1f}s "
                  f"(elapsed {time.perf_counter() - t_all:.1f}s)", flush=True)
            continue
        t0 = time.perf_counter()
        call(audit, stage, out_dir)
        result["seconds"][stage] = time.perf_counter() - t0
        print(f"  {stage} {result['seconds'][stage]:.3f}s "
              f"(elapsed {time.perf_counter() - t_all:.1f}s)", flush=True)
        Path(args.out).write_text(json.dumps(result, indent=2))

    result["total_measured"] = time.perf_counter() - t_all
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"done total {result['total_measured']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
