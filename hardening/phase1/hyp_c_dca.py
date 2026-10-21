"""Hypothesis c: does dcurves.dca scale badly with the number of subgroups?

Fixed n = 20,000, B = 20; one attribute with k levels. Counts dca calls and the
time spent inside them by wrapping isitfair.audit.dca.
"""
import faulthandler
import time

faulthandler.dump_traceback_later(600, repeat=True)
import numpy as np
from scipy.special import expit

import isitfair.audit as A
from isitfair import FairnessAudit

calls = {"n": 0, "t": 0.0, "thresholds": []}
_orig = A.dca


def wrapped(*args, **kw):
    t0 = time.perf_counter()
    out = _orig(*args, **kw)
    calls["t"] += time.perf_counter() - t0
    calls["n"] += 1
    calls["thresholds"].append(len(kw.get("thresholds", [])))
    return out


A.dca = wrapped
rng = np.random.default_rng(42)
n = 20_000
p = expit(rng.normal(-2.6, 1.0, n))
y = rng.binomial(1, p).astype(float)
print("k  dca_calls  thresholds_per_call  t_dca_s  t_decision_curve_s  t_per_subgroup_s")
for k in (2, 8, 32, 128):
    g = rng.integers(0, k, n).astype(str)
    calls.update(n=0, t=0.0, thresholds=[])
    a = FairnessAudit(y_true=y, y_proba=p, sensitive_features={"g": g}, threshold=0.2,
                      n_bootstrap=20, random_state=42, event_floors=None)
    t0 = time.perf_counter()
    a.decision_curve()
    dt = time.perf_counter() - t0
    print(f"{k:<3} {calls['n']:>9}  {sorted(set(calls['thresholds']))}  {calls['t']:8.2f}  {dt:8.2f}  {dt/(k+1):8.3f}", flush=True)
