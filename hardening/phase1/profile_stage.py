"""Profile one FairnessAudit stage at a given n (cProfile top 30 + pyinstrument).

Usage: python hardening/phase1/profile_stage.py N STAGE OUT_PREFIX
"""
import cProfile
import faulthandler
import io
import pstats
import sys
import time

faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)

sys.path.insert(0, "hardening/phase1")
from bench import call, make_data  # noqa: E402
from isitfair import FairnessAudit  # noqa: E402
from pyinstrument import Profiler  # noqa: E402

n, stage, prefix = int(sys.argv[1]), sys.argv[2], sys.argv[3]
y, p, attrs = make_data(n)
kw = dict(y_true=y, y_proba=p, sensitive_features=attrs, threshold=0.2, n_bootstrap=20, random_state=42)

a = FairnessAudit(**kw)
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
call(a, stage, None)
pr.disable()
print(f"cProfile run {time.perf_counter() - t0:.1f}s", flush=True)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
open(prefix + "_cprofile.txt", "w").write(s.getvalue())

a = FairnessAudit(**kw)
prof = Profiler(interval=0.01)
prof.start()
call(a, stage, None)
prof.stop()
open(prefix + "_pyinstrument.txt", "w", encoding="utf-8").write(prof.output_text(unicode=True, color=False, show_all=False))
print("done", flush=True)
