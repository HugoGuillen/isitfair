"""Sequential job queue for the rest of the hardening run (one heavy job at a time).

Usage: python hardening/tools/jobqueue.py   (run with nohup; log in hardening/logs/queue.log)

Waits for the jobs started earlier to finish, then runs, strictly one after the
other: golden post-fix check, Phase 5 oracles on the final code, the nine
Phase 3 notebooks (B = 50, n_jobs = -1), Phase 6 (B = 200, n_jobs = -1) on the
seven real datasets, and the uncontended Phase 1 "after" sweep. Before each
Phase 6 run its projected wall time (Phase 3 stage seconds, excluding the
input-variant stage, x 200/50) is appended to STATUS.md; after it, the actual.
Each Phase 6 run is killed at 4x its projection (minimum 30 min).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
H = REPO / "hardening"
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
BASH = r"C:\Program Files\Git\bin\bash.exe"
STATUS = H / "STATUS.md"
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
           WEASYPRINT_DLL_DIRECTORIES=r"C:\Users\hugoa\Desktop\isitfair-pango\Library\bin",
           PATH=r"C:\Users\hugoa\Desktop\isitfair-pango\Library\bin;" + os.environ["PATH"])
ENV.pop("PYTHONPATH", None)
P3 = ["support2", "compas", "german", "heart_cleveland", "oracle_sparse", "oracle_calib",
      "adult", "bank", "diabetes130"]
P6 = ["support2", "compas", "german", "heart_cleveland", "adult", "bank", "diabetes130"]


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def status_row(stage: str, start: str, end: str, wall: str, eta: str, basis: str) -> None:
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write(f"| {stage} | {start} | {end} | {wall} | {eta} | {basis} |\n")


def busy(patterns: list[str]) -> list[str]:
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
                        capture_output=True, text=True).stdout
    return [p for p in patterns if p in ps]


def run(stage: str, cmd: list[str], logfile: Path, timeout: float | None = None, eta: str = "",
        basis: str = "") -> tuple[int, float]:
    start, t0 = now(), time.perf_counter()
    log(f"START {stage} -> {logfile.name}" + (f" (timeout {timeout:.0f}s)" if timeout else ""))
    with logfile.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=REPO)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill this job's own process tree only (venv launcher -> interpreter -> workers).
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            proc.wait()
            rc = -9
    wall = time.perf_counter() - t0
    log(f"END   {stage} rc={rc} wall={wall:.0f}s")
    status_row(stage, start, now(), f"{wall:.0f}" + ("" if rc == 0 else f" (rc={rc})"), eta, basis)
    return rc, wall


def phase3_seconds(name: str) -> float:
    f = H / "notebooks" / "results" / f"{name}_B50.json"
    r = json.loads(f.read_text())
    return sum(v["seconds"] or 0 for k, v in r["stages"].items() if k != "variants")


def main() -> None:
    waits = ["golden.sh", "oracle_calib_coverage.py", "before_50000.json"]
    while b := busy(waits):
        log(f"waiting for {b}")
        time.sleep(120)

    run("P0 golden narrowed compare of HEAD baseline outputs", [PY, str(H / "tools" / "golden_narrow.py"),
                                                              str(REPO.parent / "isitfair-golden")],
        H / "logs" / "golden_baseline_narrow.log")
    run("P0/P4 golden check on fixed src (example + nb02)", [BASH, str(H / "tools" / "golden.sh"), "postfix", "all"],
        H / "logs" / "golden_postfix.log")
    run("P4 golden narrowed compare", [PY, str(H / "tools" / "golden_narrow.py"), str(REPO.parent / "isitfair-golden")],
        H / "logs" / "golden_postfix_narrow.log")
    (H / "phase5" / "results" / "oracles_final.json").unlink(missing_ok=True)
    run("P5 oracles on final code", [PY, str(H / "phase5" / "oracles.py"),
                                     str(H / "phase5" / "results" / "oracles_final.json")],
        H / "logs" / "phase5_oracles_final.log")

    for name in P3:
        run(f"P3 final B=50 n_jobs=-1 {name}",
            [BASH, "-c", f"HARDEN_NJOBS=-1 bash hardening/notebooks/run_nb.sh {name} final_B50 50 '' 43200"],
            H / "logs" / f"nb_{name}_final_B50.log")

    for name in P6:
        try:
            proj = phase3_seconds(name) * 200 / 50
        except Exception:
            proj = 3600.0
        tmo = max(1800.0, 4 * proj)
        status_row(f"P6 B=200 {name} (projection)", now(), "", "", f"{proj:.0f} s projected, timeout {tmo:.0f} s",
                   "Phase 3 final stage seconds (excl. variants) x 200/50")
        run(f"P6 B=200 n_jobs=-1 {name}", [PY, str(H / "phase6" / "pipeline.py"), name, "200", "-1"],
            H / "logs" / f"phase6_{name}_B200.log", timeout=tmo, eta=f"projected {proj:.0f}",
            basis="phase6/results JSON")

    run("P1 after sweep (uncontended)", [BASH, str(H / "phase1" / "run_after.sh")],
        H / "logs" / "phase1_after_uncontended.log")
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
