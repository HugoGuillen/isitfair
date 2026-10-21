"""Run a jupytext percent-format script cell by cell, printing per-cell wall time.

Usage: python hardening/tools/run_cells.py SCRIPT [--cwd DIR] [--timings OUT.csv]

The script is executed in one shared namespace, exactly as `python SCRIPT`
would, except that each `# %%` code cell is timed separately. A faulthandler
watchdog dumps every thread's stack every 600 s, so a stall prints a traceback
instead of staying silent.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)


def cells(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in text.splitlines(keepends=True):
        if line.startswith("# %%"):
            if buf:
                out.append((kind, "".join(buf)))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        out.append((kind, "".join(buf)))
    return [(k, s) for k, s in out if k == "code" and s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--timings", default=None)
    args = ap.parse_args()

    script = Path(args.script).resolve()
    if args.cwd:
        os.chdir(args.cwd)
    ns: dict = {"__name__": "__main__", "__file__": str(script)}
    rows = []
    t_all = time.perf_counter()
    for i, (_, src) in enumerate(cells(script.read_text(encoding="utf-8"))):
        first = next(l for l in src.splitlines() if l.strip())[:70]
        t0 = time.perf_counter()
        print(f"[cell {i:02d}] start  {first}", flush=True)
        exec(compile(src, f"{script.name}:cell{i}", "exec"), ns)
        dt = time.perf_counter() - t0
        print(f"[cell {i:02d}] done   {dt:9.2f} s   (elapsed {time.perf_counter() - t_all:9.2f} s)", flush=True)
        rows.append({"cell": i, "first_line": first, "seconds": round(dt, 3)})
    print(f"[total] {time.perf_counter() - t_all:.2f} s", flush=True)
    if args.timings:
        with open(args.timings, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["cell", "first_line", "seconds"])
            w.writeheader()
            w.writerows(rows)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
