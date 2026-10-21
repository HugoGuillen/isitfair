"""Compare a tree's results/ against results/MANIFEST.csv and the committed tables.

Usage: python hardening/tools/golden_compare.py TREE_ROOT [--only PREFIX ...]

For every MANIFEST artifact present under TREE_ROOT, recompute sha256[:16] and
compare. For every results/tables/*.csv, also compare numerically against the
committed copy in the main repository (so a hash miss can be told apart from a
number change). Exit status 0 only if every checked artifact matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    tree = Path(args.tree).resolve()

    with open(REPO / "results" / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))

    n_ok = n_bad = n_missing = 0
    for row in manifest:
        art = row["artifact"]
        if args.only and not any(art.startswith(p) for p in args.only):
            continue
        p = tree / art
        if not p.exists():
            n_missing += 1
            print(f"MISSING  {art}")
            continue
        got = sha16(p)
        if got == row["sha256_16"]:
            n_ok += 1
        else:
            n_bad += 1
            print(f"HASH     {art}: manifest {row['sha256_16']} got {got}")
            if art.endswith(".csv") and art.startswith("results/tables/"):
                ref = pd.read_csv(REPO / art)
                new = pd.read_csv(p)
                if ref.shape != new.shape or list(ref.columns) != list(new.columns):
                    print(f"         shape/columns differ: {ref.shape} vs {new.shape}")
                    continue
                num = ref.select_dtypes("number").columns
                diff = np.nanmax(np.abs(ref[num].to_numpy(float) - new[num].to_numpy(float))) if len(num) else 0.0
                nonnum = [c for c in ref.columns if c not in num]
                text_eq = ref[nonnum].astype(str).equals(new[nonnum].astype(str))
                print(f"         max |numeric diff| = {diff:.3g}; text columns equal: {text_eq}")
    print(f"\nmatched {n_ok}, differ {n_bad}, missing {n_missing}")
    return 0 if n_bad == 0 and n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
