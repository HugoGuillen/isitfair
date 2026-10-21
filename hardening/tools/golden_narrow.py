"""Narrowed golden check (agreed after hardening/baseline.md showed that HEAD
itself does not reproduce the committed report's mitigation-section ECE on this
machine).

Usage: python hardening/tools/golden_narrow.py TREE_ROOT

PASS requires:
- every results/tables/*.csv: same shape and columns, text columns identical,
  numeric columns equal within 1e-12 (CRLF normalized);
- every results/tables/*.md: identical after CRLF normalization;
- results/support2_audit_report.html with embedded images stripped: identical,
  except ECE cells in the "After" (mitigation) section, which are listed;
- in the three tables built from isotonic-recalibrated (heavily tied)
  predictions, ECE columns are exempt (agreed with the user after the post-fix
  check: tie order made them machine-specific before E05); differing ECE cells
  are listed, every other cell must still match;
- every MANIFEST PNG: same decoded pixels, or reported with its max pixel
  difference (sub-visual differences are reported, not failed).
The PDF is not compared (WeasyPrint/Pango build and fonts differ by machine).
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-12
ECE_EXEMPT = {"results/tables/table4_mitigation_before_after", "results/tables/table4_recalibration_ladder",
              "results/tables/tableS_mitigation_wasserstein"}


def md_cells(text: str) -> list[list[str]]:
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in text.splitlines() if ln.startswith("|")]


def committed(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"],
                          capture_output=True, check=True).stdout


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    fail = False
    with open(REPO / "results" / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
        arts = [r["artifact"] for r in csv.DictReader(fh)]

    print("## tables")
    for art in [a for a in arts if a.startswith("results/tables/")]:
        new_b = (tree / art).read_bytes().replace(b"\r\n", b"\n")
        old_b = committed(art)
        if new_b == old_b:
            continue
        exempt = art.rsplit(".", 1)[0] in ECE_EXEMPT
        if art.endswith(".md"):
            o, n = md_cells(old_b.decode("utf-8")), md_cells(new_b.decode("utf-8"))
            keep = [k for k, h in enumerate(o[0]) if not h.startswith("ece")] if exempt and o else None
            same_rest = (exempt and len(o) == len(n) and o[0] == n[0]
                         and all([r[k] for k in keep] == [m[k] for k in keep] for r, m in zip(o, n))
                         and [ln for ln in old_b.decode().splitlines() if not ln.startswith("|")]
                         == [ln for ln in new_b.decode().splitlines() if not ln.startswith("|")])
            if same_rest:
                print(f"ok   {art}: differs only in exempt ECE columns")
            else:
                print(f"FAIL {art}: markdown differs"); fail = True
            continue
        old, new = pd.read_csv(io.BytesIO(old_b)), pd.read_csv(io.BytesIO(new_b))
        if exempt:
            ece_cols = [c for c in old.columns if c.startswith("ece")]
            for c in ece_cols:
                for r in np.where(old[c].to_numpy(float) != new[c].to_numpy(float))[0]:
                    print(f"     allowed {art} row {r} {c}: {old[c].iloc[r]} -> {new[c].iloc[r]}")
            old, new = old.drop(columns=ece_cols), new.drop(columns=ece_cols)
        if old.shape != new.shape or list(old.columns) != list(new.columns):
            print(f"FAIL {art}: shape {old.shape} vs {new.shape}"); fail = True; continue
        num = old.select_dtypes("number").columns
        txt = [c for c in old.columns if c not in num]
        d = np.abs(old[num].to_numpy(float) - new[num].to_numpy(float))
        both_nan = np.isnan(old[num].to_numpy(float)) & np.isnan(new[num].to_numpy(float))
        nan_mismatch = int((np.isnan(d) & ~both_nan).sum())
        worst = float(np.nanmax(d)) if d.size else 0.0
        text_ok = old[txt].astype(str).equals(new[txt].astype(str))
        ok = worst <= TOL and nan_mismatch == 0 and text_ok
        fail |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {art}: max|diff| {worst:.3g}, nan mismatches {nan_mismatch}, text equal {text_ok}")
        if not ok:
            rows, cols = np.where((d > TOL) | (np.isnan(d) & ~both_nan))
            for r, c in list(zip(rows, cols))[:20]:
                key = ", ".join(str(old.iloc[r][k]) for k in txt[:4])
                print(f"     row {r} [{key}] {num[c]}: committed {old[num[c]].iloc[r]!r} -> {new[num[c]].iloc[r]!r}")

    print("## report html")
    strip = lambda s: re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "IMG", s)
    old_h = strip(committed("results/support2_audit_report.html").decode("utf-8")).splitlines()
    new_h = strip((tree / "results/support2_audit_report.html").read_text(encoding="utf-8")).splitlines()
    if len(old_h) != len(new_h):
        print(f"FAIL line count {len(old_h)} vs {len(new_h)}"); fail = True
    else:
        section = ""
        ece_cells = 0
        headers: list[str] = []
        for i, (a, b) in enumerate(zip(old_h, new_h)):
            if re.search(r"<h[1-4]", a):
                section = re.sub("<[^>]+>", "", a).strip()
            if "<thead>" in a:
                headers = []
            headers += re.findall(r"<th>([^<]*)</th>", a)
            if a == b:
                continue
            ca, cb = re.findall(r"<td>([^<]*)</td>", a), re.findall(r"<td>([^<]*)</td>", b)
            diff_cols = [k for k, (x, y) in enumerate(zip(ca, cb)) if x != y] if len(ca) == len(cb) else None
            # Rows span two lines; locate the ECE cell by the table header, offset by the
            # cells already on the row's first line.
            first = re.findall(r"<td>([^<]*)</td>", old_h[i - 1]) if "<tr>" not in old_h[i - 1] else []
            col = (len(first) + diff_cols[0]) if diff_cols is not None and len(diff_cols) == 1 else None
            is_ece = (section == "After" and col is not None and col < len(headers)
                      and headers[col] == "ECE")
            if is_ece:
                ece_cells += 1
                print(f"     allowed ECE cell line {i + 1}: {ca[diff_cols[0]]} -> {cb[diff_cols[0]]}")
            else:
                fail = True
                print(f"FAIL line {i + 1} [{section}]\n   - {a.strip()[:220]}\n   + {b.strip()[:220]}")
        print(f"html: {ece_cells} mitigation-section ECE cell(s) differ (allowed); other lines checked")

    print("## figures")
    for art in [a for a in arts if "support2_audit_report_figures/" in a and not (tree / a).exists()]:
        print(f"FAIL {art}: missing (example did not run?)"); fail = True
    for art in [a for a in arts if a.endswith(".png") and (tree / a).exists()]:
        a = mpimg.imread(io.BytesIO(committed(art)))
        b = mpimg.imread(tree / art)
        if a.shape != b.shape:
            print(f"FAIL {art}: shape {a.shape} vs {b.shape}"); fail = True; continue
        d = np.abs(a.astype(float) - b.astype(float)).max(axis=-1)
        if d.max() > 0:
            print(f"note {art}: max pixel diff {d.max():.3f}, pixels differing {np.mean(d > 0):.5f}")
            if d.max() > 0.05 or np.mean(d > 0) > 0.001:
                fail = True
                print(f"FAIL {art}: visible difference")

    print("\nGOLDEN (narrowed):", "FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
