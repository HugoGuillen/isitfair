# Phase 0 baseline

Environment: Windows 11 (10.0.26120), Intel Haswell-class CPU with AVX2 and no
AVX-512, 12 logical cores, 31.8 GB. CPython 3.10.20 (uv), venv from
`requirements-lock.txt` plus `pip install -e ".[examples]"`, then pytest,
fairlearn, pyinstrument, py-spy (none of which moved a locked version; see
`env.txt`). WeasyPrint 70.0 uses Pango from conda-forge in
`C:\Users\hugoa\Desktop\isitfair-pango` (PATH + `WEASYPRINT_DLL_DIRECTORIES`,
`hardening/tools/env.sh`).

## pytest -x -q

`31 passed in 48.34s` (`logs/pytest_baseline.log`).

## Golden run (HEAD code, committed seed 99 and B = 2000)

Run in a `git worktree` of HEAD (`../isitfair-golden`) so committed `results/`
is never written; `hardening/tools/golden.sh`. Both the example and notebook 02
imported `src/` before the first fix was applied (example 20:03, nb02 kernel
20:42:47; first edit to `src/` after 20:43:37), so this is the unmodified code.

Before running, every MANIFEST artifact on disk hashed to its MANIFEST value
(79 / 79), in the main tree and in the worktree.

### Wall time per stage, `examples/support2_mortality_audit.py`

Contended: up to ~13 other processes were sharing the 12 hardware threads, so
these are upper bounds (the user guide quotes about 12 minutes in total).

| cell | stage | seconds |
|---|---|---:|
| 1-6 | setup, load, model, split, construct | 9.1 |
| 7 | `discrimination()` | 79.6 |
| 9 | `roc_curves()` | 21.8 |
| 10 | `calibration()` | 114.7 |
| 11 | `calibration_curves()` | 1134.4 |
| 12-13 | `hosmer_lemeshow()`, detail | 1.4 |
| 14 | `decision_curve(zero_crossing=True)` | 28.4 |
| 15-16 | `estimability()`, intersection audit, `cell_diagnostics()` | 0.1 |
| 17 | second audit + `mitigate()` + before/after `calibration()` | 261.8 |
| 18 | `report(..., pdf=True)` | 703.7 |
| | total | 2354.7 |

`calibration_curves()` is 48% of the run even at n = 1,821, and most of the
report cell is `calibration_curves()` again on the three mitigated audits.

### Does HEAD reproduce the MANIFEST on this machine? No, in three ways

1. **results/tables/*.csv and *.md (notebook 02): reproduced in substance.**
   24 of 26 files are byte-identical once line endings are normalized (pandas
   writes CRLF on Windows). The other two (`table1_by_race_effect_sizes.csv`,
   `table1_by_sex_effect_sizes.csv`) differ only in the last digit of four
   floats, for example `0.05967210561035813` vs `0.05967210561035812`
   (platform math library). Every numeric column is equal to within 1e-16.

2. **Figures: pixel-identical or sub-visual.** 14 of 20 PNGs decode to
   identical pixels (bytes differ in PNG encoding only). The calibration and
   mitigation-calibration plots differ in at most 0.02% of pixels by at most
   0.008 intensity.

3. **Report HTML/PDF: seven ECE values differ.** All seven are in the
   "After" (mitigation) section, the ECE column:

   | committed | this machine |
   |---:|---:|
   | 0.0442 | 0.0412 |
   | 0.0389 | 0.0309 |
   | 0.0709 | 0.0630 |
   | 0.0328 | 0.0243 |
   | 0.0450 | 0.0664 |
   | 0.0627 | 0.0477 |
   | 0.0278 | 0.0191 |

   Cause: isotonic recalibration produces heavily tied predictions (8 to 80
   distinct values per subgroup), and `_compute_ece` bins by
   `np.argsort(y_proba)` (unstable), so tied rows straddle bin edges according
   to the sort kernel's tie order. NumPy 2.x dispatches `argsort` to
   CPU-specific SIMD kernels; on this AVX2 machine the default sort reproduces
   the right-hand column exactly, while no ordering available here (default,
   stable, or tie-snapped) reproduces the committed column. The committed
   values are therefore specific to the machine that produced them. This is
   ERRORLOG E05; its fix makes ECE deterministic but gives a third set of
   values for these seven cells.

Every other number in the report HTML (images stripped) is identical.
