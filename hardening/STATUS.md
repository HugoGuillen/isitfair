# Hardening status

Times are local (Europe, UTC+02:00). Wall seconds are measured, not estimated.
ETA column is a projection for the remaining stages, from the measured timings
listed under "Basis".

Note on rule 5: this Claude Code session has no TaskCreate/TaskUpdate tool, so
the task list is kept below in this file (one task per phase and per dataset).

## Task list

| task | status |
|---|---|
| P0 baseline | pytest done; golden example done; golden nb02 running |
| P1 diagnosis | sweep before: 1k/5k/20k done, 50k running, 100k running (pinned HEAD); after sweep running; profile, hyp b, hyp c done |
| P2 registry | done (registry.py, CARDS.md) |
| P3 notebooks pre-fix (pinned HEAD): support2, compas, german, heart_cleveland, oracle_sparse done; diabetes130 running | |
| P3 notebooks post-fix: adult, bank, oracle_calib running; others pending | |
| P4 fixes | E01 (LOESS) and E03 (ladder message) applied with tests; golden check pending |
| P5 oracle verification | pending |
| P6 full pipeline B=2000: support2, diabetes130, adult, compas, bank, german, heart_cleveland | pending |
| P7 acceptance | pending |

## Stage log

| stage | start | end | wall s | ETA remaining | basis |
|---|---|---|---|---|---|
| P0 env setup (uv, CPython 3.10.20, venv, lock, Pango via micromamba) | 2026-10-20 19:40 | 2026-10-20 20:03 | ~1380 | n/a | |
| P0 pytest -x -q baseline | 2026-10-20 20:03:31 | 2026-10-20 20:04:19 | 48 | | 31 passed |
| P0 golden example (B=2000, contended) | 2026-10-20 20:03:31 | 2026-10-20 20:42:47 | 2355 | nb02 B=2000: unknown, first run | logs/golden_baseline_example_cells.csv |
| P1 sweep before n=1000 | 2026-10-20 20:05:28 | 2026-10-20 20:06:03 | 35 | | phase1/results/before_1000.json |
| P1 sweep before n=5000 | 2026-10-20 20:06:03 | 2026-10-20 20:07:46 | 103 | | before_5000.json |
| P1 sweep before n=20000 | 2026-10-20 20:07:46 | 2026-10-20 20:27:08 | 1162 | 50k: ~2 h (cc ~ 914 s x 6.25 + calibration/HL ~ 6.25x) | before_20000.json |
| P1 profile n=20000 calibration_curves (cProfile + pyinstrument) | 2026-10-20 20:08 | 2026-10-20 20:38 | ~1800 | | profiles/ |
| P1 hyp c (dca vs k) | 2026-10-20 20:11 | 2026-10-20 20:12 | ~40 | | logs/hyp_c_dca.log |
| P1 hyp b (n_jobs, script + kernel) | 2026-10-20 20:13 | 2026-10-20 20:22 | ~560 | | logs/hyp_b_njobs.log |
| P2 registry build (9 datasets) | 2026-10-20 20:10 | 2026-10-20 20:11 | ~70 | | datasets/predictions |
| P3 pre-fix B=50 support2 / compas / german / heart / oracle_sparse (parallel) | 2026-10-20 20:18 | 2026-10-20 20:24 | 362 / 339 / 218 / 217 / 254 (sum of stages) | | notebooks/results/*_B50.json |
| P1 hyp/oracle Phase 5 oracles, 9 datasets x 2 models | 2026-10-20 20:54:50 | 2026-10-20 21:43 | ~2900 | | phase5/results/oracles.json (pre-E04/E05 code; rerun later ~50 min) |
| P3 post-fix B=50 oracle_calib | 2026-10-20 20:51:58 | 2026-10-20 21:07:14 | 916 (stages 890) | | pre-E04/E05 code |
| P3 post-fix B=50 adult | 2026-10-20 20:51:58 | 2026-10-20 21:31:09 | 2351 (stages 2618 incl. variants 1418) | | pre-E04/E05 code |
| P3 post-fix B=50 bank | 2026-10-20 20:51:58 | 2026-10-20 21:37:33 | 2735 (stages 2690 incl. variants 1327) | | pre-E04/E05 code |

## ETA snapshot 2026-10-20 21:46 (from the rows above and the log tails)

| item | state | projected remaining | basis |
|---|---|---|---|
| golden baseline nb02 (HEAD) | running since 20:42:47, 38 of ~40 table files written | under 30 min (no per-cell timing for nb02) | file count in ../isitfair-golden/results/tables |
| golden post-fix (example + nb02) | not started | ~1 h 45 min after baseline ends | example 2355 s + nb02 >= 63 min |
| Phase 5 coverage, 200 sims | 64 / 200 at 21.4 min | ~45 min | phase5_coverage.log |
| Phase 1 before 50k (HEAD) | in calibration_curves since ~20:33 | ~1 h | 20k cc 914 s x (50/20)^2 = 5700 s |
| Phase 1 before 100k (HEAD, extrapolated cc/report) | calibration done (2616 s) | ~2-3 h | B=1 cc ~ 2 x exact point cost; HL ~ calibration |
| Phase 1 after 50k then 100k | 50k at report | ~3-4 h | 100k ~ 4x 50k (exact point LOESS still O(n^2)) |
| Phase 3 rerun on final code, 9 notebooks B=50 | not started | ~3 h wall (diabetes130 bound) | adult 2618 s at n=14.6k; diabetes n=30.5k ~ 2x-4x |
| Phase 6 B=2000, 7 real datasets | not started | ~35 h wall, upper bound | rule: Phase 3 stage time (excl. variants) x 2000/50: adult 13 h, bank 15 h, diabetes ~33 h, small sets ~1-4 h each; run 4-5 in parallel. Overstates: exact point LOESS does not grow with B |
| Phase 7 acceptance | after Phase 6 | ~4 h | notebooks + pytest + golden |

## Plan change 2026-10-20 21:52 (user instruction)

Phase 6 runs at **B = 200** (not 2,000), **one dataset at a time**, each audit
with **n_jobs = -1**. All remaining heavy work goes through one sequential
queue (`hardening/tools/jobqueue.py`, log `logs/queue.log`). Stopped: the
pre-fix diabetes130 notebook (hang already reproduced and its stack captured)
and the contended Phase 1 "after" sweep (re-queued last, uncontended). Still
running outside the queue: golden baseline nb02, coverage study, Phase 1
"before" 50k and 100k (pinned HEAD).

## ETA snapshot 2026-10-20 21:52

| queue item | projected | basis |
|---|---|---|
| wait for golden baseline nb02, coverage, before-50k | ~45 min | coverage 64/200 at 21.4 min; before-50k cc ~5700 s from 20:33 |
| golden narrowed compare (baseline) | ~2 min | |
| golden post-fix (example + nb02) | ~1 h 45 min | example 2355 s contended; nb02 >= 63 min |
| Phase 5 oracles (final code) | ~50 min | first run 20:54-21:43 |
| Phase 3 final, 9 notebooks sequential | ~4 h | stage sums: adult 2618 s, bank 2690 s, oracle_calib 890 s, small ~300 s each; diabetes130 ~2x-3x adult |
| Phase 6 B=200, 7 datasets sequential | ~7 h (upper bound) | Phase 3 stages excl. variants x 200/50: adult 4800 s, bank 5450 s, diabetes130 ~12000 s, small sets < 30 min each; n_jobs=-1 shortens bootstrap stages |
| Phase 1 after sweep (uncontended) | ~2 h | contended 1k-20k 650 s; 50k/100k dominated by exact point LOESS |
| Phase 7 acceptance | ~4 h | notebooks re-executed + pytest + golden |
| **total remaining** | **~20 h** | |
| P0 golden narrowed compare of HEAD baseline outputs | 2026-10-20 22:26:10 | 2026-10-20 22:26:20 | 10 (rc=1) |  |  |
| P0/P4 golden check on fixed src (example + nb02) | 2026-10-20 22:26:20 | 2026-10-20 22:57:05 | 1845 |  |  |
| P4 golden narrowed compare | 2026-10-20 22:57:05 | 2026-10-20 22:57:15 | 10 (rc=1) |  |  |
| P5 oracles on final code | 2026-10-20 22:57:15 | 2026-10-20 23:14:36 | 1041 |  |  |
| P3 final B=50 n_jobs=-1 support2 | 2026-10-20 23:14:36 | 2026-10-20 23:16:57 | 141 |  |  |
| P3 final B=50 n_jobs=-1 compas | 2026-10-20 23:16:57 | 2026-10-20 23:19:06 | 130 |  |  |
| P3 final B=50 n_jobs=-1 german | 2026-10-20 23:19:06 | 2026-10-20 23:20:29 | 83 |  |  |
| P3 final B=50 n_jobs=-1 heart_cleveland | 2026-10-20 23:20:29 | 2026-10-20 23:21:50 | 80 |  |  |
| P3 final B=50 n_jobs=-1 oracle_sparse | 2026-10-20 23:21:50 | 2026-10-20 23:23:05 | 76 |  |  |
| P3 final B=50 n_jobs=-1 oracle_calib | 2026-10-20 23:23:05 | 2026-10-20 23:28:52 | 346 |  |  |
| P3 final B=50 n_jobs=-1 adult | 2026-10-20 23:28:52 | 2026-10-20 23:41:01 | 730 |  |  |
| P3 final B=50 n_jobs=-1 bank | 2026-10-20 23:41:01 | 2026-10-20 23:52:14 | 672 |  |  |
| P3 final B=50 n_jobs=-1 diabetes130 | 2026-10-20 23:52:14 | 2026-10-21 00:23:31 | 1877 |  |  |
| P6 B=200 support2 (projection) | 2026-10-21 00:23:31 |  |  | 367 s projected, timeout 1800 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 support2 | 2026-10-21 00:23:31 | 2026-10-21 00:25:37 | 126 | projected 367 | phase6/results JSON |
| P6 B=200 compas (projection) | 2026-10-21 00:25:37 |  |  | 343 s projected, timeout 1800 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 compas | 2026-10-21 00:25:37 | 2026-10-21 00:26:48 | 71 | projected 343 | phase6/results JSON |
| P6 B=200 german (projection) | 2026-10-21 00:26:48 |  |  | 207 s projected, timeout 1800 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 german | 2026-10-21 00:26:48 | 2026-10-21 00:27:25 | 37 | projected 207 | phase6/results JSON |
| P6 B=200 heart_cleveland (projection) | 2026-10-21 00:27:25 |  |  | 202 s projected, timeout 1800 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 heart_cleveland | 2026-10-21 00:27:25 | 2026-10-21 00:28:02 | 37 | projected 202 | phase6/results JSON |
| P6 B=200 adult (projection) | 2026-10-21 00:28:02 |  |  | 1204 s projected, timeout 4815 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 adult | 2026-10-21 00:28:02 | 2026-10-21 00:31:14 | 192 | projected 1204 | phase6/results JSON |
| P6 B=200 bank (projection) | 2026-10-21 00:31:14 |  |  | 1316 s projected, timeout 5263 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 bank | 2026-10-21 00:31:14 | 2026-10-21 00:35:17 | 243 | projected 1316 | phase6/results JSON |
| P6 B=200 diabetes130 (projection) | 2026-10-21 00:35:17 |  |  | 1800 s projected, timeout 7202 s | Phase 3 final stage seconds (excl. variants) x 200/50 |
| P6 B=200 n_jobs=-1 diabetes130 | 2026-10-21 00:35:17 | 2026-10-21 00:38:36 | 198 (rc=1) | projected 1800 | phase6/results JSON |
| P1 after sweep (uncontended) | 2026-10-21 00:38:36 | 2026-10-21 02:06:26 | 5270 |  |  |
| P6 B=200 n_jobs=-1 diabetes130 rerun (unseen-level rows dropped) | 2026-10-21 07:39:01 | 2026-10-21 07:45:01 | 360 | projected ~6 min | first run reached mitigate at 194 s; adult/bank mitigate+report <= 1 min |
| P4 golden narrowed compare, table ECE columns exempt (user-approved) | 2026-10-21 07:41:34 | 2026-10-21 07:41:34 | 10 | | logs/golden_postfix_narrow_v2.log: PASS |
| P7 acceptance (projection) | 2026-10-21 09:24:37 | | | ~1 h 45 min: notebooks ~69 min, pytest ~1 min, golden ~31 min | P3 final 4135 s; golden post-fix 1845 s |

Note 2026-10-21 09:26: the post-fix golden run of 2026-10-20 22:26 did not execute the example
(hardening/tools/queue.py shadowed the stdlib `queue` module, ImportError after 4 s; logs/golden_postfix.log).
Its tables (nb02) are valid; the report HTML compared was the committed file. Renamed to tools/jobqueue.py;
the Phase 7 golden step reruns the example.
| P7 nbconvert support2 | 2026-10-21 09:24:37 | 2026-10-21 09:26:28 | 111 | | logs/p7_nb_support2.log |
| P7 acceptance (revised projection) | 2026-10-21 09:26:58 | | | finish ~11:35: notebooks ~69 min + pytest ~1 min + nb02 ~31 min + example ~25-40 min | example now actually runs; baseline example 2355 s contended, calibration_curves unchanged at n=1821 (< 2000 cutoff) |
| P7 nbconvert compas | 2026-10-21 09:26:28 | 2026-10-21 09:28:17 | 109 | | logs/p7_nb_compas.log |
| P7 nbconvert german | 2026-10-21 09:28:17 | 2026-10-21 09:29:32 | 74 | | logs/p7_nb_german.log |
| P7 nbconvert heart_cleveland | 2026-10-21 09:29:32 | 2026-10-21 09:30:48 | 76 | | logs/p7_nb_heart_cleveland.log |
| P7 nbconvert oracle_sparse | 2026-10-21 09:30:48 | 2026-10-21 09:32:01 | 73 | | logs/p7_nb_oracle_sparse.log |
| P7 nbconvert oracle_calib | 2026-10-21 09:32:01 | 2026-10-21 09:38:11 | 370 | | logs/p7_nb_oracle_calib.log |
| P7 nbconvert adult | 2026-10-21 09:38:11 | 2026-10-21 09:49:23 | 672 | | logs/p7_nb_adult.log |
| P7 nbconvert bank | 2026-10-21 09:49:23 | 2026-10-21 10:00:13 | 649 | | logs/p7_nb_bank.log |
| P7 nbconvert diabetes130 | 2026-10-21 10:00:13 | 2026-10-21 10:31:17 | 1864 (rc=1) | | logs/p7_nb_diabetes130.log |
| P7 pytest -q | 2026-10-21 10:31:17 | 2026-10-21 10:32:23 | 65 | | 42 passed in 63.52s (0:01:03) |
| P7 diabetes130 rerun (projection) | 2026-10-21 10:32:24 | | | starts after the golden step (~11:35), ~31 min, done ~12:10 | P7 diabetes130 1864 s |
| P7 golden check (narrowed) | 2026-10-21 10:32:23 | 2026-10-21 11:24:47 | 3144 | | GOLDEN (narrowed): FAIL |
| P7 golden compare v2 (ECE column located by header; v1 missed 2 ECE cells in rows without a Brier CI) | 2026-10-21 11:25:59 | 2026-10-21 11:25:59 | 10 | | logs/p7_golden_narrow_v2.log: PASS |
| P7 nbconvert diabetes130 rerun (unseen-level rows dropped) | 2026-10-21 11:25:01 | 2026-10-21 12:01:16 | 2175 | | logs/p7_nb_diabetes130_rerun.log |
| E07/E08 check (projection) | 2026-10-21 12:25:32 | | | pytest ~1 min, golden ~52 min, bench 20k ~2 min, 50k ~10 min, 100k ~35 min: done ~14:00 | P7 golden 3134 s; Phase 1 after sweep with ICI/curve fits now once each |
| E07/E08 pytest -q | 2026-10-21 12:25:32 | 2026-10-21 12:26:40 | 68 | | 51 passed in 66.68s (0:01:06) |
| E07/E08 golden check (narrowed) | 2026-10-21 12:26:40 | 2026-10-21 13:19:46 | 3185 | | GOLDEN (narrowed): PASS |
| E07/E08 Phase 1 bench n=20000 | 2026-10-21 13:19:46 | 2026-10-21 13:21:04 | 78 | | phase1/results/after_e07e08_20000.json |
| E07/E08 Phase 1 bench n=50000 | 2026-10-21 13:21:04 | 2026-10-21 13:25:26 | 262 | | phase1/results/after_e07e08_50000.json |
| E07/E08 Phase 1 bench n=100000 | 2026-10-21 13:25:26 | 2026-10-21 13:39:49 | 863 | | phase1/results/after_e07e08_100000.json |
