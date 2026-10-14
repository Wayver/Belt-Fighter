# Belt Fighter — Comment Trim Tracker

Started: 2026-10-13 (post-11.x-smooth-clock, commit 8a14f33)
Goal: make comments short and functional. Kill session-history narrative
("Session 6.6", "10.3d handoff fix", "the 7.x netcode ran...") — that lives
in git history. Verify-or-kill factual claims (rates, tuple field counts).
Condense-and-keep the load-bearing invariants (authority boundaries, wire
shape, interpolation/handoff rules) to 1-2 lines each.

## Rules
1. STRICTLY BEHAVIOR-NEUTRAL: comments/docstrings only. No code changes.
   test_determinism must stay bit-identical; full suite must match baseline.
2. One file per commit. Run the test gates after each file.
3. Load-bearing invariants that would be lost get condensed into
   ARCHITECTURE.md (created before touching netcode.py/net.py).
4. Never span blank lines in patch search blocks (game.py has inconsistent
   blank lines with trailing whitespace).

## Baseline (pre-trim test results, 2026-10-13, commit 8a14f33)
All green: test_determinism (bit-identical ref), test_snapshot, test_render_model,
test_m2a_world, test_m2b_local, test_sim_thread, test_10_1..10_7, test_10_9 (its
exit=1 is the known pre-existing pose failures), test_10_10, test_10_12,
net self-test, test_network (final verdict ALL PASS; the 7.5b-battery
"client ghost fires local bullets" FAIL is the known load-flaky informational
check). Known pre-existing FAILs (NOT regressions): test_10_9_ghost_pose,
test_10_11c_phase_lock, test_interpolation "bullets — vacuous".
Run: cd /mnt && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m ship5.<test>

## File log
| file | status | commit | notes |
|------|--------|--------|-------|
| game.py | pending | | |
| __main__.py | pending | | |
| net.py | pending | | |
| netcode.py | pending | | |
| ship.py | pending | | |
| sim_thread.py | pending | | |
| config.py | pending | | |
| other small files | pending | | |

## Per-file notes
(Findings as I go: stale claims found, invariants preserved, etc.)