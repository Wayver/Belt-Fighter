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
| game.py | DONE | b46c276 | 3515→3126 lines (−389); all narrative markers gone; gates green |
| __main__.py | DONE | 3e2ee20 | 1161→1126 lines; all narrative markers gone; gates green |
| net.py | pending | | |
| netcode.py | pending | | |
| ship.py | pending | | |
| sim_thread.py | pending | | |
| config.py | pending | | |
| other small files | pending | | |

## game.py — DONE (commit b46c276, 2026-10-13)

3515 → 3126 lines (−389). All session-history narrative removed; the
load-bearing invariants condensed to 1-2 lines each. Gates green:
test_determinism bit-identical, test_snapshot lossless, test_render_model
ALL PASS. Invariants preserved (see the "Invariants to PRESERVE" list
below — still valid as the reference for what the trimmed comments now
say).

## NEXT: net.py (1479 lines, 318 comment lines)

Approach (same as game.py):
1. Grep for narrative markers:
   `Session [0-9]|10\.[0-9]+[a-g]?:|7\.[0-9]+[a-c]?:|6\.[0-9]+[a-c]?:|8\.[0-9]+[a-c]?:|9\.x|5b\.`
   (the 8.1 async-worker block at ~line 620 and the 10.10 input-echo
   block at ~line 97 are the biggest).
2. Condense the big blocks (the NetWorker async-worker architecture, the
   input-echo, the hull/loadout wire mapping → 2-4 lines keeping the
   invariant: worker owns the socket + JSON off the render thread; input
   applied by the worker via the sink; echo = the input the host ACTUALLY
   applied per tick).
3. Sweep inline markers.
4. Test gate: test_determinism + test_network + net self-test.
5. Commit + update this tracker.

NOTE: before netcode.py (the highest-value, highest-risk file), create
ARCHITECTURE.md extracting the surviving invariants (ghost clock,
reconcile/rewind, render point, interpolation buffer, handoff rules) so
the trimmed comments have a single current-state home.

**Invariants to PRESERVE in game.py (condense, don't delete):**
- RenderModel is plain-data; render thread reads ONLY the model, never live sim state.
- Ghost/stand-in pattern: presentation Ship synced from model per-frame, never stepped.
- Enemy stand-in gets shield flash + thruster flames; impact sparks / brownout arcs NOT yet in model.
- Bullet/missile handoff: ghost draws until buffer copy is visible, then decoupled
  render offset decays (MISSILE_HANDOFF_DECAY / BULLET_HANDOFF_DECAY).
- Remote beams are EVENTS (0.15 s flash), not snapshot state — drawn immediately,
  re-anchored to firing ship's interpolated pose.
- Per-player death: dead ships frozen, not targeted, respawn via _respawn_player.

## Per-file notes
(Findings as I go: stale claims found, invariants preserved, etc.)