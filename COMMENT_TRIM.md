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
| net.py | DONE | 19a2f35 | 1479→1458 lines; all narrative markers gone; gates green (test_determinism, net self-test 18 PASS, test_network ALL PASS) |
| netcode.py | pending | | ARCHITECTURE.md FIRST (rule 3) |
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

## net.py — DONE (commit 19a2f35, 2026-10-14)

1479 → 1458 lines (−21). All session-history narrative removed (Session
6.3/6.5/6.7/6.9/7.11/8.1, 10.1/10.4/10.10/10.11/10.11b/10.11d/10.12/10.13,
the 7.8 log ref). Stale factual claims fixed while in there: "30 Hz
snapshots" → 60 Hz (SNAPSHOT_INTERVAL=1 tick), "no threads" → NetWorker
owns the socket, "Four message types" → all 8 T_* tags listed, "~366 KB/s"
→ ~370 KB/s (measured: 6422 B wire × 60 Hz ≈ 376 KB/s), "7.3 KB MT state"
verified (json.dumps = 7340 B). Gates green: test_determinism exit=0,
net self-test 18 PASS / 0 FAIL, test_network ALL PASS (the 7.5b
informational FAIL is the known load-flaky one).

## NEXT: netcode.py (~1700+ lines) — CREATE ARCHITECTURE.md FIRST

Rule 3: before touching netcode.py, create ARCHITECTURE.md extracting the
surviving invariants (ghost clock, reconcile/rewind, render point,
interpolation buffer, handoff rules) so the trimmed comments have a single
current-state home. Then:
1. Grep for narrative markers (same pattern as before).
2. Condense the big blocks (reconcile_rewind, PredictedShip,
   SnapshotBuffer, interp_positions) to 1-2 lines each, pointing at
   ARCHITECTURE.md for the full contract.
3. Sweep inline markers.
4. Test gate: test_determinism + test_network + test_interpolation +
   test_10_1..10_12 (the netcode tests).
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

### net.py
- Stale claims found & fixed: "30 Hz snapshots" (now 60 Hz, per-step
  publish), "no threads, no shared state" (NetWorker owns the socket),
  "Four message types" (8 T_* tags now), "~366 KB/s" (measured ~376).
- Verified claims kept: 7.3 KB MT state (7340 B json.dumps), 11-tuple
  snapshot (indices 5/6/9/10 all correct), MAX_SPEED 520 px/s,
  SNAPSHOT_BUFFER_MAX 24 brackets INTERP_DELAY_MAX 0.35 s at 60 Hz.
- "pinned #5" references KEPT — they point at the durable pinned-plan
  note (user note "Belt Fighter — Network Transport Plan"), not a
  session.
- Invariants preserved (condensed): worker owns socket+JSON, main thread
  only queues/polls; drain-to-newest snapshot queue; input sink runs on
  the worker thread (GIL-atomic ref swap, no pygame); pong reflected on
  the worker thread to keep RTT honest; partial-frame send keeps the
  UNSENT remainder; resync past corrupted regions.