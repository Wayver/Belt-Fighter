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
| netcode.py | DONE | a3a2d92 | 2540→2286 lines (−254); AST-verified code-identical; all 69 markers gone; gates green |
| ship.py | DONE | acbba3c | 1138→1107 lines (−31); AST-verified code-identical; all markers gone; gates green |
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

## netcode.py — DONE (commit a3a2d92, 2026-10-14)

2540 → 2286 lines (−254). ARCHITECTURE.md created first (commit a818714)
as the single current-state home for the netcode invariants (topology &
authority, wire, interpolation buffer, render point, adaptive delay, host
clock estimate, ghost clock contract, reconcile/rewind, render pose,
ghost presentation entities & handoff). All 69 session-history markers
killed (Session 5b/6.8/7.x/8.x, 10.x, 11.x). Stale claims fixed:
_match_bullets "10 Hz cadence / 86 px" → 60 Hz / ~14 px; advance()
branch-2 docstring `now()` → `now_smooth()` (the smooth estimate is what
feeds the ghost's clock); reconcile_rewind "STEP" → TICK; removed the
dead "Known limitation (v1): reconciliation is a FULL SNAP" paragraph
(reconcile_rewind is the live path) and the "10.16 pre-step REMOVED
(10.17)" narrative block. Big blocks (PredictedShip class docstring,
reconcile_rewind, handback_missiles, advance GHOST CLOCK CONTRACT,
render_pose, _render_pre_step) condensed to invariants pointing at
ARCHITECTURE.md. BEHAVIOR-NEUTRAL PROVEN: AST dump with comments +
docstrings stripped is byte-identical to HEAD. Gates: test_determinism
bit-identical (exit 0), all test_10_x ALL PASS (the 3 known pre-existing
FAILs unchanged: test_interpolation "bullets — vacuous",
test_10_9_ghost_pose, test_10_11c_phase_lock), net self-test ALL PASS,
test_network ALL PASS.

## ship.py — DONE (commit acbba3c, 2026-10-14)

1138 → 1107 lines (−31). All session-history markers killed (10.1,
10.3b, 10.3c, 10.6, 10.7, Phase 2, For now, Before 10.3c, "unchanged",
"promotes"). Stale claims fixed: `flame_mags = {} # presentation only —
never serialized` (it IS synced now), "the enemy's Ship snapshot is the
same 24-tuple" (now 25 — verified via AST), arc-interval comment
"0.25s -> 0.10s" (the formula 0.45 - 0.15*severity gives 0.45s -> 0.30s),
_allocate pipeline line "For now: always 1.0" (it resolves the
power/compute budgets now), _fire docstring "The cooldown advances only
for shots that actually fire" (contradicted the code — cooldown always
ticks down; firing resets it; also dropped the stale "capped by the
caller" — no caller caps shots), update() docstring stray leading "...".
Invariants condensed in place (nothing moved to ARCHITECTURE.md —
ship.py is sim-side): raw unbounded angle round-trip (snapshot +
classification), seq-counter dedup key (missile_seq/bullet_seq merged
into one entry), power_used sensor-signature rationale, flame_mags
sync rationale, the full SYNCED vs PRESENTATION-ONLY classification.
BEHAVIOR-NEUTRAL PROVEN: AST dump with comments + docstrings stripped
is byte-identical to HEAD. Gates: test_determinism bit-identical (exit
0), test_m2a_world / test_m2b_local / test_sim_thread ALL PASS,
test_10_x spot-check matches baseline (the 3 known pre-existing FAILs
unchanged: test_10_9_ghost_pose, test_10_11c_phase_lock,
test_interpolation "bullets — vacuous").

## NEXT: sim_thread.py

1. Grep for narrative markers (same pattern).
2. Condense big blocks; invariants that would be lost go to
   ARCHITECTURE.md (append a section) if they're netcode-side, else
   stay in-file.
3. Sweep inline markers.
4. Test gate: test_determinism (bit-identical) + test_sim_thread +
   test_m2a_world / test_m2b_local + spot-check test_10_x.
5. Commit + update this tracker.

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

### ship.py
- Stale claims found & fixed: `flame_mags = {} # presentation only —
  never serialized` (flame_mags is in the snapshot — synced); "the
  enemy's Ship snapshot is the same 24-tuple" (25 fields now —
  bullet_seq added; verified by counting the snapshot() return tuple
  via AST); `_update_arcs` interval comment "0.25s -> 0.10s as it
  deepens" (0.45 - 0.15*severity is 0.45s -> 0.30s); module docstring
  pipeline line "_allocate ... For now: always 1.0" (it resolves the
  power/compute budgets); _fire docstring "The cooldown advances only
  for shots that actually fire, so a ship that can't fire (capped by
  the caller, or browned out) fires the instant it can" (cooldown
  always ticks down unconditionally; firing resets it to
  fire_cooldown; "capped by the caller" has no caller — dropped).
- Invariants preserved (condensed, in-file): raw unbounded angle
  round-trip (snapshot() comment + SYNCED classification), the
  missile_seq/bullet_seq dedup-key rationale (merged into one
  classification entry + one __init__ comment), power_used as the
  client-side sensor-contact signature (not derivable client-side;
  power_idle_total is a loadout constant), flame_mags sync rationale
  (client draws remote player + enemy exhaust; recomputed every tick —
  always the current frame's exhaust), the whole SYNCED vs
  PRESENTATION-ONLY classification (kept, de-tagged).
- Trailing-whitespace lines in ship.py: 136, 331, 563, 962, 974, 979
  (pre-trim line numbers) — none were spanned in a search block.

### netcode.py
- Stale claims found & fixed: _match_bullets "860 * 0.1 = 86 px at the
  10 Hz cadence" (now 60 Hz → ~14 px/window); advance() branch-2
  docstring said `host_time.now(real_time)` but the code uses
  `now_smooth` (fixed the docstring, not the code); reconcile_rewind
  "snap_time + STEP" → TICK (code uses TICK); SnapshotBuffer "16
  snapshots / 30 Hz" → 24 / 60 Hz (done in the pre-commit session).
- Dead narrative removed: "Known limitation (v1): reconciliation is a
  FULL SNAP ... dead-reckoning rewind is OUT of scope" (reconcile_rewind
  IS the live path now); "10.16 pre-step REMOVED (10.17)" block (pure
  history — the surviving invariant, pose and clock agree, was already
  stated above it).
- Invariants preserved (condensed, pointing at ARCHITECTURE.md): ghost
  clock contract (3 branches, now_smooth + rtt_ow), reconcile/rewind
  (echo-first input source, n = max(0, ...), _sim_time = max(now,
  snap_time + n*TICK)), render pose (kinematic pre-step, alpha = L/TICK,
  continuous across reconcile), handback (seq < host_seq AND buffer
  carries the id; ghost keeps drawing pending; handoff at visibility),
  re-fire guards (bullet + missile id dedup), sensor-state capture/
  restore (client_sensor_authoritative), diag_sink class-attribute
  descriptor gotcha.
- Verified claims kept: ship snapshot field indices (missile_seq=21,
  bullet_seq=24, dead=20, brownout=16, flame_mags=23, power_factor=17,
  shield_dump=6, shield_clock=7, power_used=22), missile snapshot id at
  field 8, player-bullet id at field 6, INPUT_BUFFER_MAX 2.0 s,
  BEAM_TTL 0.15 s, MATCH_RADIUS 30 px.

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