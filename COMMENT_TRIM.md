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
| game.py | IN PROGRESS (uncommitted) | — | 3515→3233 lines done; 95 narrative-marker lines remain (see below) |
| __main__.py | pending | | |
| net.py | pending | | |
| netcode.py | pending | | |
| ship.py | pending | | |
| sim_thread.py | pending | | |
| config.py | pending | | |
| other small files | pending | | |

## game.py — RESUME POINT (2026-10-13, uncommitted in working tree)

**Done so far (all in working tree, NOT yet committed):**
- Top-of-file narrative blocks (M2a/M2b render helpers, ~37+22 lines each) → condensed to 5-7 lines.
- Snapshot section header (line ~877) → condensed.
- Various inline "Session X" / "10.x" references in class docstrings and
  method bodies → shortened or removed.
- Net effect: 337 insertions, 619 deletions (−282 lines).

**Still remaining (95 lines with narrative markers):**
The bulk is in the `draw()` method (lines ~2300–2750) — the render section:
- 37-line block @ ~2634: "Remote ships (Session 6.6, hulls drawn in 6.8)..."
- 32-line block @ ~2374: "Remote projectiles (Session 7.2, the D3 fix)..."
- 21-line block @ ~2693: "10.3a: the player's OWN laser beams..."
- 20-line block @ ~2495: "10.3d: the decoupled render offset (the handback fix)..."
- 20-line block @ ~2408: "10.3g: hand off only when the buffer's copy is MOVING..."
- 16-line block @ ~2451: "10.7: BULLET HANDOFF..."
- 16-line block @ ~2328: "The REMOTE player's laser beams — received as T_BEAM events..."
- Plus ~30 scattered inline "Session 6.x" / "10.x" one-liners in
  docstrings and method bodies (lines 58–1533).

**To finish game.py:**
1. Condense the 7 big blocks in draw() (each → 2-4 lines keeping the
   invariant, dropping the session narrative).
2. Sweep the remaining ~30 inline markers (grep pattern:
   `Session [0-9]|10\.[0-9]+[a-g]?:|7\.[0-9]+[a-c]?:|6\.[0-9]+[a-c]?:|8\.[0-9]+[a-c]?:|9\.x`).
3. Run the test gate:
   `cd /mnt && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m ship5.test_determinism`
   (must be bit-identical) + spot-check test_snapshot, test_render_model.
4. Commit: `git add game.py && git commit -m "comment-trim: game.py (3515→~3100 lines)"`
5. Update this tracker's file log row.

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