# Plan 18 — Client-Side Weapon-vs-Asteroid Hit FX (2P: bullets/missiles stop at rocks)

**Status:** Session 1 (trigger + telemetry) DONE + VERIFIED (2026-10-21).
Session 2 (tests + docs) PENDING.
**Scope:** the CLIENT's presentation of the LOCAL player's weapons hitting
asteroids in 2P — the option-A hit-FX trigger in `predicted_view` (game.py)
gains a third target class (rocks), mirroring how it already handles remote
ships and AI enemies. The authoritative sim, the wire, the interpolation
buffer, the ghost's physics, and the reconcile are untouched. This is
presentation only: the authoritative hit, damage, and rock death still come
from the host snapshot.

**Session fit:** two focused sessions (trigger + telemetry, then tests +
docs), each comfortably inside a 64k context window — the same shape as
plan_17.

---

## 1. The problem

In 2P the client's option-A trigger (game.py ~L2885-3118) tests the local
player's bullets (ghost-drawn + buffer-drawn) against two target classes:
the other player's ship and AI enemies. On a hit it spawns a spark and
culls the bullet (`_hit_fx_cull_bids`) so it stops at the hull instead of
passing through. **Asteroids are not in the target list.** So when the
player shoots a rock:

- the host removes the bullet on the hit tick and bursts at the rock
  (`_collisions`, bullet-vs-asteroid, game.py ~L1960), but
- the client keeps drawing the bullet (ghost-drawn at the PRESENT,
  `D = _last_step_time - rp` ahead of the rock's drawn position) until the
  host's authoritative cull arrives with a snapshot ~100-200 ms later.

The bullet visibly **flies through the rock** (or past it) before
disappearing. Plan 17 made this worse to notice: the client now draws real
spinning rock polygons, so a bullet punching clean through a detailed rock
stands out.

The same artifact exists for the local player's **missiles** (ghost-drawn,
buffer copy deduped by id): the host detonates a missile on a rock
(`_collisions`, missile-vs-asteroid), the client keeps drawing the ghost
missile until the buffer copy arrives.

Remote-player and enemy projectiles do NOT have this artifact: they are
drawn from the buffer at the render point, the same time domain as the
rocks, so the host's cull (which arrives with the snapshot) makes them
stop at the rock's drawn position, one tick early at most. No work needed.

## 2. Design

### 2.1 The hit region: the host's circle, not the drawn polygon

The host's authoritative bullet/missile-vs-rock test is a **circle**
(`b.pos.distance_to(a.pos) < a.collision_radius`, radius =
`ROCK_SIZES[size]['radius']` — 50/28/15). The drawn polygon's vertex
radii are `R × U(0.72, 1.25)` and the polygon is **not necessarily
convex** (a 0.72 vertex between two 1.25 vertices is concave), so the
convex-polygon helpers in collision.py (`segment_hits_convex_poly`,
`point_in_convex_poly`) must NOT be used on raw rock verts.

Decision: **mirror the host's circle exactly** — the same principle the
enemy-shield fix pinned ("make the client's hit region match the host's
exactly"). Consequences:

- The client's cull decision agrees with the host's hit decision in
  (nearly) all cases, so the buffer copy's cull lines up with the
  authoritative cull and the rock's death/split arrives consistently.
- A polygon-accurate test would disagree with the host at the vertex tips
  (1.25R: client culls, host's bullet keeps flying → the client hides a
  bullet the host still shows for up to the cull TTL) and in the notches
  (0.72R: the reverse). The circle avoids both.
- Tunneling: the host's per-tick point test already cannot tunnel a rock
  (bullet step 860/60 ≈ 14.3 px < 2×15 px, the smallest diameter), so a
  swept segment-vs-circle on the client covers the same region.

**Deriving R on the client (no wire change):** the wire sends verts but
not `size`/`radius` (plan_17 pruned them; the wire is frozen). The vertex
radii are `R × U(0.72, 1.25)`, so estimate

```
R_est = (max_i |v_i| + min_i |v_i|) / 2
```

(max+min)/2 is a robust center-of-range estimator: E ≈ 0.986R with
~±3% sample error (≤ 1.5 px on a large rock) — small enough that the
residual client/host disagreement is a few px at the boundary, the same
tolerance class as the existing ship/enemy trigger.

### 2.2 The impact point: the circle ENTRY point

The existing `_seg_circle_hit` (game.py:312) returns the CLOSEST point on
the segment to the center — fine for ships (the segment ends at the hull),
but a bullet through a rock's center would spark at the center, deep
inside the drawn rock. The rock test instead solves the segment-circle
quadratic and takes the **first intersection (entry point)**, so the spark
lands on the rock's near surface — the same look as the host's beam path
(`_beam_hit_asteroid` bursts at `hit_pt` on the surface). The host's
bullet path bursts at `a.pos` (the center); the surface point is strictly
better visually and the difference is presentation-only.

### 2.3 What fires, where

New module-level helper in game.py (pure math, testable, no pygame):

```python
def _seg_circle_entry(x0, y0, x1, y1, cx, cy, r):
    """First intersection of segment p0->p1 with circle (cx, cy, r).
    Returns (hit, hx, hy) — the ENTRY point on the circle boundary, or
    (False, None, None). A segment starting inside the circle returns
    (True, p0). Pure math — mirrors _seg_circle_hit's role."""
```

In `predicted_view`, the option-A trigger gains:

1. **Rock target list** (built once per frame, alongside
   `_hit_fx_ships` / `_hit_fx_enemies`):
   `[(aid, x, y, R_est) for (aid, x, y, ang) in pos['asteroids']]`
   with `R_est` from the new registry (below); rocks with no registered
   shape are skipped (registry miss — should be impossible, same graceful
   degradation as the draw fallback). No viewport culling (the host
   doesn't cull; the test is a cheap quadratic — ~26 bullets × ~85 rocks
   ≈ 2k solves per firing frame, negligible).
2. **Rock pass** for bullets: AFTER the ships pass and the enemies pass
   (the host's `_collisions` order is enemies-then-rocks; a bullet is
   consumed by the first hit), for each not-yet-culled local bullet
   (ghost or buffer source, same segment the existing passes use):
   `_seg_circle_entry` vs each rock; on hit →
   - `_hit_fx_cull[bid] = _hit_fx_now` (the EXISTING cull set — the
     existing cull checks in the ghost-draw loop, the buffer-draw loop,
     and the fog-light loop all keep working unchanged),
   - `burst(self._hit_fx, pygame.Vector2(hx, hy), 20, rng=random)` —
     mirrors the host's `burst(self.particles, a.pos, a.radius, ...)`;
     NOTE the `radius` arg is currently UNUSED in `particles.burst`
     (count depends only on `big`), so any value renders identically —
     do not chase it,
   - `self._sfx("small_explosion")` (the host plays exactly this,
     unthrottled, on a bullet rock-hit),
   - `_hftel.log_hit(bid, src, "rock:%d" % aid, hx, hy, d)`.
3. **Rock pass** for ghost missiles: the ghost's `local_missiles`
   (`m.prev_pos -> m.pos` segment) vs the same rock list; on hit →
   - a NEW cull set `self._hit_fx_cull_mids` (mid -> wall time; TTL
     MISSILE_LIFE, pruned alongside the bullet cull at the top of the
     trigger block),
   - the same burst + SFX + telemetry,
   - the ghost-missile draw loop skips culled ids and the list is
     filtered after the loop (the exact pattern the bullet cull already
     uses at ~L3356), BEFORE the fog-light loop (so a culled missile
     gets no lingering glow — same ordering as the bullets).

### 2.4 R registry (client presentation state)

`Game.__init__` / `reset()`: `self._rock_coll_r = {}` (id -> R_est).
Populated in `push_snapshot` in the SAME loop that fills `rock_shapes`
(plan_17 seam, game.py ~L2366): when `verts is not None`,
`self._rock_coll_r[a[0]] = (max_r + min_r) / 2` over the vertex radii.
The `rock_shapes` entry shape is UNCHANGED (call sites + test_17
assertions stay as they are).

### 2.5 Telemetry (hit_fx_telemetry.py)

Extend the FRAME row with `n_rocks`, `min_d_rock`, `min_r_rock`,
`rock_hit` (same shape as the ship/enemy columns — the established
diagnostic pattern: one playtest shows why a spark did or didn't fire).
HIT rows already carry a free-text `tgt` label (`"rock:<id>"`). The
`log_frame` signature gains the rock args (default None — no other
callers).

## 3. Work breakdown

### Session 1 — trigger + telemetry (DONE + VERIFIED 2026-10-21)

1. **game.py**: `_seg_circle_entry` helper (next to `_seg_circle_hit`,
   ~L312) — DONE. Segment-circle quadratic, first (entry) root; a
   start-inside segment returns p0; a degenerate segment is a point test.
   Unit-checked (entry on the boundary, miss, start-inside, degenerate
   in/out, end-on-boundary).
2. **game.py**: `self._rock_coll_r = {}` in `__init__` + `reset()`;
   populate in `push_snapshot` (the plan_17 verts loop) — DONE.
   `R_est = (max|v| + min|v|) / 2` over the vertex radii.
3. **game.py**: `predicted_view` trigger — DONE:
   - rock target list (`_hit_fx_rocks`, from `pos['asteroids']` +
     `_rock_coll_r`; registry-miss rocks skipped),
   - bullet rock pass (AFTER the enemy pass, before `_hftel.log_frame` —
     the host's enemies-then-rocks order; on hit: existing
     `_hit_fx_cull[bid]` + `burst(..., 20)` + `small_explosion` SFX +
     `_hftel.log_hit(..., "rock:<id>")`),
   - ghost-missile rock pass (`_hit_fx_cull_mids`, TTL-pruned at
     MISSILE_LIFE alongside the bullet cull; the ghost-missile draw loop
     skips culled mids and the list is filtered after the loop, before
     the fog lights — the exact bullet-cull pattern).
   - **Deviation from the draft (missile cull coverage):** the draft only
     culled the GHOST missile. The buffer's copy of a culled missile would
     otherwise reappear (the dedup skip is keyed on `local_missile_ids`,
     which loses the id once the ghost missile is removed). So the cull
     also covers the buffer-draw loop (skip the buffer missile whose mid
     is in `_hit_fx_cull_mids`) and `_remote_fog_lights` (new
     `local_missile_cull` param, default `()` — no other callers). Same
     persistent-cull rationale as the bullets: the buffer copy stays
     culled until the host's authoritative detonation arrives.
4. **hit_fx_telemetry.py**: rock columns (`n_rocks`, `min_d_rock`,
   `min_r_rock`, `rock_hit`) + `log_frame` args (`n_rocks=0`,
   `min_rock=None` — backward compatible) — DONE. Columns inserted into
   the FRAME schema (no positional CSV analyzer exists for this file —
   checked tools/).

**Verification (2026-10-21):**
- Required suite green: `test_10_7_bullet_dedup`,
  `test_14_2_ghost_lockstep`, `test_14_3_missile_lockstep`,
  `test_10_3b_ghost_missiles`, `test_determinism`, `test_snapshot`,
  `test_net_process`, `test_render_model`, `test_17_rock_shapes` (the
  registry seam is touched — load-bearing), plus the full 10.x/15.x
  client-path set (`test_15_1..15_4`, `test_10_13`, `test_10_4`,
  `test_10_6`, `test_10_12`, `test_10_10`, `test_10_5`, `test_10_2`,
  `test_10_2b`, `test_10_1`, `test_10_3c`, `test_10_3d`) — all PASS.
- `test_interpolation`: 2 FAILs (`rewind zero-loss`, `rewind
  echo-driven`, 8.563px) — PRE-EXISTING (verified by stashing the
  changes: identical failures on the clean tree; a ghost-rewind issue in
  netcode, unrelated to this work).
- Known pre-existing failures unchanged (identical assertion messages):
  `test_10_3a_ghost_beams`, `test_10_9_ghost_pose`,
  `test_10_11c_phase_lock`, `test_network`.
- End-to-end smoke (throwaway, since the formal test is Session 2's
  scope): scratch client + one wire rock (stable id across a 3-snapshot
  stream) — registry R_est derived from the verts (27.52 for a medium
  rock, within [0.72R, 1.25R]); ghost bullet at the rock → burst(14) +
  cull + removed from local_bullets; buffer bullet crossing the rock →
  burst + cull; a rock 400 px beyond the segment → no spark, no cull;
  ghost missile at the rock → burst + cull + removed from
  local_missiles; CSV records a `src='buf'` `rock:<id>` HIT row + a
  FRAME row with `n_rocks=1`/`rock_hit=1`. ALL PASS.

### Session 2 — tests + docs

5. **New test** `test_18_rock_hits.py` (standalone `main()`, the
   test_15_4 / test_10_7 scratch-client scaffolding):
   - **HELPER**: `_seg_circle_entry` — entry point on the boundary
     (|p - c| ≈ r), a clean miss, a start-inside case, a degenerate
     (zero-length) segment.
   - **R_est**: a known vert set (e.g. radii {10, 20}) → 15.0; a real
     `Asteroid._make_rock` output → within [0.72R, 1.25R] of the true R.
   - **END-TO-END BULLET**: scratch client Game, wire 5-tuple rock
     (verts on first sight) pushed via `push_snapshot` ×3, ghost seeded
     with a local bullet aimed at the rock; run `predicted_view` until
     the trigger fires → assert: a burst spawned (`len(self._hit_fx) >
     0`), the spark is on the circle boundary (|impact - rock center|
     ≈ R_est ± 1 px), the bullet is culled (gone from
     `ghost.local_bullets`, not drawn).
   - **MISS**: a bullet aimed through the gap beside the rock → no cull,
     no burst, bullet still drawn.
   - **MISSILE**: a ghost missile on a rock course → burst + the missile
     removed from `local_missiles` + no fog light for it.
   - **TTL**: a cull entry older than BULLET_LIFE / MISSILE_LIFE is
     pruned (the sets are bounded).
6. **Docs**: ARCHITECTURE.md — one paragraph under "Ghost presentation
   entities & handoff" (or a new "Client hit FX" bullet): the option-A
   trigger now covers rocks with the host's circle region (R derived from
   the wire verts, (max+min)/2), entry-point impact, same cull set;
   presentation only. PERFORMANCE.md: no bandwidth change (wire
   untouched); per-frame cost note (~2k quadratic solves while firing).
7. **Manual 2P check** (playtest): fire the burst gun at rocks — bullets
   stop at the rock surface with a spark + small_explosion, no
   pass-through; missiles detonate on rocks; F3 + hit_fx_debug.csv shows
   `rock_hit` rows with `d` ≈ rock radius; sustained fire at a rock field
   shows no cull-set growth.

## 4. What is NOT changing (scope boundary)

- **The wire** — untouched (plan_17's format is frozen + verified; R is
  derived, not sent).
- **The authoritative sim, the ghost's PHYSICS, the reconcile, the
  interpolation buffer** — untouched. The ghost still does NO rock
  collision (plan_17 §4's deferred "ghost rock collision" stays deferred —
  a predicted clip-through is still corrected by the next reconcile; this
  plan only fixes what the player SEES, not where the ghost's ship flies).
- **Remote-player / enemy projectiles vs rocks** — already consistent at
  the render point (host cull arrives with the snapshot); no work.
- **Beams** — the host resolves beam-vs-rock before emitting T_BEAM (the
  endpoint is already on the rock surface); the client just draws the
  frozen endpoint. The local ghost's beam has no rock awareness (its
  laser target comes from enemy proxies only) — a ghost beam aimed at an
  enemy behind a rock draws through the rock for its 0.15 s flash;
  pre-existing, cosmetic, out of scope.
- **`rock_shapes` entry shape, `_draw_remote_rock`, viewport culling** —
  untouched (the new R registry is a separate dict).
- **`particles.burst`** — untouched (its unused `radius` arg stays).

## 5. Risks / edge cases

- **R_est error** (±~3%): the client's circle can differ from the host's
  by ≤ ~1.5 px on a large rock. A boundary graze can disagree one frame —
  the same tolerance class as the existing ship/enemy trigger; the
  authoritative outcome always comes from the host.
- **Time-domain mismatch**: the bullet is drawn at the PRESENT, the rock
  at the render point (D ≈ 0.1-0.35 s in the past) — the same accepted
  trade-off as the ship/enemy passes (impact error = the rock's
  displacement in D, 2-13 px for 20-130 px/s rocks vs radii 15-50 px).
- **Rock dies before the render point reaches it**: the client already
  burst + culled; the rock drops out of the buffer and the split children
  pop in with fresh ids + verts (plan_17 handles the verts). No extra
  state needed.
- **Registry miss** (no verts for an id): the rock is skipped by the
  trigger (no R_est) — degrades to today's behavior, never a crash.
- **Cull-set growth**: both cull sets are TTL-pruned every frame (the
  bullet set already is; the missile set mirrors it).
- **SFX volume**: sustained fire at a rock field plays small_explosion per
  hit — the host does exactly the same; if it is loud in the playtest,
  route it through `_sfx_throttled` (one-line change, flagged for the
  playtest rather than pre-decided).