# Plan 15 — Bullet Time-Domain Lock (the pass-through / early-disappear fix)

**Status:** BRIEF (Stage 1 only). Not started.
**Scope:** the client's rendering of the LOCAL player's own gun shots
(`predicted_view` + `_remote_fog_lights` in game.py). The authoritative
sim, the ghost's physics, the interpolation buffer, the reconcile, and the
dedup/handoff *mechanism* are all untouched. This is a pure
presentation-layer change: it moves WHERE the ghost's bullet is DRAWN, not
where it is SIMULATED.

**Session fit:** one focused edit cluster (one function + one helper + one
test update + one new test). Comfortably inside a single 64k session.

---

## 1. The problem

In 2P, the client's own bullets appear to **pass through** enemies or
**disappear before reaching them**. The authoritative sim is correct — the
collision happens at the right time on the host. The bug is purely visual:
the client draws three elements on three different time references.

| Element | Time reference | Code |
|---|---|---|
| Ghost's bullet | `_last_step_time` (the last tick boundary the ghost stepped) | game.py:2783, `b.pos` |
| Local ship | `_sim_time` (render time = host clock + rtt_ow) | game.py:2775, `render_pose_smoothed` |
| Enemy | `rp = newest_arrived_stamp − delay` (the render point) | game.py:2367 / 2400 |

When the host registers a hit, the client draws the **enemy at the render
point** (`rp`, which is `delay + rtt_ow` in the past) but the **bullet at
the ghost's `_last_step_time`** (≈ the present). The bullet leads the
enemy's drawn position by ≈ (relative closing speed) · (delay + rtt_ow) —
tens of px for a moving enemy. That is the pass-through / early-disappear.

The root cause is the same class as the two deferred bugs in plan_14 §5
("early enemy-bullet hit" and "ghost bullet spawn misalignment at rest"):
**render vs. sim time-domain mismatch.** This plan fixes the bullet-vs-enemy
case (the dominant, most-visible one).

---

## 2. The fix (age-based time offset — the standard technique)

Draw the ghost's bullet at the **render point**, not at the ghost's
`_last_step_time`. The bullet's stored position `b.pos` is at
`_last_step_time`; the enemy is drawn at `rp`. The exact gap is

```
D = _last_step_time − rp        (seconds; the render point is D in the past)
```

Both operands are already in scope in `predicted_view`:
`rp = self.render_point.now()` (game.py:2367) and
`self.ghost._last_step_time`. So `D = self.ghost._last_step_time − rp`.

The bullet's age is `a = BULLET_LIFE − b.life`. Its muzzle is
`b.pos − b.vel·a`. The bullet's true position at the render point is
`b.pos − b.vel·D`, clamped so it never renders behind the muzzle (a bullet
younger than `D` has not left the muzzle yet at the render point):

```
t_off      = min(a, max(D, 0.0))
render_pos = b.pos − b.vel · t_off
```

- `a < D`  → `t_off = a` → `render_pos = b.pos − b.vel·a = muzzle` (frozen
  at the muzzle — the bullet really had not left yet at the render point).
- `a ≥ D`  → `t_off = D` → `render_pos = b.pos − b.vel·D` (flying at full
  speed, phase-locked to the enemy's time domain).

This is the standard commercial technique for local projectiles against
interpolated remotes: the local projectile is drawn in the remote's past
time domain so it strikes the enemy exactly where the player sees it.

**Two corrections vs. the original idea (both load-bearing):**

1. **Use the exact gap `D = _last_step_time − rp`, not `delay` (or
   `INTERP_DELAY`).** `delay` misses `rtt_ow` and the `_acc` phase, leaving
   a residual of `rtt_ow ± _acc` (a few px on a LAN, more on a bad
   connection). The exact gap is what guarantees the bullet crosses the
   enemy's drawn position on the same frame the enemy dies on screen.
2. **Remove the now-redundant handoff pull-forward.** The existing
   pull-forward (`x += vx·delay` for local bullets) exists to close the gap
   between where the ghost drew the bullet (the present) and where the
   buffer's copy sits (the render point). With this fix the ghost draws the
   bullet *at* the render point, so that gap is zero and the handoff is
   already seamless. If the pull-forward is kept, the buffer's copy jumps
   *forward* by `V·delay` on the handoff frame. Delete it in both places.

**The cost (flag for the playtest):** the bullet is frozen at the muzzle for
`D` seconds (~0.1–0.35 s). For the burst weapon (6 shots at 0.049 s apart),
the first ~2 shots pile up at the muzzle (stationary ship) or trail along
the recent path (moving ship), then fly off as a stream. That is the
standard cost of the technique and is acceptable, but it is the one thing to
judge by feel in the playtest.

**Known minor trade-off:** the old pull-forward made the ghost's bullet and
the buffer's copy coincide *exactly* at handoff. Without it, they coincide
to within the ghost-vs-host integration difference (~0 in steady state —
the ghost is in lockstep with the host after plan_14), so the handoff is
seamless in steady state and carries at most a small (reconcile-scale) snap
during a hiccup. Acceptable; the playtest judges it.

---

## 3. Exact changes (file by file)

### 3.0 `game.py` — add `BULLET_LIFE` to the config import (game.py:14)

`BULLET_LIFE` is **not** currently imported in game.py (confirmed). Add it to
the first `from .config import (...)` block (game.py:14, the one that already
carries `BULLET_SPEED` / `BULLET_COLOR`):

```python
from .config import (WIDTH, HEIGHT, SPAWN_PROTECT, BULLET_POOL_MAX,
                    BULLET_SPEED, BULLET_LIFE,
                    ROCK_SPLIT, ROCK_SIZES, BG, STAR_COLOR,
                    ...
```

### 3.1 `game.py` — `predicted_view`, the ghost bullet draw (game.py:2783-2785)

**BEFORE:**
```python
        for b in self.ghost.local_bullets:
            s = self.cam.to_screen(b.pos)
            pygame.draw.circle(screen, BULLET_COLOR, (int(s.x), int(s.y)), 3)
```

**AFTER:**
```python
        # Draw the ghost's OWN gun shots at the RENDER POINT (rp), not at
        # the ghost's _last_step_time — phase-locked to the enemy's time
        # domain so the bullet strikes the enemy where it is DRAWN (the
        # pass-through / early-disappear fix, plan_15). D is the exact gap
        # between the bullet's stored time (_last_step_time) and the render
        # point; t_off clamps so a bullet younger than D renders at the
        # muzzle (it had not left yet at the render point).
        _D = self.ghost._last_step_time - rp
        for b in self.ghost.local_bullets:
            _a = BULLET_LIFE - b.life
            _t_off = min(_a, max(_D, 0.0))
            _rx = b.pos.x - b.vel.x * _t_off
            _ry = b.pos.y - b.vel.y * _t_off
            s = self.cam.to_screen(pygame.Vector2(_rx, _ry))
            pygame.draw.circle(screen, BULLET_COLOR, (int(s.x), int(s.y)), 3)
```

(`BULLET_LIFE` is added in 3.0. `rp` is in scope from game.py:2367. Hoist
the `_D = ...` line to just before this loop so 3.2's fog light shares it.)

### 3.2 `game.py` — `predicted_view`, the ghost bullet fog light (game.py:2824)

The ghost bullet's fog light is added in `predicted_view` (NOT in
`_remote_fog_lights`). Offset it by the same amount so the glow follows the
sprite. **BEFORE:**
```python
        for b in self.ghost.local_bullets:
            lights.append(LightSource(b.pos, 30, 0.5))
```
**AFTER:**
```python
        for b in self.ghost.local_bullets:
            _a = BULLET_LIFE - b.life
            _t_off = min(_a, max(_D, 0.0))
            lights.append(LightSource(
                pygame.Vector2(b.pos.x - b.vel.x * _t_off,
                               b.pos.y - b.vel.y * _t_off), 30, 0.5))
```
(`_D` is computed in 3.1, in the same function scope — hoist the
`_D = self.ghost._last_step_time - rp` line to just before the bullet-draw
loop so both 3.1 and 3.2 share it.)

### 3.3 `game.py` — `predicted_view`, REMOVE the local-bullet pull-forward (game.py:2664)

In the buffer-bullet draw loop, delete the local-bullet pull-forward so the
handed-off buffer copy is drawn at its raw interpolated position (the ghost
already drew it at the render point, so the handoff is seamless). **BEFORE:**
```python
                _buf_drawn.append(_bid_t)
                _buf_bdrawn.append(_bid_t)
                if _bid_t[0] == self.local_index:
                    x += vx * _bdelay
                    y += vy * _bdelay
```
**AFTER:**
```python
                _buf_drawn.append(_bid_t)
                _buf_bdrawn.append(_bid_t)
                # (plan_15: the local-bullet pull-forward is GONE — the
                # ghost now draws its own bullets at the render point, so
                # the buffer's handed-off copy is already seamless here.)
```
Leave `_bdelay = self.latency.delay` (game.py:2556) in place — it is still
passed to `_remote_fog_lights` (harmless once 3.4 removes its only use) and
removing the assignment is a separate, optional cleanup.

### 3.4 `game.py` — `_remote_fog_lights`, REMOVE the local-bullet pull-forward (game.py:3055)

Same deletion in the fog-light helper. **BEFORE:**
```python
                if _bid_t in local_bullet_ids:
                    continue   # glows from the ghost's predicted pos
                if _bid_t[0] == local_index:
                    x += vx * delay
                    y += vy * delay
```
**AFTER:**
```python
                if _bid_t in local_bullet_ids:
                    continue   # glows from the ghost's predicted pos
                # (plan_15: local-bullet pull-forward removed — see 3.3.)
```
The `delay` / `local_index` parameters become unused for the local-bullet
pull (they are still part of the signature; leave them — other callers and
the missile path are unaffected). Update the docstring line that describes
the local-bullet pull-forward to note it is removed.

### 3.5 `test_10_7_bullet_dedup.py` — update the PULL-FORWARD assertion (~396-408)

The existing test asserts the handed-off bullet is drawn **pulled forward**
at `raw + V*delay`. That behavior is removed. Update the final section
("PULL-FORWARD") to assert the handed-off bullet is drawn at its **raw
interpolated buffer position** (no pull):

- Replace `_ex, _ey = _raw[0] + _raw[2]*_delay, _raw[1] + _raw[3]*_delay`
  with `_ex, _ey = _raw[0], _raw[1]`.
- Replace the `bullet_px(screen, g2.cam, _ex, _ey) > 0` assertion's message
  ("DRAWN PULLED FORWARD at ... raw + V*delay") with "DRAWN at its raw
  interpolated position (no pull-forward — plan_15)".
- DELETE the `_ex > _raw[0]` "forward-only" assertion (it no longer holds —
  the position IS the raw position).
- Update the final `print("ALL PASS: 10.7 ...")` string to drop
  "closed-form pull-forward handoff" → "seamless render-point handoff".

Everything else in that test (the dedup, the pending-mark, the
wait-until-advancing handoff, the removal from `local_bullets`) is
unchanged and must still pass.

### 3.6 NEW test — `test_15_1_bullet_time_lock.py`

A focused, standalone `main()` script (same style as `test_10_7_bullet_dedup`)
that proves the time-domain lock. Three checks:

1. **OFFSET FORMULA.** Build a scratch client `Game` with a seeded ghost and
   a single local bullet at a known `pos`/`vel`/age. Drive
   `predicted_view` for one frame. Capture the rendered bullet pixel and
   assert it equals `cam.to_screen(b.pos − b.vel·min(age, D))` where
   `D = ghost._last_step_time − render_point.now()` (recompute `D` in the
   test the same way). Tolerance: 1 px (camera rounding).
2. **FROZEN AT MUZZLE.** A fresh bullet (`age < D`) renders at the muzzle
   (`b.pos − b.vel·age`), i.e. it has not left the muzzle at the render
   point. Assert the rendered pixel is at the muzzle, not ahead of it.
3. **NO PULL-FORWARD AT HANDOFF.** Reuse the `test_10_7` gate-5 setup
   (a pending local bullet whose buffer copy has advanced). After
   `predicted_view`, assert the buffer's copy is drawn at its **raw**
   interpolated position (NOT `raw + V*delay`) — the handoff is seamless
   because the ghost drew at the render point.

The test needs the same scratch-game scaffolding `test_10_7` uses
(`push_snapshot` ×3, `latency.tick`, `render_point.advance`, a `bullet_px`
helper that counts `BULLET_COLOR` pixels near a world point). Reuse
`test_10_7`'s helpers by import or copy.

---

## 4. What is NOT changing (scope boundary)

- The authoritative sim, the ghost's **physics** (`b.pos` is still stepped
  at `_last_step_time` — only the DRAW is offset), the reconcile, and the
  ghost clock — untouched.
- The interpolation buffer, `RenderPoint`, `interp_positions` — untouched.
- The **dedup** (skip the buffer's copy of a ghost-drawn bullet) — untouched.
- The **handoff mechanism** (pending-mark, wait-until-advancing, removal
  from `local_bullets`) — untouched; only the pull-forward *offset* is
  removed.
- **Missiles** — untouched (they have their own offset/decay handoff; this
  plan is bullets only). Revisit missiles only if the playtest shows the
  same pass-through for them.
- **Enemy bullets** — untouched (not player-fired; always drawn from the
  buffer at the render point, already phase-locked).
- The **remote player's** bullets — untouched (drawn at their raw
  interpolated position; they are not predicted by the local ghost).
- The deferred "ghost bullet spawn misalignment at rest" (plan_14 §5) —
  **still deferred.** It is a `render_pose_smoothed` lag (the smoothed pose
  trails the physics pose after a hard stop), a different mechanism. If it
  is still visible after this plan, it is a separate small change.

---

## 5. Verification checklist

- [ ] `test_15_1_bullet_time_lock` (new) PASS — offset formula, frozen-at-
      muzzle, no pull-forward at handoff.
- [ ] `test_10_7_bullet_dedup` PASS (updated PULL-FORWARD section → raw
      position; all other gates unchanged).
- [ ] Full required suite PASS: `test_10_7_bullet_dedup`,
      `test_14_2_ghost_lockstep`, `test_14_3_missile_lockstep`,
      `test_10_3b_ghost_missiles`, `test_interpolation`, `test_determinism`,
      `test_snapshot`, `test_net_process`, `test_render_model`. (Known
      pre-existing failures unchanged: `test_10_3a_ghost_beams`,
      `test_10_9_ghost_pose`, `test_10_11c_phase_lock`, `test_network`.)
- [ ] Live 2P playtest (user): fire at a **moving** enemy — the bullet
      strikes the enemy where it is drawn (no pass-through, no early
      disappear). Judge the muzzle-freeze feel (the first ~2 burst shots
      pile up at the muzzle before flying off). F3 telemetry: no new
      double-draw, `snap_px` / `render_snap_px` still small.

If the playtest confirms the strike alignment and the muzzle-freeze feel is
acceptable, Stage 1 is complete. The deferred hard-stop spawn bug and the
missile pass-through (if any) are the follow-ups.