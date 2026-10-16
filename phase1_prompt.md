# Phase 1 — Host Fire Path (plan_14)

You are continuing the bullet-cap foundation fix for Belt Fighter (2P
networked). **Read `plan_14.md` first** — it is the source of truth. This
prompt is the Phase 1 brief; the plan has the full context, design, and
verification checklist.

## Where we are

- **Phase 0 (measure) is DONE and committed.** The telemetry
  (`bullet_telemetry.py` → `bullet_debug.csv`, F3-gated, client-only) +
  `tools/analyze_bullet_drift.py` are in place. A 63 s 2P capture
  **confirmed the drift**: the ghost's `bullet_seq` runs exactly **+1** ahead
  of the host's (bounded — resynced each reconcile, then the ghost fires one
  more), and **554 host-never-fired bullets** were captured (always the ghost's
  "13th", `1:12` when the host's seq is 12). The root cause is the host's
  WORLD bullet cap (`len(self.bullets) >= MAX_BULLETS`) suppressing fire in a
  way the ghost (which gates on its LOCAL-DRAW count) does not reproduce.
- **Phase 1 (this session): change the HOST fire path only.** Do NOT touch the
  ghost yet (that's Phase 2). Do NOT change `FIRE_COOLDOWN` or add
  `BULLET_POOL_MAX` yet (that's Phase 4, balance).

## The change (host fire path, `game.py` `Game._step`)

Make the host's fire decision **input-local** (drop the world-dependent gate)
and add a **cull-oldest performance safety net** that never binds in normal
play.

1. **Remove the two world fire gates** in `Game._step` (search for
   `MAX_BULLETS` / `MAX_MISSILES` in `game.py`):
   ```python
   if p_inp.fire and len(self.bullets) >= MAX_BULLETS:
       p_inp = replace(p_inp, fire=False)
   if p_inp.missile_fire and len(self.missiles) >= MAX_MISSILES:
       p_inp = replace(p_inp, missile_fire=False)
   ```
   Delete both. The fire decision then reduces to `Ship._fire` /
   `_update_missiles`, which are already input-local (cooldown + power).

2. **Add the cull-oldest safety net** after the bullet/missile append loops in
   `Game._step`. The lists are in fire order (newest at the back), so
   `pop(0)` is the oldest:
   ```python
   while len(self.bullets) > BULLET_POOL_MAX:
       self.bullets.pop(0)
   while len(self.missiles) > MISSILE_POOL_MAX:
       self.missiles.pop(0)
   ```
   Introduce `BULLET_POOL_MAX` / `MISSILE_POOL_MAX` in `config.py` as a
   **temporary** safety net sized well above the natural count so it never
   binds in normal play (natural = 90/player bullets, 180 world at the current
   60/s rate; use e.g. `BULLET_POOL_MAX = 400`, `MISSILE_POOL_MAX = 12`).
   These get finalized in Phase 4 when the fire rate is chosen. The point now
   is just: the fire decision no longer depends on the pool.

3. **Leave the input-echo block as-is.** It echoes the post-gate `p_inp`; with
   the gate gone, `p_inp` is the raw input — which is now exactly what the host
   applies, so the echo is still correct.

## Key facts (verified — don't re-derive)

- **Fire rate is 60/s, not 100/s.** `FIRE_COOLDOWN` (0.01) < `TICK` (0.0167),
  so the gun fires once per tick. Natural live bullets = 90/player, 180 world.
- **The gun cooldown can go NEGATIVE.** In `Ship._fire`, `w.cooldown -= dt`
  runs unconditionally and only resets to `fire_cooldown` when the gun fires.
  A ready-but-suppressed gun runs its cooldown negative. This is existing
  behavior — don't "fix" it; it's why the Phase 0 cooldown-delta signal showed
  large negative host values.
- **`bullet_seq` / `missile_seq`** are per-ship counters, synced via the ship
  snapshot (fields 24 / 21). The host bumps them at fire; the ghost resyncs
  them on reconcile. The id `(player_index, seq)` is the dedup key.
- **The ghost is NOT changed in this phase.** After Phase 1 the host fires
  every shot its input-local decision allows; the ghost (still gated on its
  local-draw count) will now fire *fewer* than the host in some cases — that's
  expected and is what Phase 2 fixes. Don't be alarmed by a temporary
  asymmetry; Phase 1's tests are single-player + determinism, not 2P lockstep.

## Tests to write / run (Phase 1 scope)

1. **Single-player sustained fire:** hold fire for N ticks, assert the
   sustained rate is `min(1/FIRE_COOLDOWN, 60)` shots/s (no
   burst-then-dead-zone), and the live bullet count grows to the natural ~90
   (not capped at 12). A new test or an extension of an existing single-player
   test.
2. **Determinism:** run `test_determinism` — the host must stay deterministic
   (the fire decision + cull are pure functions of state). This is the critical
   regression guard for this phase.
3. **Snapshot round-trip:** `test_snapshot` still passes (the bullet list is
   longer now; the snapshot/apply path must handle it).
4. **Full existing suite:** `test_interpolation`, `test_10_7_bullet_dedup`,
   `test_10_3b_ghost_missiles`, `test_net_process` — all must still pass
   (they exercise the host fire path indirectly).

Run tests from `/mnt` (the package is `ship5`, a symlink to the workspace):
`cd /mnt && python -m ship5.test_determinism` etc.

## Definition of done (Phase 1)

- [ ] The two world fire gates are removed from `Game._step`.
- [ ] The cull-oldest safety net is in place (temporary `BULLET_POOL_MAX` /
      `MISSILE_POOL_MAX` in `config.py`).
- [ ] Single-player sustained fire = 60/s, no burst-then-dead-zone, pool grows
      to ~90 (not 12).
- [ ] `test_determinism`, `test_snapshot`, `test_interpolation`,
      `test_10_7_bullet_dedup`, `test_10_3b_ghost_missiles`, `test_net_process`
      all pass.
- [ ] Committed with a clear message.

## Explicitly OUT of scope for Phase 1

- The ghost fire path (Phase 2).
- `FIRE_COOLDOWN` / damage tuning / final `BULLET_POOL_MAX` (Phase 4).
- The "early enemy-bullet hit" presentation bug (separate effort).
- The 2P lockstep / no-double-draw test (Phase 2 — it can't pass until the
  ghost is fixed).

## After Phase 1

Phase 2 (ghost fire path) is the natural next session: remove the ghost's
local-draw gate in `PredictedShip.step` (`netcode.py`), add the matching cull,
and add the 2P lockstep test (ghost `bullet_seq` + gun cooldown == host's
across N reconciles; no ghost-only id ever drawn). Then re-run
`tools/analyze_bullet_drift.py` on a fresh 2P capture to confirm the seq gap
stays 0 and the true-extra count drops to 0.