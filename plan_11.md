# Belt Fighter 11.x — Smooth 2P (eliminate the local-ship snap/jitter)

Scope: the LOCAL player's ship on the client (the prediction ghost). Remote
entities (other ship, enemies, asteroids, projectiles) already render
smoothly via the interpolation buffer + render point; they are out of scope.

Everything below is derived from the CURRENT code (config.py, net.py,
sim_thread.py, __main__.py, game.py, netcode.py) and the fresh
`net_debug.csv` (2591 reconciles). No reliance on older sessions.

---

## A. Current netcode (from code only)

### A.1 Constants (config.py)
- `TICK = 1/60` s (16.67 ms) — the fixed sim step.
- `SNAPSHOT_INTERVAL = 1` tick — the host publishes a snapshot every tick
  (60 Hz).
- `INTERP_DELAY = 0.1` s (base render lag); `INTERP_DELAY_MIN = 0.1`,
  `INTERP_DELAY_MAX = 0.35`, `ADAPT_K = 2.0`.
- `SNAPSHOT_BUFFER_MAX = 24`.
- `MAX_FRAME_DT = 0.25`.

### A.2 Host (sim_thread.py, net.py NetWorker)
- The sim thread runs a fixed-step loop at 60 Hz with a dynamic sleep to the
  next tick boundary (one tick per wake-up in steady state).
- After each step it calls `worker.publish_snapshot(g.sim_time,
  g.snapshot())` (60 Hz).
- The worker thread (a ~1 ms loop) drains the snapshot queue to the NEWEST
  and sends one `T_SNAP` per iteration (stale ones dropped). The wire
  therefore carries ~60 Hz snapshots, each stamped with the host's
  `sim_time` at the step.
- `T_ECHO`: after each step, the input the host actually applied to player 1
  (the client's ship) that tick — one `(sim_time, input)` entry per tick,
  batched into one `T_ECHO` per step.
- `T_BEAM`: one per laser discharge.
- `T_PONG`: reflected on the worker thread the moment a `T_INPUT` lands
  (no sim work).
- The host applies the LATEST received input each tick.

### A.3 Client frame loop (__main__.py run_client)
Per frame (`dt` clamped to 0.05 s):
1. Sample the local input once (`ShipInput.from_keys`).
2. Send `T_INPUT` (tagged for RTT) every frame.
3. `ghost.record_input(host_time.now(now), inp)` into the ghost's input
   buffer.
4. `worker.poll()`:
   - `T_SNAP`: `host_time.record(now, sim_time)` + `latency.update(jitter)`;
     then `push_snapshot(sim_time, snap, now=sim_time + rtt_ow,
     real_time=now, rate)`.
   - `T_ECHO`: `ghost.record_echo(sim_time, input)` per entry.
   - `T_PONG`: `rtt = now - t0`; `rtt_ow = EMA(rtt/2)`, clamped
     `[0.002, 0.25]`.
5. `latency.tick(dt)`; `render_point.advance(dt, snap_buf.newest_time())`.
6. `predicted_view(dt, keys, host_time, real_time)`:
   - `rp = render_point.now()`; `pos = snap_buf.positions_at(rp)` (remote
     entities).
   - `ghost.advance(dt, inp, enemies, real_time, host_time, rtt_ow)`.
   - `gpos, gangle = ghost.render_pose()` (the local ship).
   - Draw the local ship at `gpos`; draw remote entities from `pos`.

### A.4 The prediction ghost (netcode.py PredictedShip)
State:
- `_sim_time` — the ghost's physics clock (the time of its pose `s.pos`).
- `_last_step_time` — the tick boundary the last physics step ran at.
- `_acc = _sim_time - _last_step_time` — the render interpolation phase, in
  `[0, TICK)`.
- `_prev_pos` / `_prev_angle` — the pose captured immediately before the
  last physics step.

`advance(dt, ...)`:
- Clock (branch 2, the live client): `_sim_time = host_time.now(real_time)
  + rtt_ow`.
- `while _sim_time - _last_step_time >= TICK`: capture `_prev`,
  `step(TICK, inp)`, `_last_step_time += TICK`.
- `_acc = _sim_time - _last_step_time`.

`reconcile_rewind(ship_s, snap_time, now, ...)`:
- `apply_snapshot(ship_s)` → `s.pos` = the host's pose at `snap_time`.
- `n = int((now - snap_time) / TICK)`; replay `n` ticks from the echo buffer
  (else the input buffer).
- `_sim_time = max(now, snap_time + n*TICK)`;
  `_acc = _sim_time - (snap_time + n*TICK)`;
  `_last_step_time = snap_time + n*TICK`.

`render_pose()`:
- `alpha = clamp(_acc / TICK, 0, 1)`;
  return `lerp(_prev_pos, s.pos, alpha)`,
  `lerp_angle(_prev_angle, s.angle, alpha)`.

### A.5 The render point (netcode.py RenderPoint) — remote entities only
- `render_t = newest_arrived_stamp - delay`, chased at `dt`/frame (catch-up
  `0.2 * gap` when the gap exceeds `5*dt`).
- `delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter), MIN, MAX)`, moving at
  most `1/60` s per frame.
- The local ship does NOT use the render point — it uses the ghost.

### A.6 The interpolation buffer (netcode.py SnapshotBuffer)
- `(sim_time, snap)` pairs, max 24. `positions_at(render_t)` returns the
  bracketing window's interpolated poses (remote entities only).

### A.7 The host-time estimator (netcode.py HostTimeEstimator)
- Point-slope anchor on the newest sample: `host_time ≈ h_anchor +
  rate * (local - t_anchor)`.
- `rate` = EMA of `Δhost/Δlocal`, clamped `[0.4, 1.5]`.
- `now()` capped at `newest + MAX_LEAD` (`MAX_LEAD = SNAPSHOT_INTERVAL*TICK`
  = 1 tick).
- `last_jitter` = the per-arrival prediction residual (feeds the adaptive
  delay).

---

## B. Current observed behavior (fresh net_debug.csv, 2591 reconciles)
- Reconciles every ~24 ms.
- `replay_ticks` = 0 for 99.9% (p99 = 0, max = 1).
- `replay_span` ≈ `rtt_ow` ≈ 0.0108 s (L ≈ 10.8 ms < TICK).
- `phase_ticks` = 1.0 (the ghost's step grid sits 1 tick behind the
  snapshot).
- `advance_steps` mean 0.38 per interval (the ghost is almost never
  stepping).
- `snap_px` mean 8.75, p50 8.42, p90 8.67 (≈ 1 tick of motion at
  MAX_SPEED 520 px/s).
- `render_snap_px` mean 5.79 (the player sees ~5.8 px of it).

---

## C. The defect (traced through the current code)
1. `now = snap_time + L` (L = `rtt_ow` ≈ 10.8 ms). `n = int(L/TICK) =
   int(0.65) = 0`. The replay is 0 ticks: `apply_snapshot` leaves `s.pos` at
   the host's pose at `snap_time`, and `_last_step_time = snap_time`.
2. After the reconcile, `_sim_time = max(now, snap_time) = snap_time + L`, so
   `_acc = _sim_time - _last_step_time = L ≈ 10.8 ms`.
3. `render_pose()` = `lerp(_prev_pos, s.pos, alpha = L/TICK ≈ 0.65)`. But
   `_prev_pos` is the pose captured before the ghost's last `advance()` step
   (a pre-reconcile, predicted pose) and `s.pos` is the snapshot's pose (at
   `snap_time`). Both are at or before `snap_time`. So the render pose lands
   at or before `snap_time` — it does NOT reach the render clock's time
   (`snap_time + L`).
4. Between reconciles, `advance()` sets `_sim_time = host_time.now(real_time)
   + rtt_ow` (advancing at 1x real time), but the while-loop only fires when
   `_sim_time - _last_step_time >= TICK`. Since `_last_step_time` was just
   set to `snap_time` (the newest tick boundary) and `_sim_time` is only
   `L ≈ 10.8 ms` ahead, the loop fires at most once per interval. The ghost
   takes ~0.4 steps per interval, not 6.
5. Net effect: the render pose is pinned near `snap_time` and only moves when
   a new snapshot (1 tick ahead) lands → the 8.67 px sawtooth.

The root cause: the render pose is derived from the physics pose (which the
reconcile pins to `snap_time`), and the interpolation is BACKWARD
(`_prev_pos → s.pos`), so it sits behind the render clock. The render and
physics are coupled through the snapshot time.

---

## D. The fix (11.x) — decouple the render pose from the physics pose

The physics pose is correct: it sits at `snap_time` (the replay end). The
render pose must be at the render clock's time (`_sim_time =
_last_step_time + _acc`), which is FORWARD of the current physics pose, not
behind it.

Change `render_pose()` to a FORWARD interpolation:
- `P0 = s.pos` (the current physics pose, at `_last_step_time`).
- `P1 =` the pose one tick ahead (at `_last_step_time + TICK`), computed by
  stepping a COPY of the ship one tick with the current input (render-only —
  it does not mutate the physics state).
- `render = lerp(P0, P1, alpha)` with `alpha = clamp(_acc / TICK, 0, 1)`.

This puts the render pose exactly at `_last_step_time + _acc = _sim_time`
(the render clock's time). It is smooth because:
- Between reconciles, `_acc` grows from `L` toward `TICK` as `_sim_time`
  advances at 1x real time, so the render pose creeps forward from `P0`
  toward `P1` at 1x real time.
- When `_acc` reaches `TICK`, `advance()` fires a step: `s.pos` becomes the
  old `P1`, `_last_step_time += TICK`, `_acc` wraps to ~0. The render pose is
  continuous (it was at `P1` = the new `P0`, and now starts at the new `P0`).
- At a reconcile, `s.pos` is set to the snapshot's pose (at `snap_time`),
  `_last_step_time = snap_time`, `_acc = L`. The render pose =
  `lerp(snapshot pose, step(snapshot pose), L/TICK)`. The correction is ~0
  when the prediction was accurate (the snapshot's pose ≈ the predicted pose
  at `snap_time`).

### D.1 Implementation notes
- Compute `P1` by stepping a copy of the ship one tick with the current
  input. Cache it and recompute only when `s.pos`/`s.angle`/`s.vel` or the
  input changes (i.e., after a step or a reconcile) — not every frame.
- The angle uses the same shortest-arc rule as the current `render_pose()`.
- `P1` is presentation-only: it must not feed the sim, the handback math, or
  the no-teleport/rewind gates (those read `s.pos`, the physics pose).
- Keep the physics pose (`s.pos`) and the clock (`_sim_time`,
  `_last_step_time`, `_acc`) exactly as they are — only the RENDER pose
  changes.

### D.2 Why not just extrapolate `s.pos + s.vel * _acc`?
It is cheaper (no full step) and exact at `_acc = 0`, but it is an
extrapolation (the codebase's rule is to never extrapolate into the future),
and it drifts on a hard turn (the ship curves, so a straight-line
extrapolation by velocity is off by a few px over a tick). The forward
interpolation to a simulated `P1` is bounded and turn-accurate. Use the
velocity extrapolation only if the per-step copy is too expensive in
practice.

---

## E. Verification
1. Unit: a ghost seeded + reconciled with `now = snap_time + L` (L < TICK)
   must have `render_pose()` at `snap_time + L` (within a px of
   `s.pos + s.vel * L`), and `render_pose()` must advance at 1x real time
   between reconciles (no jump at the step boundary).
2. Live 2P: re-run the F3 net-debug capture. Expected: `snap_px` and
   `render_snap_px` drop from ~8.7 / ~5.8 px to ~0-1 px; the local ship
   glides instead of sawtooth-ing.
3. Regression: the existing ghost-clock / no-teleport / rewind gates must
   still pass (they read the physics pose, which is unchanged).