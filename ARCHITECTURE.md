# Belt Fighter — Netcode Architecture (current state)

The single home for the load-bearing invariants of the 2P netcode.
Comments in netcode.py / net.py / game.py point here instead of
re-explaining. History (why a rule changed) lives in git, not here.

## Topology & authority
- 2P, host-authoritative. The host runs the sim; the client is a
  PRESENTATION peer — it never runs the sim and never calls
  `Game.apply_snapshot` on world state. The only exception is the
  prediction ghost, which applies ship snapshots to its OWN private Ship.
- The client renders remote entities from the interpolation buffer at a
  render point in the PAST; it renders its OWN ship from the ghost, which
  is stepped with local input and reconciled to authority.
- The host applies the LATEST received input each tick (pinned #5).
- Snapshots are published at the sim STEP (60 Hz, one per step) and sent
  immediately by the NetWorker — no send timer. The snapshot's sim_time
  IS its send time. The client reconciles at the ghost's OWN clock
  (`ghost.sim_time`, advanced to the current frame BEFORE the poll
  loop) — NOT at `sim_time + L` (that anchor sat ~3 ms ahead of the raw
  clock and each reconcile jumped the clock forward -> a per-reconcile
  snap).

## Host process topology (the sim is a SEPARATE PROCESS)
- The host's authoritative sim runs in its OWN process (sim_process.py),
  NOT a thread in the render process. The render thread holds the GIL
  for its whole draw (~18-25 ms — longer than the 16.7 ms sim tick), so
  a same-process sim thread is starved and BATCHES 2 ticks (ragged
  17/33 ms snapshot stamps). The client's ghost clock (anchored on the
  previous snapshot) can't anticipate a 33 ms jump, so it sits ~7 ms
  behind the new stamp and the reconcile pulls the clock+pose forward —
  the residual ~8 px snap tail. A separate process has no shared GIL:
  the sim steps exactly one tick per iteration (measured: 0% batched,
  every gap 16.7 ms, 1.00x rate) even while the render process draws.
- The sim process is HEADLESS: SDL_VIDEODRIVER/AUDIODRIVER=dummy set
  before pygame is imported, a 1x1 dummy display, and the Game is built
  with None display surfaces. The sim path (_step / snapshot /
  render_model) never touches the display surfaces — only draw() does —
  so this is safe. draw() runs in the RENDER process on a shell Game
  (M3).
- The render->sim command channel is a Pipe (setup / input / command /
  remote_input / quit — small, low-rate, ordered). `remote_input` (M4)
  carries the CLIENT's input: the worker's input sink (worker thread)
  sends it the moment a T_INPUT lands, and the sim process swaps
  `game.remote_input` (the in-process `set_remote_input` reference swap).
  The command Pipe has TWO senders in the render process (the main
  thread + the worker thread), so the send side is serialized by a lock
  (two concurrent senders would interleave the Pipe's 4-byte length
  prefixes). The sim->render channels are TWO BOUNDED Queues with
  put_nowait DROP-WHEN-FULL — NOT Pipes. A Pipe blocks the child's
  bridge on send() when the render process is busy drawing, and a
  blocked bridge can't drain commands. Dropping the oldest item when
  full is latest-wins — exactly what the render thread wants (it draws
  the newest model, never a stale one); a dropped snapshot is a network
  loss the client's interpolation buffer already handles.
  * the MODEL channel (maxsize 4): the render model (60 Hz) + lifecycle
    (ready/crash/done). Drained by the RENDER thread.
  * the NET channel (maxsize 16): the wire traffic (snap/beam/echo) the
    SimThread hands to its worker. Drained by a DEDICATED net-pump
    thread (run_host's _NetPump) — NOT the render thread.
- The render loop DRAINS the model queue in the SAME thread as the draw,
  every frame. A separate drain thread would be starved by the draw's
  GIL hold, the queue would fill, and the sim process would block —
  defeating the whole point.
- The NetWorker + Connection stay in the RENDER process (a live socket
  is not passed across the spawn boundary). The sim process hands the
  render process the snapshot / beam / echo data over the DEDICATED net
  Queue; a dedicated net-pump thread (run_host's _NetPump) drains it and
  routes each message to the worker, which serializes + sends it the
  moment it lands (M4). WHY a dedicated thread: the first M4 rode the
  wire traffic on the model Queue, so the render thread's per-frame
  drain (the SAME thread as the draw) was in the snapshot's send path —
  a busy draw delayed (up to a full frame) or dropped snapshots. The
  pump drains the net Queue every ~1 ms regardless of the draw, so the
  render thread's draw is OUT of the send path. (The pump shares the
  render process's GIL with the draw, so under a heavy draw its sends
  can lag by up to the GIL switch interval — far better than a full
  frame, and the sim process itself is GIL-free, so the snapshot STAMPS
  stay smooth either way.)

## Wire
- TCP, non-blocking, 4-byte big-endian length prefix + JSON. Message
  types: join, welcome, input (carries `tag`), snap, respawn, beam,
  echo, pong (see net.py for shapes).
- The snapshot is an 11-tuple (game.py `Game.snapshot`); PRUNED on the
  wire: asteroids -> `[id, x, y]` (1-decimal), rng state -> None
  placeholder. ~6.4 KB/snapshot, ~370 KB/s at 60 Hz.
- The socket I/O + JSON run on the NetWorker daemon thread; the main
  thread only queues/polls it. The worker owns the Connection after
  start(). The host's input sink + pong reflection run on the worker
  thread (GIL-atomic ref swap; no pygame).
- Partial frames are sent in parts; the UNSENT remainder is kept (never
  re-send a whole frame — it would duplicate bytes). The receive side
  resyncs past corrupted regions instead of raising.

## Interpolation buffer (SnapshotBuffer)
- Holds (sim_time, snap) pairs, oldest first, bounded to
  SNAPSHOT_BUFFER_MAX (24 — brackets INTERP_DELAY_MAX 0.35 s at 60 Hz).
  Out-of-order arrivals are dropped.
- `positions_at(render_t)` interpolates between the two snapshots that
  BRACKET render_t; render_t is clamped to the window — never
  extrapolate. The bracketing span is passed as `dt` to
  `interp_positions` (projectile matching needs the real window).
- Membership follows the CURRENT snapshot: new entities pop in at their
  curr position; dead ones drop out.
- Entity matching: player ships by INDEX (ships don't turn over),
  enemies by ship id, asteroids by rock id, projectiles by PREDICTED
  POSITION within a kind (`_match_bullets` — index matching is wrong:
  the sim's bullet lists are sliding windows).
- `lerp` is exact at the endpoints (t==0/1 return a/b by identity —
  IEEE-754 makes `a + (b-a)*1.0` not always bit-identical to b);
  `lerp_angle` lerps the WRAPPED delta (raw angles are unbounded).
- Latches are taken from the CURRENT snapshot, not lerp'd: `dead`,
  `brownout`, `flame_mags`. Lerp'd: pose, angle, shield state,
  `power_factor`, enemy `power_used`.

## Render point (RenderPoint)
- `render_t = newest_arrived_snapshot_stamp - delay` — anchored on the
  DATA, not on a model of the host clock. The point can never exceed the
  newest snapshot (delay >= INTERP_DELAY_MIN > 0).
- The point CHASES the anchor: it moves toward `newest - delay` at at
  most `dt` (the frame's REAL time) per frame — 1x real time on any
  refresh rate (a fixed 1/60 cap drifts behind on sub-60 FPS clients).
  `dt` is clamped to MAX_FRAME_DT by the caller.
- Catch-up: when behind by more than ~5 frames (d > 5*dt), advance by
  CATCHUP * d (0.2) per frame so a large gap closes in ~5 frames; the
  step is always bounded by d (no overshoot). Steady-state gap ~5*dt.

## Adaptive delay (LatencyTracker)
- `delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter),
  INTERP_DELAY_MIN, INTERP_DELAY_MAX)` — renders further in the past
  when the arrival pattern is ragged, so a late packet lands inside the
  window instead of behind the render point.
- Input: the HostTimeEstimator's per-arrival jitter sample (consumed
  only on a RECORDED arrival).
- The delay moves at most MAX_STEP = 1/60 s per frame (the render point
  can never jump) and never shrinks below INTERP_DELAY_MIN.

## Host clock estimate (HostTimeEstimator)
- Point-slope fit anchored on the NEWEST sample:
  `host_time ≈ h_anchor + rate * (local_time - t_anchor)`. The offset is
  a small delta (time since the newest sample), so the estimate stays
  well-conditioned for the whole session (an absolute intercept
  `A + rate * local_time` amplifies rate error by the unbounded
  local_time — the error grew over a session).
- `rate` = EMA of Δhost/Δlocal, clamped to [0.4, 1.5] (a struggling
  host simulates slower than real time; the estimate must follow).
- `now()` is capped at newest + MAX_LEAD (one snapshot interval) — the
  estimate stays anchored on the data when the host starves.
- `last_jitter` = PREDICTION RESIDUAL: how far this arrival's stamp is
  from where the PREVIOUS anchor + rate predicted it (computed with the
  rate BEFORE the update — the innovation). Feeds the LatencyTracker.
- `now` (the raw capped estimate) is the GHOST's clock (see the GHOST
  CLOCK CONTRACT). `now_smooth` is retained for diagnostics/tests but is
  NOT the ghost's clock: the reconcile re-anchors the ghost to the raw
  estimate each snapshot, so a smoothed clock lags the anchor by its
  correction time-constant (~6 ms at SMOOTH_ALPHA 0.05) and that lag
  reads as a per-reconcile snap. The raw estimate is already smooth
  between arrivals (fixed anchor + fitted rate, continuous re-anchor).

## Ghost clock contract (PredictedShip.advance)
- Three quantities: `_sim_time` (the time of the physics pose),
  `_last_step_time` (the tick boundary the last step ran at), `_acc`
  (= `_sim_time - _last_step_time`, in [0, TICK) — the render
  interpolation phase).
- The clock sets `_sim_time` each call, in priority order:
  1. `real_time is None` (tests): free-run `_sim_time += clamped dt`.
  2. `host_time` ready (the LIVE client): `_sim_time =
     host_time.now(real_time) + rtt_ow` — the ghost's clock IS the
     host's clock (the RAW capped estimate, NOT now_smooth — see the
     Host clock estimate section) plus one-way latency, so the step grid
     locks to the host's tick grid and the reconcile's pose correction
     is ~0.
  3. `real_time` given, host_time not ready (pre-first-snapshot window,
     some tests): free-run from the last reconcile anchor
     (`_sim_base + _rate * (real_time - _real_base)`; `_rate` is always
     1.0). These anchor fields exist ONLY for this branch.
- Physics steps run on tick-boundary crossings
  (`_sim_time - _last_step_time >= TICK`), so the ghost integrates at
  the SIM's rate regardless of display refresh. `dt` is clamped to
  MAX_FRAME_DT (no catch-up spiral).
- A DEAD ghost does not advance (the host freezes a dead ship); its
  beams/missiles still age out.

## Reconcile / rewind (PredictedShip.reconcile_rewind)
- Apply the authoritative ship snapshot taken at `snap_time`, then
  REPLAY the local inputs the host applied between `snap_time` and
  `now` (the ghost's OWN current sim time — the caller passes
  `ghost.sim_time`, advanced to the current frame BEFORE the poll loop;
  a 1-frame-stale clock sat 12 ms behind the fresh stamp and caused the
  ~8 px snap). Reconciling at the ghost's own clock (NOT
  `snap_time + rtt_ow`, which sat ~3 ms ahead of the raw clock and made
  each reconcile jump the clock forward) means the reconcile only
  corrects the pose + re-anchors the tick grid. The prediction is
  REBUILT, not snapped (pinned #4).
- The replay span is ~one-way latency: sub-tick on a LAN, so `n=0` in
  steady state and the render lerp (snapshot pose -> one tick ahead, at
  `_acc/TICK`) rebuilds the prediction. The replay fires when the span
  is >= 1 tick (after a hiccup, or higher latency) — and when it does,
  it reconstructs the pose to ~0.06 px (measured).
- Input source: the ECHO buffer (the inputs the host ACTUALLY applied,
  T_ECHO, stamped with the host's own sim clock) when it has anything,
  else the client's own sent input (initial transient / loss). Both are
  (time, ShipInput) pairs ordered by time; selection is a monotone
  pointer — the newest buffered input with time <= tick time (mirrors
  the host's LATEST-received-input rule, pinned #5). A dropped echo
  holds the last echoed input — exactly what the host did.
- `n = max(0, int((now - snap_time) / TICK + 1e-9))` whole ticks
  replayed (the snapshot reflects the state AFTER the tick that ended at
  snap_time, so the first replayed tick STARTS at snap_time).
- After the replay: `_sim_time = max(now, snap_time + n*TICK)` (never
  pull the clock back), `_acc = _sim_time - (snap_time + n*TICK)`,
  `_last_step_time = snap_time + n*TICK`.
- The replay re-runs the fire ticks after apply_snapshot has RESET the
  seq counters — the re-fire guard in `step` skips a shot/missile whose
  id is already in flight (the prediction's copy is the fresher one).
- The client's sensor state (V/G/T) is captured before apply_snapshot
  and restored after when `client_sensor_authoritative` is set — the
  host never saw the client's V/G/T, so its snapshot carries the host's
  own sensor state. Everything else stays host-authoritative.

## Render pose (PredictedShip.render_pose)
- The render pose sits at the CLOCK's time (`_sim_time`), not the
  physics pose's time (the replay end). The clock is L = rtt_ow ahead
  of the physics pose after a reconcile, so the render lerps the
  replay-end pose toward the pose ONE TICK ahead (a KINEMATIC pre-step:
  pos += vel*TICK, angle += turn*ROT_SPEED*turn_rate_factor*TICK — no
  full physics, no side effects) at alpha = `_acc / TICK` (clamped to
  [0, 1]). Continuous across the reconcile (the new base is where the
  old "next" was). Render helper only — never touches `self._ship.pos`.

## Ghost presentation entities & handoff
- The ghost keeps its OWN shots as presentation objects: `local_bullets`
  (Bullets), `local_beams` ([hull-local muzzle, world end, age, ttl]),
  `local_missiles` (GhostMissiles). Presentation only: no collision, no
  sim feedback. They exist so the player sees their own fire
  immediately instead of ~100 ms later.
- Each carries the SAME unique id the host assigns — `(player_index,
  seq)` — and the ghost's seq counters are resynced to the host's via
  the ship snapshot on reconcile. The id is the dedup key:
  predicted_view skips the buffer's copy of a projectile whose id is in
  the ghost's local list.
- Beams: 0.15 s TTL (BEAM_TTL, mirrors the host). The muzzle is the
  HULL-LOCAL offset, re-anchored to the ghost's CURRENT pose at draw
  time (a frozen world-space muzzle is left behind as the ghost moves).
- Handback: a ghost projectile is marked `pending` (NOT culled) once
  BOTH hold: `seq < host_seq` (the snapshot's authoritative seq counts
  launches as of the snapshot — exact, phase-free) AND the buffer's
  NEWEST snapshot actually carries a projectile with that id (the first
  snapshot to carry it arrives up to one interval after the launch).
  The ghost KEEPS DRAWING a pending projectile; predicted_view hands it
  off on the frame the buffer copy is actually VISIBLE (the render
  point reaches the carrying snapshot ~INTERP_DELAY later — culling at
  snapshot arrival left a 6-15 frame blink):
  - missiles: remove from local_missiles + seed a decoupled render
    offset (ghost_pos - interp_pos) that decays to the buffer's
    position (MISSILE_HANDOFF_DECAY);
  - bullets: remove from local_bullets + draw the buffer copy pulled
    forward by V*delay (closed form — a bullet is a straight line).
  Bullets additionally wait until the buffer copy has ADVANCED before
  the handoff.
- GhostMissile homing is client-side: it steers toward the buffer's
  enemy PROXY for its target id (re-resolved each step; a gone target
  coasts), mirroring the host's Missile.update (boost ramp, then
  turn-rate-limited seek toward the lead point).

## Misc
- Enemy proxies: the caller builds lightweight proxies from the buffer's
  enemy entries (`.pos`, `.vel`, `.id`, `.ship` with power fields) and
  feeds them to the ghost's step/replay so the ghost's laser/missile
  targeting + sensor contacts match the host's given the same inputs.
- `diag_sink` (class-level, NOT instance — a function stored as an
  instance attribute would bind as a method): optional ghost-clock
  records to ghost_clock.csv, set by the client loop only while
  debug_net is on.