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
- Snapshots are published at the sim STEP (60 Hz, one per step). The
  host's NET process sends them on a STEADY 60 Hz cadence (SEND_TICK =
  16.7 ms) using a deadline + busy-wait hybrid, and LATEST-WINS when
  more than one snap is pending. THE CADENCE (plan_20): the loop paces
  itself to SEND_TICK — it sleeps until SEND_BUSY_MARGIN (2 ms) before
  the next deadline, then busy-waits the last 2 ms, so the send lands on
  the deadline within ~1 ms REGARDLESS of the OS timer resolution
  (stdlib only — time.monotonic + a spin, no timeBeginPeriod). WHY the
  hybrid: the OLD loop paced itself with a bare time.sleep(0.001) and
  let the OS timer set the cadence — on a coarse timer (Windows' default
  ~15.6 ms, or a coalesced virtualized timer on the Mac-hosted Linux
  host) that 1 ms sleep returned after ~15-17 ms, so the loop only
  iterated ~54x/sec and the client saw ragged 17-26 ms gaps (the
  residual stutter). The hybrid owns the cadence: the deadline grid is
  SEND_TICK and the busy-wait absorbs the timer's coarseness (2 ms of
  spin per 16.7 ms tick burns ~12% of one core). A best-effort, wrapped
  priority elevation (os.nice on Linux/macOS, SetPriorityClass on
  Windows — all try/except with a normal-priority fallback) is the
  secondary lever for the descheduling component; the hybrid works
  without it. LATEST-WINS (kept from the staleness fix, commit 8c32708):
  each loop iteration drains the net Queue, and if more than one snap
  landed since the last iteration (a hiccup — a slow body, an OS
  deschedule), only the NEWEST is sent (the older ones are dropped). A
  FIFO (oldest-first) would queue stale snaps behind the newest — the
  client would render seconds-old state and the ship would lag far
  behind the input (unplayable). Latest-wins guarantees the client
  always receives the most current state; a dropped snap is a 1-tick
  loss the client's interpolation buffer already handles (it holds the
  last frame), and the rock-shape tracker re-sends full verts for any
  rock it has not serialized, so a drop is a one-time bandwidth cost,
  not a correctness issue. The cadence and latest-wins are ORTHOGONAL:
  latest-wins decides WHICH snap to send (the newest — no staleness),
  the pacing decides WHEN (a steady 16.7 ms — no ragged gaps).
  Beams/echoes are still sent the moment they land (low-rate events, no
  cadence). The snapshot's sim_time is still its stamp (latest-wins
  drops the SEND of older snaps, not the stamp). The client
  reconciles at the ghost's OWN clock
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
  quit — small, low-rate, ordered). The CLIENT's input does NOT ride this
  Pipe: the host's NET process receives the T_INPUT off the socket and
  sends it to the sim over the SEPARATE net->sim remote_input Pipe
  (net_process.py) the moment it lands (~1 ms), and the sim process
  swaps `game.remote_input` (the `set_remote_input` reference swap). The
  command Pipe therefore has a SINGLE sender (the render thread — the
  worker's input sink is gone, plan_13), so the send side needs no lock.
  The sim->render channels are TWO BOUNDED Queues with
  put_nowait DROP-WHEN-FULL — NOT Pipes. A Pipe blocks the child's
  bridge on send() when the render process is busy drawing, and a
  blocked bridge can't drain commands. Dropping the oldest item when
  full is latest-wins — exactly what the render thread wants (it draws
  the newest model, never a stale one); a dropped snapshot is a network
  loss the client's interpolation buffer already handles.
  * the MODEL channel (maxsize 4): the render model (60 Hz) + lifecycle
    (ready/crash/done). Drained by the RENDER thread.
  * the NET channel (maxsize 16): the wire traffic (snap/beam/echo) the
    SimThread hands to the net Queue. Drained by the host's DEDICATED NET
    PROCESS (net_process.py, plan_13) — NOT the render thread. The net
    process owns the socket end to end (listen + accept + handshake +
    send + receive) and runs its own loop paced to a steady 60 Hz
    (SEND_TICK, the deadline + busy-wait hybrid — see the snapshot
    bullet), so the render thread's draw is completely out of the send
    path (no shared GIL, no GIL-switch lag). It sends beams/echoes the
    moment they land and sends snaps LATEST-WINS (only the newest
    pending snap per loop iteration — see the snapshot bullet above).
- The render loop DRAINS the model queue in the SAME thread as the draw,
  every frame. A separate drain thread would be starved by the draw's
  GIL hold, the queue would fill, and the sim process would block —
  defeating the whole point.
- The socket + the send live in the host's DEDICATED NET PROCESS
  (net_process.py, plan_13) — NOT the render process. A live socket
  can't be pickled across the spawn boundary, so the net process owns
  it end to end (listen + accept + the join/welcome handshake + send +
  receive). The sim process hands it the snapshot / beam / echo data
  over the DEDICATED net Queue; the net process drains it in its own
  loop, paced to a steady 60 Hz (SEND_TICK, the deadline + busy-wait
  hybrid — see the snapshot bullet), and sends each message
  (beams/echoes immediately, snaps latest-wins — see the snapshot
  bullet). WHY a separate process (not
  a thread in the render process): the first M4 rode the wire traffic on
  the model Queue, so the render thread's per-frame drain (the SAME
  thread as the draw) was in the snapshot's send path — a busy draw
  delayed (up to a full frame) or dropped snapshots. A net-pump THREAD
  was the next fix, but it still shared the render process's GIL with
  the draw, so under a heavy draw its sends could lag by up to the GIL
  switch interval. The separate process has its OWN GIL (shared only
  with itself), so the render thread's draw can hold the render GIL for
  as long as it wants and the net process keeps draining + sending —
  the render thread is COMPLETELY out of the send path. (The sim
  process is GIL-free too, so the snapshot STAMPS stay smooth either
  way; latest-wins keeps the SEND current on top of that.)

## Client process topology (the net is a SEPARATE PROCESS)
- The client's socket I/O + JSON run in its OWN process
  (client_net_process.py) — NOT a NetWorker thread in the render
  process. The net process owns the socket end to end (connect + the
  join/welcome handshake + the game loop's input send + snapshot
  receive) and talks to the render process only through two Pipes:
  render->net (input / respawn / quit) and net->render (connected /
  snap / beam / echo / pong / disconnected / error / crash). The render
  process learns the handshake outcome by polling the msg Pipe
  (non-blocking poll()) while drawing the "connecting" screen — the
  render thread never touches the socket.
- WHY a process, not a thread (the GIL fix): a NetWorker THREAD shares
  the render process's GIL with the draw. When a snapshot's bytes land,
  the worker's recv() returns (it releases the GIL while blocked) but it
  must RE-ACQUIRE the GIL to stamp the wire-arrival time + queue the
  message, and the render thread holds the GIL for the whole draw frame
  (~16-25 ms). So a snap arriving at time T got stamped at T + (one
  frame) — the client's client_snap_debug.csv went BIMODAL (76% at
  10-18 ms, then a sharp 13% peak at 30-36 ms) that the host's UNIMODAL
  send (55% at 10-18 ms, 31% at 18-22 ms, 1.5% at 30-36 ms) never
  produced. The extra ~666 32 ms gaps were manufactured by the GIL
  re-acquisition, not the wire. A separate process has its OWN GIL
  (shared only with itself): the render thread's draw can hold the
  render GIL as long as it wants and the net process keeps draining the
  socket + stamping the wire-arrival time on time. The wire-arrival
  stamp (the "receive" time in client_snap_debug.csv) is now the TRUE
  wire cadence, decoupled from the render frame.
- The net process is HEADLESS (SDL_VIDEODRIVER/AUDIODRIVER=dummy set
  before the net import, inside the function so the parent keeps its
  real video driver) — it does no draw. A best-effort, wrapped priority
  elevation (os.nice on Linux/macOS, SetPriorityClass on Windows — all
  try/except with a normal-priority fallback) is the secondary lever for
  the OS-descheduling component; the separate GIL is the primary fix and
  works without it.
- Shutdown: the render process sends ("quit",) on the cmd Pipe, closes
  both Pipes, join(timeout=2), terminate() if still alive. The net
  process ALSO exits on cmd_conn EOF (the parent died), so a missed quit
  can't leak it. Idempotent — safe on every exit path (ESC/QUIT,
  disconnect, connect-phase error).

## Wire
- TCP, non-blocking, 4-byte big-endian length prefix + JSON. Message
  types: join, welcome, input (carries `tag`), snap, respawn, beam,
  echo, pong (see net.py for shapes). The `beam` (T_BEAM) message carries a
  `target` field (plan_19) — `None` (rock) / an enemy ship id (int) /
  `"p<i>"` (the other player) — so the client knows which ship a beam hit
  (it fires the local ship's hit FX when `target == "p<local_index>"`).
  A missing field reads as None (the rock path) — backward-tolerant.
- The snapshot is an 11-tuple (game.py `Game.snapshot`); PRUNED on the
  wire: asteroids -> `[id, x, y, angle, verts]` (x/y 1-decimal, angle
  3-decimal, `verts` DELTA-COMPRESSED — the rock's 10-vertex shape is sent
  full-precision the FIRST time its id appears and `None` afterwards, driven
  by a per-connection `RockShapeTracker` in the send path; the client keeps
  a persistent id->verts registry, `Game.rock_shapes`, populated in
  `push_snapshot`), rng state -> None placeholder. ~6.9 KB/snapshot steady
  state (one angle float per rock over the old `[id, x, y]` form), ~415 KB/s
  at 60 Hz; a NEW rock (a spawn/split child) costs one extra ~411 B of verts
  (spawns are capped at 4/tick, splits rare — amortized ~1-2 KB/s). See
  plan_17_client_rock_shapes.md.
- The socket I/O + JSON run in a DEDICATED NET PROCESS on BOTH peers
  (host: net_process.py, client: client_net_process.py) — NOT a thread
  in the render process. Each net process owns its Connection end to
  end (the host: listen + accept + handshake; the client: connect +
  handshake) and talks to its render process only through Pipes. WHY a
  process, not a thread: a NetWorker THREAD shares the render process's
  GIL with the draw — when a snapshot's bytes land, the worker's
  recv() returns but it must RE-ACQUIRE the GIL to stamp the
  wire-arrival time, and the render thread holds the GIL for the whole
  draw frame (~16-25 ms), so a snap arriving at time T got stamped at
  T + (one frame) — the client's receive cadence went BIMODAL (76% at
  10-18 ms, then a sharp 13% peak at 30-36 ms) that the host's
  UNIMODAL send (55% at 10-18 ms, 31% at 18-22 ms, 1.5% at 30-36 ms)
  never produced. A separate process has its OWN GIL (shared only with
  itself): the render thread's draw can hold the render GIL as long as
  it wants and the net process keeps draining the socket + stamping on
  time. The host's input forward + pong reflection run in the host net
  process; the client's input send + snap/beam/echo/pong forward run in
  the client net process (no pygame in either). The NetWorker class
  remains in net.py (its self-test still exercises it) but the game no
  longer uses it.
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
- Client hit FX (option-A trigger, plan_18): the trigger (game.py
  `predicted_view`) now covers ROCKS as a third target class, alongside
  the remote ship and AI enemies. It mirrors the host's
  bullet/missile-vs-rock test EXACTLY — a CIRCLE, not the drawn polygon
  (the host's test is `dist < collision_radius`; the polygon is
  non-convex, so the convex-poly helpers must not be used on raw verts).
  R is DERIVED from the wire verts as `(max|v| + min|v|) / 2` (the wire
  prunes size/radius — plan_17 — so it is not sent; the registry
  `_rock_coll_r` is filled in `push_snapshot`'s verts loop). The impact
  is the circle ENTRY point (`_seg_circle_entry`, the first segment
  crossing), so the spark lands on the rock's near surface, not its
  center. On a hit the trigger spawns a burst + `small_explosion` SFX and
  culls the projectile in the SAME persistent cull set the ship/enemy
  passes use (`_hit_fx_cull_bids` for bullets — ghost + buffer-drawn;
  `_hit_fx_cull_mids` for ghost missiles, which also culls the buffer's
  copy + fog light), so the projectile stops at the rock instead of
  flying through until the host's authoritative cull arrives.
  Presentation only: the wire is untouched, the ghost's PHYSICS does no
  rock collision (a predicted clip-through is corrected by the next
  reconcile), and the authoritative hit/damage/rock death still come from
  the host snapshot.

## 2P player combat (plan_19)
The two human players can fight each other. The core abstraction is a
`team` field on the ship + a single team-based targeting predicate, which
replaces the old hardcoded "loop over `self.enemies`".

- **Team model** — `Ship.team: int`. Player `i`'s ship -> `TEAM_i` (== its
  index, set by `Game` in `reset()` / `set_player_ship()`); AI enemies ->
  `TEAM_NEUTRAL` (-1, set in `AIEnemy.__init__`). `team` is NOT serialized —
  the client derives the same assignment from its player index, so the ship
  snapshot stays 25 fields.
- **One predicate** — `can_target(attacker, target)` (game.py): True iff the
  target is ALIVE and off-team (`target.team != attacker.team and not
  getattr(target, "dead", False)`). Enemies have no `.dead` (always
  targetable); a frozen corpse is not a target. Single-player reduces to
  "enemies only" (the other player does not exist) — bit-identical.
- **Unified targetable surface** — an AI enemy and a player ship both expose
  `.pos/.vel/.angle/.team/.dead/.collision_radius/.shield_on/.shield_contains/
  .shield_impact_point/.register_hit/.lead_point`. The one new piece was
  `Ship.lead_point` (constant-velocity intercept, the same math as
  `AIEnemy.lead_point`). Host targeting (`_pick_laser_target` /
  `_pick_missile_target`) loops `_targetable_candidates(ship)` (enemies + the
  other player, filtered by `can_target`); host damage (`_collisions` /
  `_resolve_beam`) tests the other player's ship with the SAME shield-oval /
  swept-hull criteria as the enemy path, routing hits through
  `_handle_ship_hit` (per-player death + respawn already exist).
- **Client lock-on** — the ghost sees the remote player via a
  `_GhostPlayerProxy` (built from `pos['ships'][remote_index]`, `.id` =
  `"p<index>"`, `.team` = the remote index, `.lead_point` = constant-velocity
  intercept). `PredictedShip._pick_*` / `step_local_missiles` take the proxy
  alongside the enemy proxies (the `by_id` map includes it), so the ghost
  locks + charges + launches + homes on the other player on the same tick the
  host does (lockstep preserved).
- **Two additive wire changes** (the snapshot is untouched):
  1. **T_BEAM gains `target`** — `None` (rock) / an enemy ship id (int) /
     `"p<i>"` (the other player). The host's `_emit_beam_event` sets it from
     the resolved target; the client stores it as the 6th field of the
     `remote_beams` entry. A missing field reads as None (the rock path) —
     backward-tolerant.
  2. **Missile snapshot target encoding** — `Missile.snapshot` stores the
     target as an enemy ship id (int) / `"p<i>"` (a string) / None.
     `Game.apply_snapshot` re-wires it: int -> the enemy via `id_map`;
     `"p<i>"` -> a `_PlayerHomingTarget` adapter over `self.players[i]`
     exposing `.hp/.pos/.lead_point` (so `Missile.update`'s existing
     `target.hp > 0` + `target.lead_point(...)` work unchanged). Host-side
     only — the client's ghost uses `_GhostPlayerProxy` instead.
- **Client hit FX** — the other player's weapons impact the local ship with
  shield flash + burst + SFX, on the client:
  - **Bullets + missiles** — the existing shield_dump-jump detection
    (`_local_incoming_hit_fx`) is weapon-agnostic, so it already fires for the
    other player's bullets + missiles. The impact-point search was generalized
    to also consider incoming PLAYER bullets (`kind == 'player'`,
    `owner == remote_index`), so the spark lands on the true impact point
    (missiles have no buffer bullet to project from — the oval-front fallback
    applies, same as today's enemy-missile case).
  - **Beams** — a T_BEAM whose `target` is the local player
    (`"p<local_index>"`) fires the local ship's hit FX in `_remote_beam_hit_fx`
    (called from `predicted_view`): a shield_burst at the beam's endpoint (the
    `end` the host computed on the local ship's shield oval) + the throttled
    `shield_hit` SFX, in the same frame the beam is drawn. A beam is hitscan —
    there is no in-flight projectile for the shield_dump-jump detection to
    watch, so this is the one new detection. Each beam fires its FX exactly
    once (a `_beam_fx_fired` id-set, pruned to the live beams each frame).
  Presentation only: no sim, no authority (the authoritative hit + damage
  already come from the host snapshot).

## 2P asteroid field (plan_21)
The asteroid field is host-authoritative. In 2P, the field is the UNION of a
3x3 sector region around each alive player (not a single centroid blob). A
rock is kept if it is within FIELD_KEEP_RADIUS (2200 px) of at least one
alive player. The host culls rocks outside the client's viewport
(+ WIRE_CULL_MARGIN 640 px) before serializing the snapshot, reducing wire
bandwidth by ~60% vs the original centroid field (measured: 144.5 -> 23.2
rocks on the wire, 8873 -> 3560 B/snap, 2P at 1500 px separation — see
PERFORMANCE.md for the table).

- **Per-player field (S1)** — `spawning.update_field` takes the list of
  ALIVE player positions and tops up the union of each player's 3x3 sector
  region (overlapping regions are not double-filled; the global
  MAX_SPAWNS_PER_TICK cap bounds the fill). Culling is per-player
  min-distance, not centroid: a rock is kept if it is within
  FIELD_KEEP_RADIUS of ANY alive player. With no alive players (2P, both
  dead) the field is left as-is — no cull, no top-up.
- **Wire culling (S2)** — `serialize_snapshot(snap, tracker, cull_center)`
  drops rocks outside the cull rect (viewport + WIRE_CULL_MARGIN) centered
  on the remote player's snapshot position. Only the production 2P send
  path (`net_process.py`, via `_remote_player_pos`) passes `cull_center`;
  the in-process `NetWorker` (self-test + test_10_12 only) and 1P keep the
  default `cull_center=None` (no cull). The margin covers camera look-ahead
  (312 px) + viewport cull margin (80) + interp delay x max rock speed (46)
  + safety (200).

The client is unaffected: it interpolates rocks by id and handles new/missing
rocks gracefully (a rock that drops out of the wire is simply not drawn; a
rock that re-enters the cull rect pops in at its current position, well
off-screen due to the 640 px margin). `RockShapeTracker` keeps the id in
`_sent` across a cull-rect exit/re-entry, so the immutable verts are never
re-sent.

## Misc
- Enemy proxies: the caller builds lightweight proxies from the buffer's
  enemy entries (`.pos`, `.vel`, `.id`, `.ship` with power fields) and
  feeds them to the ghost's step/replay so the ghost's laser/missile
  targeting + sensor contacts match the host's given the same inputs.
- `diag_sink` (class-level, NOT instance — a function stored as an
  instance attribute would bind as a method): optional ghost-clock
  records to ghost_clock.csv, set by the client loop only while
  debug_net is on.