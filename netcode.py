"""Remote-interpolation helpers.

The authoritative peer sends a `Game.snapshot()` every
`config.SNAPSHOT_INTERVAL` ticks. The remote peer does NOT run the sim for
remote entities — it renders BETWEEN snapshots by interpolating, instead of
teleporting each time a snapshot lands.

`PredictedShip` is the client-side prediction ghost: the remote peer steps
a private Ship with its OWN input and reconciles it to the authoritative
ship snapshot when one lands, so the player's own ship does not feel the
INTERP_DELAY. Remote entities (other ship, enemies, asteroids) keep using
`SnapshotBuffer`.

`interp_positions()` is the core: a PURE function of two snapshots + alpha.
It reads plain data and returns plain data. It must never touch sim state —
the sim stays authoritative on one peer; this only feeds the render.
Entities are matched by IDENTITY (enemies by ship id, asteroids by rock
id) so interpolation survives entity turnover — see `interp_positions`
for the membership rules.

The buffer returns (x, y, angle, dead, shield_dump, shield_clock,
flame_mags) per ship, with the angle lerp'd the same way
`Ship.sync_render` does (the wrapped delta, via `lerp_angle`), so the
remote peer can draw the remote hull at its interpolated orientation
instead of a dot. The `dead` flag lets the remote peer skip a dead ship
(it disappears until it respawns) and spawn a client-side explosion on
the alive->dead transition. `flame_mags` is the per-thruster exhaust
(current snapshot, not lerp'd) so the remote player ship's exhaust
renders (parity with the host).

Enemies carry their ANGLE, TAG, VELOCITY, and ID too — (tag, x, y, angle,
vx, vy, id) — so the remote peer can draw each enemy as its REAL hull
(the hull is fixed on both peers by construction: ENEMY_HULL/MOTE_HULL +
fixed loadouts) at its interpolated orientation, build lightweight
targeting proxies (lead_point needs pos + vel), and track a specific
enemy across frames (by id — enemies churn, so there is no stable index
like ship 0). And ALL projectiles are interpolated: snapshot indices 2
(player bullets), 3 (enemy bullets), 4 (missiles) come back under the
'bullets' key as (x, y, vx, vy, kind, owner, boost) entries, so the
remote peer draws real bullets/missiles instead of nothing. Projectiles
are matched by PREDICTED POSITION, not index — see `_match_bullets` for
why index matching (sliding lists) is wrong.

Snapshot layout (see Game.snapshot in game.py):
    [0] players_s tuple of one Ship.snapshot() per player:
                  ship_s = pos.x=[0], pos.y=[1], vel.x=[2], vel.y=[3],
                  angle=[4], ...
    [1] enemies_s tuple of (tag, e_s); e_s = AIEnemy.snapshot() =
                  (ship_s, hp, id, acc_x, acc_y) -> enemy pos = e_s[0][0], e_s[0][1]
    [2] bullets_s, [3] enemy_bullets_s, [4] missiles_s
                  b_s = (pos.x, pos.y, vel.x, vel.y, owner, life, id)
                       (id = (player_index, bullet_seq), the client's
                       dedup key; enemy bullets keep the 6-field form —
                       the ghost only dedups its OWN shots)
                  m_s = (pos.x, pos.y, vel.x, vel.y, owner, life, boost,
                         target_id, id)
    [5] asteroids_s  tuple of rock entries. TWO FORMS are accepted (the
                     buffer is a presentation store fed either the
                     authoritative sim snapshot — loopback/tests — or the
                     pruned wire snapshot — the network):
                       SIM  (9-tuple): (id, x, y, vel.x, vel.y, size,
                                       angle, spin, verts)
                       WIRE (5-tuple): [id, x, y, angle, verts_or_None]
                     (net.py serialize_snapshot). interp_positions reads
                     id=[0], x=[1], y=[2], angle=[6] sim / [3] wire —
                     detected by len (the two forms differ only in which
                     fields the wire keeps).
    [6] rng_state, [7] game_over, [8] protect_timer, [9] next_id,
    [10] rock_next_id
"""
import math

import pygame

from .config import (INTERP_DELAY, INTERP_DELAY_MIN, INTERP_DELAY_MAX,
                    ADAPT_K, SNAPSHOT_INTERVAL, SNAPSHOT_BUFFER_MAX, TICK,
                    MAX_FRAME_DT, ROT_SPEED, MAX_SPEED,
                    BULLET_POOL_MAX, MISSILE_POOL_MAX,
                    MISSILE_SPEED, MISSILE_ACCEL, MISSILE_BOOST_TIME,
                    MISSILE_TURN_RATE, MISSILE_LIFE,
                    SENSOR_SIGNATURE_THRESHOLD, SENSOR_SIG_FULL)
from .ship import Ship
from .bullets import Bullet
from . import missile_telemetry as _mtel
from . import bullet_telemetry as _btel

# The ghost's beam visuals age out at the host's beam ttl, so the
# client's local beams fade on the same clock as the host's.
BEAM_TTL = 0.15

__all__ = ["interp_positions", "lerp", "lerp_angle", "SnapshotBuffer",
           "PredictedShip", "HostTimeEstimator", "LatencyTracker",
           "RenderPoint"]


def lerp(a, b, t):
    """Linear interpolation: a + (b - a) * t.

    Exact at the endpoints by identity: a + (b - a) * 1.0 is not always
    bit-identical to b in IEEE-754, so the render must sit EXACTLY on
    the snapshot when a snapshot becomes the window boundary (a ~1e-13
    offset reads as a tiny pop every SNAPSHOT_INTERVAL).
    """
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    return a + (b - a) * t


def lerp_angle(a0, a1, t):
    """Interpolate between two RAW (unbounded) angles by lerp-ing the
    WRAPPED delta — exactly what Ship.sync_render does. The sim's angle
    accumulates forever, so the delta is wrapped to [-pi, pi] first; the
    result stays anchored on a0, so it inherits a0's unwrapped offset.
    Never lerp the raw angles directly."""
    # Exact at the endpoints (same IEEE-754 rationale as lerp): a0 + da * 1.0
    # is not always bit-identical to a1, so the render would sit ~1 ulp off
    # the snapshot at alpha=1 — a tiny pop every SNAPSHOT_INTERVAL.
    if t == 0.0:
        return a0
    if t == 1.0:
        return a1
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return a0 + da * t


class HostTimeEstimator:
    """The client's estimate of the AUTHORITATIVE peer's sim clock.

    The client never runs the sim, so it has no sim clock of its own — but
    the ghost's clock needs one. Each snapshot is stamped with the host's
    sim time at send and arrives at a known local time, so every arrival is
    a sample of the mapping from local time to host sim time. This class
    fits that mapping as a line in POINT-SLOPE form, anchored on the
    NEWEST sample:

        host_time ≈ h_anchor + rate * (local_time - t_anchor)

    The offset is a small DELTA (the time since the newest sample), not an
    absolute intercept — an intercept `A + rate * local_time` amplifies any
    rate error by the unbounded local_time, so the estimate's error would
    grow over the session. Re-anchoring to the newest sample keeps the
    estimate well-conditioned for the whole session.

    `rate` — the host's sim rate in local-time units, from consecutive
    samples (Δhost/Δlocal). A healthy host is 1.0; a struggling host
    simulates slower than real time, and the estimate must follow that or
    it drifts away from the host's clock. Clamped to [RATE_MIN, RATE_MAX].

    `now(local_time)` is CAPPED at the newest sample's stamp + MAX_LEAD
    (one snapshot interval). The cap anchors the estimate on the DATA, not
    the model: if the host starves, samples arrive a full second apart and
    extrapolating at the clamped rate would run the estimate far ahead of
    the data. Capped, the estimate holds within one snapshot interval of
    the newest stamp, so the render point sits at or behind the newest
    snapshot and the remote render holds the last frame while the host
    catches up. In steady state the cap never binds.

    The per-arrival jitter sample (`last_jitter`) is the PREDICTION
    RESIDUAL: how far this arrival's stamp is from where the PREVIOUS
    model (the previous anchor + rate) predicted it would be. A steady
    arrival sits near 0; a late/early packet spikes. It is the input the
    LatencyTracker consumes.

    `local_time` is any monotonically increasing seconds value the caller
    has (pygame.time.get_ticks()/1000.0 in the game loop; plain t in tests).
    It may be large (it grows for the whole session) — the point-slope form
    only ever computes the SMALL difference (local_time - t_anchor), so a
    large local_time is harmless.

    This estimate is NOT the client's render clock. The render point is
    anchored on the DATA — the newest ARRIVED snapshot's stamp minus the
    adaptive delay (see `RenderPoint`) — because a model of the host clock
    and the data disagree under jitter/loss, and the render must not wander
    by the disagreement. The estimator still runs for two reasons: its
    per-arrival jitter sample (`last_jitter`) is the input the
    LatencyTracker consumes, and `now()` (the raw capped estimate) is the
    ghost's clock (see the GHOST CLOCK CONTRACT in PredictedShip.advance).
    `now_smooth()` is retained for diagnostics/tests but is NOT the ghost's
    clock — a smoothed clock lags the raw anchor the reconcile re-anchors
    to, and that lag reads as a per-reconcile snap.
    """

    RATE_ALPHA = 0.1
    RATE_MIN, RATE_MAX = 0.4, 1.5
    # The SMOOTH render clock (now_smooth) — see its docstring.
    # SMOOTH_ALPHA is the per-frame correction rate toward the raw (capped)
    # estimate: a slow pull, not a snap, so a noisy anchor can't jerk the
    # render. SMOOTH_MAX_JUMP bounds the per-frame correction (s) so a single
    # large residual (a hiccup / a burst after a stall) can't yank the clock
    # more than this in one frame.
    SMOOTH_ALPHA = 0.05
    SMOOTH_MAX_JUMP = 0.02
    MAX_LEAD = SNAPSHOT_INTERVAL * TICK   # one snapshot interval

    def __init__(self):
        # Point-slope anchor (the newest recorded sample), not an absolute
        # intercept — see the class docstring.
        self._anchor = None  # (local_time, host_time) of the newest sample
        self._rate = 1.0     # EMA host sim rate (local-time units)
        self._last_jitter = None  # per-arrival jitter sample
        # The smooth render clock's state (see now_smooth). _smooth is
        # the smoothed host-time estimate; _smooth_t is the local time it
        # was last advanced to (None until the first now_smooth call).
        self._smooth = None
        self._smooth_t = None

    @property
    def ready(self):
        """True once at least one snapshot has been recorded."""
        return self._anchor is not None

    @property
    def offset(self):
        """The current anchor's HOST stamp (None before the first
        snapshot). This is the point-slope anchor's host_time (h_anchor) —
        the estimate is `offset + rate * (local_time - anchor_local_time)`.
        Exposed for the debug overlay."""
        if self._anchor is None:
            return None
        return self._anchor[1]

    @property
    def rate(self):
        """Current host sim-rate estimate (1.0 = real-time). Exposed for
        the debug overlay."""
        return self._rate

    @property
    def last_jitter(self):
        """The per-arrival jitter sample of the LAST recorded snapshot, or
        None until one exists.

        This is the PREDICTION RESIDUAL — how far this arrival's stamp is
        from where the PREVIOUS model (the previous anchor + rate)
        predicted it would be:
        `|host_sim_time - (h_anchor_prev + rate * (local_time - t_anchor_prev))|`.
        A steady arrival sits near 0; a late/early packet spikes. Exposed
        so LatencyTracker can consume it (and for the debug overlay)."""
        return self._last_jitter

    def record(self, local_time, host_sim_time):
        """Record a snapshot that was taken at `host_sim_time` (its wire
        stamp) and observed at `local_time`. Out-of-order stamps (should not
        happen over TCP) are dropped, like SnapshotBuffer.push.

        Returns True if the sample was recorded, False if it was dropped
        (the caller feeds the jitter sample to the LatencyTracker only on
        a recorded arrival).

        The model is `host = h_anchor + rate * (local_time - t_anchor)`,
        anchored on the NEWEST sample. On each recorded arrival:
          1. the jitter sample is the PREDICTION RESIDUAL — how far this
             arrival's stamp is from where the PREVIOUS anchor + PREVIOUS
             rate predicted it;
          2. the RATE EMA is updated from the Δhost/Δlocal of the two
             newest samples;
          3. the anchor is MOVED to this newest sample (t_anchor, h_anchor)
             = (local_time, host_sim_time).
        Because the anchor is always the newest sample, `now()`
        extrapolates forward from real data at the fitted rate — and the
        offset it adds is a small delta (time since the newest sample), so
        the estimate stays well-conditioned for the whole session."""
        if self._anchor is not None and host_sim_time <= self._anchor[1]:
            return False
        if self._anchor is not None:
            lt0, ht0 = self._anchor
            # The jitter sample is the PREDICTION RESIDUAL — how far this
            # arrival's stamp is from where the PREVIOUS model (previous
            # anchor + PREVIOUS rate) predicted it. Computed with the rate
            # BEFORE this sample updates it (the standard innovation:
            # measurement minus the prior model's prediction), so a late
            # packet spikes the full amount instead of being partially
            # absorbed into the rate EMA.
            predicted = ht0 + self._rate * (local_time - lt0)
            self._last_jitter = abs(host_sim_time - predicted)
            if local_time > lt0:
                r = (host_sim_time - ht0) / (local_time - lt0)
                r = max(self.RATE_MIN, min(self.RATE_MAX, r))
                self._rate += self.RATE_ALPHA * (r - self._rate)
        else:
            self._last_jitter = 0.0
        # Move the anchor to this newest sample (point-slope re-anchor).
        self._anchor = (local_time, host_sim_time)
        return True

    def now(self, local_time):
        """Estimated host sim time at `local_time` (None before the first
        snapshot). Capped at the newest sample's stamp + MAX_LEAD — see
        the class docstring for why the estimate must stay anchored on the
        data when the host starves.

        Point-slope form — extrapolate FORWARD from the newest sample (the
        anchor) at the fitted rate:
        `h_anchor + rate * (local_time - t_anchor)`. The offset is a small
        delta (time since the newest sample), so the estimate stays
        well-conditioned for the whole session."""
        if self._anchor is None:
            return None
        t_anchor, h_anchor = self._anchor
        est = h_anchor + self._rate * (local_time - t_anchor)
        newest = h_anchor          # the anchor IS the newest sample
        if est > newest + self.MAX_LEAD:
            return newest + self.MAX_LEAD
        return est

    def now_raw(self, local_time):
        """The UNCAPPED estimate (diagnostics): same as `now()` but without
        the MAX_LEAD cap. Exposed so the ghost-clock instrumentation
        (ghost_clock.csv) can see whether the cap is binding — if
        `now_raw` is ahead of `now` by the cap, the estimate is being held
        at `newest + MAX_LEAD`. None before the first snapshot."""
        if self._anchor is None:
            return None
        t_anchor, h_anchor = self._anchor
        return h_anchor + self._rate * (local_time - t_anchor)

    def now_smooth(self, local_time):
        """The SMOOTH host-time estimate — the render clock.

        `now()` (the raw point-slope estimate) is correct on average but
        NOISY frame-to-frame: the anchor re-anchors on every snapshot
        arrival, the rate EMA wobbles, and the MAX_LEAD cap freezes the
        estimate at `newest + MAX_LEAD` and then jumps it when a new
        snapshot releases it. The ghost's render pose is
        `physics_pose + v * _acc`, and `_acc` is driven by this clock —
        so every wobble in the clock is a wobble in the rendered ship.

        This is the render's clock instead. It advances at the smoothed
        rate (1x real time in steady state) and applies a SLOW, bounded
        correction toward the raw capped estimate (`now`), clamped so it
        never moves more than SMOOTH_MAX_JUMP per frame and never goes
        BACKWARD (monotonic). The result is a clock that:
          - tracks the host's clock (the slow correction pulls it to the
            data, so it can't drift away over a session);
          - is smooth (a noisy anchor can't jerk it — the correction is a
            slow pull, and the per-frame jump is bounded);
          - is monotonic (no backward jumps — the render never reverses).
        The raw `now()` / `now_raw()` are unchanged (diagnostics + the
        existing estimator tests); only the RENDER reads this.
        """
        if self._anchor is None:
            return None
        target = self.now(local_time)  # the raw capped estimate
        if self._smooth is None:
            # First call: seed on the raw estimate (no correction yet).
            self._smooth = target
            self._smooth_t = local_time
            return self._smooth
        prev = self._smooth
        # Advance at the smoothed rate (1x real time in steady state) —
        # this is the smooth part. `local_time` is monotonic, so this
        # never goes backward.
        dt = max(0.0, local_time - self._smooth_t)
        self._smooth += self._rate * dt
        self._smooth_t = local_time
        # Slow, bounded correction toward the raw estimate (the tracking
        # part). The residual is small in steady state (the anchor is
        # recent), so this is a gentle pull; on a hiccup it's large, so
        # the clamp keeps it to a bounded per-frame nudge.
        err = target - self._smooth
        corr = self.SMOOTH_ALPHA * err
        corr = max(-self.SMOOTH_MAX_JUMP, min(self.SMOOTH_MAX_JUMP, corr))
        self._smooth += corr
        # Monotonic: never go backward (a backward clock reverses the
        # render). The advance is non-negative; the correction can be
        # negative, so clamp the net to >= the previous value.
        if self._smooth < prev:
            self._smooth = prev
        return self._smooth


class LatencyTracker:
    """The adaptive interpolation delay.

    A fixed INTERP_DELAY (0.1 s) is the right lag for a clean connection,
    but on a jittery link a snapshot that arrives late strands the render
    point ahead of the data: the buffer clamps to the newest snapshot and
    the remote render stutters. The fix is to render FURTHER in the past
    when the arrival pattern is ragged: the delay grows with the measured
    jitter, so a late packet lands inside the window instead of behind the
    render point.

    The delay is a PURE FUNCTION OF THE ARRIVAL PATTERN — `RenderPoint`
    reads `.delay` every frame. The input is the HostTimeEstimator's
    per-arrival jitter sample (`HostTimeEstimator.last_jitter`): a steady
    stream sits near 0, a late/early packet spikes.

    The law (config knobs, see config.py):

        delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter),
                      INTERP_DELAY_MIN, INTERP_DELAY_MAX)

    with EMA alpha = 0.2 (the sample stream settles within a few
    snapshots). Two disciplines on top of the clamp:

      - the delay only ever moves by at most MAX_STEP = 1/60 s per
        frame, so the render point (newest_snap - delay) can never jump —
        a jitter spike raises the delay over ~20 frames instead of
        teleporting the render ~100 ms into the past;
      - the delay never shrinks below INTERP_DELAY_MIN (= INTERP_DELAY,
        the base) — adaptation only adds lag, it never goes under the
        clean-connection value.

    Wiring: the client calls `estimator.record(now, stamp)` on each
    snapshot arrival and, when it returns True,
    `tracker.update(estimator.last_jitter)`; every frame it calls
    `tracker.tick(dt)` (the per-frame smoothing step) and reads
    `tracker.delay`.
    """

    ALPHA = 0.2          # jitter EMA
    MAX_STEP = 1.0 / 60.0  # the delay may move at most this far per frame

    def __init__(self, base=INTERP_DELAY, k=ADAPT_K,
                 delay_min=INTERP_DELAY_MIN, delay_max=INTERP_DELAY_MAX):
        self._base = base
        self._k = k
        self._min = delay_min
        self._max = delay_max
        self._jitter_ema = 0.0
        self._delay = base
        self._samples = 0

    @property
    def delay(self):
        """The current adaptive delay in seconds. Always within
        [INTERP_DELAY_MIN, INTERP_DELAY_MAX]; INTERP_DELAY before the
        first sample."""
        return self._delay

    @property
    def jitter_ema(self):
        """The current EMA of the per-arrival jitter samples (None before
        the first sample). Exposed for the debug overlay."""
        if self._samples == 0:
            return None
        return self._jitter_ema

    @property
    def samples(self):
        """Number of jitter samples consumed (diagnostics)."""
        return self._samples

    def update(self, jitter_sample):
        """Consume one per-arrival jitter sample (a non-negative seconds
        value from HostTimeEstimator.last_jitter). Only called on a
        RECORDED snapshot arrival (estimator.record returned True) — a
        dropped out-of-order stamp carries no new information."""
        if self._samples == 0:
            self._jitter_ema = jitter_sample
        else:
            self._jitter_ema += self.ALPHA * (jitter_sample - self._jitter_ema)
        self._samples += 1
        self._target()

    def tick(self, dt):
        """Per-frame smoothing step: move the delay toward its target by
        at most MAX_STEP (1/60 s), so the render point it drives can
        never jump. `dt` is the real frame time; the step is frame-BOUND
        (one frame may move at most MAX_STEP regardless of dt), which is
        what keeps the render point continuous on any display refresh
        rate."""
        self._step_toward(self._target())

    def _target(self):
        """The unsmoothed delay the EMA wants: BASE + k * EMA(jitter),
        clamped to [MIN, MAX]."""
        d = self._base + self._k * self._jitter_ema
        return max(self._min, min(self._max, d))

    def _step_toward(self, target):
        if target > self._delay:
            self._delay = min(target, self._delay + self.MAX_STEP)
        elif target < self._delay:
            self._delay = max(target, self._delay - self.MAX_STEP)


class RenderPoint:
    """The client's render point in HOST SIM TIME.

    The render point is anchored on the DATA, not on a model of the host
    clock (a model and the data disagree under jitter/loss, and the
    render must not wander by the disagreement):

        render_t = newest_arrived_snapshot_stamp - delay

    a fixed distance behind the newest snapshot in the buffer. A late
    packet then simply holds the point on the last window (a <=1-frame
    stall) instead of the point outrunning the data and
    clamp-stuttering, and the point can never exceed the newest
    snapshot (delay >= INTERP_DELAY_MIN > 0).

    One subtlety: `newest - delay` on its own is a STAIRCASE. The newest
    stamp is flat between arrivals and jumps by one snapshot interval
    when a packet lands, so the raw anchor would jump forward by a whole
    snapshot interval every arrival — a telegraph, not a render. The point
    therefore CHASES the anchor: each frame it moves toward
    `newest - delay` at at most `dt` (the frame's REAL time), so the
    point advances at 1x real time — the same rate the anchor creeps
    forward at (the host's sim rate, ~1x).

    Capping by `dt` (NOT a fixed 1/60 s) is what keeps the point from
    drifting BEHIND the data on a client running below 60 FPS. A fixed
    1/60 cap advances the point by only 1/60 s per frame, but a 47 FPS
    frame lasts 1/47 s of real time, so the point falls behind by
    (1/47 - 1/60) every frame and the remote render looks at ever-older
    snapshots. Capped by `dt`, the point advances 1/47 s per frame at
    47 FPS — exactly 1x real time — and never drifts. The cap still
    smooths the staircase (a one-interval anchor jump is covered over a
    couple of frames at 60 FPS) and still bounds a hiccup (the caller
    clamps `dt` to MAX_FRAME_DT, so one frame moves the point at most
    that far). The net per-frame advance is bounded by `dt` +
    MAX_STEP_DELAY (the chase step plus the delay's own per-frame
    movement) — the render point is continuous on any refresh rate, and
    a jitter spike or a late packet can never teleport it. In steady
    state the chase is slack (the anchor creeps forward at the host's
    sim rate, ~1x, and the point tracks it within ~one frame's step);
    under a stall the anchor holds and the point converges onto it and
    holds there.

    `advance(dt, newest)` is called once per frame with the newest
    snapshot stamp in the buffer (None before the first arrival — the
    point is None until then, and the caller draws its waiting state).
    `now()` returns the render point (host sim time) or None.
    """

    # The chase cap is the frame's REAL time (`dt`), not a fixed 1/60 s —
    # see the class docstring. `dt` is clamped by the caller to
    # MAX_FRAME_DT, which bounds a hiccup.
    MAX_STEP_DELAY = LatencyTracker.MAX_STEP  # the delay's own per-frame move
    # Catch-up fraction. When the point is BEHIND the target by more than
    # ~5 frames (d > 5*dt), it advances by CATCHUP * d per frame (20% of
    # the gap) instead of just dt (1x real time), so a large gap (a
    # snapshot burst after a wire stall, or the buffer filling at the
    # start) closes in ~5 frames (~83 ms at 60 FPS). For small gaps
    # (d < 5*dt) the cap is dt (1x real time), preserving the staircase
    # smoothing. The steady-state gap (where the catch-up rate equals the
    # target's 1x advance) is 5*dt (~0.083 s at 60 FPS) — the point
    # settles just behind the target, inside the buffer's window.
    CATCHUP = 0.2

    def __init__(self, tracker=None):
        self._tracker = tracker
        self._t = None         # the render point (host sim time), or None

    @property
    def tracker(self):
        """The LatencyTracker this point reads its delay from (the debug
        overlay reads both). Settable so tests can swap trackers."""
        return self._tracker

    @tracker.setter
    def tracker(self, trk):
        self._tracker = trk

    def advance(self, dt, newest):
        """One frame of real time `dt`. `newest` is the newest snapshot
        stamp in the buffer (host sim time) or None. Returns the render
        point (or None while there is no anchor yet).

        The chase is capped by `dt` (the frame's REAL time), NOT a fixed
        1/60 s: the point advances at 1x real time on any refresh rate,
        so it never drifts behind the data on a client running below
        60 FPS. `dt` is clamped by the caller to MAX_FRAME_DT, which
        bounds a hiccup (one frame moves the point at most that far).

        Catch-up: a pure 1x cap has a flaw — it closes a LARGE gap at
        only 1x real time, and if the target also advances at ~1x (the
        host's sim rate), the gap NEVER closes. So when the point is
        behind by more than ~5 frames (d > 5*dt), advance by CATCHUP * d
        per frame (20% of the gap) instead of dt, so the gap closes
        exponentially and settles at the steady-state gap of 5*dt
        (~0.083 s at 60 FPS — just behind the target, inside the buffer's
        window). For small gaps (d < 5*dt) the cap is dt (1x real time),
        so the staircase smoothing is unchanged and the point never
        teleports. The step is always bounded by d (min(d, ...)), so the
        point can never overshoot the target. See the CATCHUP class attr
        for the rationale."""
        if newest is None:
            return None
        if self._t is None:
            # First anchor: sit at the anchor (it is behind the newest
            # snapshot by the delay, so inside the buffer's window).
            self._t = newest - self._delay()
            return self._t
        target = newest - self._delay()
        d = target - self._t
        if d > 0.0:
            # Catch-up: when behind by more than ~5 frames (d > 5*dt),
            # advance by CATCHUP * d per frame (20% of the gap) instead of
            # just dt (1x real time), so a large gap closes in ~5 frames.
            # For small gaps (d < 5*dt) the cap is dt (1x real time),
            # preserving the staircase smoothing. See the CATCHUP class
            # attr for the rationale.
            step = min(d, max(dt, self.CATCHUP * d))
            self._t += step
        elif d < 0.0:
            # Ahead of the target (the delay shrank): retreat at 1x real
            # time (no catch-up needed — the point is already ahead).
            self._t -= min(-d, dt)
        return self._t

    def now(self):
        """The current render point (host sim time), or None before the
        first snapshot arrived."""
        return self._t

    def _delay(self):
        trk = self._tracker
        if trk is None:
            return INTERP_DELAY
        return trk.delay


def _enemy_pos(e_s):
    # e_s = (ship_s, hp, id, acc_x, acc_y); the ship snapshot is first.
    return (e_s[0][0], e_s[0][1])


def _enemy_id(e_s):
    # The ship id: assigned once from the AIEnemy._next_id counter and
    # never reused within a run — a stable identity across snapshots.
    return e_s[2]


def _asteroid_key(a_s):
    # a_s = (id, pos.x, pos.y, vel.x, vel.y, size, angle, spin, verts).
    # The id is assigned once from the Asteroid._next_id class counter
    # and never reused within a run — a stable identity across snapshots.
    # Split children get fresh ids -> they read as "new".
    return a_s[0]


def _beam_blocked_end(beam, rocks):
    """Client-side mirror of the host's `_beam_blocked` +
    `_beam_hit_asteroid` visual-endpoint math: clamp a laser beam's
    endpoint to the nearest asteroid that intersects the beam segment, so
    the client's OWN beam stops at the rock's surface (matching the host)
    instead of punching through to the target.

    `beam` is a live `Beam` (`.start` / `.end` world-space Vector2s — the
    muzzle and the raw target position). `rocks` is a list of
    `_GhostRockProxy` (`.pos` / `.collision_radius`) built from the
    buffer's asteroid entries. Returns the beam's visual endpoint as a
    Vector2: the rock's surface point when a rock blocks the segment (the
    SAME nearest-by-t + closest-point + surface-point math the host uses),
    or `beam.end` unchanged when nothing blocks it.

    Presentation only — the ghost is never fed back to the sim, so this
    only affects what the client draws. Mirrors the host exactly so the
    client's beam and the host's beam stop at the same point."""
    seg = beam.end - beam.start
    seg_len2 = seg.length_squared()
    if seg_len2 < 1e-6 or not rocks:
        return beam.end
    best_t, best_a, best_pt = 1.0, None, None
    for a in rocks:
        t = (a.pos - beam.start).dot(seg) / seg_len2
        t = max(0.0, min(1.0, t))          # clamp: only between muzzle and target
        closest = beam.start + seg * t
        if closest.distance_to(a.pos) < a.collision_radius and t < best_t:
            best_t, best_a, best_pt = t, a, closest
    if best_a is None:
        return beam.end
    # Surface point on the rock facing the beam's closest approach — the
    # host's `_beam_hit_asteroid` vis_end math.
    d = best_pt - best_a.pos
    if d.length_squared() < 1e-6:
        d = seg
    d = d.normalize()
    return best_a.pos + d * best_a.collision_radius


def _match_bullets(prev_list, curr_list, dt):
    """Match curr bullets to prev bullets by PREDICTED POSITION. Returns a
    list, one entry per curr bullet: the prev twin's (x, y) or None (a
    new bullet — pops in at its curr position, the same membership rule
    as enemies/asteroids).

    Why not match by INDEX within a kind? The sim's bullet lists are
    sliding windows: a fired bullet appends, an expired one is culled
    from the FRONT (`self.bullets = [b for b in self.bullets if b.life >
    0]`), so every later index slides down by one each time a bullet
    dies. Index matching would lerp a bullet from a stranger's position
    and make it jump.

    Bullets fly in straight lines (no steering — only missiles steer,
    and they are matched with the same method), so a prev bullet's
    position at the curr snapshot is exactly prev.pos + prev.vel * dt.
    Each curr bullet is matched to the nearest prev bullet's predicted
    position within MATCH_RADIUS; a prev bullet is used at most once
    (two bullets can't occupy the same predicted spot). dt is the
    snapshot window (t_curr - t_prev); at dt == 0 the prediction is the
    prev position itself.

    MATCH_RADIUS: a bullet moves BULLET_SPEED * dt per window (~14 px at
    the 60 Hz cadence). The radius must exceed that (a real twin is
    always within ~1 window of motion of its prediction) but be small
    enough that two DIFFERENT bullets never both fall inside it of each
    other's predictions. Bullets from the same gun fire from the same
    muzzle ~10 px apart at 100 Hz, so they can be as close as ~10 px;
    30 px is well clear of that while a mis-match would require two
    bullets' predicted positions to be within 30 px of the same curr
    bullet — only possible for near-coincident fire, where swapping the
    lerp pair is visually a no-op (same muzzle, same velocity).
    """
    MATCH_RADIUS = 30.0
    # Predicted curr-time position of each prev bullet.
    preds = [((p[0] + p[2] * dt, p[1] + p[3] * dt), i)
             for i, p in enumerate(prev_list)]
    used = set()
    out = []
    for c in curr_list:
        best, best_d = None, MATCH_RADIUS
        for (px, py), i in preds:
            if i in used:
                continue
            d = math.hypot(c[0] - px, c[1] - py)
            if d <= best_d:
                best, best_d = i, d
        if best is None:
            out.append(None)          # new bullet: pops in at curr pos
        else:
            used.add(best)
            out.append((prev_list[best][0], prev_list[best][1]))
    return out


def interp_positions(prev_s, curr_s, alpha, dt=None):
    """Interpolated poses for each player ship ((x, y, angle)), each enemy
    ((tag, x, y, angle, vx, vy, id)), each asteroid ((id, x, y, angle)),
    and each projectile ((x, y, vx, vy, kind, owner, boost, mid)), between
    two Game snapshots.

    prev_s / curr_s are the 11-tuples from Game.snapshot(); alpha in [0, 1]
    is clamped (never extrapolate into the future). Returns a plain dict —
    no pygame objects, no sim state touched:

        {'ships': [(x, y, angle, dead, shield_dump, shield_clock,
                    flame_mags, brownout, power_factor), ...],
         'enemies': [(tag, x, y, angle, vx, vy, id,
                      shield_dump, shield_clock, power_used,
                      flame_mags, shield_charge), ...],
         'asteroids': [(id, x, y, angle), ...],
         'bullets': [(x, y, vx, vy, kind, owner, boost, mid), ...]}

    LERP'd across the window: pose, angle (via `lerp_angle` — the wrapped
    delta, never a plain lerp of the raw unbounded angles), shield state
    (shield_dump / shield_clock), `power_factor`, and enemy `power_used`.
    Taken from the CURRENT snapshot (a latch/dict has no meaningful
    cross-window blend): `dead` (ship_s[20]), `brownout` (ship_s[16]),
    `flame_mags` (ship_s[23]).

    Each projectile entry carries `mid`: the unique id (player_index, seq)
    for kind 'missile' and 'player' (the ghost's own gun shots), else None
    (kind 'enemy') — the client's dedup key against the buffer's copy
    (see predicted_view).

    Enemy entries: angle lerp'd with the same wrapped-delta rule as player
    ships; TAG and VELOCITY from curr (presentation data for the
    targeting proxy); ID carried so a caller can track a specific enemy
    across frames (enemies churn — no stable index).

    Projectiles: snapshot indices 2 (player bullets), 3 (enemy bullets),
    4 (missiles) are all interpolated. Position lerp'd (matched by
    predicted position — see `_match_bullets`), velocity/owner/boost from
    curr. A bullet fired mid-window pops in at its curr position; one
    that expires drops out.

    Entity matching (full contract in ARCHITECTURE.md): player ships by
    INDEX (ships don't turn over), enemies by ship id, asteroids by rock
    id, projectiles by predicted position within a kind (index matching
    is wrong — the sim's lists slide). Membership follows the CURRENT
    snapshot: a new entity pops in at its curr position, a dead one drops
    out. Output order is curr_s's order.

    Asteroids: the rock entry is read in EITHER form (see the module
    docstring's [5] note) — the authoritative sim 9-tuple (loopback/tests)
    or the pruned wire 5-tuple (the network). id/x/y are at the same
    indices in both; the angle is [6] in the sim form and [3] in the wire
    form (detected by len). The angle is lerp'd with lerp_angle (the same
    wrapped-delta rule as ships/enemies) so the client's rock spins
    smoothly; x/y are lerp'd as before. The client's rock-shape registry
    (Game.rock_shapes) is populated separately from the wire verts —
    interp_positions never touches verts.
    """
    a = min(1.0, max(0.0, alpha))

    # Player ships: matched by index (see docstring). A ship missing from
    # prev_s[0] (shouldn't happen — ships don't turn over) pops in at its
    # curr pose. The prev ship carries (x, y, angle, shield_dump,
    # shield_clock, power_factor) — lerp'd with the pose; flame_mags
    # (ship_s[23]) and brownout (ship_s[16]) come from the CURRENT
    # snapshot (like `dead`), NOT lerp'd.
    prev_ships = {i: (p[0], p[1], p[4], p[6], p[7], p[17])
                  for i, p in enumerate(prev_s[0])}
    ships = []
    for i, c in enumerate(curr_s[0]):
        cp = (c[0], c[1], c[4], c[6], c[7], c[17])
        pp = prev_ships.get(i)
        if pp is None:
            ships.append((cp[0], cp[1], cp[2], c[20], cp[3], cp[4],
                          dict(c[23]), c[16], cp[5]))
        else:
            ships.append((lerp(pp[0], cp[0], a),
                          lerp(pp[1], cp[1], a),
                          lerp_angle(pp[2], cp[2], a),
                          c[20],
                          lerp(pp[3], cp[3], a),
                          lerp(pp[4], cp[4], a),
                          dict(c[23]),
                          c[16],
                          lerp(pp[5], cp[5], a)))

    # Enemies: carry (tag, x, y, angle, vx, vy). The angle is lerp'd with
    # lerp_angle (the same wrapped-delta rule as player ships); the tag
    # and velocity come from the CURRENT snapshot. The prev enemy carries
    # (x, y, angle, shield_dump, shield_clock, power_used) — lerp'd with
    # the pose; a new enemy pops in at its curr values. flame_mags
    # (ship_s[23]) and shield_charge (ship_s[5]) come from the CURRENT
    # snapshot, NOT lerp'd (a state latch, like `dead`/`brownout` — the
    # client's option-A trigger reads shield_charge to decide whether to
    # spark at the shield boundary or the hull).
    prev_enemies = {}
    for _tag, p in prev_s[1]:
        es = p[0]
        prev_enemies[_enemy_id(p)] = (es[0], es[1], es[4], es[6], es[7],
                                      es[22])
    enemies = []
    for tag, c in curr_s[1]:
        ce = c[0]
        cpos = (ce[0], ce[1])
        eid = _enemy_id(c)
        pp = prev_enemies.get(eid)
        if pp is None:
            enemies.append((tag, cpos[0], cpos[1], ce[4], ce[2], ce[3],
                            eid, ce[6], ce[7], ce[22], dict(ce[23]),
                            ce[5]))
        else:
            enemies.append((tag,
                            lerp(pp[0], cpos[0], a),
                            lerp(pp[1], cpos[1], a),
                            lerp_angle(pp[2], ce[4], a),
                            ce[2], ce[3],
                            eid,
                            lerp(pp[3], ce[6], a),
                            lerp(pp[4], ce[7], a),
                            lerp(pp[5], ce[22], a),
                            dict(ce[23]),
                            ce[5]))

    # Asteroids: the rock entry is read in EITHER form (sim 9-tuple or wire
    # 5-tuple — see the module docstring's [5] note). id/x/y are at the
    # same indices in both; the angle is [6] in the sim form and [3] in
    # the wire form (detected by len). The prev rock carries (x, y, angle);
    # a new rock pops in at its curr values. x/y lerp'd, angle via
    # lerp_angle (the same wrapped-delta rule as ships/enemies).
    def _rock_angle(r):
        return r[6] if len(r) > 6 else r[3]
    prev_rocks = {_asteroid_key(p): (p[1], p[2], _rock_angle(p))
                  for p in prev_s[5]}
    asteroids = []
    for c in curr_s[5]:
        cp = (c[1], c[2], _rock_angle(c))
        pp = prev_rocks.get(_asteroid_key(c))
        if pp is None:
            asteroids.append((c[0], cp[0], cp[1], cp[2]))
        else:
            asteroids.append((c[0],
                              lerp(pp[0], cp[0], a),
                              lerp(pp[1], cp[1], a),
                              lerp_angle(pp[2], cp[2], a)))

    # Projectiles: all three kinds, matched by predicted position within a
    # kind (see _match_bullets). `dt` is the time BETWEEN the two
    # snapshots — the span the prev bullets travel before the curr
    # snapshot. `positions_at` passes the bracketing span it already
    # computed (exact); a direct call (tests) falls back to the nominal
    # snapshot cadence. The position lerp itself uses alpha (exact); dt
    # only scales the predicted-position matching, and MATCH_RADIUS is
    # generous enough that a slightly-off dt still matches correctly.
    if dt is None:
        dt = SNAPSHOT_INTERVAL * TICK
    bullets = []
    # Missiles and PLAYER bullets carry their unique id (player_index, seq)
    # — the client's dedup key (its own ghost projectiles are suppressed
    # against the buffer's copy of the same id; the other player's
    # projectiles are drawn from the buffer). Enemy bullets carry None
    # (the ghost never dedups them — always drawn from the buffer).
    for kind, idx, boost_field, id_field in (("player", 2, None, 6),
                                             ("enemy", 3, None, None),
                                             ("missile", 4, 6, 8)):
        prev_list = prev_s[idx]
        curr_list = curr_s[idx]
        matches = _match_bullets(prev_list, curr_list, dt)
        for c, pp in zip(curr_list, matches):
            if pp is None:
                x, y = c[0], c[1]
            else:
                x = lerp(pp[0], c[0], a)
                y = lerp(pp[1], c[1], a)
            boost = c[boost_field] if boost_field is not None else 0.0
            mid = c[id_field] if id_field is not None else None
            bullets.append((x, y, c[2], c[3], kind, c[4], boost, mid))

    return {'ships': ships, 'enemies': enemies, 'asteroids': asteroids,
            'bullets': bullets}


class SnapshotBuffer:
    """The remote peer's interpolation store.

    The authoritative peer stamps each `Game.snapshot()` with the sim time
    it was taken at and sends it on. The remote peer pushes them here, in
    arrival order, and renders `INTERP_DELAY` seconds in the PAST: it asks
    `positions_at(sim_time - INTERP_DELAY)` for the interpolated pose
    (x, y, angle) of each ship and the (x, y) of each enemy and each
    asteroid.

    Why the delay: a snapshot taken at sim time T is only usable once it has
    ARRIVED, which is at least one round-trip later. Rendering at
    `now - INTERP_DELAY` (with INTERP_DELAY >= the snapshot interval) means
    the render point always sits inside a window whose two bounding
    snapshots are already in hand — no extrapolation into the future, no
    rubber-banding. See config.INTERP_DELAY.

    Membership and matching are by IDENTITY (same rules as
    `interp_positions`): an entity present in the newer snapshot is rendered;
    a brand-new one pops in at its newer position; one that died drops out.
    This is what makes the buffer trackable across entity turnover — the
    no-teleport test (test_interpolation part d) relies on it.

    The buffer is a PURE presentation store: it holds plain snapshot tuples
    and sim times, never touches sim state, and never mutates its inputs.
    """

    def __init__(self, max_snapshots=None):
        # (sim_time, snapshot) pairs, oldest first. Bounded so a stalled
        # authoritative peer can't grow the buffer without limit.
        # The default depth is config.SNAPSHOT_BUFFER_MAX (24) — deep
        # enough to bracket the render point at the 60 Hz cadence even
        # when the adaptive delay is at its INTERP_DELAY_MAX cap (see
        # config). Pass an explicit value to override (tests use 200 for
        # long sweeps).
        if max_snapshots is None:
            max_snapshots = SNAPSHOT_BUFFER_MAX
        self._snaps = []
        self._max = max_snapshots

    def __len__(self):
        return len(self._snaps)

    def newest_time(self):
        """The sim time of the newest snapshot in the buffer (None when
        empty). The client's render clock anchors on this — the host's
        own clock as carried by the wire — instead of the display's wall
        clock."""
        if not self._snaps:
            return None
        return self._snaps[-1][0]

    def latest_snapshot(self):
        """The newest snapshot in the buffer (None when empty).

        The ghost's handback decision needs the missile list of the
        NEWEST snapshot — the one that actually carries a just-launched
        missile — not just its missile_seq. The seq alone is necessary
        but not sufficient: a missile the host launched after the newest
        snapshot is provably NOT in the buffer yet, even though its seq
        is below any future snapshot's seq."""
        if not self._snaps:
            return None
        return self._snaps[-1][1]

    def push(self, sim_time, snap):
        """Record a snapshot taken at `sim_time`. Out-of-order arrivals are
        dropped (the remote render only ever moves forward in sim time)."""
        if self._snaps and sim_time <= self._snaps[-1][0]:
            return
        self._snaps.append((sim_time, snap))
        if len(self._snaps) > self._max:
            del self._snaps[:len(self._snaps) - self._max]

    def positions_at(self, render_t):
        """Interpolated positions at render time `render_t`.

        Returns the same dict shape as `interp_positions`
        ({'ships': [(x, y, angle, dead, shield_dump, shield_clock,
                     flame_mags), ...],
        'enemies': [(tag, x, y, angle, vx, vy, id,
                     shield_dump, shield_clock, power_used,
                     flame_mags), ...],
        'asteroids': [(id, x, y, angle), ...],
        'bullets': [(x, y, vx, vy, kind, owner, boost, mid), ...]}) or None
        when there is not yet a window to interpolate between (fewer than
        two snapshots, or render_t before the first snapshot).

        The window is the two snapshots that BRACKET render_t: the newest
        snapshot at or before render_t is `prev`, the next one is `curr`,
        and alpha = (render_t - t_prev) / (t_curr - t_prev) in [0, 1].
        render_t is clamped to the window — never extrapolate. The
        bracketing span (t_curr - t_prev) is passed to interp_positions as
        `dt` so projectile predicted-position matching uses the REAL
        window, not the nominal cadence.
        """
        n = len(self._snaps)
        if n < 2:
            return None
        t0, s0 = self._snaps[0]
        if render_t <= t0:
            # Before the first snapshot: sit exactly on it (alpha 0).
            # prev is curr, so the projectile window is 0 (a bullet's
            # predicted position IS its curr position — exact match).
            return interp_positions(s0, s0, 0.0, dt=0.0)
        tN, sN = self._snaps[-1]
        if render_t >= tN:
            # At/after the newest: sit exactly on it (alpha 1). The remote
            # render lags the sim by INTERP_DELAY, so this is the steady
            # state between snapshots, not a look-ahead. prev is curr, so
            # the projectile window is 0 (exact match, as above).
            return interp_positions(sN, sN, 1.0, dt=0.0)
        # Find the bracketing pair: the newest snapshot at or before
        # render_t is prev; the one after it is curr.
        for i in range(n - 1):
            ti, si = self._snaps[i]
            tj, sj = self._snaps[i + 1]
            if ti <= render_t <= tj:
                span = tj - ti
                alpha = 0.0 if span <= 0.0 else (render_t - ti) / span
                # Pass the REAL bracketing span as the projectile window:
                # predicted-position matching needs the time the prev
                # bullets travel before the curr snapshot.
                return interp_positions(si, sj, alpha, dt=span)
        # Unreachable: render_t is strictly inside (t0, tN).
        return None


class GhostMissile:
    """A presentation missile the prediction ghost fired.

    The client never runs the sim, so without this the player's own
    missiles appear only when the host's next snapshot arrives (~100 ms
    later) — and the buffer's copy of the SAME missile would then be drawn
    a second time (the double-draw). This is the client-side twin of the
    host's `bullets.Missile`: it carries the SAME unique id the host
    assigns at launch (`(player_index, seq)` — the ghost's `missile_seq`
    is resynced to the host's via the ship snapshot on reconcile), so the
    client can dedup it against the buffer's copy (predicted_view skips a
    buffer missile whose id matches a ghost missile).

    HOMING is client-side: the ghost has no live AIEnemy list, so the
    missile steers toward the buffer's enemy PROXY for its target id
    (re-resolved each step — if the target died, it is gone from the
    buffer and the missile coasts). The steering math mirrors the host's
    `Missile.update` (boost ramp, then turn-rate-limited seek toward the
    lead point) so the predicted path tracks the host's. It is
    PRESENTATION-ONLY: no collision, no sim feedback — the authoritative
    missile (host) is what actually hits.

    The target is a `_GhostEnemyProxy` (exposes `.pos`, `.vel`, `.id`);
    `lead_point` is the same intercept math the host's `AIEnemy.lead_point`
    uses (constant-velocity lead — the buffer carries no smoothed accel).
    """

    def __init__(self, pos, vel, mid, target=None):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.vel = vel.copy()
        self.id = mid                    # (player_index, seq) — the dedup key
        self.target = target             # _GhostEnemyProxy (or None)
        self.life = MISSILE_LIFE
        self.boost = MISSILE_BOOST_TIME
        # Set by handback_missiles once the buffer's copy is in the NEWEST
        # snapshot. The missile is NOT culled then — the buffer's copy
        # only becomes VISIBLE when the render point reaches the carrying
        # snapshot (~INTERP_DELAY later). Until then the ghost keeps
        # drawing it (no gap). predicted_view hands it off (removes it +
        # seeds the decoupled render offset) on the frame the buffer copy
        # is visible. Cleared implicitly when the missile leaves
        # local_missiles (handed off or life-culled).
        self.pending = False

    def update(self, dt, target=None):
        """Advance one step. `target` is the CURRENT buffer proxy for this
        missile's target id (re-resolved by the caller each step; None when
        the target is gone from the buffer — the missile then coasts).
        Mirrors the host's `Missile.update` (boost ramp, then
        turn-rate-limited seek toward the lead point). The host gates the
        seek on `self.target.hp > 0`; the ghost's equivalent is "the target
        is still in the buffer" — so `self.target` is set to the current
        proxy (or None) each step, and a None target coasts."""
        self.prev_pos = self.pos.copy()
        self.target = target
        if self.boost > 0:
            # straight-line boost, ramp to cruise speed (host's Missile)
            self.boost -= dt
            self.vel += self.vel.normalize() * MISSILE_ACCEL * dt
            if self.vel.length() > MISSILE_SPEED:
                self.vel.scale_to_length(MISSILE_SPEED)
        elif self.target is not None:
            # seek: steer toward the live intercept point, turn-rate limited
            aim = self.target.lead_point(self.pos, self.vel.length())
            if aim is None:
                aim = self.target.pos     # fallback: chase current pos
            desired = aim - self.pos
            if desired.length() > 1:
                desired.normalize_ip()
            cur = self.vel.copy()
            speed = cur.length()
            cur.normalize_ip()
            ang = math.atan2(desired.y, desired.x)
            cur_ang = math.atan2(cur.y, cur.x)
            diff = (ang - cur_ang + math.pi) % (2 * math.pi) - math.pi
            max_turn = MISSILE_TURN_RATE * dt
            turn = max(-max_turn, min(max_turn, diff))
            new_ang = cur_ang + turn
            self.vel = pygame.Vector2(math.cos(new_ang), math.sin(new_ang)) * speed
        self.pos += self.vel * dt
        self.life -= dt


class PredictedShip:
    """Client-side prediction ghost for the local player's ship.

    The remote peer renders its OWN ship from the interpolation buffer at
    `sim_time - INTERP_DELAY`, which feels ~100 ms laggy. The fix is to
    predict the local ship: step a private Ship with the player's OWN
    input at the sim's fixed rate, and reconcile it to the authoritative
    ship snapshot when one lands. This class is that ghost — a full `Ship`
    (not a minimal kinematic stand-in) so reconciliation reuses the
    existing `apply_snapshot` round-trip and the physics stay faithful.

    Clock discipline: the ghost must step at the SIM's fixed rate, not the
    display's. The caller feeds it real frame times via `advance(dt, inp)`;
    the internal accumulator converts that to whole `TICK` steps. Stepping
    once per display frame would make the ghost integrate
    `refresh_rate / 60` times the sim's motion (on a 144 Hz monitor it
    raced 2.4x ahead and every reconcile yanked it back).

    The ghost is a PRESENTATION object: it is stepped with the local input
    and reconciled to authority, but it never feeds the sim. The sim stays
    authoritative on one peer; this only feeds the render.

    Local bullets: the ghost also keeps its OWN gun shots as presentation
    `Bullet`s (`self.local_bullets`), so the player sees their own fire
    immediately instead of waiting ~100 ms for the host's next snapshot.
    They are stepped at the sim's rate (via `advance`) and culled on
    expiry. They are NOT reconciled to the authoritative list: the 2P
    snapshot carries no stable player id (both ships are ship_id 0), so the
    local player's authoritative bullets cannot be isolated by owner — and
    the ghost's own bullets ARE the local player's by construction (same
    input + weapon state the host applies). They are presentation only:
    they never collide, never feed the sim, and the remote peer's own
    bullets still come from the interpolation buffer.

    The Game-fed per-tick fields (`tracked`, `laser_target`,
    `missile_target`, `contacts`) are reset to their idle values before each
    step: the ghost has no enemy list to target, and none of these fields
    affect the ship's pos/vel/angle motion except through the weapon
    charge/lock power draw (which can flip the brownout latch). That power
    draw is exactly the drift the ghost is allowed to have — it is bounded
    by the test, not eliminated.

    `laser_target` and `missile_target` are picked from the caller's
    `enemies` proxies (the interpolation buffer's enemy entries) when
    given — the SAME nearest-in-range math the host's `_pick_laser_target`
    / `_pick_missile_target` use — so the ghost charges + fires a beam and
    locks + launches a missile on the same tick the host does. The fired
    beams are kept as presentation entries in `self.local_beams` (hull-
    local muzzle -> world target, aged at the host's 0.15 s beam ttl;
    predicted_view re-anchors the muzzle to the ghost's current pose at
    draw time). The LAUNCHED missiles are kept as presentation
    `GhostMissile`s in `self.local_missiles` (each with the same unique id
    the host assigns — `(player_index, seq)` — so the client dedups it
    against the buffer's copy; homing is client-side, steering toward the
    buffer's enemy proxy for the target id). Call `step_local_beams` /
    `step_local_missiles` after `step` to advance + cull the lists.

    Full contract (ghost clock, reconcile/rewind, render pose, handoff)
    in ARCHITECTURE.md.
    """

    # How much host time of local input to keep for rewind replay. 2 s is
    # far more than the worst-case snapshot age (the adaptive delay caps
    # the render lag at INTERP_DELAY_MAX = 0.35 s, and a snapshot is at
    # most one interval old), so the buffer never runs out of the inputs
    # a rewind needs — while staying small (~120 entries at 60 Hz input
    # sampling).
    INPUT_BUFFER_MAX = 2.0

    # Diagnostics: an optional callable the game loop sets (when debug_net
    # is on) to receive per-call ghost-clock records. Signature:
    # fn(kind, rec) where kind is 'ADV' (one per advance() call) or
    # 'REC' (one per reconcile_rewind() call) and rec is a dict of plain
    # floats/ints. None by default (tests + the host's ghost): the hook is
    # a no-op. The client loop (run_client) sets it to a writer that
    # appends to ghost_clock.csv.
    diag_sink = None

    # SMOOTHED render pose (render_pose_smoothed): a VELOCITY-BASED
    # controller (feedforward + proportional), not a bounded-rate chase.
    # The smoothed pose's velocity = the target's velocity (feedforward:
    # track the ship's motion exactly, zero steady-state lag) + SMOOTH_GAIN
    # * the gap (proportional: close any stall gap). The gap dynamics are
    # d(gap)/dt = -SMOOTH_GAIN * gap, so the gap closes with a time constant
    # of 1/SMOOTH_GAIN. 15 1/s means the gap closes in ~0.07 s (4 frames at
    # 60 Hz) — fast enough to settle before the player reacts, smooth enough
    # to avoid a lurch.
    #
    # Why a velocity-based controller and not the alternatives:
    # - A BOUNDED-RATE CHASE capped at the ship's ACTUAL speed has a cap of
    #   0 at rest, so a gap a stall left behind could never close (the
    #   "fires from empty space, never gets closer" bug).
    # - Flooring that cap at the ship's MAX speed closes the gap, but makes
    #   the catch-up a constant-velocity slide that rubber-bands when the
    #   player starts moving (the slide and the target's acceleration
    #   disagree — there is no feedforward).
    # - A plain proportional catch-up (no feedforward) settles to a
    #   persistent lag of v/GAIN in steady state (the ship drawn behind
    #   where it is, the muzzle drifting off the bullets).
    # The velocity-based controller gets the best of each: the feedforward
    # tracks the ship's motion exactly (zero steady-state lag, no
    # rubber-band), the proportional term closes any stall gap (works at
    # rest, damped, no overshoot), and there is no constant-velocity slide.
    SMOOTH_GAIN = 15.0            # 1/s, proportional gain (gap time constant = 1/GAIN)
    # Cap on the CATCH-UP velocity (the proportional term), in units of the
    # ship's max speed / max turn rate. Prevents a lurch on a large stall
    # gap (the catch-up is capped at one tick/frame of motion at max speed).
    # The feedforward (the target's velocity) is NOT capped — it always
    # tracks the ship exactly. Capping the catch-up (not the total velocity)
    # means the gap still closes while the ship is at max speed.
    SMOOTH_MAX_CATCH = 1.0        # × max_speed (position), × max_turn_rate (angle)
    # HYBRID snap threshold (in units of one tick of motion at max speed):
    # when the gap is <= SMOOTH_SNAP_FACTOR * max_speed * TICK, the smoothed
    # pose SNAPS to the target exactly (the old behavior — no vibration,
    # because the rendered pose is exactly the render target every frame).
    # When the gap is larger (a stall), the velocity controller is used
    # (smooth catch-up). The factor is 2.0: the steady-state gap is at most
    # 2 ticks (the sim clock can advance 2 ticks in one frame), so 2.0
    # always snaps in steady state (no vibration), and the transition lurch
    # (when a stall gap closes to <= 2 ticks) is at most 2 ticks — the same
    # as a normal burst-frame move, so it is not perceived as a lurch.
    SMOOTH_SNAP_FACTOR = 2.0

    def __init__(self, hull=None, loadout=None, local_index=0):
        # A self-contained Ship: __init__ builds components/thrusters/
        # weapons/shield/sensors/collision from the hull+loadout and holds
        # no reference to Game, so it can be stepped in isolation.
        self._ship = Ship(hull=hull, loadout=loadout)
        # plan_19: the ghost's team == the local player's index — the
        # team predicate (can_target) needs it to tell the remote player
        # (off-team, targetable) from the ghost's own ship (same team,
        # never a target). The host's player ships get the same
        # assignment in Game (team == index); the ghost mirrors it.
        self._ship.team = local_index
        self._seeded = False
        # The render interpolation phase: _sim_time - _last_step_time, in
        # [0, TICK). advance() recomputes it each call from the clock
        # (see the GHOST CLOCK CONTRACT in advance()); render_pose lerps
        # _prev -> curr at alpha = _acc / TICK. 0.0 until the first
        # advance.
        self._acc = 0.0
        # --- branch-3 (test-only) free-run clock state --------------------------
        # These three fields exist ONLY for advance()'s branch 3 (real_time
        # given but host_time not ready — the pre-first-snapshot window and
        # some tests, e.g. test_10_11c). They are NOT the live client clock:
        # on a real client host_time is always ready, so branch 2
        # (_sim_time = host_time.now_smooth(real_time) + rtt_ow) is used and
        # these are never read. See the GHOST CLOCK CONTRACT in advance().
        #
        # _real_base: the ghost's REAL-TIME base (pygame ticks / 1000) of
        # the last reconcile. Branch 3 free-runs at 1x real time from it:
        # _sim_time = _sim_base + _rate * (real_time - _real_base). 0.0
        # until the first reconcile.
        self._real_base = 0.0
        # _sim_base: the SIM time at the last reconcile (= now =
        # snap_time + L). The branch-3 anchor's sim time. Set in
        # reconcile_rewind; 0.0 until the first reconcile.
        self._sim_base = 0.0
        # _rate: the branch-3 free-run rate (sim time per real time).
        # Always 1.0 (the host sim runs at 1x real time). Set in
        # reconcile_rewind / seed; 1.0 until the first reconcile.
        self._rate = 1.0
        # The last sim time at which a physics step was executed (the most recent
        # crossed tick boundary). The step trigger in `advance` is
        # `floor(_sim_time / TICK) > floor(_last_step_time / TICK)` — a step
        # runs once per crossed boundary, and `_acc` is `_sim_time -
        # _last_step_time` (the render interpolation phase). Set in seed()
        # and reconcile_rewind() to the post-reconcile `_sim_time` (no step
        # is owed at the reconcile itself — the replay/apply already put the
        # pose at `_sim_time`).
        self._last_step_time = 0.0
        # The ghost's current PHYSICS sim time (the time of its pose,
        # `s.pos`). The ghost advances in real time between reconciles (via
        # `advance`), so `_sim_time` is AHEAD of the newest snapshot's stamp
        # by the amount of time the ghost has advanced since that snapshot
        # was taken. `reconcile_rewind` uses `_sim_time` as `now` (the replay
        # target) instead of the host-time estimate (which is anchored on the
        # newest snapshot and equals `snap_time` exactly — a 0-tick replay).
        # Updated in `advance` (add n*TICK per step) and `reconcile_rewind`
        # (set to snap_time + n*TICK after the replay). 0.0 until the first
        # seed/reconcile.
        self._sim_time = 0.0
        # The ghost's pose at the START of the current in-progress step, for
        # sub-step render interpolation (render_pose). The ghost is drawn at
        # lerp(_prev, curr, acc/TICK) — the same interpolation the host's
        # local ship gets from _ship_pose(step_alpha) — instead of its raw
        # post-step pos (which moves in 60 Hz jumps on a ~62 Hz display).
        # Set in advance() before each step; reset to curr in
        # seed()/reconcile_rewind() so a reconcile never adds a phantom lerp
        # from the pre-reconcile pose. None until the first seed
        # (render_pose is only called once seeded).
        self._prev_pos = None
        self._prev_angle = None
        # RENDER-ONLY pre-step state (the sawtooth fix). The physics pose (s.pos)
        # sits at the replay end (snap_time + n*TICK), but the render clock
        # (_sim_time) is L = rtt_ow AHEAD of it (the reconcile sets
        # _sim_time = max(now, snap_time + n*TICK) with now = snap_time + L).
        # Rendering the physics pose directly pins the render to snap_time
        # while the clock is at snap_time + L — each reconcile then jumps
        # the pose 1 tick (the sawtooth). The fix: render at the CLOCK's
        # time. _render_base is the pose at the replay end (== s.pos at
        # reconcile time); _render_next is the pose ONE TICK ahead (a
        # kinematic pre-step — see _render_pre_step). render_pose lerps
        # _render_base -> _render_next at alpha = L/TICK, so the render pose
        # sits at snap_time + L (the clock's time) and is continuous across
        # the reconcile. None until the first seed/reconcile.
        self._render_base = None
        self._render_base_angle = None
        self._render_next = None
        self._render_next_angle = None
        # The SMOOTHED render pose (the decoupled visual layer). The
        # physics pose (s.pos) is ragged on burst frames (it jumps 2
        # ticks when the sim clock jumps 2 ticks), so rendering it
        # directly would lurch the ship 2 ticks for one frame (the
        # shudder). The smoothed render pose chases the physics pose
        # (at the clock's time) at a bounded rate (1x real time), so it
        # never jumps more than 1 tick per frame. None until the first
        # render_pose call (then it is initialized to the target).
        self._smooth_pos = None
        self._smooth_angle = None
        # Whether the ship was DEAD on the last render_pose_smoothed
        # call. When the ship respawns (dead->alive), the target pose
        # jumps from the death location to the spawn point, and the
        # smoothed pose would chase it at max_speed * dt per frame —
        # the ship visibly "flies back" to the spawn point. The fix:
        # on the dead->alive transition, reset the smoothed pose to the
        # target instantly (the ship appears at the spawn point, not
        # flying back). Set by render_pose_smoothed.
        self._smooth_dead = False
        # The last ShipInput the ghost was stepped with (set in step()).
        # The render pre-step reads its .turn to predict the angular motion
        # over the L gap. None before the first step (the pre-step then
        # coasts straight — no turn).
        self._last_inp = None
        # The local input buffer — (host_time, ShipInput) pairs, oldest
        # first, bounded to INPUT_BUFFER_MAX seconds of host time.
        # `host_time` is the client's estimate of the host's sim clock at
        # the moment the input was sampled (the HostTimeEstimator), so a
        # snapshot arriving at sim time T can replay the inputs the host
        # actually applied between T and now (see `reconcile_rewind`).
        self._input_buffer = []
        # The input-ECHO buffer — (sim_time, ShipInput) pairs, oldest first,
        # bounded to INPUT_BUFFER_MAX seconds of sim time (the same bound as
        # _input_buffer). These are the inputs the host ACTUALLY applied to
        # the local player (player 1), echoed per tick over the wire
        # (T_ECHO, see net.py) and stamped with the tick's sim_time —
        # authoritative, unlike _input_buffer's client-estimated stamps.
        # reconcile_rewind prefers them for the replay (the host applied the
        # latest input it had RECEIVED, up to one-way latency stale — the
        # client's own sent input diverges from that on an input change).
        # Falls back to _input_buffer when no echo covers a tick (initial
        # transient / loss — a dropped echo holds the last echoed input,
        # exactly what the host itself did).
        self._echo_buffer = []
        # The LAST reconcile's replay length, for the net-debug CSV.
        # _last_replay_ticks = the whole-tick count `n` the rewind replayed;
        # _last_replay_span = the raw (now - snap_time) seconds that `n` was
        # truncated from. Both are 0/0 until the first reconcile (and on the
        # no-replay path).
        self._last_replay_ticks = 0
        self._last_replay_span = 0.0
        # The ghost's OWN gun shots, as presentation Bullets. The client never
        # runs the sim, so without this the player's own shots appear only
        # when the host's next snapshot arrives (~100 ms later).
        # `local_index` is the local player's index (0 = host, 1 = client);
        # it is carried for diagnostics and so a future session can match
        # the authoritative local-owner bullets (the 2P snapshot carries no
        # stable player id — both ships are ship_id 0 — so the ghost's own
        # bullets are the local player's by construction).
        self.local_index = local_index
        self.local_bullets = []
        # The ghost's OWN laser discharges, as presentation entries
        # [start, end, age, ttl] in WORLD space (muzzle -> target). The
        # client never runs the sim, so without this the player's own beam
        # appears only when the host's next snapshot arrives (~100 ms
        # later). Like local_bullets these are presentation only: no
        # collision, no sim feedback.
        self.local_beams = []
        # The ghost's OWN missiles, as presentation GhostMissiles. The
        # client never runs the sim, so without this the player's own
        # missiles appear only when the host's next snapshot arrives
        # (~100 ms later) — and the buffer's copy of the SAME missile would
        # then be drawn a second time (the double-draw). Each carries the
        # SAME unique id the host assigns at launch ((player_index, seq) —
        # the ghost's missile_seq is resynced to the host's via the ship
        # snapshot on reconcile), so predicted_view dedups it against the
        # buffer's copy. Homing is client-side (see GhostMissile).
        self.local_missiles = []
        # V/G/T (option A): when True, the ghost's SENSOR state
        # (targeting_on, sensor_on, scan_cd, scan_reveal, scan_dump,
        # scan_pulse) is CLIENT-AUTHORITATIVE — the client's V/G/T keys
        # mutate it directly (via the command sink) and reconcile must
        # PRESERVE it across apply_snapshot, which would otherwise
        # overwrite it with the host's values (the host never saw the
        # client's V/G/T, so its snapshot carries the host's own sensor
        # state). The host's ship stays authoritative for everything else
        # (pose, weapons, power, shield) — only the sensor fields the
        # client controls locally are preserved. Off by default (the host's
        # ghost and tests keep the host-authoritative behavior).
        self.client_sensor_authoritative = False

    @property
    def ship(self):
        """The underlying ghost Ship (read access for tests/diagnostics)."""
        return self._ship

    @property
    def seeded(self):
        """True once an authoritative snapshot has been applied."""
        return self._seeded

    @property
    def sim_time(self):
        """The ghost's current PHYSICS sim time — the time of its pose, `s.pos`.
        The ghost advances in real time between reconciles (via `advance`),
        so this is AHEAD of the newest snapshot's stamp by the amount of
        time the ghost has advanced since that snapshot was taken.
        `push_snapshot` passes this as `now` to `reconcile_rewind` (the
        replay target) instead of the host-time estimate — the estimate is
        anchored on the newest snapshot and equals `snap_time` exactly,
        which would make the replay 0 ticks and leave the ghost's real-time
        advance un-replayed. Updated in `advance` (adds n*TICK per step)
        and `reconcile_rewind` (sets snap_time + n*TICK after the replay).
        0.0 until the first seed/reconcile."""
        return self._sim_time

    def seed(self, ship_s, snap_time=0.0, real_time=None, rate=1.0):
        """Apply the FIRST authoritative ship snapshot (index 0 of the Game
        snapshot) to the ghost. Subsequent snapshots use `reconcile`.

        `snap_time` (the snapshot's sim time) initializes the ghost's
        physics clock (`_sim_time`) so the first `reconcile_rewind` has a
        valid `now` (the ghost's current sim time) to replay from.

        `real_time` (the client's real time at the seed) anchors the
        branch-3 free-run clock (see the GHOST CLOCK CONTRACT in
        advance())."""
        self._ship.apply_snapshot(ship_s)
        self._seeded = True
        self._sim_time = snap_time   # start the physics clock
        # Anchor the branch-3 free-run clock to the seed. The pose is at
        # snap_time, so _last_step_time = snap_time (render phase 0).
        self._sim_base = snap_time
        self._real_base = real_time if real_time is not None else 0.0
        self._last_step_time = snap_time
        # Branch 3 free-runs at exactly 1x real time (the host sim runs at
        # 1x real time overall). The `rate` param is kept for API
        # compatibility but ignored.
        self._rate = 1.0
        # Start the render interpolation from the seeded pose (no phantom
        # lerp from a pre-seed pose).
        self._prev_pos = pygame.Vector2(self._ship.pos)
        self._prev_angle = self._ship.angle

    def reconcile(self, ship_s):
        """Snap the ghost to an authoritative ship snapshot (full snap, no
        replay). Idempotent with `seed` — a caller may use either for the
        first snapshot. The game loop uses `reconcile_rewind` instead
        (dead-reckoning); this stays for tests and as the simple fallback."""
        self._capture_client_sensor_state()
        self._ship.apply_snapshot(ship_s)
        self._seeded = True
        self._restore_client_sensor_state()
        # The reconcile moved the pose — restart the render interpolation
        # from the new pose (no phantom lerp).
        self._prev_pos = pygame.Vector2(self._ship.pos)
        self._prev_angle = self._ship.angle

    def _capture_client_sensor_state(self):
        """V/G/T (option A): save the ghost's sensor fields before an
        apply_snapshot overwrites them. No-op when the client is not
        authoritative for its sensor state (host's ghost + tests)."""
        if not self.client_sensor_authoritative:
            return
        s = self._ship
        self._client_targeting_on = s.targeting_on
        self._client_sensor_on = s.sensor_on
        self._client_scan_cd = s.scan_cd
        self._client_scan_reveal = s.scan_reveal
        self._client_scan_dump = s.scan_dump
        self._client_scan_pulse = s.scan_pulse

    def _restore_client_sensor_state(self):
        """V/G/T (option A): re-apply the ghost's own sensor state after an
        apply_snapshot, when the client is authoritative for it.

        apply_snapshot restores the host's sensor fields (sensor_on,
        scan_cd, scan_reveal, scan_dump, scan_pulse, targeting_on) — but
        the host never saw the client's V/G/T presses, so those values are
        the host's OWN ship's sensor state, not the client's. When
        `client_sensor_authoritative` is set, the ghost's pre-snapshot
        sensor state (mutated by the client's command sink) is the truth,
        so it is restored on top of the authoritative snap. Everything
        else (pose, weapons, power, shield) stays host-authoritative.
        No-op when the flag is off (host's ghost + tests)."""
        if not self.client_sensor_authoritative:
            return
        s = self._ship
        s.targeting_on = self._client_targeting_on
        s.sensor_on = self._client_sensor_on
        s.scan_cd = self._client_scan_cd
        s.scan_reveal = self._client_scan_reveal
        s.scan_dump = self._client_scan_dump
        s.scan_pulse = self._client_scan_pulse

    def record_input(self, host_time, inp):
        """Record the local input sampled at host time `host_time`.
        `host_time` is the client's estimate of the host's sim clock at
        the moment the input was sampled (the HostTimeEstimator) — NOT the
        local wall clock — so that when a snapshot arrives at sim time T,
        `reconcile_rewind` can replay the inputs the host actually applied
        between T and now: the host applies the LATEST received input each
        tick (pinned #5), so the replay must select inputs by their
        host-time stamp, and the stamp must be in the host's clock for
        that selection to line up.

        The buffer is bounded to INPUT_BUFFER_MAX seconds of host time
        (oldest entries dropped). `inp` is stored by reference; the caller
        passes a fresh ShipInput each frame (ShipInput.from_keys builds a
        new one), so no copy is needed."""
        self._input_buffer.append((host_time, inp))
        # Drop entries older than INPUT_BUFFER_MAX seconds of host time.
        cutoff = host_time - self.INPUT_BUFFER_MAX
        while self._input_buffer and self._input_buffer[0][0] < cutoff:
            self._input_buffer.pop(0)

    def record_echo(self, sim_time, inp):
        """Record one input-echo entry: the input the host ACTUALLY applied to
        the local player on the tick that STARTS at `sim_time` (the T_ECHO
        message's stamp — the host's own sim clock, not the client's
        estimate). The client loop calls this once per T_ECHO entry, in
        arrival order (the wire is in-order TCP, so the buffer stays
        ordered by sim_time).

        Bounded to INPUT_BUFFER_MAX seconds of sim time (oldest dropped) —
        the same bound as `record_input`: a rewind replays at most
        (now - snap_time) of sim time, which is far less than 2 s.

        `inp` is a fresh ShipInput (deserialize_input builds one per
        entry), so it is stored by reference like record_input."""
        self._echo_buffer.append((sim_time, inp))
        cutoff = sim_time - self.INPUT_BUFFER_MAX
        while self._echo_buffer and self._echo_buffer[0][0] < cutoff:
            self._echo_buffer.pop(0)

    def reconcile_rewind(self, ship_s, snap_time, now, enemies=None,
                     real_time=None, rate=1.0, rocks=None,
                     player_proxy=None):
        """Dead-reckoning reconcile. Apply the authoritative ship snapshot taken
        at sim time `snap_time`, then REPLAY the local inputs the host
        applied between `snap_time` and `now`, so the ghost ends up at
        exactly `authority @ snap_time + local inputs since snap_time` —
        the 100 ms of prediction is REBUILT, not snapped (pinned #4).
        Full contract in ARCHITECTURE.md.

        `enemies` (a list of lightweight enemy proxies, see `step`) is fed
        to each replayed step so the replayed laser charge/lock state
        matches the host's (the host's replayed ticks saw the live enemies
        in the wedge). The caller passes the buffer's enemy proxies at the
        snapshot's render point; a None (tests) replays with no targets.

        `rocks` (a list of lightweight asteroid proxies, see `step`) is
        fed to each replayed step so a beam re-fired during the replay is
        clamped to the nearest asteroid that intersects its segment (the
        client-side mirror of the host's `_beam_blocked`) — the same
        occlusion the live steps get.

        `player_proxy` (plan_19) is the remote-player proxy, fed to each
        replayed step as an additional targeting candidate (see `step`) —
        the host's replayed ticks saw the other player's ship as a
        candidate, so the replay must too, or the re-run lock/charge
        desyncs.

        The replay mirrors the host's own tick loop: for each sim tick T
        from snap_time to now (step TICK), the host applies the LATEST
        received input (pinned #5), so the replay steps the ghost with the
        newest buffered input whose host_time <= T. That input-selection
        is what makes the replay reproduce the host's motion — the
        test_interpolation (l) battery proves it to ~1e-9 with the same
        input the host used.

        The local bullet list is rebuilt the same way: `step` re-fires
        from the ghost's re-seeded weapon state (apply_snapshot restores
        cooldown/charge/lock), so the ghost's own shots resume from the
        authoritative fire cadence instead of the predicted one.

        `now` is the ghost's OWN current sim clock (the caller passes
        `ghost.sim_time` — the clock the ghost was advanced to this
        frame, NOT snap_time + rtt_ow). Reconciling at the ghost's own
        clock means the reconcile does NOT jump the clock forward (the
        old snap_time + rtt_ow anchor sat ~3 ms ahead of the raw clock
        and each reconcile jumped it -> a per-reconcile snap); it only
        corrects the pose to the snapshot and re-anchors the tick grid.
        The replay runs at most (now - snap_time)/TICK steps — ~one-way
        latency in steady state (sub-tick on a LAN, so n=0 and the
        render lerp rebuilds the prediction; the replay fires when the
        span is >= 1 tick, e.g. after a hiccup). The fixed-step
        accumulator is reset so the next `advance` starts clean from the
        replayed state.

        V/G/T (option A): the client's sensor state is captured before the
        apply_snapshot (which would overwrite it with the host's) and
        restored after, so the replay advances the CLIENT's sensor state
        forward (the host never saw the client's V/G/T).
        """
        self._capture_client_sensor_state()
        # Diagnostics: the pre-reconcile clock state (the REC record).
        _pre_sim, _pre_last = self._sim_time, self._last_step_time
        # The replay (step, not advance) moves the physics state but does
        # NOT touch `_acc` or `_sim_time`. After the replay, the physics
        # clock is set to max(now, snap_time + n*TICK) and the accumulator
        # to the time since the last replayed step (see the end of this
        # method).
        self._ship.apply_snapshot(ship_s)
        self._seeded = True
        self._restore_client_sensor_state()
        self._acc = 0.0
        # The replay's input source is the host's ACTUAL applied input —
        # the echo buffer (T_ECHO, stamped with the host's own sim clock)
        # — when it has anything, else the client's own sent input
        # (initial transient before the first echo, or total loss). Both
        # buffers are (time, ShipInput) pairs ordered by time, so the
        # pointer math below is a drop-in swap of the source. The echo
        # kills the in-flight-input snapback: on an input CHANGE the
        # client's own input is one-way-latency ahead of what the host
        # actually applied, and the old replay (own input) diverged
        # ~MAX_SPEED x latency before the reconcile yanked it back. A
        # dropped echo degrades gracefully — the monotone pointer holds
        # the last echoed input across the gap, exactly what the host
        # itself did (pinned #5).
        buf = self._echo_buffer or self._input_buffer
        # Record the replay length for the net-debug CSV. `n` (whole ticks
        # replayed) + the raw span (now - snap_time) are the two numbers
        # that decide the ghost's post-reconcile pose. Set to 0/0 on the
        # no-replay path.
        self._last_replay_ticks = 0
        self._last_replay_span = (now - snap_time)
        if not buf:
            # No replay — the pose is the snapshot's; anchor the render
            # interpolation on it (no phantom lerp).
            self._prev_pos = pygame.Vector2(self._ship.pos)
            self._prev_angle = self._ship.angle
            # No replay — the pose IS the snapshot's (at snap_time). Set
            # the physics clock to max(now, snap_time) and the accumulator
            # to the time since the snapshot's pose (the same logic as the
            # replayed path with n=0). Keeping the clock at `now` (not
            # snap_time) avoids pulling it back, which would make the ghost
            # fall behind the host (the negative replay_ticks / lag spikes).
            self._sim_time = max(now, snap_time)
            self._acc = self._sim_time - snap_time
            # Anchor the branch-3 free-run clock to this reconcile (see the
            # GHOST CLOCK CONTRACT in advance()). _last_step_time = the
            # pose's time (snap_time, n=0) — the render interpolation phase
            # is _sim_time - snap_time = L.
            self._sim_base = self._sim_time
            self._real_base = real_time if real_time is not None else 0.0
            self._last_step_time = snap_time
            # Branch 3 free-runs at exactly 1x real time (see seed()).
            self._rate = 1.0
            # Diagnostics: the no-replay REC record.
            self._diag('REC', {'real_time': real_time, 'snap_time': snap_time,
                                'now': now, 'n': 0,
                                'pre_sim': _pre_sim, 'pre_last': _pre_last,
                                'post_sim': self._sim_time,
                                'post_last': self._last_step_time,
                                'post_acc': self._acc})
            return
        # Walk the sim ticks from snap_time to now. `i` is the tick index
        # (0 = snap_time itself, the snapshot's own tick — already applied
        # by the snapshot, so the first replayed tick is snap_time + TICK).
        # Index of the newest buffered input with host_time <= tick time.
        # The buffer is ordered by host_time (input is sampled in order),
        # so this is a monotone pointer.
        j = -1
        # The number of whole TICK steps between snap_time and now. The
        # snapshot reflects the state AFTER the tick that ended at
        # snap_time, so the first replayed tick is the one that STARTS at
        # snap_time (the host applies the input stamped at a tick's start
        # during that tick — the client records the input at the same
        # host-time the host samples it).
        n = int((now - snap_time) / TICK + 1e-9)
        # Clamp n to >= 0. When the ghost's clock falls behind the
        # snapshot (now < snap_time, from a lag spike), the replay span is
        # negative. Replaying negative ticks is meaningless (the ghost is
        # already behind the snapshot), so skip the replay (n=0) and let
        # the accumulator clamp to 0 below. The ghost will catch up on the
        # next reconcile (when now > snap_time again).
        n = max(0, n)
        self._last_replay_ticks = n   # net-debug CSV
        for i in range(n):
            t = snap_time + i * TICK
            while j + 1 < len(buf) and buf[j + 1][0] <= t + 1e-9:
                j += 1
            inp = buf[j][1] if j >= 0 else buf[0][1]
            self.step(TICK, inp, enemies=enemies, rocks=rocks,
                      player_proxy=player_proxy)
            self.step_local_bullets(TICK)
            self.step_local_beams(TICK)
            self.step_local_missiles(TICK, enemies, player_proxy)
        # The replay moved the physics state to snap_time + n*TICK. Set the
        # physics clock to max(now, snap_time + n*TICK):
        #   - normal case (now >= snap_time): the clock stays at `now` (the
        #     ghost's current sim time, AHEAD of the snapshot). This avoids
        #     pulling the clock back (which caused the negative
        #     replay_ticks / lag spikes).
        #   - lag-spike case (now < snap_time, n=0): the clock advances to
        #     snap_time (the ghost was behind the snapshot; the snapshot's
        #     pose is authoritative, so the clock jumps forward to match).
        # The accumulator is the time since the last step:
        #   _sim_time - (snap_time + n*TICK), which is in [0, TICK) for the
        #   normal case and 0 for the lag-spike case.
        self._sim_time = max(now, snap_time + n * TICK)
        self._acc = self._sim_time - (snap_time + n * TICK)
        # Anchor the branch-3 free-run clock to this reconcile (see the
        # GHOST CLOCK CONTRACT in advance()). _last_step_time = the pose's
        # time (snap_time + n*TICK, the replay end) — the render
        # interpolation phase is _sim_time - (snap_time + n*TICK) = the
        # sub-tick remainder (0 to TICK).
        self._sim_base = self._sim_time
        self._real_base = real_time if real_time is not None else 0.0
        self._last_step_time = snap_time + n * TICK
        # Branch 3 free-runs at exactly 1x real time (see seed()).
        self._rate = 1.0
        # Diagnostics: the replayed REC record.
        self._diag('REC', {'real_time': real_time, 'snap_time': snap_time,
                            'now': now, 'n': n,
                            'pre_sim': _pre_sim, 'pre_last': _pre_last,
                            'post_sim': self._sim_time,
                            'post_last': self._last_step_time,
                            'post_acc': self._acc})
        # The pose is at snap_time + n*TICK (the replay end), and
        # _last_step_time is the same, so the render interpolation phase
        # is _acc = _sim_time - _last_step_time (in [0, TICK)).

    def step(self, dt, inp, enemies=None, rocks=None, player_proxy=None):
        """Advance the ghost ONE fixed step with the LOCAL input.

        `dt` is passed by the caller (the sim's TICK) because netcode.py
        must not import it from game.py (game.py imports netcode.py). The
        Game-fed per-tick fields are reset to idle before the step — the
        ghost has no enemy list to target (see class docstring).

        `rocks` (a list of `_GhostRockProxy`, see `advance`) is fed to the
        beam-occlusion clamp below: the ghost's own laser beam's endpoint
        is clamped to the nearest asteroid that intersects the beam
        segment (the client-side mirror of the host's `_beam_blocked`), so
        the client's beam stops at the rock's surface like the host's
        instead of punching through to the target. None (tests / no
        window) leaves the beam's endpoint at the raw target position.

        The SHOTS the ghost's gun fires this step are kept as presentation
        `Bullet`s in `self.local_bullets` so the player sees their own
        shots immediately instead of waiting ~100 ms for the host's next
        snapshot. The list is uncapped by the fire decision (plan_14
        Phase 2 — the old local-draw cap mirrored the host's old world
        cap and desynced the lockstep); a cull-oldest safety net at
        BULLET_POOL_MAX polices it, mirroring the host's. Call
        `step_local_bullets` after `step` to advance + cull the list
        (the game loop does this via `advance`).

        `enemies` is a list of lightweight enemy proxies (objects exposing
        `.pos`) built by the caller from the interpolation buffer's enemy
        entries. When given, the ghost's laser target is picked from them
        (the SAME nearest-in-range math the host's `_pick_laser_target`
        uses) so the ghost charges + fires a beam on the same tick the
        host does.

        `player_proxy` (plan_19) is the REMOTE-PLAYER proxy
        (`_GhostPlayerProxy`, built by the caller from the buffer's
        `pos['ships'][remote_index]` entry). When given, it joins the
        enemy proxies as a targeting candidate (the host's
        `_targetable_candidates` returns the same set: enemies + the
        other player's ship), so the ghost locks + charges + launches on
        the other player on the same tick the host does. None (tests,
        single-player, or before the buffer has a window) leaves the
        candidates as the enemies only.

        The BEAMS `s.update()` returns are kept as presentation
        entries in `self.local_beams` (hull-local muzzle -> world target;
        predicted_view re-anchors the muzzle to the ghost's current pose
        at draw time), so the player sees their own beam immediately
        instead of waiting ~100 ms for the host's next snapshot. Call
        `step_local_beams` after `step` to age + cull the list. When
        `enemies` is None (tests, or before the buffer has a window) the
        laser target stays None.

        The game loop does NOT call this directly — it calls `advance`,
        which decides how many fixed steps a real frame warrants. Tests
        call it directly to step the ghost at exactly the sim's rate.
        """
        s = self._ship
        # Remember the input for the render pre-step (it reads .turn to
        # predict the angular motion over the L gap).
        self._last_inp = inp
        s.tracked = 0
        s.laser_target = self._pick_laser_target(s, enemies, player_proxy)
        # Pick the ghost's missile target from the proxies (the SAME
        # nearest-in-range math the host's _pick_missile_target uses), so
        # the ghost locks + launches on the same tick the host does.
        s.missile_target = self._pick_missile_target(s, enemies, player_proxy)
        s.contacts = []
        shots, beams, missiles = s.update(dt, inp)
        # Build the ghost's sensor contacts AFTER s.update — the ghost's
        # own sensor state (sensor_on, scan_reveal, scan_cd) is this
        # tick's value (synced from the snapshot on reconcile, and mutated
        # immediately by the client's V/G command sink), so the contacts
        # match what the host's _update_contacts would produce given the
        # same sensor state + the same enemy positions. The enemies are
        # the buffer's proxies (at the render point, ~INTERP_DELAY behind
        # the host's live positions) — acceptable for presentation (the
        # contacts are 10 Hz blips/arrows).
        self._update_contacts(s, enemies)
        # Keep the ghost's own gun shots. The fire decision is
        # INPUT-LOCAL (plan_14 Phase 2): every shot the ghost's weapon
        # state produces is kept — the old local-draw cap
        # (len(local_bullets) < MAX_BULLETS) mirrored the host's old
        # WORLD cap, which the ghost could not reproduce, so it fired a
        # different set of shots than the host and desynced bullet_seq /
        # cooldown (the double-bullet). The list is policed by the
        # cull-oldest safety net below (BULLET_POOL_MAX, mirroring the
        # host's) — performance only, never a fire gate.
        # Each ghost bullet gets the SAME unique id the host assigns at
        # fire — (player_index, per-ship bullet_seq) — and the ghost's
        # bullet_seq is bumped to match (it is resynced to the host's via
        # the ship snapshot on reconcile, so the ids stay aligned). The id
        # is the no-double-draw dedup key: predicted_view skips the
        # buffer's copy of a bullet whose id is in local_bullets (the
        # ghost draws it, predicted, immediate). `pending` (set by
        # handback_bullets) marks a bullet whose buffer copy is visible in
        # the newest snapshot — predicted_view hands it off to the buffer
        # once that copy has ADVANCED (the wait-until-advancing rule,
        # reused from the missile handoff).
        # The ghost's gun cooldown (the ballistic weapon's). Read before the
        # fire loop so the FIRE telemetry can record the pre-fire value.
        _g_cd = 0.0
        for _w in s.weapons:
            if _w.comp.bullet_speed > 0:
                _g_cd = _w.cooldown
                break
        for shot in shots:
            seq_before = s.bullet_seq
            bid = (self.local_index, seq_before)
            s.bullet_seq += 1
            # The reconcile REWIND (reconcile_rewind) re-runs the fire
            # tick after apply_snapshot has RESET bullet_seq to the
            # host's pre-fire value — so the replay re-fires the SAME
            # bullet with the SAME id the prediction already emitted in
            # advance(). Without this guard the ghost would hold two
            # Bullets of one id (the in-flight one + a fresh one at the
            # muzzle): the player sees a double-draw and, because the
            # buffer's copy is suppressed by the id-dedup, BOTH visible
            # ones are collision-less ghost bullets. Skip the re-fire
            # when the id is already in flight (the prediction's copy
            # is the fresher one).
            if any(tuple(b.id) == bid for b in self.local_bullets):
                continue
            b = Bullet(shot.pos, shot.vel, owner=shot.owner, bid=bid)
            b.pending = False   # handback state (client-only)
            self.local_bullets.append(b)
            # TELEMETRY (Phase 0, plan_14): a ghost Bullet was CREATED.
            # f_cap_before = the local-draw count BEFORE this append —
            # kept for CSV compatibility (the ghost cap it measured is
            # gone as of Phase 2; the column now just tracks the
            # presentation list's depth). f_guard is always 0 here (a
            # guard hit skips the append).
            _btel.log("FIRE", f_bid=_btel.fmt_ids([bid]),
                      f_seq_before=seq_before,
                      f_seq_after=s.bullet_seq, f_guard=0,
                      f_local_ids=_btel.fmt_ids(
                          (tuple(b.id) for b in self.local_bullets)),
                      f_cap_before=len(self.local_bullets) - 1)
        # Cull-oldest performance safety net (plan_14 Phase 2, mirrors the
        # host's in Game._step): the list is in fire order (newest at the
        # back), so pop(0) is the oldest. BULLET_POOL_MAX sits well above
        # the natural count (90/player at the current 60/s rate), so this
        # never binds in normal play — it polices the presentation list
        # without ever changing whether a shot fires.
        while len(self.local_bullets) > BULLET_POOL_MAX:
            self.local_bullets.pop(0)
        # Keep the ghost's own laser discharges as presentation entries
        # [local_start, end, age, ttl]. The muzzle is the HULL-LOCAL offset
        # (beam.local_start), NOT the frozen world-space point (beam.start)
        # — predicted_view re-anchors it to the ghost's CURRENT pose each
        # frame (the exact math the host's _draw_world_beam uses for its
        # own beams). Storing the world-space muzzle drew the origin at
        # where the muzzle WAS at fire time, so as the ghost moved during
        # the 0.15 s flash the origin was left behind in empty space. A
        # LIST (not tuple) so step_local_beams can age `b[2]` in place
        # (mirrors the host's `self.beams` list-of-lists).
        #
        # The END is clamped to the nearest asteroid that intersects the
        # beam segment (the client-side mirror of the host's
        # `_beam_blocked` + `_beam_hit_asteroid` vis_end math): the host's
        # `_resolve_beam` stops the beam at the rock's surface when a rock
        # is in the way, so the client's own beam must stop there too —
        # storing the raw beam.end (the target's position) made the client
        # beam visually punch THROUGH the asteroid to the target. `rocks`
        # are the buffer's asteroid proxies (None -> no clamp, the raw
        # target endpoint).
        for beam in beams:
            end = _beam_blocked_end(beam, rocks)
            self.local_beams.append(
                [beam.local_start, end.copy(), 0.0, BEAM_TTL])
        # Keep the ghost's own LAUNCHED missiles as presentation
        # GhostMissiles. The fire decision is INPUT-LOCAL (plan_14 Phase 3):
        # every missile the ghost's weapon state produces is kept — the old
        # local-draw cap (len(local_missiles) < MAX_MISSILES) mirrored the
        # host's old WORLD cap, which the ghost could not reproduce, so it
        # launched a different set of missiles than the host and desynced
        # missile_seq (the double-missile). The list is policed by the
        # cull-oldest safety net below (MISSILE_POOL_MAX, mirroring the
        # host's) — performance only, never a fire gate. The lock-rebuild
        # (lock_progress resets to 0 on launch, 0.8 s to rebuild) is the
        # real rate limiter. Each gets the SAME unique id the host assigns
        # at launch — (player_index, per-ship seq) — and the ghost's
        # missile_seq is bumped to match (it is resynced to the host's via
        # the ship snapshot on reconcile, so the ids stay aligned). The
        # target is the buffer proxy the ghost locked (m.target is a
        # _GhostEnemyProxy); homing re-resolves it by id each step.
        for m in missiles:
            seq_before = s.missile_seq
            mid = (self.local_index, seq_before)
            s.missile_seq += 1
            # The reconcile REWIND (reconcile_rewind) re-runs the fire
            # tick after apply_snapshot has RESET missile_seq to the
            # host's pre-fire value — so the replay re-fires the SAME
            # missile with the SAME id the prediction already emitted
            # in advance(). Without this guard the ghost would hold two
            # GhostMissiles of one id (the in-flight one + a fresh one
            # at the muzzle): the player sees "two missiles" (one right
            # behind the other) and, because the buffer's copy is
            # suppressed by the id-dedup, BOTH of the visible ones are
            # collision-less ghost missiles that pass through the
            # target and orbit it. Skip the re-fire when the id is
            # already in flight (the prediction's copy is the fresher
            # one — it has already integrated the host's flight).
            guard = any(tuple(g.id) == mid for g in self.local_missiles)
            if guard:
                # TELEMETRY: a re-fire was SUPPRESSED. If the
                # suppressed id differs from the in-flight id, the seq
                # resync drifted — capture it.
                _mtel.log("FIRE", f_mid=_mtel.fmt_ids([mid]),
                          f_seq_before=seq_before,
                          f_seq_after=s.missile_seq, f_guard=1,
                          f_local_ids=_mtel.fmt_ids(
                              (tuple(g.id) for g in self.local_missiles)))
                continue
            self.local_missiles.append(
                GhostMissile(m.pos, m.vel, mid, target=m.target))
            # TELEMETRY: a GhostMissile was CREATED. This is the id the
            # ghost will dedup + handback on. seq_before/after show the
            # counter move; f_local_ids is the full in-flight set.
            _mtel.log("FIRE", f_mid=_mtel.fmt_ids([mid]),
                      f_seq_before=seq_before,
                      f_seq_after=s.missile_seq, f_guard=0,
                      f_local_ids=_mtel.fmt_ids(
                          (tuple(g.id) for g in self.local_missiles)))
        # Cull-oldest performance safety net (plan_14 Phase 3, mirrors the
        # host's in Game._step): the list is in launch order (newest at the
        # back), so pop(0) is the oldest. MISSILE_POOL_MAX sits well above
        # the natural count (~3.75/player at the 0.8 s lock-rebuild
        # cadence), so this never binds in normal play — it polices the
        # presentation list without ever changing whether a missile
        # launches.
        while len(self.local_missiles) > MISSILE_POOL_MAX:
            self.local_missiles.pop(0)

    def _targetable_candidates(self, s, enemies, player_proxy=None):
        """plan_19: the alive, off-team proxies `s` may target — the enemy
        proxies + the remote-player proxy. Mirrors the host's
        `Game._targetable_candidates` (enemies + the other player's ship,
        filtered by the `can_target` team predicate) so the ghost picks the
        SAME target the host does, given the same buffer window. The
        predicate is inlined (netcode.py cannot import game.py — circular):
        a proxy is targetable iff its `.team != s.team` and it is not
        `.dead` (enemy proxies have no `.dead` — `getattr(..., False)`
        keeps them always-targetable, exactly like the host's AIEnemies).
        """
        out = []
        for e in (enemies or []):
            if e.team != s.team and not getattr(e, "dead", False):
                out.append(e)
        if player_proxy is not None:
            if (player_proxy.team != s.team
                    and not getattr(player_proxy, "dead", False)):
                out.append(player_proxy)
        return out

    def _pick_laser_target(self, s, enemies, player_proxy=None):
        """Nearest targetable proxy within the max laser range; None if no
        laser is fitted or there are no candidates. Mirrors the host's
        `Game._pick_laser_target` (nearest-in-range over
        `_targetable_candidates`) so the ghost charges + fires on the same
        tick the host does, given the same target in the wedge. Reads only
        `.pos` off each proxy — the buffer's enemy proxies and the
        remote-player proxy expose that."""
        cands = self._targetable_candidates(s, enemies, player_proxy)
        if not cands:
            return None
        max_range = max((w.comp.laser_range for w in s.weapons
                         if w.comp.laser_range > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for t in cands:
            d = t.pos.distance_to(s.pos)
            if d <= best_d:
                best, best_d = t, d
        return best

    def _pick_missile_target(self, s, enemies, player_proxy=None):
        """Nearest targetable proxy within the max missile lock range; None
        if no missile is fitted or there are no candidates. Mirrors the
        host's `Game._pick_missile_target` (nearest-in-range over
        `_targetable_candidates`) so the ghost locks + launches on the same
        tick the host does, given the same target in the wedge. Reads only
        `.pos` off each proxy."""
        cands = self._targetable_candidates(s, enemies, player_proxy)
        if not cands:
            return None
        max_range = max((w.comp.missile_lock_range for w in s.weapons
                         if w.comp.missile_speed > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for t in cands:
            d = t.pos.distance_to(s.pos)
            if d <= best_d:
                best, best_d = t, d
        return best

    def _update_contacts(self, s, enemies):
        """Build the ghost's sensor contacts from the fitted sensor.

        Mirrors the host's `Game._update_contacts` EXACTLY so the ghost's
        contacts match the host's given the same sensor state + the same
        enemy positions. The host iterates `self.enemies` (live AIEnemies)
        and reads `e.ship.power_used - e.ship.power_idle_total`; the ghost
        iterates the buffer's enemy PROXIES and reads the same off
        `proxy.ship` (a `_GhostEnemyShip` stand-in whose power fields are
        filled by `_ghost_enemy_proxies` — power_used from the buffer,
        power_idle_total derived from the tag's loadout).

        Passive: enemies in sensor_range whose active power draw
        (power_used - idle) crosses SENSOR_SIGNATURE_THRESHOLD. Active:
        while a ping's reveal lasts, everything in scan_range is
        confirmed. A confirmed contact replaces a passive one (the host
        keys `found` by the enemy object; the ghost keys by proxy id —
        the same 1:1 mapping, since there is one proxy per enemy).

        `enemies` is the list of proxies (or None) — the ghost has no
        enemy list of its own. When None (tests, or before the buffer has
        a window) the contacts stay empty.
        """
        c = s.sensor_comp
        s.contacts = []
        if c is None:
            return
        if not enemies:
            return
        found = {}
        if s.sensor_on and c.sensor_range > 0:
            for e in enemies:
                d = e.pos.distance_to(s.pos)
                if d <= c.sensor_range:
                    sig = e.ship.power_used - e.ship.power_idle_total
                    if sig >= SENSOR_SIGNATURE_THRESHOLD:
                        found[e.id] = [e.pos.copy(), d,
                                       min(1.0, sig / SENSOR_SIG_FULL),
                                       False]
        if s.scan_reveal > 0 and c.scan_range > 0:
            for e in enemies:
                d = e.pos.distance_to(s.pos)
                if d <= c.scan_range:
                    found[e.id] = [e.pos.copy(), d, 1.0, True]
        s.contacts = list(found.values())

    def step_local_beams(self, dt):
        """Age the ghost's local beams by `dt` and cull the expired. Each entry
        is [start, end, age, ttl] (a list — age is mutated in place).
        Mirrors the host's `for b in self.beams: b[4] += dt` +
        `self.beams = [b for b in self.beams if b[4] < b[5]]`. Called by
        `advance` after each ghost step; tests may call it directly. `dt`
        is the sim's TICK (passed in, not imported)."""
        for b in self.local_beams:
            b[2] += dt
        self.local_beams = [b for b in self.local_beams if b[2] < b[3]]

    def step_local_bullets(self, dt):
        """Advance the ghost's local bullets by `dt` and cull the dead. Mirrors
        the host's `for b in self.bullets: b.update(dt)` +
        `self.bullets = [b for b in ... if b.life > 0]`. Called by
        `advance` after each ghost step; tests may call it directly. `dt`
        is the sim's TICK (passed in, not imported)."""
        for b in self.local_bullets:
            b.update(dt)
        self.local_bullets = [b for b in self.local_bullets if b.life > 0]

    def step_local_missiles(self, dt, enemies=None, player_proxy=None):
        """Advance the ghost's local missiles by `dt` and cull the dead. Mirrors
        the host's `for m in self.missiles: m.update(dt)` +
        `self.missiles = [m for m in ... if m.life > 0]` — except the
        homing target is the buffer's enemy PROXY for the missile's target
        id (re-resolved each step from `enemies`; None when the target has
        left the buffer, so the missile coasts). Called by `advance` after
        each ghost step; tests may call it directly. `dt` is the sim's TICK
        (passed in, not imported).

        `player_proxy` (plan_19) is the remote-player proxy: a missile
        locked on the other player carries its `"p<index>"` id, which is a
        different key space from the enemy numeric ids, so the `by_id` map
        includes the proxy too — a missile locked on the remote player
        homes on it (the proxy is always present while the ship is in the
        buffer; ships don't turn over)."""
        # Map target id -> proxy for this step (the buffer's current view).
        by_id = {e.id: e for e in (enemies or [])}
        if player_proxy is not None:
            by_id[player_proxy.id] = player_proxy
        for m in self.local_missiles:
            # Re-resolve the target by the id the missile locked. m.target
            # is the proxy it was launched at; its .id is the key.
            tid = m.target.id if m.target is not None else None
            m.update(dt, by_id.get(tid))
        self.local_missiles = [m for m in self.local_missiles
                               if m.life > 0]

    def handback_missiles(self, host_seq, latest_missiles=None):
        """Mark the ghost's OWN missiles as handed back to the buffer.

        A ghost missile is marked `pending` (NOT culled) once BOTH hold:
          * `seq < host_seq` — the snapshot's authoritative
            `missile_seq` (ship snapshot field 21) counts the missiles
            the host had launched as of the snapshot, so a lower seq was
            launched BEFORE the snapshot (the seq comparison is exact and
            phase-free: it does not depend on the reconcile replay span
            the way an age cutoff would); AND
          * the buffer's NEWEST snapshot actually carries a missile with
            this id (`latest_missiles` — the newest snapshot's missile
            list, each entry's id at field 8; None when the buffer is
            empty).

        Why the second condition: the snapshot that arrives right after a
        launch is the PRE-fire one — the host snapshots every
        SNAPSHOT_INTERVAL ticks, so the first snapshot to carry the
        missile arrives up to one interval (~100 ms) later. The seq alone
        says "hand back" the moment ANY snapshot reports a higher seq,
        but the buffer's copy only becomes drawable when the CARRYING
        snapshot enters the buffer — and the render point (newest stamp -
        INTERP_DELAY) reaches it only ~INTERP_DELAY after that.

        The ghost copy is kept (marked `pending`) and the ghost KEEPS
        DRAWING it; predicted_view hands it off on the frame the buffer
        copy is actually visible — removing it from local_missiles (so
        the buffer copy is no longer dedup-skipped) and seeding the
        decoupled render offset from the ghost's position AT THAT MOMENT
        (ghost_pos - interp_pos), which the MISSILE_HANDOFF_DECAY then
        smooths to the buffer's position. The missile is drawn
        continuously, by the ghost until the buffer copy appears and by
        the buffer (offset-corrected) after.

        (If the host's copy had already expired by the snapshot, the
        ghost's copy — same age — would be dead too, so the pending mark
        is a no-op: the missile is life-culled by step_local_missiles
        before the buffer copy is ever visible.)

        RETURNS an empty dict (the offset is seeded by predicted_view at
        the handoff frame, not at snapshot arrival)."""
        if latest_missiles is None:
            ids = None
        else:
            ids = {tuple(m[8]) for m in latest_missiles if m[8] is not None}
        before = [tuple(m.id) for m in self.local_missiles
                  if m.id is not None]
        # Buffer copy position per id (the newest snapshot's missile list —
        # each entry is (px, py, vx, vy, owner, life, boost, target_id, mid)).
        buf_pos = {}
        if latest_missiles is not None:
            for e in latest_missiles:
                if e[8] is not None:
                    buf_pos[tuple(e[8])] = (e[0], e[1])
        # The ghost missile is NOT culled here. It is marked `pending` and KEPT
        # in local_missiles so the ghost keeps DRAWING it. The buffer's
        # copy only becomes VISIBLE when the render point reaches the
        # carrying snapshot (~INTERP_DELAY after this snapshot), so
        # culling here left a 6-15 frame gap where the missile was drawn
        # by NEITHER the ghost nor the buffer — the "blink".
        # predicted_view hands it off on the frame the buffer copy is
        # visible: it removes the pending missile from local_missiles (so
        # the buffer copy is no longer dedup-skipped) and seeds the
        # decoupled render offset from the ghost's position AT THAT
        # MOMENT (ghost_pos - interp_pos), which the existing decay then
        # smooths to the buffer's position.
        #
        # Safe against the reconcile re-fire: a pending missile has
        # seq < host_seq, i.e. it was launched BEFORE this snapshot, so
        # its fire tick is not in the rewind replay span [snap_time, now]
        # — apply_snapshot's seq reset + replay cannot re-fire it (the
        # re-fire guard would catch a duplicate anyway).
        #
        # TELEMETRY: h_pending lists the ids just marked pending (the
        # handback decision). h_jump is the distance (px) between the
        # ghost's position and the buffer's NEWEST-snapshot copy position
        # — the offset the decay smooths out (informational; the handoff
        # now happens ~INTERP_DELAY later, at the render point).
        pending = []
        jump_parts = []
        for m in self.local_missiles:
            ready = (m.id is not None and m.id[0] == self.local_index
                     and m.id[1] < host_seq
                     and (ids is None or tuple(m.id) in ids))
            if ready and not m.pending:
                m.pending = True
                mid = tuple(m.id)
                pending.append(mid)
                bp = buf_pos.get(mid)
                if bp is not None:
                    jump_parts.append(
                        "%s:%.1f" % (_mtel.fmt_ids([mid]),
                                     math.hypot(m.pos.x - bp[0],
                                                m.pos.y - bp[1])))
        _mtel.log("HANDBACK", h_host_seq=host_seq,
                  h_buf_ids=_mtel.fmt_ids(sorted(ids)) if ids else "",
                  h_before=_mtel.fmt_ids(before),
                  h_after=_mtel.fmt_ids(
                      tuple(m.id) for m in self.local_missiles),
                  h_culled=_mtel.fmt_ids(pending),
                  h_jump="|".join(jump_parts))
        return {}

    def handback_bullets(self, host_seq, latest_bullets=None):
        """Mark the ghost's OWN bullets as handed back to the buffer (the
        lightweight mirror of `handback_missiles`).

        A ghost bullet is marked `pending` (NOT culled) once BOTH hold:
          * `seq < host_seq` — the snapshot's authoritative
            `bullet_seq` (ship snapshot field 24) counts the bullets the
            host had fired as of the snapshot, so a lower seq was fired
            BEFORE the snapshot (exact, phase-free — same logic as the
            missile seq comparison); AND
          * the buffer's NEWEST snapshot actually carries a bullet with
            this id (`latest_bullets` — the newest snapshot's player-
            bullet list, each entry's id at field 6; None when the
            buffer is empty).

        The ghost KEEPS DRAWING a pending bullet; predicted_view hands it
        off on the frame the buffer's copy is visible AND has ADVANCED
        (the wait-until-advancing rule, reused) — removing it from
        local_bullets (so the buffer copy is no longer dedup-skipped) and
        drawing the buffer copy PULLED FORWARD by V*delay (the closed-form
        handoff — no offset state, no decay, no backward cap, because a
        bullet is a straight line). The bullet is drawn continuously, by
        the ghost until the buffer copy appears and by the buffer
        (pulled forward) after.

        (If the host's copy had already expired by the snapshot, the
        ghost's copy — same age — would be dead too, so the pending mark
        is a no-op: the bullet is life-culled by step_local_bullets before
        the buffer copy is ever visible.)

        RETURNS an empty dict (parity with handback_missiles)."""
        if latest_bullets is None:
            ids = None
        else:
            ids = {tuple(b[6]) for b in latest_bullets if b[6] is not None}
        before = [tuple(b.id) for b in self.local_bullets
                  if b.id is not None]
        pending = []
        for b in self.local_bullets:
            ready = (b.id is not None and b.id[0] == self.local_index
                     and b.id[1] < host_seq
                     and (ids is None or tuple(b.id) in ids))
            if ready and not b.pending:
                b.pending = True
                pending.append(tuple(b.id))
        _mtel.log("HANDBACK", h_host_seq=host_seq,
                  h_buf_ids=_mtel.fmt_ids(sorted(ids)) if ids else "",
                  h_before=_mtel.fmt_ids(before),
                  h_after=_mtel.fmt_ids(
                      tuple(b.id) for b in self.local_bullets),
                  h_culled=_mtel.fmt_ids(pending),
                  h_jump="")
        # TELEMETRY (Phase 0, plan_14): the bullet handback decision in the
        # bullet lifecycle file (h_host_bseq = the snapshot's authoritative
        # bullet_seq; h_culled = the ids just marked pending).
        _btel.log("HANDBACK", h_host_bseq=host_seq,
                  h_buf_bids=_btel.fmt_ids(sorted(ids)) if ids else "",
                  h_before=_btel.fmt_ids(before),
                  h_after=_btel.fmt_ids(
                      tuple(b.id) for b in self.local_bullets),
                  h_culled=_btel.fmt_ids(pending))
        return {}

    def advance(self, dt, inp, enemies=None, real_time=None,
                host_time=None, rtt_ow=0.0, rocks=None,
                player_proxy=None):
        """Advance the ghost by real time `dt`.

        THE GHOST CLOCK CONTRACT — the single source of truth for how the
        ghost's sim clock and step grid work. Read this before touching
        advance / reconcile_rewind / render_pose (also summarized in
        ARCHITECTURE.md).

        The ghost is a presentation copy of the local ship. It must sit at
        the HOST's sim time (not the client's display time) so that when a
        snapshot lands, the reconcile's pose correction is ~0. Three
        quantities define its state:

          _sim_time        the ghost's current sim time (the time of its
                           physics pose; advanced by the clock below).
          _last_step_time  the tick boundary the last physics step ran at
                           (the step grid). A step runs when
                           _sim_time - _last_step_time >= TICK.
          _acc             _sim_time - _last_step_time, in [0, TICK) — the
                           render interpolation phase. render_pose lerps
                           _prev -> curr at alpha = _acc / TICK.

        The clock sets _sim_time each call. Three branches, in priority
        order:

          1. real_time is None (tests that drive advance with a bare dt
             and no reconcile): _sim_time += clamped dt — a simple
             free-run. The only branch the pure ghost-clock tests
             (n_steps == floor(total/TICK)) use.
          2. host_time is ready (the LIVE client — host_time is always set
             on a real client): _sim_time =
             host_time.now(real_time) + rtt_ow. The ghost's clock IS the
             host's clock (the HostTimeEstimator's RAW capped estimate)
             plus the one-way latency. This locks the step grid to the
             host's tick grid, so the phase offset at the next reconcile
             is 0 and the reconcile's pose correction is ~0. The RAW
             estimate (not now_smooth) is the ghost's clock: the
             reconcile re-anchors to the raw estimate each snapshot, so
             a smoothed clock would lag the anchor and snap. This is the
             branch that matters for 2P smoothness.
          3. real_time given but host_time not ready (the window before the
             first snapshot, and some tests — e.g. test_10_11c): free-run
             from the last reconcile anchor, _sim_time = _sim_base +
             _rate * (real_time - _real_base). _sim_base/_real_base/_rate
             exist ONLY for this branch; they are NOT the live clock
             (branch 2 supersedes them once host_time is ready).

        Physics steps run on tick-boundary crossings (the while loop
        below), so the ghost integrates at the SIM's rate no matter the
        display's refresh rate. `dt` is clamped to MAX_FRAME_DT so a
        hiccup can't trigger a catch-up spiral. `inp` is the CURRENT local
        input; every step this call takes uses it (the host applies the
        latest received input each tick). Returns the number of fixed
        steps taken.

        `enemies`: a list of lightweight enemy proxies from the
        interpolation buffer, fed to every step so the ghost's laser
        target tracks the buffer's enemies and it fires a beam on the same
        tick the host does.

        `player_proxy` (plan_19): the remote-player proxy
        (`_GhostPlayerProxy`) — fed to every step as an additional
        targeting candidate (see `step`), so the ghost's lock/charge/
        launch on the other player stays in lockstep with the host.

        `rocks`: a list of lightweight asteroid proxies from the
        interpolation buffer, fed to every step so the ghost's own laser
        beam's endpoint is clamped to the nearest asteroid that intersects
        the beam segment (the client-side mirror of the host's
        `_beam_blocked`) — the client's beam stops at the rock's surface
        like the host's instead of punching through to the target.

        A DEAD ghost does not advance: the host freezes a dead ship, so
        predicting it with the local input would race ahead and every
        reconcile yank it back. Freezing matches authority exactly (snap
        size 0 while dead); the accumulator is discarded so the first
        advance after a respawn starts clean.
        """
        if self._ship.dead:
            self._acc = 0.0
            # A dead ghost is frozen — keep the render interpolation
            # anchored on the (static) pose so render_pose returns it
            # exactly (alpha 0 would otherwise lerp from a stale _prev).
            self._prev_pos = pygame.Vector2(self._ship.pos)
            self._prev_angle = self._ship.angle
            # A beam is a 0.15 s flash — age + cull it even while the
            # ghost is dead, mirroring the host (Game._step ages
            # self.beams outside the game_over guard). Without this a beam
            # fired at the moment of death would hang frozen in the air
            # until the respawn.
            self.step_local_beams(min(dt, MAX_FRAME_DT))
            # A missile in flight at the moment of death coasts + ages out
            # too (no enemies while dead — it can no longer home).
            self.step_local_missiles(min(dt, MAX_FRAME_DT),
                                     player_proxy=player_proxy)
            self._diag('ADV', {'real_time': real_time,
                                'pre_sim': self._sim_time,
                                'pre_last': self._last_step_time,
                                'pre_acc': self._acc,
                                'ht_raw': None, 'ht_cap': None,
                                'rtt_ow': rtt_ow, 'clock': None,
                                'steps': 0,
                                'post_last': self._last_step_time,
                                'post_acc': self._acc, 'dead': 1})
            return 0
        # Diagnostics: the pre-call clock state (the ADV record).
        _pre_sim, _pre_last, _pre_acc = (self._sim_time, self._last_step_time,
                                         self._acc)
        # The clock (see the GHOST CLOCK CONTRACT in the docstring above).
        # Branch 1: tests that drive advance with a bare dt, no reconcile.
        _ht_raw = _ht_cap = None
        if real_time is None:
            self._sim_time += min(dt, MAX_FRAME_DT)
        # Branch 2: the LIVE client. The ghost's clock IS the host's clock
        # (the estimator's point-slope model) plus the one-way latency —
        # the host's sim time at the moment the client is rendering. This
        # locks the step grid to the host's tick grid, so the phase offset
        # at the next reconcile is 0 and the reconcile's pose correction is
        # ~0. The estimator re-anchors on every snapshot arrival, so this
        # tracks the host's clock without any client-side drift.
        elif host_time is not None and host_time.ready:
            # Diagnostics: capture the estimator's UNCAPPED estimate
            # (now_raw) alongside the capped one (now) so the CSV can show
            # whether the MAX_LEAD cap is binding (now_raw > now).
            _ht_raw = host_time.now_raw(real_time)
            _ht_cap = host_time.now(real_time)
            # The SIM clock is the RAW capped estimate (now), NOT now_smooth.
            # The raw estimate is phase-locked to the host's tick grid
            # (it is anchored on the newest snapshot's stamp), so the
            # reconcile's pose correction is ~0 (the ghost's physics
            # pose is at the same time as the snapshot's data). A
            # smoothed clock (now_smooth) lags the raw estimate by up to
            # ~16 ms after a burst, so the reconcile's correction would
            # be ~1 tick (~8.67 px) — the phase-lock is broken.
            #
            # The raw estimate IS ragged on burst frames (it jumps 2
            # ticks on a double-arrival frame), but that raggedness is
            # absorbed by the RENDER pose's smoothing (see render_pose):
            # the sim clock (and the physics pose) can be ragged, as long
            # as the RENDER pose is smooth. The render pose chases the
            # physics pose at a bounded rate (1x real time), so it never
            # jumps more than 1 tick per frame, even when the physics
            # pose jumps 2 ticks.
            self._sim_time = host_time.now(real_time) + rtt_ow
        # Branch 3: real_time given but host_time not ready (the window
        # before the first snapshot, and some tests — e.g. test_10_11c).
        # Free-run from the last reconcile anchor. _sim_base/_real_base/
        # _rate exist ONLY for this branch; they are NOT the live clock
        # (branch 2 supersedes them once host_time is ready).
        else:
            self._sim_time = self._sim_base + self._rate * (
                real_time - self._real_base)
        # Trigger physics steps when the sim time crosses a tick boundary
        # (_sim_time - _last_step_time >= TICK). The step advances the pose
        # by TICK and moves _last_step_time to the new boundary. The
        # number of steps per snapshot interval is exactly 2 (the sim time
        # advances 2*TICK per 2-tick interval, since the host sim runs at
        # 1x real time) — not 1/2/3.
        n = 0
        while self._sim_time - self._last_step_time >= TICK - 1e-9:
            # Capture the pose BEFORE the step so render_pose can
            # interpolate across the in-progress step (the host's
            # _ship_pose does the same from the model's prev/curr).
            self._prev_pos = pygame.Vector2(self._ship.pos)
            self._prev_angle = self._ship.angle
            self.step(TICK, inp, enemies=enemies, rocks=rocks,
                      player_proxy=player_proxy)
            self.step_local_bullets(TICK)   # advance + cull
            self.step_local_beams(TICK)     # age + cull
            self.step_local_missiles(TICK, enemies, player_proxy)   # home + cull
            self._last_step_time += TICK
            n += 1
        # The render interpolation phase: the time since the last step
        # (0 to TICK). render_pose lerps _prev -> curr at alpha = _acc/TICK.
        self._acc = self._sim_time - self._last_step_time
        # Diagnostics: the post-call ADV record (one per advance()).
        self._diag('ADV', {'real_time': real_time,
                            'pre_sim': _pre_sim, 'pre_last': _pre_last,
                            'pre_acc': _pre_acc,
                            'ht_raw': _ht_raw, 'ht_cap': _ht_cap,
                            'rtt_ow': rtt_ow, 'clock': self._sim_time,
                            'steps': n,
                            'post_last': self._last_step_time,
                            'post_acc': self._acc, 'dead': 0})
        return n

    def _diag(self, kind, rec):
        """Diagnostics: forward one clock record to the class-level `diag_sink`
        (a no-op when it is None — tests + the host's ghost). The sink is
        set by the client loop only while debug_net is on.

        Read via the CLASS (`PredictedShip.diag_sink`), NOT `self.diag_sink`:
        a function stored as a class attribute is a descriptor, so instance
        access (`self.diag_sink`) would bind it as a method and call it with
        `self` as a third argument (TypeError). Class access returns the raw
        function."""
        sink = PredictedShip.diag_sink
        if sink is not None:
            sink(kind, rec)

    def _render_pre_step(self):
        """Compute the RENDER-ONLY pre-step pose (the sawtooth fix).

        The physics pose (s.pos) sits at the replay end (snap_time +
        n*TICK), but the render clock (_sim_time) is L = rtt_ow AHEAD of
        it. Rendering the physics pose pins the render to snap_time while
        the clock is at snap_time + L, so each reconcile jumps the pose 1
        tick (the sawtooth). The fix: render at the CLOCK's time by
        lerping the pose at the replay end toward the pose ONE TICK ahead
        at alpha = L/TICK (full contract in ARCHITECTURE.md).

        The "one tick ahead" pose is a KINEMATIC pre-step, NOT a full
        physics step: it integrates the CURRENT velocity (pos += vel*TICK)
        and applies the CURRENT turn input (angle += turn*ROT_SPEED*
        turn_rate_factor*TICK) — no side effects, no shots/beams/missiles.
        It is RECOMPUTED every frame from the current physics pose +
        input, so it always tracks the live pose.

        Sets _render_base/_render_base_angle (the pose at the replay end,
        == s.pos) and _render_next/_render_next_angle (the pose one tick
        ahead). Called by render_pose each frame.
        """
        s = self._ship
        # The pose at the replay end (the physics pose, at snap_time +
        # n*TICK). This is the "base" of the render lerp.
        self._render_base = pygame.Vector2(s.pos)
        self._render_base_angle = s.angle
        # The pose ONE TICK ahead: integrate the current velocity + apply
        # the current turn input (the kinematic pre-step). `inp` is the
        # last input the ghost was stepped with (set in advance/step);
        # None before the first step (then the pre-step is pure coasting).
        inp = self._last_inp
        turn = inp.turn if inp is not None else 0.0
        next_pos = s.pos + s.vel * TICK
        next_angle = s.angle + turn * ROT_SPEED * s.hull.turn_rate_factor * TICK
        self._render_next = next_pos
        self._render_next_angle = next_angle

    def render_pose(self):
        """The ghost's INTERPOLATED render pose — (pos, angle).

        The render pose sits at the CLOCK's time (_sim_time), NOT the
        physics pose's time (the replay end, snap_time + n*TICK). The
        reconcile sets _sim_time = max(now, snap_time + n*TICK) with
        now = snap_time + L (L = rtt_ow), so the clock is L AHEAD of the
        physics pose. Rendering the physics pose directly pins the render
        to snap_time while the clock is at snap_time + L — each reconcile
        then jumps the pose 1 tick (the sawtooth).

        The fix: lerp the pose at the replay end (snap_time + n*TICK, ==
        s.pos) toward the pose ONE TICK ahead (snap_time + (n+1)*TICK, the
        kinematic pre-step) at alpha = L/TICK, where
        L = _sim_time - (snap_time + n*TICK) = _acc. That puts the render
        pose at snap_time + L (the clock's time) and makes it CONTINUOUS
        across the reconcile: the pre-reconcile render pose (at the old
        snap_time + L) and the post-reconcile render pose (at the new
        snap_time + L) are the SAME point in the ship's trajectory (the
        new snap_time is 1 tick ahead, so the new base is where the old
        "next" was), so there is no jump.

        This is a RENDER helper only — it does NOT touch self._ship.pos
        (the ghost's actual simulated pose, which the no-teleport/rewind
        gates and the handoff math read). Before the first seed the render
        state is None: return the raw pose (the caller only renders once
        seeded).
        """
        s = self._ship
        if self._render_base is None:
            return pygame.Vector2(s.pos), s.angle
        # Recompute the pre-step from the CURRENT physics pose + input so
        # it tracks the live pose (the pose advances in TICK chunks between
        # reconciles; the pre-step advances with it).
        self._render_pre_step()
        # alpha = L/TICK, where L = _sim_time - (snap_time + n*TICK) =
        # _acc (the render interpolation phase, in [0, TICK)). Clamped to
        # [0, 1] (on a hiccup the acc can exceed one step; render the
        # latest simulated state, never extrapolate past the pre-step).
        alpha = max(0.0, min(1.0, self._acc / TICK))
        rpos = self._render_base.lerp(self._render_next, alpha)
        a0, a1 = self._render_base_angle, self._render_next_angle
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        rangle = a0 + da * alpha
        return rpos, rangle

    def render_pose_smoothed(self, dt):
        """The ghost's SMOOTHED render pose — (pos, angle).

        The render pose (from render_pose) is ragged on burst frames: the
        sim clock (phase-locked to the host's raw tick grid) jumps 2 ticks
        when two snapshots arrive in one frame, so the physics pose jumps
        2 ticks, and the render pose (at the clock's time) jumps 2 ticks
        too. Rendering that directly lurches the ship 2 ticks for one
        frame (the shudder).

        The fix: a HYBRID of a snap and a VELOCITY-BASED CONTROLLER (feedforward +
        proportional), selected by the gap size:
        - GAP SMALL (<= SMOOTH_SNAP_FACTOR ticks of motion at max speed,
          i.e. steady state): SNAP to the target exactly. The rendered pose
          is then EXACTLY the render target every frame — zero per-frame
          error, no vibration. (The render target itself is already smooth:
          it is the physics pose lerped across the in-progress tick by
          render_pose, so snapping to it does not re-introduce the 60 Hz
          stepping.)
        - GAP LARGE (a stall/burst left the smoothed pose behind): the
          velocity controller. The smoothed pose's velocity = the ship's
          velocity (feedforward: track the ship's motion exactly, no
          rubber-band when the player starts moving) + a catch-up term
          proportional to the RESIDUAL gap (the gap minus the ship's motion
          during the frame, which the feedforward already handles). The
          residual-gap dynamics are d(residual)/dt = -SMOOTH_GAIN *
          residual, so a stall gap closes with a time constant of
          1/SMOOTH_GAIN (exponential, monotonic, no overshoot). The
          catch-up is capped at SMOOTH_MAX_CATCH * max_speed to avoid a
          lurch on a large stall gap.

        Why the hybrid and not either half alone:
        - A BOUNDED-RATE CHASE capped at the ship's ACTUAL speed has a cap
          of 0 at rest, so a gap a stall left behind could never close
          (the "fires from empty space, never gets closer" bug).
        - Flooring that cap at the ship's MAX speed closes the gap, but
          makes the catch-up a constant-velocity slide that rubber-bands
          when the player starts moving (no feedforward).
        - A velocity controller ALONE (no snap) has a constant per-frame
          vibration: the feedforward integrates over the SMOOTH real frame
          time dt, but the render target advances in RAGGED tick chunks
          (the sim clock jumps 0, 1, or 2 ticks per frame), so the
          feedforward's per-frame advance differs from the target's by up
          to ~1 tick — a vibration visible from game start. The snap
          eliminates it in steady state (the pose is exactly the target).
        The snap threshold (2 ticks at max speed) covers the worst-case
        steady-state gap (a burst frame advances the target 2 ticks), so
        the snap always engages in steady state, and the transition lurch
        when a stall gap closes to <= 2 ticks is at most 2 ticks — the same
        as a normal burst-frame move, not perceived as a lurch.

        Subtleties that matter (each was a bug when missed):
        - The feedforward must apply EVERY frame of the controller branch
          (even when the residual is ~0). If it's skipped, the smoothed
          pose stays put while the target moves by v*dt — a per-frame
          jitter.
        - The catch-up reacts to the RESIDUAL gap (gap - ship's motion this
          frame), not the full gap. The full gap includes the ship's motion
          during the frame, which the feedforward already handles; reacting
          to the full gap double-counts that motion, over-corrects, and
          overshoots. The catch-up DIRECTION is toward the target
          (delta / dist, always well-defined), not along the residual (a
          near-zero residual's direction is numerically unstable).

        This is a RENDER helper only — it does NOT touch self._ship.pos
        (the ghost's actual simulated pose, which the no-teleport/rewind
        gates and the handoff math read). Before the first call the
        smoothed pose is None: return the render pose (the caller only
        renders once seeded).
        """
        target_pos, target_angle = self.render_pose()
        # The ship's dead state (the ghost is frozen while dead — advance()
        # takes 0 steps, so the render pose is the death pose, static).
        _dead = self._ship.dead
        if self._smooth_pos is None:
            # First call: initialize to the target (no smoothing yet).
            self._smooth_pos = pygame.Vector2(target_pos)
            self._smooth_angle = target_angle
            self._smooth_dead = _dead
            return self._smooth_pos, self._smooth_angle
        # Dead->alive transition (respawn): the target pose jumps from the
        # death location to the spawn point. The smoothed pose would chase
        # it at max_speed * dt per frame — the ship visibly "flies back" to
        # the spawn point. The fix: reset the smoothed pose to the target
        # instantly (the ship appears at the spawn point, not flying back).
        if self._smooth_dead and not _dead:
            self._smooth_pos = pygame.Vector2(target_pos)
            self._smooth_angle = target_angle
            self._smooth_dead = _dead
            return self._smooth_pos, self._smooth_angle
        # While dead: the render pose is the death pose (static — the ghost
        # is frozen). The smoothed pose tracks it exactly (no chase needed;
        # the target is static). This keeps the smoothed pose at the death
        # location while dead, so the dead->alive transition (above) resets
        # it to the spawn point.
        if _dead:
            self._smooth_pos = pygame.Vector2(target_pos)
            self._smooth_angle = target_angle
            self._smooth_dead = _dead
            return self._smooth_pos, self._smooth_angle
        # Alive->alive: bounded-rate chase. Move the smoothed pose toward
        # the target, but at most `max_speed * dt` per frame (1 tick of
        # motion at the ship's max speed). The ship's max speed is
        # MAX_SPEED * the hull's max_speed_factor (the same cap the sim
        # applies in Ship.update). In steady state the target moves <= this
        # cap (the ship never exceeds max_speed), so the smoothed pose
        # tracks with zero lag. On a burst frame the target jumps 2 ticks
        # (2 * max_speed * TICK), but the smoothed pose moves only 1 tick
        # (max_speed * dt, with dt ~ TICK at 60 FPS) — the jump is spread
        # over 2 frames.
        self._smooth_dead = _dead
        # VELOCITY-BASED CONTROLLER (feedforward + proportional). The
        # smoothed pose's velocity = the ship's velocity (feedforward —
        # track the ship's motion exactly, zero steady-state lag) + a
        # catch-up term proportional to the gap (closes any stall gap,
        # damped). The gap dynamics are d(gap)/dt = -GAIN * gap, so the gap
        # closes with a time constant of 1/GAIN (exponential, no overshoot).
        # When the gap is small (<= the snap threshold) the smoothed pose
        # snaps to the target exactly (cuts off the exponential tail,
        # guarantees exact tracking in steady state).
        #
        # Why this and not the alternatives (see the SMOOTH_GAIN comment):
        # the feedforward tracks the ship's motion exactly (zero steady-state
        # lag, no rubber-band when the player starts moving), the
        # proportional term closes any stall gap (works at rest, damped),
        # and there is no constant-velocity slide (the catch-up is
        # proportional to the gap, not a fixed rate).
        #
        # The feedforward is the ship's PHYSICS velocity (self._ship.vel) —
        # the SAME velocity the render target advances at (see
        # _render_pre_step: next_pos = s.pos + s.vel * TICK), so
        # feedforward + 0 catch-up tracks the target exactly in steady
        # state. The catch-up is capped at SMOOTH_MAX_CATCH * max_speed so a
        # large stall gap doesn't lurch in one frame (the cap is on the
        # catch-up, NOT the total velocity — the feedforward always tracks
        # the ship exactly).
        ship_vel = self._ship.vel
        max_speed = MAX_SPEED * self._ship.hull.max_speed_factor
        # HYBRID: snap to the target when the gap is small (steady state —
        # the old behavior, no vibration), use the velocity controller when
        # the gap is large (stall recovery — smooth catch-up).
        #
        # Why the hybrid: the velocity controller's feedforward
        # (smooth += ship_vel * dt) approximates the target's advance, but
        # the target advances in RAGGED tick chunks (the sim clock jumps 0,
        # 1, or 2 ticks per frame) while the feedforward integrates over the
        # SMOOTH real frame time dt. That mismatch is a per-frame error of
        # up to ~1 tick (8.67 px) — a constant vibration visible from game
        # start. The old code snapped to the target in steady state (gap
        # <= cap), so the rendered pose was EXACTLY the render target every
        # frame — zero error, no vibration. The hybrid restores that: snap
        # when the gap is small (steady state), use the controller only when
        # the gap is large (a stall).
        #
        # The threshold is SMOOTH_SNAP_FACTOR * max_speed * TICK (2 ticks of
        # motion at max speed). The steady-state gap is at most 2 ticks (the
        # sim clock can advance 2 ticks in one frame), so 2.0 always snaps in
        # steady state (no vibration). When a stall gap closes to <= 2 ticks,
        # the snap lurch is at most 2 ticks — the same as a normal burst-frame
        # move, so it is not perceived as a lurch.
        delta = target_pos - self._smooth_pos
        dist = delta.length()
        snap_threshold = self.SMOOTH_SNAP_FACTOR * max_speed * TICK
        if dist <= snap_threshold:
            # Steady state: snap to the target exactly (no vibration).
            self._smooth_pos = pygame.Vector2(target_pos)
        else:
            # Stall recovery: velocity controller (feedforward + proportional
            # catch-up). The catch-up reacts to the RESIDUAL gap (the gap
            # MINUS the ship's motion during the frame), not the full gap —
            # the full gap includes the ship's motion during the frame
            # (v*dt), which the feedforward is already handling. Reacting to
            # the full gap would double-count that motion, over-correct, and
            # overshoot. The catch-up DIRECTION is toward the target
            # (delta / dist, always well-defined), not along the residual
            # (a near-zero residual's direction is numerically unstable).
            residual = delta - ship_vel * dt
            rdist = residual.length()
            # ALWAYS apply the feedforward (track the ship's motion exactly).
            self._smooth_pos += ship_vel * dt
            # Apply the catch-up only if there's a residual gap (proportional
            # to the residual, capped, toward the target).
            if rdist > 1e-9 and dist > 1e-9:
                catch = min(self.SMOOTH_GAIN * rdist,
                            self.SMOOTH_MAX_CATCH * max_speed)
                self._smooth_pos += (delta / dist) * catch * dt
        # Angle: same discipline. The feedforward is the ship's angular
        # velocity (the render target advances at the ship's turn rate, see
        # _render_pre_step: next_angle = s.angle + turn * ROT_SPEED *
        # turn_rate_factor * TICK). The catch-up is proportional to the
        # angle gap, capped at SMOOTH_MAX_CATCH * max_turn_rate.
        # self._last_inp is the last input the ghost was stepped with (set
        # in step()). .turn is -1, 0, or 1 (the turn input).
        turn = self._last_inp.turn if self._last_inp is not None else 0.0
        ship_omega = turn * ROT_SPEED * self._ship.hull.turn_rate_factor
        max_turn_rate = ROT_SPEED * self._ship.hull.turn_rate_factor
        da = (target_angle - self._smooth_angle + math.pi) % (2 * math.pi) - math.pi
        # HYBRID (same as the position): snap to the target when the angle
        # gap is small (steady state — no vibration), use the velocity
        # controller when the gap is large (stall recovery — smooth
        # catch-up). The threshold is SMOOTH_SNAP_FACTOR * max_turn_rate *
        # TICK (2 ticks of rotation at max turn rate) — the steady-state
        # angle gap is at most 2 ticks, so 2.0 always snaps in steady state.
        ang_snap_threshold = (self.SMOOTH_SNAP_FACTOR * max_turn_rate * TICK)
        if abs(da) <= ang_snap_threshold:
            # Steady state: snap to the target exactly (no vibration).
            self._smooth_angle = target_angle
        else:
            # Stall recovery: velocity controller. The catch-up reacts to
            # the RESIDUAL angle gap (the gap MINUS the ship's rotation
            # during the frame), not the full gap — the full gap includes
            # the ship's rotation during the frame (omega*dt), which the
            # feedforward is already handling. Reacting to the full gap
            # would double-count that rotation, over-correct, and overshoot.
            residual_da = da - ship_omega * dt
            # ALWAYS apply the feedforward (track the ship's rotation
            # exactly).
            self._smooth_angle += ship_omega * dt
            # Apply the catch-up only if there's a residual angle gap.
            if abs(residual_da) > 1e-9:
                catch = min(self.SMOOTH_GAIN * abs(residual_da),
                            self.SMOOTH_MAX_CATCH * max_turn_rate)
                self._smooth_angle += math.copysign(catch, residual_da) * dt
        return self._smooth_pos, self._smooth_angle

    def pos(self):
        """The ghost's current (x, y, angle) for rendering."""
        s = self._ship
        return (s.pos.x, s.pos.y, s.angle)