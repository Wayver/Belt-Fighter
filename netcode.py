"""Remote-interpolation helpers (Session 5a; id-based matching 5b-1).

The authoritative peer sends a `Game.snapshot()` every
`config.SNAPSHOT_INTERVAL` ticks. The remote peer does NOT run the sim for
remote entities — it renders BETWEEN snapshots by interpolating, instead of
teleporting each time a snapshot lands.

`PredictedShip` (Session 5b.4a) is the client-side prediction ghost: the
remote peer steps a private Ship with its OWN input and reconciles it to
the authoritative ship snapshot when one lands, so the player's own ship
does not feel the INTERP_DELAY. Remote entities (other ship, enemies,
asteroids) keep using `SnapshotBuffer`.

`interp_positions()` is the core: a PURE function of two snapshots + alpha.
It reads plain data and returns plain data. It must never touch sim state —
the sim stays authoritative on one peer; this only feeds the render.
Entities are matched by IDENTITY (enemies by ship id, asteroids by rock
id) so interpolation survives entity turnover — see `interp_positions`
for the membership rules.

Player ships carry their ANGLE too (Session 6.8): the buffer returns
(x, y, angle, dead, shield_dump, shield_clock) per ship, with the angle
lerp'd the same way `Ship.sync_render` does (the wrapped delta, via
`lerp_angle`), so the remote peer can draw the remote hull at its
interpolated orientation instead of a dot. The `dead` flag (Session
10.1) lets the remote peer skip a dead ship (it disappears until it
respawns) and spawn a
client-side explosion on the alive->dead transition.

Session 7.2 (full remote rendering): enemies carry their ANGLE, TAG,
VELOCITY, and ID too — (tag, x, y, angle, vx, vy, id) — so the remote
peer can draw each enemy as its REAL hull (the hull is fixed on both
peers by construction: ENEMY_HULL/MOTE_HULL + fixed loadouts) at its
interpolated orientation, build lightweight targeting proxies
(lead_point needs pos + vel), and track a specific enemy across frames
(by id — enemies churn, so there is no stable index like ship 0). And ALL projectiles are interpolated:
snapshot indices 2 (player bullets), 3 (enemy bullets), 4 (missiles)
come back under the 'bullets' key as (x, y, vx, vy, kind, owner,
boost) entries, so the remote peer draws real bullets/missiles instead
of nothing (the D3 defect). Projectiles are matched by PREDICTED
POSITION, not index — see `_match_bullets` for why index matching
(sliding lists) is wrong.

Snapshot layout (see Game.snapshot in game.py):
    [0] players_s tuple of one Ship.snapshot() per player (Session 6.1):
                  ship_s = pos.x=[0], pos.y=[1], vel.x=[2], vel.y=[3],
                  angle=[4], ...
    [1] enemies_s tuple of (tag, e_s); e_s = AIEnemy.snapshot() =
                  (ship_s, hp, id, acc_x, acc_y) -> enemy pos = e_s[0][0], e_s[0][1]
    [2] bullets_s, [3] enemy_bullets_s, [4] missiles_s
                  b_s = (pos.x, pos.y, vel.x, vel.y, owner, life)
                  m_s = (pos.x, pos.y, vel.x, vel.y, owner, life, boost, target_id)
    [5] asteroids_s  tuple of Asteroid.snapshot() =
                     (id, pos.x, pos.y, vel.x, vel.y, size, angle, spin, verts)
    [6] rng_state, [7] game_over, [8] protect_timer, [9] next_id,
    [10] rock_next_id
"""
import math

import pygame

from .config import (INTERP_DELAY, INTERP_DELAY_MIN, INTERP_DELAY_MAX,
                    ADAPT_K, SNAPSHOT_INTERVAL, TICK, MAX_FRAME_DT,
                    MAX_BULLETS, MAX_MISSILES,
                    MISSILE_SPEED, MISSILE_ACCEL, MISSILE_BOOST_TIME,
                    MISSILE_TURN_RATE, MISSILE_LIFE)
from .ship import Ship
from .bullets import Bullet
from . import missile_telemetry as _mtel

# 10.3a: the ghost's beam visuals age out at the host's beam ttl
# (Game._step: `self.beams = [b for b in self.beams if b[4] < b[5]]`
# with b[5] = 0.15). Mirrored here so the client's local beams fade on
# the same clock as the host's.
BEAM_TTL = 0.15

__all__ = ["interp_positions", "lerp", "lerp_angle", "SnapshotBuffer",
           "PredictedShip", "HostTimeEstimator", "LatencyTracker",
           "RenderPoint"]


def lerp(a, b, t):
    """Linear interpolation: a + (b - a) * t.

    Exact at the endpoints: t==0 returns a and t==1 returns b *by
    identity*, not by arithmetic. a + (b - a) * 1.0 is not always
    bit-identical to b in IEEE-754 (b - a rounds, then adding a rounds
    again), so the shortcut keeps alpha=0/1 lossless. That matters for
    netcode: when a snapshot lands and becomes the boundary of the
    interpolation window, the render must sit EXACTLY on the snapshot,
    not ~1e-13 off (which would read as a tiny pop every SNAPSHOT_INTERVAL).
    """
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    return a + (b - a) * t


def lerp_angle(a0, a1, t):
    """Interpolate between two RAW (unbounded) angles by lerp-ing the
    WRAPPED delta — exactly what Ship.sync_render does. The raw angles may
    be far outside [-pi, pi] (the sim's angle accumulates forever; Session 4
    fix), so the delta is wrapped to [-pi, pi] first; the result stays
    anchored on a0, so it inherits a0's unwrapped offset. Never lerp the
    raw angles directly."""
    # Exact at the endpoints (same IEEE-754 rationale as lerp): a0 + da * 1.0
    # is not always bit-identical to a1 (da rounds, then adding a0 rounds
    # again), so the render would sit ~1 ulp off the snapshot at alpha=1 —
    # a tiny pop every SNAPSHOT_INTERVAL (Session 7.2, caught by the
    # turnover battery's endpoint-exactness check).
    if t == 0.0:
        return a0
    if t == 1.0:
        return a1
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return a0 + da * t


class HostTimeEstimator:
    """The client's estimate of the AUTHORITATIVE peer's sim clock
    (Session 7.1).

    The client never runs the sim, so it has no sim clock of its own — but
    the ghost's reconcile bookkeeping (and, before 7.5b, the render point)
    need one. Deriving it from the LOCAL wall clock
    (the 6.6 `sim_time += dt` fix) is wrong in principle: the host's
    sim_time is a fixed-step accumulator and the client's frame dt is the
    display clock, so two independent clocks drift and the render point
    wanders inside the interpolation window (stutter).

    Instead: each snapshot is stamped with the host's sim time at send and
    arrives at a known local time, so every arrival is a sample of the
    mapping from local time to host sim time. This class fits that mapping
    as a line in POINT-SLOPE form, anchored on the NEWEST sample
    (Session 7.9):

        host_time ≈ h_anchor + rate * (local_time - t_anchor)

    where (t_anchor, h_anchor) is the newest recorded sample's
    (local_time, stamp) and `rate` is an EMA of the host's sim rate in
    local-time units (one sample per snapshot, 10 Hz). In steady state this
    is the same line as the 7.1 affine fit `A + rate * local_time` — but
    the offset is now a small DELTA (the time since the newest sample, at
    most a few snapshot intervals), not an absolute intercept.

    Why point-slope, not the 7.1 absolute intercept `A + rate * local_time`?
    The 7.1 form computed `A = host_time - rate * local_time` — a difference
    of two LARGE numbers (local_time is pygame ticks since init, which grows
    unboundedly: 86 s in the 7.8 log). Any small error in `rate` is then
    amplified by that large local_time (a 0.01 rate error at local_time 86 s
    is a 0.86 s offset error), so the estimate's error GREW over the session
    (the 7.8 log: |est - newest| went 0.45 s -> 1.0 s on the 4090, 0.69 s ->
    1.24 s on the Mac). Re-anchoring to the newest sample makes the offset a
    small delta, so the estimate stays well-conditioned for the WHOLE session
    — the leverage is bounded by the time since the newest sample, not by the
    session length.

    `rate` — the host's sim rate in local-time units, from consecutive
    samples (Δhost/Δlocal). A healthy 60 FPS host is 1.0; a struggling host
    (loop slower than the dt clamp) simulates slower than real time, and the
    estimate must follow that or it drifts away from the host's clock.
    Clamped to [RATE_MIN, RATE_MAX] as a sanity bound. (The rate EMA is
    unchanged from 7.1 — only the offset representation was re-anchored.)

    `now(local_time)` = h_anchor + rate * (local_time - t_anchor) — the
    host's own clock as carried by the wire, not the display's — CAPPED at
    the newest sample's stamp + MAX_LEAD (one snapshot interval). The cap
    anchors the estimate on the DATA, not the model: if the host starves (a
    contended loop simulating at 0.1x real time), samples arrive a full
    second apart and extrapolating at the clamped rate would run the estimate
    far ahead of the data, making the render point clamp-stutter. Capped, the
    estimate holds within one snapshot interval of the newest stamp, so the
    render point (sim_time - INTERP_DELAY) sits at or behind the newest
    snapshot and the remote render simply holds the last frame while the host
    catches up. In steady state (1x host, 10 Hz samples) the cap never binds
    — the estimate is at most ~0.01 s ahead of the newest sample.

    The per-arrival jitter sample (`last_jitter`) is the PREDICTION RESIDUAL
    (Session 7.9): how far this arrival's stamp is from where the PREVIOUS
    model (the previous anchor + rate) predicted it would be. A steady
    arrival sits near 0; a late/early packet spikes. This is the
    well-conditioned replacement for the 7.1 offset sample
    (`host_time - rate * local_time`), which had the same leverage problem as
    the intercept. It is the input the LatencyTracker consumes (7.5a).

    `local_time` is any monotonically increasing seconds value the caller
    has (pygame.time.get_ticks()/1000.0 in the game loop; plain t in tests).
    It may be large (it grows for the whole session) — the point-slope form
    only ever computes the SMALL difference (local_time - t_anchor), so a
    large local_time is harmless.

    Session 7.5b: this estimate is NO LONGER the client's render clock.
    The render point is now anchored on the DATA — the newest ARRIVED
    snapshot's stamp minus the adaptive delay (see `RenderPoint`) —
    because a model of the host clock and the data disagree under
    jitter/loss, and the render must not wander by the disagreement.
    The estimator still runs for two reasons: its per-arrival jitter
    sample (`last_jitter`) is the input the LatencyTracker consumes, and
    `now()` remains available for diagnostics (the 7.8 overlay).
    """

    RATE_ALPHA = 0.5
    RATE_MIN, RATE_MAX = 0.4, 1.5
    # Session 7.9: no OFFSET_ALPHA — the offset is no longer an EMA of
    # `host_time - rate * local_time` (that form's leverage made the error
    # grow over the session). The offset is the point-slope anchor itself
    # (the newest sample), which is exact by construction.
    MAX_LEAD = SNAPSHOT_INTERVAL * TICK   # one snapshot interval

    def __init__(self):
        # Session 7.9: point-slope anchor (the newest recorded sample), not
        # an absolute intercept — see the class docstring for why the
        # absolute-intercept form's leverage made the estimate's error grow
        # over the session.
        self._anchor = None  # (local_time, host_time) of the newest sample
        self._rate = 1.0     # EMA host sim rate (local-time units)
        self._last_jitter = None  # Session 7.5a: per-arrival jitter sample

    @property
    def ready(self):
        """True once at least one snapshot has been recorded."""
        return self._anchor is not None

    @property
    def offset(self):
        """The current anchor's HOST stamp (None before the first
        snapshot). Session 7.9: this is the point-slope anchor's host_time
        (h_anchor), NOT the 7.1 absolute intercept A — the estimate is now
        `offset + rate * (local_time - anchor_local_time)`. Exposed for the
        7.8 debug overlay."""
        if self._anchor is None:
            return None
        return self._anchor[1]

    @property
    def rate(self):
        """Current host sim-rate estimate (1.0 = real-time). Exposed for
        the 7.8 debug overlay."""
        return self._rate

    @property
    def last_jitter(self):
        """The per-arrival jitter sample of the LAST recorded snapshot
        (Session 7.5a), or None until one exists.

        Session 7.9: this is the PREDICTION RESIDUAL — how far this
        arrival's stamp is from where the PREVIOUS model (the previous
        anchor + rate) predicted it would be:
        `|host_sim_time - (h_anchor_prev + rate * (local_time - t_anchor_prev))|`.
        A steady arrival sits near 0; a late/early packet spikes. This is
        the well-conditioned replacement for the 7.1 offset sample
        (`host_time - rate * local_time`), which had the same leverage
        problem as the intercept (it was a difference of two large numbers).
        Exposed so LatencyTracker can consume it (and for the 7.8 debug
        overlay)."""
        return self._last_jitter

    def record(self, local_time, host_sim_time):
        """Record a snapshot that was taken at `host_sim_time` (its wire
        stamp) and observed at `local_time`. Out-of-order stamps (should not
        happen over TCP) are dropped, like SnapshotBuffer.push.

        Returns True if the sample was recorded, False if it was dropped
        (Session 7.5a: the caller feeds the jitter sample to the
        LatencyTracker only on a recorded arrival).

        Session 7.9 (point-slope re-anchor): the model is
        `host = h_anchor + rate * (local_time - t_anchor)`, anchored on the
        NEWEST sample. On each recorded arrival:
          1. the jitter sample is the PREDICTION RESIDUAL — how far this
             arrival's stamp is from where the PREVIOUS anchor + PREVIOUS
             rate predicted it (a small, well-conditioned number, unlike the
             7.1 offset sample);
          2. the RATE EMA is updated from the Δhost/Δlocal of the two newest
             samples (unchanged from 7.1);
          3. the anchor is MOVED to this newest sample (t_anchor, h_anchor)
             = (local_time, host_sim_time).
        Because the anchor is always the newest sample, `now()` extrapolates
        forward from real data at the fitted rate — and the offset it adds is
        a small delta (time since the newest sample), so the estimate stays
        well-conditioned for the whole session (no leverage growth)."""
        if self._anchor is not None and host_sim_time <= self._anchor[1]:
            return False
        if self._anchor is not None:
            lt0, ht0 = self._anchor
            # Session 7.9: the jitter sample is the PREDICTION RESIDUAL —
            # how far this arrival's stamp is from where the PREVIOUS model
            # (previous anchor + PREVIOUS rate) predicted it. Computed with
            # the rate BEFORE this sample updates it (the standard
            # innovation: measurement minus the prior model's prediction),
            # so a late packet spikes the full amount instead of being
            # partially absorbed into the rate EMA. A steady arrival sits
            # near 0; a late/early packet spikes. Well-conditioned (a small
            # delta), unlike the 7.1 offset sample.
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

        Session 7.9: point-slope form — extrapolate FORWARD from the newest
        sample (the anchor) at the fitted rate:
        `h_anchor + rate * (local_time - t_anchor)`. The offset is a small
        delta (time since the newest sample), so the estimate stays
        well-conditioned for the whole session (no leverage growth)."""
        if self._anchor is None:
            return None
        t_anchor, h_anchor = self._anchor
        est = h_anchor + self._rate * (local_time - t_anchor)
        newest = h_anchor          # the anchor IS the newest sample
        if est > newest + self.MAX_LEAD:
            return newest + self.MAX_LEAD
        return est


class LatencyTracker:
    """The adaptive interpolation delay (Session 7.5a).

    A fixed INTERP_DELAY (0.1 s) is the right lag for a clean connection,
    but on a jittery wifi link a snapshot that arrives late strands the
    render point ahead of the data: the buffer clamps to the newest
    snapshot and the remote render stutters (the 7.1 MAX_LEAD cap turns
    the outrun into a hold, but a hold is still a stall). The fix is to
    render FURTHER in the past when the arrival pattern is ragged: the
    delay grows with the measured jitter, so a late packet lands inside
    the window instead of behind the render point.

    The delay is a PURE FUNCTION OF THE ARRIVAL PATTERN (7.5a scope:
    compute the value; 7.5b re-anchored the render point to it — see
    `RenderPoint`, which reads `.delay` every frame). The input is the
    HostTimeEstimator's per-arrival jitter sample
    (`HostTimeEstimator.last_jitter` — `|offset_sample - offset_ema|`,
    the 7.1 offset samples repurposed): a steady 10 Hz stream sits near
    0, a late/early packet spikes.

    The law (config knobs, see config.py):

        delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter),
                      INTERP_DELAY_MIN, INTERP_DELAY_MAX)

    with EMA alpha = 0.2 (the same time constant as the estimator's
    offset EMA — a 10 Hz sample stream settles in ~0.1 s). Two
    disciplines on top of the clamp:

      - the delay only ever moves by at most MAX_STEP = 1/60 s per
        frame, so the render point (7.5b: newest_snap - delay) can never
        jump — a jitter spike raises the delay over ~20 frames instead of
        teleporting the render ~100 ms into the past;
      - the delay never shrinks below INTERP_DELAY_MIN (= INTERP_DELAY,
        the base) — adaptation only adds lag, it never goes under the
        clean-connection value.

    Wiring (the 7.5b seam, documented here in 7.5a): the client calls
    `estimator.record(now, stamp)` on each snapshot arrival and, when it
    returns True, `tracker.update(estimator.last_jitter)`; every frame it
    calls `tracker.tick(dt)` (the per-frame smoothing step) and reads
    `tracker.delay`.
    """

    ALPHA = 0.2          # jitter EMA (same time constant as the 7.1 offset EMA)
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
        the first sample). Exposed for the 7.8 debug overlay."""
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
        at most MAX_STEP (1/60 s), so the render point it drives (7.5b)
        can never jump. `dt` is the real frame time; the step is
        frame-BOUND (one frame may move at most MAX_STEP regardless of
        dt), which is what keeps the render point continuous on any
        display refresh rate."""
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
    """The client's render point in HOST SIM TIME (Session 7.5b).

    7.1 rendered at `host_time_estimate - INTERP_DELAY`, where the
    estimate extrapolates the newest snapshot's stamp forward at the
    fitted host rate. That is a MODEL of the host clock: on a clean link
    it sits ~0.01 s ahead of the newest ARRIVED snapshot, but under
    jitter/loss the model and the data disagree — the estimate can lag
    the newest stamp (a late packet pulls the offset EMA down) or lead
    it (the MAX_LEAD cap holds it one interval ahead), and the render
    point wanders by as much as the disagreement. The fix (this
    session): anchor the render point on the DATA, not the model —

        render_t = newest_arrived_snapshot_stamp - delay

    a fixed distance behind the newest snapshot in the buffer. A late
    packet then simply holds the point on the last window (a <=1-frame
    stall) instead of the point outrunning the data and
    clamp-stuttering, and the point can never exceed the newest
    snapshot (delay >= INTERP_DELAY_MIN > 0).

    This REPLACES the 7.1 host-time estimate for the render point — the
    genuinely risky re-anchor of the core render path, isolated in this
    session. The estimator is NOT deleted: it still runs, and its
    per-arrival jitter sample (`last_jitter`) is what feeds the
    LatencyTracker (7.5a). It is no longer a clock the render reads.

    One subtlety: `newest - delay` on its own is a STAIRCASE. The newest
    stamp is flat between arrivals and jumps by one snapshot interval
    (0.1 s) when a packet lands, so the raw anchor would jump 0.1 s
    forward every 100 ms — a 6x telegraph, not a render. The point
    therefore CHASES the anchor: each frame it moves toward
    `newest - delay` at at most `dt` (the frame's REAL time), so the
    point advances at 1x real time — the same rate the anchor creeps
    forward at (the host's sim rate, ~1x).

    Capping by `dt` (NOT a fixed 1/60 s) is what keeps the point from
    drifting BEHIND the data on a client running below 60 FPS. A fixed
    1/60 cap advances the point by only 1/60 s per frame, but a 47 FPS
    frame lasts 1/47 s of real time, so the point falls behind by
    (1/47 - 1/60) every frame and the remote render looks at ever-older
    snapshots (the 7.8 log's 4.3 s drift on the Mac, ~4.6 s of stale
    data). Capped by `dt`, the point advances 1/47 s per frame at
    47 FPS — exactly 1x real time — and never drifts. The cap still
    smooths the staircase (a 0.1 s anchor jump is covered over ~6 frames
    at 60 FPS, ~5 at 47) and still bounds a hiccup (the caller clamps
    `dt` to MAX_FRAME_DT, so one frame moves the point at most that far).
    The net per-frame advance is bounded by `dt` + MAX_STEP_DELAY (the
    chase step plus the delay's own per-frame movement) — the render
    point is continuous on any refresh rate, and a jitter spike or a late
    packet can never teleport it. In steady state the chase is slack
    (the anchor creeps forward at the host's sim rate, ~1x, and the point
    tracks it within ~one frame's step); under a stall the anchor holds
    and the point converges onto it and holds there.

    `advance(dt, newest)` is called once per frame with the newest
    snapshot stamp in the buffer (None before the first arrival — the
    point is None until then, and the caller draws its waiting state).
    `now()` returns the render point (host sim time) or None.
    """

    # Session 7.9: the chase cap is the frame's REAL time (`dt`), not a
    # fixed 1/60 s — see the class docstring. A fixed 1/60 cap made the
    # point advance at (frame_rate/60) x real time, so any client below
    # 60 FPS drifted behind the data (the 7.8 log: 4.3 s on the Mac).
    # `dt` is clamped by the caller to MAX_FRAME_DT, which bounds a hiccup.
    MAX_STEP_DELAY = LatencyTracker.MAX_STEP  # the delay's own per-frame move
    # Session 7.10: catch-up fraction. When the point is BEHIND the target by
    # more than ~5 frames (d > 5*dt), it advances by CATCHUP * d per frame
    # (20% of the gap) instead of just dt (1x real time), so a large gap
    # (a snapshot burst after a wire stall, or the buffer filling at the
    # start) closes in ~5 frames (~83 ms at 60 FPS) instead of ~10 (166 ms).
    # For small gaps (d < 5*dt) the cap is dt (1x real time), preserving the
    # staircase smoothing. The steady-state gap (where the catch-up rate
    # equals the target's 1x advance) is 5*dt (~0.083 s at 60 FPS) — the
    # point settles just behind the target, inside the buffer's window.
    # Session 7.10b: raised from 0.1 to 0.2 to halve the catch-up duration
    # after a snapshot burst (the "big lag spike" in the 7.10 re-test: a
    # 1.4 s gap took ~10 frames to close at CATCHUP=0.1; at 0.2 it takes
    # ~5). The per-frame fast-forward is 20% of the gap (e.g. 280 ms of sim
    # time in one 16.7 ms frame for a 1.4 s gap) — a visible but brief
    # fast-forward, better than a 166 ms stall at 1x.
    CATCHUP = 0.2

    def __init__(self, tracker=None):
        self._tracker = tracker
        self._t = None         # the render point (host sim time), or None

    @property
    def tracker(self):
        """The LatencyTracker this point reads its delay from (the 7.8
        debug overlay reads both). Settable so tests can swap trackers."""
        return self._tracker

    @tracker.setter
    def tracker(self, trk):
        self._tracker = trk

    def advance(self, dt, newest):
        """One frame of real time `dt`. `newest` is the newest snapshot
        stamp in the buffer (host sim time) or None. Returns the render
        point (or None while there is no anchor yet).

        The chase is capped by `dt` (the frame's REAL time), NOT a fixed
        1/60 s (Session 7.9): the point advances at 1x real time on any
        refresh rate, so it never drifts behind the data on a client
        running below 60 FPS. `dt` is clamped by the caller to
        MAX_FRAME_DT, which bounds a hiccup (one frame moves the point at
        most that far).

        Session 7.10 (catch-up): a pure 1x cap has a flaw — it closes a
        LARGE gap at only 1x real time, and if the target also advances at
        ~1x (the host's sim rate), the gap NEVER closes. The 7.10 log
        showed exactly this: the buffer filled with ~1 s of snapshots at
        the start, the point was born 1.0 s behind its target, and it
        stayed there for the whole session (d = 0.96 -> 1.08 s, stable),
        rendering 700 ms of stale data (the buffer's oldest snapshot —
        the point was clamped to the buffer's back edge). The fix: when
        the point is behind by more than ~5 frames (d > 5*dt), advance
        by CATCHUP * d per frame (20% of the gap) instead of dt, so the
        gap closes exponentially (halving every ~3.5 frames at 60 FPS)
        and settles at the steady-state gap of 5*dt (~0.083 s at 60 FPS —
        just behind the target, inside the buffer's window). For small
        gaps (d < 5*dt) the cap is dt (1x real time), so the staircase
        smoothing is unchanged and the point never teleports. The step is
        always bounded by d (min(d, ...)), so the point can never
        overshoot the target. See the CATCHUP class attr for the
        rationale."""
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
            # Session 7.10: catch-up. When the point is behind the target by
            # more than ~5 frames (d > 5*dt), advance by CATCHUP * d per
            # frame (20% of the gap) instead of just dt (1x real time), so a
            # large gap (a snapshot burst after a wire stall, or the buffer
            # filling at the start) closes in ~5 frames (~83 ms at 60 FPS).
            # For small gaps (d < 5*dt) the cap is dt (1x real time),
            # preserving the staircase smoothing. The steady-state gap
            # (where the catch-up rate equals the target's 1x advance) is
            # 5*dt (~0.083 s at 60 FPS) — the point settles just behind the
            # target, not 1.0 s behind it (the 7.9 behavior, where the gap
            # never closed). See the CATCHUP class attr for the rationale.
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
    # (Session 5b.1) and never reused within a run — a stable identity
    # across snapshots. Split children get fresh ids -> they read as
    # "new".
    return a_s[0]


def _match_bullets(prev_list, curr_list, dt):
    """Match curr bullets to prev bullets by PREDICTED POSITION (Session
    7.2). Returns a list, one entry per curr bullet: the prev twin's
    (x, y) or None (a new bullet — pops in at its curr position, the
    same membership rule as enemies/asteroids).

    Why not match by INDEX within a kind (the 7.2 plan's original
    assumption)? The sim's bullet lists are sliding windows: a fired
    bullet appends, an expired one is culled from the FRONT
    (`self.bullets = [b for b in self.bullets if b.life > 0]`), so every
    later index slides down by one each time a bullet dies. A probe over
    20 snapshot windows (6 ticks each, the test's script_input) found
    ~11% of same-index pairs were NOT the same bullet — index matching
    would lerp a bullet from a stranger's position and make it jump.

    Bullets fly in straight lines (no steering — only missiles steer,
    and they are matched with the same method), so a prev bullet's
    position at the curr snapshot is exactly prev.pos + prev.vel * dt.
    Each curr bullet is matched to the nearest prev bullet's predicted
    position within MATCH_RADIUS; a prev bullet is used at most once
    (two bullets can't occupy the same predicted spot). dt is the
    snapshot window (t_curr - t_prev); at dt == 0 the prediction is the
    prev position itself.

    MATCH_RADIUS: a bullet moves BULLET_SPEED * dt per window (860 * 0.1
    = 86 px at the 10 Hz cadence). The radius must exceed that (a real
    twin is always within ~1 window of motion of its prediction) but be
    small enough that two DIFFERENT bullets never both fall inside it of
    each other's predictions. Bullets from the same gun fire from the
    same muzzle ~10 px apart at 100 Hz, so they can be as close as
    ~10 px; 30 px is well clear of that while a mis-match would require
    two bullets' predicted positions to be within 30 px of the same curr
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
    """Interpolated poses for each player ship ((x, y, angle), Session 6.8),
    each enemy ((tag, x, y, angle, vx, vy, id), Session 7.2), each asteroid
    ((x, y)), and each projectile ((x, y, vx, vy, kind, owner, boost),
    Session 7.2), between two Game snapshots.

    prev_s / curr_s are the 11-tuples from Game.snapshot(); alpha in [0, 1]
    is clamped (never extrapolate into the future). Returns a plain dict —
    no pygame objects, no sim state touched:

        {'ships': [(x, y, angle, dead, shield_dump, shield_clock), ...],
         'enemies': [(tag, x, y, angle, vx, vy, id,
                      shield_dump, shield_clock), ...],
         'asteroids': [(x, y), ...],
         'bullets': [(x, y, vx, vy, kind, owner, boost, mid), ...]}

    Each projectile entry carries `mid` (10.3b): the missile's unique id
    (player_index, seq) for kind == 'missile', else None. The client uses
    it to dedup its own ghost missiles against the buffer's copy of the
    same missile (no double-draw) — see predicted_view.

    Ships carry their `dead` flag too (Session 10.1): ship_s[20], taken
    from the CURRENT snapshot (membership follows curr). The remote peer
    uses it to skip drawing a dead ship (it disappears until it respawns)
    and to spawn a client-side explosion on the alive->dead transition.

    Ships also carry their shield-impact state (Session 10.2):
    shield_dump (ship_s[6]) and shield_clock (ship_s[7]), LERP'd between
    the two snapshots like the pose. These drive the shield-impact flash
    (Ship._draw_shield: the blue->white glow that appears when the shield
    is struck). The client's own ship (the ghost) already shows its flash
    (a real Ship, restored on reconcile); the REMOTE ship is drawn via
    Ship.draw too, so feeding it the buffer's shield state makes its
    impact flash render identically to the host's. No wire change — both
    fields are already in the ship snapshot.

    Ships carry their angle (Session 6.8): ship_s[4] is the RAW unbounded
    angle (see Ship.snapshot), so it is lerp'd with `lerp_angle` — the
    wrapped-delta lerp `Ship.sync_render` uses — never a plain lerp of the
    raw values (which would swing the wrong way around the circle once the
    raw angle leaves [-pi, pi]). The result stays anchored on the prev
    angle's unwrapped offset, exactly like sync_render's rangle.

    Enemies carry their angle too (Session 7.2): e_s[0][4] is the enemy
    ship's RAW angle, lerp'd with the SAME wrapped-delta rule (the plan's
    "same wrapped-delta rule as player ships, 6.8"), so the remote peer
    draws the real enemy hull at its interpolated orientation instead of a
    10 px dot (the D4 defect). The TAG (the first element of the
    (tag, e_s) pair — 'ai'/'mote'/'test') comes from the CURRENT
    snapshot: membership already follows curr, and the tag is constant
    for a given enemy id, so curr's tag is the right one. The VELOCITY
    (e_s[0][2], e_s[0][3]) is taken from curr (it is presentation data
    for the targeting proxy — lead_point needs pos + vel — and a 10 Hz
    stale velocity is fine for a reticle). The ID (e_s[2]) is carried so
    a caller can track a specific enemy across frames (enemies churn, so
    there is no stable index like ship 0) — presentation only.

    Projectiles (Session 7.2, the D3 defect — the client rendered no
    bullets at all): snapshot indices 2 (player bullets), 3 (enemy
    bullets), 4 (missiles) are all interpolated. Each entry is
    (x, y, vx, vy, kind, owner, boost): position lerp'd between the two
    snapshots (matched by predicted position — see `_match_bullets`),
    velocity from curr (drives the stretched-capsule render length),
    kind in {'player', 'enemy', 'missile'}, owner from curr (the bullet's
    ship id — presentation only; the client never runs the sim), and
    boost from curr (missile exhaust flicker; 0.0 for non-missiles).
    A bullet fired mid-window pops in at its curr position (no earlier
    position to lerp from); one that expires drops out (membership
    follows curr — the same rule as enemies/asteroids).

    Entity matching:
      - player ships by INDEX (Session 6.1): ships don't turn over — a dead
        ship is game over, not a respawn — so slot i of curr_s[0] is the
        same ship as slot i of prev_s[0]. Output order is curr_s[0]'s order.
      - enemies by their ship id (e_s[2]);
      - asteroids by their rock id (a_s[0]) — see _asteroid_key;
      - projectiles by predicted position within a kind — see
        `_match_bullets` (index matching is wrong: the lists slide).
    Enemy/asteroid matching is by IDENTITY, not index (Session 5b-1):
    This survives entity turnover inside the window: a rock that splits,
    a sector that tops up or culls, an enemy that dies and respawns.

    Membership follows the CURRENT snapshot: entities present in curr_s
    are rendered — a new entity pops in at its curr position (there is no
    earlier position to lerp from); an entity that died within the window
    drops out (there is no death position to lerp to — it vanishes a
    snapshot interval early, imperceptible at 10 Hz). Output order is
    curr_s's order.
    """
    a = min(1.0, max(0.0, alpha))

    # Player ships: matched by index (see docstring). A ship missing from
    # prev_s[0] (shouldn't happen — ships don't turn over) pops in at its
    # curr pose, same rule as new enemies/asteroids. The angle is lerp'd
    # with lerp_angle (wrapped delta), not a plain lerp — see docstring.
    # 10.1: the per-ship `dead` flag (ship_s[20]) is carried through the
    # buffer so the remote peer can (a) skip drawing a dead ship (it
    # "disappears" until it respawns) and (b) spawn a client-side explosion
    # on the alive->dead transition. It is taken from the CURRENT snapshot
    # (membership follows curr) — a dead ship's pose is static, so only the
    # flag, not the pose, changes at the death.
    # The prev ship carries (x, y, angle, shield_dump, shield_clock) — the
    # shield state (10.2) is lerp'd with the pose so the remote ship's
    # impact flash fades smoothly across the window.
    prev_ships = {i: (p[0], p[1], p[4], p[6], p[7])
                  for i, p in enumerate(prev_s[0])}
    ships = []
    for i, c in enumerate(curr_s[0]):
        cp = (c[0], c[1], c[4], c[6], c[7])
        pp = prev_ships.get(i)
        if pp is None:
            ships.append((cp[0], cp[1], cp[2], c[20], cp[3], cp[4]))
        else:
            ships.append((lerp(pp[0], cp[0], a),
                          lerp(pp[1], cp[1], a),
                          lerp_angle(pp[2], cp[2], a),
                          c[20],
                          lerp(pp[3], cp[3], a),
                          lerp(pp[4], cp[4], a)))

    # Enemies (Session 7.2): carry (tag, x, y, angle, vx, vy). The angle
    # is lerp'd with lerp_angle (the same wrapped-delta rule as player
    # ships, 6.8); the tag and velocity come from the CURRENT snapshot
    # (membership follows curr; velocity is presentation data for the
    # targeting proxy).
    # The prev enemy carries (x, y, angle, shield_dump, shield_clock) —
    # the shield state (10.2b) is lerp'd with the pose so the remote
    # enemy's impact flash fades smoothly across the window (same rule as
    # the player ships' shield state).
    prev_enemies = {}
    for _tag, p in prev_s[1]:
        es = p[0]
        prev_enemies[_enemy_id(p)] = (es[0], es[1], es[4], es[6], es[7])
    enemies = []
    for tag, c in curr_s[1]:
        ce = c[0]
        cpos = (ce[0], ce[1])
        eid = _enemy_id(c)
        pp = prev_enemies.get(eid)
        if pp is None:
            enemies.append((tag, cpos[0], cpos[1], ce[4], ce[2], ce[3],
                            eid, ce[6], ce[7]))
        else:
            enemies.append((tag,
                            lerp(pp[0], cpos[0], a),
                            lerp(pp[1], cpos[1], a),
                            lerp_angle(pp[2], ce[4], a),
                            ce[2], ce[3],
                            eid,
                            lerp(pp[3], ce[6], a),
                            lerp(pp[4], ce[7], a)))

    prev_rocks = {_asteroid_key(p): (p[1], p[2]) for p in prev_s[5]}
    asteroids = []
    for c in curr_s[5]:
        cp = (c[1], c[2])
        pp = prev_rocks.get(_asteroid_key(c))
        if pp is None:
            asteroids.append(cp)
        else:
            asteroids.append((lerp(pp[0], cp[0], a),
                              lerp(pp[1], cp[1], a)))

    # Projectiles (Session 7.2): all three kinds, matched by predicted
    # position within a kind (see _match_bullets). `dt` is the time
    # BETWEEN the two snapshots — the span the prev bullets travel before
    # the curr snapshot. `positions_at` passes the bracketing span it
    # already computed (exact); a direct call (tests) falls back to the
    # nominal snapshot cadence, which is the realistic window. The
    # position lerp itself uses alpha (exact); dt only scales the
    # predicted-position matching, and MATCH_RADIUS is generous enough
    # that a slightly-off dt still matches correctly.
    if dt is None:
        dt = SNAPSHOT_INTERVAL * TICK
    bullets = []
    # 10.3b: missiles carry their unique id (player_index, seq) — the
    # client's dedup key (its own ghost missiles are suppressed against the
    # buffer's copy of the same id; the other player's missiles are drawn
    # from the buffer). Bullets carry None (they are matched by position,
    # not identity — see _match_bullets).
    for kind, idx, boost_field, id_field in (("player", 2, None, None),
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
    """The remote peer's interpolation store (Session 5b.3).

    The authoritative peer stamps each `Game.snapshot()` with the sim time
    it was taken at and sends it on. The remote peer pushes them here, in
    arrival order, and renders `INTERP_DELAY` seconds in the PAST: it asks
    `positions_at(sim_time - INTERP_DELAY)` for the interpolated pose
    (x, y, angle) of each ship (Session 6.8) and the (x, y) of each enemy
    and each asteroid.

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

    def __init__(self, max_snapshots=8):
        # (sim_time, snapshot) pairs, oldest first. Bounded so a stalled
        # authoritative peer can't grow the buffer without limit.
        self._snaps = []
        self._max = max_snapshots

    def __len__(self):
        return len(self._snaps)

    def newest_time(self):
        """The sim time of the newest snapshot in the buffer (None when
        empty). Session 7.1: the client's render clock anchors on this —
        the host's own clock as carried by the wire — instead of the
        display's wall clock."""
        if not self._snaps:
            return None
        return self._snaps[-1][0]

    def latest_snapshot(self):
        """The newest snapshot in the buffer (None when empty).

        10.3b fix: the ghost's handback decision needs the missile list
        of the NEWEST snapshot — the one that actually carries a just-
        launched missile — not just its missile_seq. The seq alone is
        necessary but not sufficient: a missile the host launched after
        the newest snapshot is provably NOT in the buffer yet, even
        though its seq is below any future snapshot's seq."""
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
        ({'ships': [(x, y, angle, dead, shield_dump, shield_clock), ...],
        'enemies': [(tag, x, y, angle, vx, vy, id,
                     shield_dump, shield_clock), ...],
        'asteroids': [(x, y), ...],
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
                # Pass the REAL bracketing span as the projectile window
                # (Session 7.2): predicted-position matching needs the
                # time the prev bullets travel before the curr snapshot.
                return interp_positions(si, sj, alpha, dt=span)
        # Unreachable: render_t is strictly inside (t0, tN).
        return None


class GhostMissile:
    """A presentation missile the prediction ghost fired (10.3b).

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
    """Client-side prediction ghost for the local player's ship
    (Session 5b.4a).

    The remote peer renders its OWN ship from the interpolation buffer at
    `sim_time - INTERP_DELAY`, which feels ~100 ms laggy. The classic fix is
    to predict the local ship: step a private Ship with the player's OWN
    input at the sim's fixed rate, and snap it back to the authoritative
    ship snapshot when one lands. This class is that ghost — a full `Ship`
    (not a minimal kinematic stand-in) so reconciliation reuses the existing
    `apply_snapshot` round-trip and the physics stay faithful.

    Clock discipline (Session 7.1): the ghost must step at the SIM's fixed
    rate, not the display's. The caller feeds it real frame times via
    `advance(dt, inp)`; the internal accumulator converts that to whole
    `STEP` steps. Stepping once per display frame (the pre-7.1 behavior)
    made the ghost integrate `refresh_rate / 60` times the sim's motion —
    on a 144 Hz monitor it raced 2.4x ahead and every reconcile yanked it
    back (the reported "spiking and snapbacking").

    The ghost is a PRESENTATION object: it is stepped with the local input
    and reconciled to authority, but it never feeds the sim. The sim stays
    authoritative on one peer; this only feeds the render.

    Local bullets (Session 7.3): the ghost also keeps its OWN gun shots as
    presentation `Bullet`s (`self.local_bullets`), so the player sees their
    own fire immediately instead of waiting ~100 ms for the host's next
    snapshot. They are stepped at the sim's rate (via `advance`) and culled
    on expiry. They are NOT reconciled to the authoritative list: the 2P
    snapshot carries no stable player id (both ships are ship_id 0), so the
    local player's authoritative bullets cannot be isolated by owner — and
    the ghost's own bullets ARE the local player's by construction (same
    input + weapon state the host applies). They are presentation only:
    they never collide, never feed the sim, and the remote peer's own
    bullets still come from the interpolation buffer (Session 7.2).

    Known limitation (v1): reconciliation is a FULL SNAP — the ghost jumps
    to the authoritative position when a snapshot lands. Dead-reckoning
    rewind (replaying a local input buffer from the last reconciled state)
    is OUT of scope; the test (test_interpolation part e) measures and
    bounds the resulting drift so it is not a silent regression.

    The Game-fed per-tick fields (`tracked`, `laser_target`,
    `missile_target`, `contacts`) are reset to their idle values before each
    step: the ghost has no enemy list to target, and none of these fields
    affect the ship's pos/vel/angle motion except through the weapon
    charge/lock power draw (which can flip the brownout latch). That power
    draw is exactly the drift the ghost is allowed to have — it is bounded
    by the test, not eliminated.

    10.3a exception: `laser_target` is now picked from the caller's
    `enemies` proxies (the interpolation buffer's enemy entries) when
    given — the ghost charges + fires a beam on the same tick the host
    does, and the fired beams are kept as presentation entries in
    `self.local_beams` (world-space muzzle -> target, aged at the host's
    0.15 s beam ttl). `contacts` stays idle (10.3c).

    10.3b: `missile_target` is now picked from the caller's `enemies`
    proxies too (the SAME nearest-in-range math the host's
    `_pick_missile_target` uses), so the ghost locks + launches a missile
    on the same tick the host does. The LAUNCHED missiles are kept as
    presentation `GhostMissile`s in `self.local_missiles` (each with the
    same unique id the host assigns — `(player_index, seq)` — so the
    client dedups it against the buffer's copy; homing is client-side,
    steering toward the buffer's enemy proxy for the target id). Call
    `step_local_missiles` after `step` to advance + cull the list.
    """

    # Session 7.6: how much host time of local input to keep for rewind
    # replay. 2 s is far more than the worst-case snapshot age (the
    # adaptive delay caps the render lag at INTERP_DELAY_MAX = 0.35 s, and
    # a snapshot is at most one interval old), so the buffer never runs out
    # of the inputs a rewind needs — while staying small (a 2 s buffer at
    # 60 Hz input sampling is ~120 entries).
    INPUT_BUFFER_MAX = 2.0

    def __init__(self, hull=None, loadout=None, local_index=0):
        # A self-contained Ship: __init__ builds components/thrusters/
        # weapons/shield/sensors/collision from the hull+loadout and holds
        # no reference to Game, so it can be stepped in isolation.
        self._ship = Ship(hull=hull, loadout=loadout)
        self._seeded = False
        # Fixed-step accumulator (Session 7.1): converts real frame times
        # into whole STEP steps so the ghost runs at the sim's rate on any
        # display refresh rate.
        self._acc = 0.0
        # Session 7.6: the local input buffer — (host_time, ShipInput) pairs,
        # oldest first, bounded to INPUT_BUFFER_MAX seconds of host time.
        # `host_time` is the client's estimate of the host's sim clock at the
        # moment the input was sampled (the 7.1 HostTimeEstimator), so a
        # snapshot arriving at sim time T can replay the inputs the host
        # actually applied between T and now (see `reconcile_rewind`).
        self._input_buffer = []
        # Session 7.3: the ghost's OWN gun shots, as presentation Bullets.
        # The client never runs the sim, so without this the player's own
        # shots appear only when the host's next snapshot arrives (~100 ms
        # later) — "I fired and nothing happened for a tenth of a second".
        # `local_index` is the local player's index (0 = host, 1 = client);
        # it is carried for diagnostics and so a future session can match
        # the authoritative local-owner bullets (the 2P snapshot carries no
        # stable player id — both ships are ship_id 0 — so 7.3 keeps the
        # ghost's own bullets as the local player's by construction).
        self.local_index = local_index
        self.local_bullets = []
        # 10.3a: the ghost's OWN laser discharges, as presentation entries
        # [start, end, age, ttl] in WORLD space (muzzle -> target). The
        # client never runs the sim, so without this the player's own beam
        # appears only when the host's next snapshot arrives (~100 ms
        # later). Like local_bullets these are presentation only: no
        # collision, no sim feedback.
        self.local_beams = []
        # 10.3b: the ghost's OWN missiles, as presentation GhostMissiles.
        # The client never runs the sim, so without this the player's own
        # missiles appear only when the host's next snapshot arrives
        # (~100 ms later) — and the buffer's copy of the SAME missile would
        # then be drawn a second time (the double-draw). Each carries the
        # SAME unique id the host assigns at launch ((player_index, seq) —
        # the ghost's missile_seq is resynced to the host's via the ship
        # snapshot on reconcile), so predicted_view dedups it against the
        # buffer's copy. Homing is client-side (see GhostMissile).
        self.local_missiles = []

    @property
    def ship(self):
        """The underlying ghost Ship (read access for tests/diagnostics)."""
        return self._ship

    @property
    def seeded(self):
        """True once an authoritative snapshot has been applied."""
        return self._seeded

    def seed(self, ship_s):
        """Apply the FIRST authoritative ship snapshot (index 0 of the Game
        snapshot) to the ghost. Subsequent snapshots use `reconcile`."""
        self._ship.apply_snapshot(ship_s)
        self._seeded = True

    def reconcile(self, ship_s):
        """Snap the ghost to an authoritative ship snapshot.

        Full snap for v1 (see class docstring for the dead-reckoning
        limitation). Idempotent with `seed` — a caller may use either for
        the first snapshot. Session 7.6: the game loop uses
        `reconcile_rewind` instead (dead-reckoning); this stays for tests
        and as the v1 fallback."""
        self._ship.apply_snapshot(ship_s)
        self._seeded = True

    def record_input(self, host_time, inp):
        """Record the local input sampled at host time `host_time`
        (Session 7.6). `host_time` is the client's estimate of the host's
        sim clock at the moment the input was sampled (the 7.1
        HostTimeEstimator) — NOT the local wall clock — so that when a
        snapshot arrives at sim time T, `reconcile_rewind` can replay the
        inputs the host actually applied between T and now: the host
        applies the LATEST received input each tick (pinned #5), so the
        replay must select inputs by their host-time stamp, and the stamp
        must be in the host's clock for that selection to line up.

        The buffer is bounded to INPUT_BUFFER_MAX seconds of host time
        (oldest entries dropped). `inp` is stored by reference; the caller
        passes a fresh ShipInput each frame (ShipInput.from_keys builds a
        new one), so no copy is needed."""
        self._input_buffer.append((host_time, inp))
        # Drop entries older than INPUT_BUFFER_MAX seconds of host time.
        cutoff = host_time - self.INPUT_BUFFER_MAX
        while self._input_buffer and self._input_buffer[0][0] < cutoff:
            self._input_buffer.pop(0)

    def reconcile_rewind(self, ship_s, snap_time, now, enemies=None):
        """Dead-reckoning reconcile (Session 7.6) — replaces the v1 full
        snap. Apply the authoritative ship snapshot taken at sim time
        `snap_time`, then REPLAY the local inputs the host applied between
        `snap_time` and `now`, so the ghost ends up at exactly
        `authority @ snap_time + local inputs since snap_time` — the
        100 ms of prediction is REBUILT, not snapped (pinned #4).

        10.3a: `enemies` (a list of lightweight enemy proxies, see
        `step`) is fed to each replayed step so the replayed laser
        charge/lock state matches the host's (the host's replayed ticks
        saw the live enemies in the wedge). The caller passes the
        buffer's enemy proxies at the snapshot's render point; a None
        (tests) replays with no targets — the pre-10.3a behavior.

        The replay mirrors the host's own tick loop: for each sim tick T
        from snap_time to now (step STEP), the host applies the LATEST
        received input (pinned #5), so the replay steps the ghost with the
        newest buffered input whose host_time <= T. That input-selection
        is what makes the replay reproduce the host's motion — the
        test_interpolation (l) battery proves it to ~1e-9 with the same
        input the host used.

        The 7.3 local bullet list is rebuilt the same way: `step` re-fires
        from the ghost's re-seeded weapon state (apply_snapshot restores
        cooldown/charge/lock), so the ghost's own shots resume from the
        authoritative fire cadence instead of the predicted one.

        `now` is the client's current estimate of the host's sim clock
        (the same value the caller uses to stamp inputs). The replay runs
        at most (now - snap_time)/STEP steps — one snapshot interval
        (~6 steps) in steady state. The fixed-step accumulator is reset
        so the next `advance` starts clean from the replayed state.
        """
        self._ship.apply_snapshot(ship_s)
        self._seeded = True
        self._acc = 0.0
        buf = self._input_buffer
        if not buf:
            return
        # Walk the sim ticks from snap_time to now. `i` is the tick index
        # (0 = snap_time itself, the snapshot's own tick — already applied
        # by the snapshot, so the first replayed tick is snap_time + STEP).
        # Index of the newest buffered input with host_time <= tick time.
        # The buffer is ordered by host_time (input is sampled in order),
        # so this is a monotone pointer.
        j = -1
        # The number of whole STEP steps between snap_time and now. The
        # snapshot reflects the state AFTER the tick that ended at
        # snap_time, so the first replayed tick is the one that STARTS at
        # snap_time (the host applies the input stamped at a tick's start
        # during that tick — the client records the input at the same
        # host-time the host samples it).
        n = int((now - snap_time) / TICK + 1e-9)
        for i in range(n):
            t = snap_time + i * TICK
            while j + 1 < len(buf) and buf[j + 1][0] <= t + 1e-9:
                j += 1
            inp = buf[j][1] if j >= 0 else buf[0][1]
            self.step(TICK, inp, enemies=enemies)
            self.step_local_bullets(TICK)
            self.step_local_beams(TICK)
            self.step_local_missiles(TICK, enemies)   # 10.3b

    def step(self, dt, inp, enemies=None):
        """Advance the ghost ONE fixed step with the LOCAL input.

        `dt` is passed by the caller (the sim's STEP) because netcode.py
        must not import it from game.py (game.py imports netcode.py). The
        Game-fed per-tick fields are reset to idle before the step — the
        ghost has no enemy list to target (see class docstring).

        Session 7.3: the SHOTS the ghost's gun fires this step are kept as
        presentation `Bullet`s in `self.local_bullets` (capped at
        MAX_BULLETS, mirroring the host's world cap) so the player sees
        their own shots immediately instead of waiting ~100 ms for the
        host's next snapshot. Call `step_local_bullets` after `step` to
        advance + cull the list (the game loop does this via `advance`).

        10.3a: `enemies` is a list of lightweight enemy proxies (objects
        exposing `.pos`) built by the caller from the interpolation
        buffer's enemy entries. When given, the ghost's laser target is
        picked from them (the SAME nearest-in-range math the host's
        `_pick_laser_target` uses) so the ghost charges + fires a beam on
        the same tick the host does. The BEAMS `s.update()` returns are
        kept as presentation entries in `self.local_beams` (world-space
        muzzle -> target), so the player sees their own beam immediately
        instead of waiting ~100 ms for the host's next snapshot. Call
        `step_local_beams` after `step` to age + cull the list. When
        `enemies` is None (tests, or before the buffer has a window) the
        laser target stays None — the pre-10.3a behavior.

        The game loop does NOT call this directly — it calls `advance`,
        which decides how many fixed steps a real frame warrants. Tests
        call it directly to step the ghost at exactly the sim's rate.
        """
        s = self._ship
        s.tracked = 0
        s.laser_target = self._pick_laser_target(s, enemies)
        # 10.3b: pick the ghost's missile target from the proxies (the SAME
        # nearest-in-range math the host's _pick_missile_target uses), so
        # the ghost locks + launches on the same tick the host does.
        s.missile_target = self._pick_missile_target(s, enemies)
        s.contacts = []
        shots, beams, missiles = s.update(dt, inp)
        # Keep the ghost's own gun shots (Session 7.3). The host's world
        # cap is MAX_BULLETS total; the ghost's list holds only the local
        # player's shots, so capping it at MAX_BULLETS is a faithful
        # (slightly generous) stand-in — the gun's own cooldown is the
        # real limiter.
        for shot in shots:
            if len(self.local_bullets) < MAX_BULLETS:
                self.local_bullets.append(
                    Bullet(shot.pos, shot.vel, owner=shot.owner))
        # 10.3a: keep the ghost's own laser discharges as presentation
        # entries [start, end, age, ttl] in WORLD space. The host re-anchors
        # its beams via a hull-local start + target_id because the RenderModel
        # stores a hull-local start; the ghost's Beam already carries the
        # world-space muzzle (beam.start) and target pos (beam.end), so the
        # client draws it directly (no re-anchoring). A LIST (not tuple) so
        # step_local_beams can age `b[2]` in place (mirrors the host's
        # `self.beams` list-of-lists).
        for beam in beams:
            self.local_beams.append(
                [beam.start.copy(), beam.end.copy(), 0.0, BEAM_TTL])
        # 10.3b: keep the ghost's own LAUNCHED missiles as presentation
        # GhostMissiles. Each gets the SAME unique id the host assigns at
        # launch — (player_index, per-ship seq) — and the ghost's
        # missile_seq is bumped to match (it is resynced to the host's via
        # the ship snapshot on reconcile, so the ids stay aligned). The
        # target is the buffer proxy the ghost locked (m.target is a
        # _GhostEnemyProxy); homing re-resolves it by id each step. The
        # host's world cap (MAX_MISSILES) is mirrored as a cap on the
        # ghost's own list (a faithful stand-in — the lock-rebuild cooldown
        # is the real limiter).
        for m in missiles:
            if len(self.local_missiles) < MAX_MISSILES:
                seq_before = s.missile_seq
                mid = (self.local_index, seq_before)
                s.missile_seq += 1
                # 10.3b fix: the reconcile REWIND (reconcile_rewind) re-runs
                # the fire tick after apply_snapshot has RESET missile_seq
                # to the host's pre-fire value — so the replay re-fires the
                # SAME missile with the SAME id the prediction already
                # emitted in advance(). Without this guard the ghost would
                # hold two GhostMissiles of one id (the in-flight one + a
                # fresh one at the muzzle): the player sees "two missiles"
                # (one right behind the other) and, because the buffer's
                # copy is suppressed by the id-dedup, BOTH of the visible
                # ones are collision-less ghost missiles that pass through
                # the target and orbit it. Skip the re-fire when the id is
                # already in flight (the prediction's copy is the fresher
                # one — it has already integrated the host's flight).
                guard = any(tuple(g.id) == mid for g in self.local_missiles)
                if guard:
                    # TELEMETRY (10.3b): a re-fire was SUPPRESSED. If the
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
                # TELEMETRY (10.3b): a GhostMissile was CREATED. This is the
                # id the ghost will dedup + handback on. seq_before/after show
                # the counter move; f_local_ids is the full in-flight set.
                _mtel.log("FIRE", f_mid=_mtel.fmt_ids([mid]),
                          f_seq_before=seq_before,
                          f_seq_after=s.missile_seq, f_guard=0,
                          f_local_ids=_mtel.fmt_ids(
                              (tuple(g.id) for g in self.local_missiles)))

    def _pick_laser_target(self, s, enemies):
        """Nearest enemy proxy within the max laser range; None if no laser
        is fitted or `enemies` is empty/None. Mirrors the host's
        `Game._pick_laser_target` (nearest-in-range) so the ghost charges +
        fires on the same tick the host does, given the same target in the
        wedge. Reads only `.pos` off each proxy — the buffer's enemy
        proxies expose that (10.3a shared plumbing)."""
        if not enemies:
            return None
        max_range = max((w.comp.laser_range for w in s.weapons
                         if w.comp.laser_range > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in enemies:
            d = e.pos.distance_to(s.pos)
            if d <= best_d:
                best, best_d = e, d
        return best

    def _pick_missile_target(self, s, enemies):
        """Nearest enemy proxy within the max missile lock range; None if no
        missile is fitted or `enemies` is empty/None. Mirrors the host's
        `Game._pick_missile_target` (nearest-in-range) so the ghost locks +
        launches on the same tick the host does, given the same target in
        the wedge. Reads only `.pos` off each proxy (10.3a shared
        plumbing)."""
        if not enemies:
            return None
        max_range = max((w.comp.missile_lock_range for w in s.weapons
                         if w.comp.missile_speed > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in enemies:
            d = e.pos.distance_to(s.pos)
            if d <= best_d:
                best, best_d = e, d
        return best

    def step_local_beams(self, dt):
        """Age the ghost's local beams by `dt` and cull the expired
        (10.3a). Each entry is [start, end, age, ttl] (a list — age is
        mutated in place). Mirrors the host's `for b in self.beams:
        b[4] += dt` + `self.beams = [b for b in self.beams if b[4] <
        b[5]]`. Called by `advance` after each ghost step; tests may
        call it directly. `dt` is the sim's STEP (passed in, not
        imported)."""
        for b in self.local_beams:
            b[2] += dt
        self.local_beams = [b for b in self.local_beams if b[2] < b[3]]

    def step_local_bullets(self, dt):
        """Advance the ghost's local bullets by `dt` and cull the dead
        (Session 7.3). Mirrors the host's `for b in self.bullets:
        b.update(dt)` + `self.bullets = [b for b in ... if b.life > 0]`.
        Called by `advance` after each ghost step; tests may call it
        directly. `dt` is the sim's STEP (passed in, not imported)."""
        for b in self.local_bullets:
            b.update(dt)
        self.local_bullets = [b for b in self.local_bullets if b.life > 0]

    def step_local_missiles(self, dt, enemies=None):
        """Advance the ghost's local missiles by `dt` and cull the dead
        (10.3b). Mirrors the host's `for m in self.missiles: m.update(dt)`
        + `self.missiles = [m for m in ... if m.life > 0]` — except the
        homing target is the buffer's enemy PROXY for the missile's target
        id (re-resolved each step from `enemies`; None when the target has
        left the buffer, so the missile coasts). Called by `advance` after
        each ghost step; tests may call it directly. `dt` is the sim's STEP
        (passed in, not imported)."""
        # Map enemy id -> proxy for this step (the buffer's current view).
        by_id = {e.id: e for e in (enemies or [])}
        for m in self.local_missiles:
            # Re-resolve the target by the id the missile locked. m.target
            # is the proxy it was launched at; its .id is the key.
            tid = m.target.id if m.target is not None else None
            m.update(dt, by_id.get(tid))
        self.local_missiles = [m for m in self.local_missiles
                               if m.life > 0]

    def handback_missiles(self, host_seq, latest_missiles=None):
        """Hand the ghost's OWN missiles back to the buffer once the
        buffer's copy of each is VISIBLE (10.3b fix).

        A ghost missile is culled only when BOTH hold:
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

        Why the second condition (the "disappears, then reappears" fix):
        the snapshot that arrives right after a launch is the PRE-fire
        one — the host snapshots every SNAPSHOT_INTERVAL ticks, so the
        first snapshot to carry the missile arrives up to one interval
        (~100 ms) later. The seq alone says "cull" the moment ANY
        snapshot reports a higher seq, but the buffer's copy only becomes
        drawable when the CARRYING snapshot enters the buffer — and the
        render point (newest stamp - INTERP_DELAY) reaches it only
        ~INTERP_DELAY after that. Culling on seq alone deleted the ghost
        copy while the buffer's copy was still ~100-200 ms in the
        future: the missile VANISHED, then REAPPEARED at the position
        the carrying snapshot recorded (a "new nearby location"). With
        the id check the ghost copy stays until the buffer's copy is
        actually in the newest snapshot — the render then shows the
        buffer's copy (the one that actually hits: only the host's
        authoritative missile has collision; the ghost's copy is
        presentation-only and would otherwise pass through the target
        and orbit it for the rest of MISSILE_LIFE) from the very frame
        the ghost copy stops drawing, so the missile appears continuous.

        (If the host's copy had already expired by the snapshot, the
        ghost's copy — same age — would be dead too, so culling it is a
        no-op.)"""
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
        kept = []
        culled = []
        # TELEMETRY (10.3b): the handback decision. If a ghost missile was
        # KEPT even though the buffer already carries its id (or vice
        # versa), the dedup/handback pair is out of sync — that is the
        # double-draw. h_culled lists the ids just removed. h_jump is the
        # DISCONTINUITY: for each culled missile, the distance (px) between
        # the ghost's position (last frame drawn) and the buffer's copy
        # position (first frame drawn) — the visible jump the player sees
        # when the handback happens. This is the number the slow-launch
        # (10.3c) change is meant to shrink.
        jump_parts = []
        for m in self.local_missiles:
            cull = (m.id is not None and m.id[0] == self.local_index
                    and m.id[1] < host_seq
                    and (ids is None or tuple(m.id) in ids))
            if cull:
                culled.append(tuple(m.id))
                bp = buf_pos.get(tuple(m.id))
                if bp is not None:
                    jump_parts.append(
                        "%s:%.1f" % (_mtel.fmt_ids([tuple(m.id)]),
                                     math.hypot(m.pos.x - bp[0],
                                                m.pos.y - bp[1])))
            else:
                kept.append(m)
        self.local_missiles = kept
        _mtel.log("HANDBACK", h_host_seq=host_seq,
                  h_buf_ids=_mtel.fmt_ids(sorted(ids)) if ids else "",
                  h_before=_mtel.fmt_ids(before),
                  h_after=_mtel.fmt_ids(tuple(m.id) for m in kept),
                  h_culled=_mtel.fmt_ids(culled),
                  h_jump="|".join(jump_parts))

    def advance(self, dt, inp, enemies=None):
        """Advance the ghost by real time `dt` (Session 7.1).

        Feeds `dt` into the fixed-step accumulator and steps the ghost
        `floor(acc / STEP)` times at exactly `STEP` — the same
        accumulator pattern the authoritative `Game.update` uses. The
        ghost therefore integrates at the SIM's rate no matter the
        display's refresh rate: a 144 Hz monitor yields ~2.4 steps per
        16.7 ms of sim time on average, never 2.4 steps per display
        frame. `dt` is clamped to MAX_FRAME_DT first (imported from
        config, not game.py) so a hiccup can't trigger a catch-up spiral.

        `inp` is the CURRENT local input; every step this call takes uses
        it (the host applies the latest received input each tick — the
        same rule). Returns the number of fixed steps taken (0 when the
        frame is shorter than one STEP of accumulated time).

        10.3a: `enemies` (a list of lightweight enemy proxies from the
        interpolation buffer, see `step`) is fed to every step this call
        takes, so the ghost's laser target tracks the buffer's enemies
        and the ghost fires a beam on the same tick the host does. The
        ghost's local beams are aged + culled each step (mirrors the
        host's beam ttl).

        10.1: a DEAD ghost does not advance. The host freezes a dead
        ship (no movement, no firing — Game._step skips it), so the
        authoritative snapshot of a dead ship is STATIC. Predicting a
        static ship with the local input would race ahead and every
        reconcile yank it back — the reported "freezing and snapping".
        Freezing the ghost (no advance, no local-bullet steps) matches
        authority exactly: the snap size is 0 while dead. The accumulator
        is discarded (not kept) so the first advance after a respawn
        starts clean from the fresh snapshot.
        """
        if self._ship.dead:
            self._acc = 0.0
            # 10.3a: a beam is a 0.15 s flash — age + cull it even while
            # the ghost is dead, mirroring the host (Game._step ages
            # self.beams outside the game_over guard). Without this a beam
            # fired at the moment of death would hang frozen in the air
            # until the respawn.
            self.step_local_beams(min(dt, MAX_FRAME_DT))
            # 10.3b: a missile in flight at the moment of death coasts +
            # ages out too (no enemies while dead — it can no longer home).
            self.step_local_missiles(min(dt, MAX_FRAME_DT))
            return 0
        self._acc += min(dt, MAX_FRAME_DT)
        n = 0
        while self._acc >= TICK:
            self.step(TICK, inp, enemies=enemies)
            self.step_local_bullets(TICK)   # Session 7.3: advance + cull
            self.step_local_beams(TICK)     # 10.3a: age + cull
            self.step_local_missiles(TICK, enemies)   # 10.3b: home + cull
            self._acc -= TICK
            n += 1
        return n

    def pos(self):
        """The ghost's current (x, y, angle) for rendering."""
        s = self._ship
        return (s.pos.x, s.pos.y, s.angle)