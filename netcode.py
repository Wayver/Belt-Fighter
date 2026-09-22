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

Snapshot layout (see Game.snapshot in game.py):
    [0] players_s tuple of one Ship.snapshot() per player (Session 6.1):
                  ship_s = pos.x=[0], pos.y=[1], vel.x=[2], vel.y=[3],
                  angle=[4], ...
    [1] enemies_s tuple of (tag, e_s); e_s = AIEnemy.snapshot() =
                  (ship_s, hp, id, acc_x, acc_y) -> enemy pos = e_s[0][0], e_s[0][1]
    [2] bullets_s, [3] enemy_bullets_s, [4] missiles_s   (not interpolated here)
    [5] asteroids_s  tuple of Asteroid.snapshot() =
                     (id, pos.x, pos.y, vel.x, vel.y, size, angle, spin, verts)
    [6] rng_state, [7] game_over, [8] protect_timer, [9] next_id,
    [10] rock_next_id
"""
import math

from .config import INTERP_DELAY, SNAPSHOT_INTERVAL
from .ship import Ship

__all__ = ["interp_positions", "lerp", "lerp_angle", "SnapshotBuffer",
           "PredictedShip"]


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
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return a0 + da * t


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


def interp_positions(prev_s, curr_s, alpha):
    """Interpolated (x, y) positions for each player ship, each enemy, and
    each asteroid, between two Game snapshots.

    prev_s / curr_s are the 11-tuples from Game.snapshot(); alpha in [0, 1]
    is clamped (never extrapolate into the future). Returns a plain dict —
    no pygame objects, no sim state touched:

        {'ships': [(x, y), ...],
         'enemies': [(x, y), ...],
         'asteroids': [(x, y), ...]}

    Entity matching:
      - player ships by INDEX (Session 6.1): ships don't turn over — a dead
        ship is game over, not a respawn — so slot i of curr_s[0] is the
        same ship as slot i of prev_s[0]. Output order is curr_s[0]'s order.
      - enemies by their ship id (e_s[2]);
      - asteroids by their rock id (a_s[0]) — see _asteroid_key.
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
    # curr position, same rule as new enemies/asteroids.
    prev_ships = {i: (p[0], p[1]) for i, p in enumerate(prev_s[0])}
    ships = []
    for i, c in enumerate(curr_s[0]):
        cp = (c[0], c[1])
        pp = prev_ships.get(i)
        if pp is None:
            ships.append(cp)
        else:
            ships.append((lerp(pp[0], cp[0], a),
                          lerp(pp[1], cp[1], a)))

    prev_enemies = {_enemy_id(p[1]): _enemy_pos(p[1]) for p in prev_s[1]}
    enemies = []
    for _tag, c in curr_s[1]:
        cp = _enemy_pos(c)
        pp = prev_enemies.get(_enemy_id(c))
        if pp is None:
            enemies.append(cp)
        else:
            enemies.append((lerp(pp[0], cp[0], a),
                            lerp(pp[1], cp[1], a)))

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

    return {'ships': ships, 'enemies': enemies, 'asteroids': asteroids}


class SnapshotBuffer:
    """The remote peer's interpolation store (Session 5b.3).

    The authoritative peer stamps each `Game.snapshot()` with the sim time
    it was taken at and sends it on. The remote peer pushes them here, in
    arrival order, and renders `INTERP_DELAY` seconds in the PAST: it asks
    `positions_at(sim_time - INTERP_DELAY)` for the interpolated (x, y) of
    the ship, each enemy, and each asteroid.

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
        ({'ships': [(x, y), ...], 'enemies': [...], 'asteroids': [...]})
        or None
        when there is not yet a window to interpolate between (fewer than
        two snapshots, or render_t before the first snapshot).

        The window is the two snapshots that BRACKET render_t: the newest
        snapshot at or before render_t is `prev`, the next one is `curr`,
        and alpha = (render_t - t_prev) / (t_curr - t_prev) in [0, 1].
        render_t is clamped to the window — never extrapolate.
        """
        n = len(self._snaps)
        if n < 2:
            return None
        t0, s0 = self._snaps[0]
        if render_t <= t0:
            # Before the first snapshot: sit exactly on it (alpha 0).
            return interp_positions(s0, s0, 0.0)
        tN, sN = self._snaps[-1]
        if render_t >= tN:
            # At/after the newest: sit exactly on it (alpha 1). The remote
            # render lags the sim by INTERP_DELAY, so this is the steady
            # state between snapshots, not a look-ahead.
            return interp_positions(sN, sN, 1.0)
        # Find the bracketing pair: the newest snapshot at or before
        # render_t is prev; the one after it is curr.
        for i in range(n - 1):
            ti, si = self._snaps[i]
            tj, sj = self._snaps[i + 1]
            if ti <= render_t <= tj:
                span = tj - ti
                alpha = 0.0 if span <= 0.0 else (render_t - ti) / span
                return interp_positions(si, sj, alpha)
        # Unreachable: render_t is strictly inside (t0, tN).
        return None


class PredictedShip:
    """Client-side prediction ghost for the local player's ship
    (Session 5b.4a).

    The remote peer renders its OWN ship from the interpolation buffer at
    `sim_time - INTERP_DELAY`, which feels ~100 ms laggy. The classic fix is
    to predict the local ship: step a private Ship with the player's OWN
    input every frame, and snap it back to the authoritative ship snapshot
    when one lands. This class is that ghost — a full `Ship` (not a minimal
    kinematic stand-in) so reconciliation reuses the existing
    `apply_snapshot` round-trip and the physics stay faithful.

    The ghost is a PRESENTATION object: it is stepped with the local input
    and reconciled to authority, but it never feeds the sim. The sim stays
    authoritative on one peer; this only feeds the render.

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
    """

    def __init__(self, hull=None, loadout=None):
        # A self-contained Ship: __init__ builds components/thrusters/
        # weapons/shield/sensors/collision from the hull+loadout and holds
        # no reference to Game, so it can be stepped in isolation.
        self._ship = Ship(hull=hull, loadout=loadout)
        self._seeded = False

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
        the first snapshot."""
        self._ship.apply_snapshot(ship_s)
        self._seeded = True

    def step(self, dt, inp):
        """Advance the ghost one fixed step with the LOCAL input.

        `dt` is passed by the caller (the sim's STEP) because netcode.py
        must not import it from game.py (game.py imports netcode.py). The
        Game-fed per-tick fields are reset to idle before the step — the
        ghost has no enemy list to target (see class docstring). The
        returned (shots, beams, missiles) are discarded: the ghost's
        weapons are presentation and its projectiles are not rendered.
        """
        s = self._ship
        s.tracked = 0
        s.laser_target = None
        s.missile_target = None
        s.contacts = []
        s.update(dt, inp)

    def pos(self):
        """The ghost's current (x, y, angle) for rendering."""
        s = self._ship
        return (s.pos.x, s.pos.y, s.angle)