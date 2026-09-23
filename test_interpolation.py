"""Remote-interpolation + snapshot-cadence check (Session 5a).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_interpolation

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens. The SDL dummy-driver boilerplate, Keys, and script_input are copied
from test_snapshot.py so the input pattern is identical.

Five proofs:

  a. INTERPOLATION — interp_positions() (netcode.py) is a pure function of
     two Game snapshots, with ID-BASED entity matching (Session 5b-1):
       - alpha=0  -> exactly the prev snapshot's positions
       - alpha=1  -> exactly the curr snapshot's positions
       - alpha=0.5 -> the exact midpoint (linear)
       - alpha out of [0,1] is CLAMPED (never extrapolate into the future)
     The oracles are built INDEPENDENTLY of netcode's matching: enemies by
     ship id (e_s[2]), asteroids by rock id (a_s[0]) — so a bug in the
     matching shows up as a mismatch, not a tautology.

  b. CADENCE SANITY — the synced/presentation split holds under load:
     Game A runs at 60 Hz; every SNAPSHOT_INTERVAL ticks its snapshot is
     applied to a FRESH Game B. At each boundary B's SYNCED state (ship
     pos/vel/angle, each enemy pos, each asteroid pos, rng.getstate())
     matches A exactly. Particles/beams are deliberately NOT compared —
     they are presentation-only, not in the snapshot, and a fresh B
     correctly does not have A's particles. This proves the synced state
     is sufficient to reconstruct the sim at the coarser cadence.

  c. TURNOVER STRESS — id-based matching must survive entity churn:
     snapshots taken W=60 ticks apart (10x the snapshot interval) span
     splits, spawns, culls, and enemy respawns. The test asserts the
     window ACTUALLY contains turnover (rock ids or enemy ids differ
     between the two snapshots) — otherwise the check is vacuous — and
     then re-runs the alpha=0/1/0.5/clamp battery against the
     independent id-based oracles.

  d. NO-TELEPORT — the remote buffer (netcode.SnapshotBuffer) never makes
     the ship jump more than the sim's OWN peak per-frame displacement:
     the authoritative Game runs RUN_TICKS ticks, pushing a snapshot into
     a fresh buffer every SNAPSHOT_INTERVAL ticks; the sweep then reads
     positions_at(INTERP_DELAY + i*STEP) and compares consecutive-frame
     ship displacement against the sim's measured peak. Within a window
     the buffer's per-frame disp = dist(endpoints)/SNAPSHOT_INTERVAL <=
     the sim's peak (triangle inequality), so a matching/clamping/
     extrapolation bug shows up as a jump far beyond the sim's real
     motion. The ship is the probe: always present, no id matching
     needed, a clean signal while rocks/enemies churn.

  e. PREDICTION — the local-ship ghost (netcode.PredictedShip, wired into
     Game in Session 5b.4b) tracks the authoritative ship with bounded
     drift. The authoritative Game runs RUN_TICKS ticks with
     script_input(t); every SNAPSHOT_INTERVAL ticks the ghost is reconciled
     to the sim's snapshot (via the 5b.4b push_snapshot seam), and between
     reconciles the ghost is stepped with the same input as the sim. Two
     bounds:
       1. NO-TELEPORT — the ghost's per-frame ship displacement never
          exceeds the sim's OWN peak (+1e-3). (Flagged risk: the ghost is
          never browned out by weapon power draw, so it can move slightly
          faster than the sim; the bound is against the sim's PEAK, not the
          per-tick value, so a small excess is expected. Fallback if too
          tight: loosen to sim_peak * (1 + small) + eps.)
       2. DRIFT — after each reconcile, the ghost is within
          sim_peak * SNAPSHOT_INTERVAL + eps of the authoritative ship.
     The "ship actually moved" guard (max_auth above a floor) prevents the
     vacuous never-moved case. The max drift is REPORTED (not required to
     be non-zero — a zero-drift result is a legitimate pass, per the pinned
     decision).

  f. SHIP ANGLE (Session 6.8) — the buffer now carries the ship's ANGLE as
     well as its position, so the remote peer can draw the remote hull at
     its interpolated orientation. The alpha battery (a) and the turnover
     battery (c) already check the angle against the independent oracle
     (wrapped-delta lerp, endpoint exactness). This part adds the
     no-teleport bound for the ANGLE itself: a full 360-degree spin in one
     render frame would read as a "teleport" of the hull's orientation, so
     the buffer's per-frame angle change is bounded by the sim's OWN peak
     per-frame angle change (+1e-3 rad). The scripted input turns in 0.5 s
     bursts (Q/E cycles), so the ship does spin — the check is not vacuous.

  g. GHOST CLOCK (Session 7.1) — the prediction ghost must step at the
     SIM's fixed rate, not the display's. The pre-7.1 client stepped the
     ghost once per DISPLAY frame, so on a 144 Hz monitor it integrated
     2.4x the sim's motion and every reconcile yanked it back (the
     reported "spiking and snapbacking"). `PredictedShip.advance(dt, inp)`
     converts real frame times into whole STEP steps via an accumulator.
     This part drives advance() with a MIXED frame-dt sequence (8/16/33 ms
     — a 120 Hz, a 60 Hz, and a 30 Hz monitor interleaved) and asserts:
       1. the ghost stepped exactly floor(total/STEP) times — the count is
          what a sim-rate ghost takes over the same real time, independent
          of how the time was chunked into frames;
       2. the ghost's trajectory matches a reference ghost stepped at
          exactly the sim's rate over the same total time (max distance
          ~1e-9 — same steps, same input, same physics);
       3. a 0.5 s hiccup frame (above MAX_FRAME_DT) is clamped — the
          ghost steps at most ~MAX_FRAME_DT/STEP times for it, no
          catch-up spiral.
     The HostTimeEstimator (the client's estimate of the host's sim clock)
     is checked in the same part: with a steady 10 Hz sample stream it
     converges to the true offset (within 5 ms), tracks a host that
     simulates at 0.8x real time (within 20 ms after 2 s), and stays
     anchored on the data for a STARVED host (0.1x real time, samples a
     full second apart) — never more than one snapshot interval ahead of
     the newest stamp.
"""
import math
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT, SNAPSHOT_INTERVAL, INTERP_DELAY, \
    MAX_FRAME_DT
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .netcode import (interp_positions, SnapshotBuffer, PredictedShip,
                      HostTimeEstimator)
from .intent import ShipInput

WARMUP = 300   # ticks before the first pair of snapshots (sim is hot)
K = 6          # ticks between snap_prev and snap_curr = one snapshot
               # interval (SNAPSHOT_INTERVAL), the realistic netcode window
W = 60         # turnover-stress window: 10x the snapshot interval — long
               # enough that splits/spawns/culls/respawns are likely
RUN_TICKS = 600  # cadence test length (10 simulated seconds at 60 Hz)
SEED = 1234


class Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """A canned input pattern that exercises every fire path:
    turn cycles, thrust bursts, bullets, lasers, missiles, stop.
    Copied verbatim from test_snapshot.py so the input pattern is
    identical."""
    return Keys(
        {pygame.K_q: 1 if (t // 30) % 3 == 0 else 0,
           pygame.K_e: 1 if (t // 30) % 3 == 2 else 0,
           pygame.K_w: 1 if (t // 60) % 2 == 0 else 0,
           pygame.K_a: 1 if (t // 15) % 2 == 0 else 0,
           pygame.K_d: 1 if (t // 15) % 2 == 1 else 0,
           pygame.K_SPACE: 1 if (t // 30) % 2 == 0 else 0,
           pygame.K_r: 1 if (t % 45) < 5 else 0,
           pygame.K_2: 1 if (t % 60) < 3 else 0,
           pygame.K_b: 1 if (t % 90) < 5 else 0})


def snapshot(g):
    """Copied verbatim from test_snapshot.py — the "are two Games
    identical" oracle."""
    s = g.ship
    return (
        # player ship
        (round(s.pos.x, 6), round(s.pos.y, 6), round(s.angle, 6),
         round(s.vel.x, 6), round(s.vel.y, 6)),
        # asteroids: full kinematic + shape state
        tuple(sorted((round(a.pos.x, 6), round(a.pos.y, 6), a.size,
                      round(a.vel.x, 6), round(a.vel.y, 6),
                      round(a.angle, 6),
                      tuple(round(c, 6) for v in a.verts
                            for c in (v.x, v.y)))
              for a in g.asteroids)),
        # enemies
        tuple(sorted((round(e.pos.x, 6), round(e.pos.y, 6),
                      round(e.ship.angle, 6),
                      round(e.ship.vel.x, 6), round(e.ship.vel.y, 6),
                      e.hp)
              for e in g.enemies)),
        # projectiles
        tuple(sorted((round(b.pos.x, 6), round(b.pos.y, 6), b.owner)
                    for b in g.bullets)),
        tuple(sorted((round(b.pos.x, 6), round(b.pos.y, 6), b.owner)
                    for b in g.enemy_bullets)),
        tuple(sorted((round(m.pos.x, 6), round(m.pos.y, 6), m.owner)
                    for m in g.missiles)),
        # particles (cosmetic, but must be deterministic too)
        tuple(sorted((round(p.pos.x, 6), round(p.pos.y, 6),
                      round(p.life, 6))
                    for p in g.particles)),
        # the strongest check: the rng itself must be at the same point
        g.rng.getstate(),
    )


def synced_state(g):
    """SYNCED-ONLY oracle for the cadence check: ship pos/vel/angle, each
    enemy pos, each asteroid pos, and the rng state. Deliberately EXCLUDES
    particles/beams — they are presentation-only, not in the snapshot, and
    a fresh Game B correctly does not have A's particles."""
    s = g.ship
    return (
        (round(s.pos.x, 6), round(s.pos.y, 6), round(s.angle, 6),
         round(s.vel.x, 6), round(s.vel.y, 6)),
        tuple((round(e.pos.x, 6), round(e.pos.y, 6)) for e in g.enemies),
        tuple((round(a.pos.x, 6), round(a.pos.y, 6)) for a in g.asteroids),
        g.rng.getstate(),
    )


SYNCED_NAMES = ["ship pos/vel/angle", "enemy pos", "asteroid pos", "rng_state"]


def snap_positions(s):
    """The (x, y, angle) pose a Game snapshot implies for each player ship,
    plus the (x, y) of each enemy and each asteroid — the oracle
    interp_positions is checked against. Mirrors the layout documented in
    netcode.py (Session 6.1: index 0 is a tuple of per-player ship
    snapshots; Session 6.8: ships carry their raw angle, ship_s[4])."""
    return {
        'ships': [(p[0], p[1], p[4]) for p in s[0]],
        'enemies': [(e_s[0][0], e_s[0][1]) for _tag, e_s in s[1]],
        'asteroids': [(a_s[1], a_s[2]) for a_s in s[5]],
    }


# --- independent id-based oracles (Session 5b-1) ---
#
# These mirror the identity scheme netcode.py uses (enemies by ship id,
# asteroids by rock id) but are written INDEPENDENTLY of netcode —
# different helper names, built directly from the snapshot layout — so a
# bug in netcode's matching shows up as a mismatch instead of a tautology.

def _eid(e_s):
    return e_s[2]                     # ship id


def _akey(a_s):
    return a_s[0]                     # rock id (Session 5b.1)


def id_oracle(prev_s, curr_s, alpha):
    """What interp_positions SHOULD return: membership from curr_s, each
    entity lerped from its prev_s twin when one exists, else its curr
    position (a new entity pops in). Written independently of netcode.

    Ships carry their pose (x, y, angle) (Session 6.8): the position is
    lerped linearly and the angle is lerped by its WRAPPED delta — the
    same rule Ship.sync_render and netcode.lerp_angle use, written
    independently here (the sim's raw angle is unbounded, so a plain lerp
    of the raw values would swing the wrong way around the circle).

    Endpoint EXACTNESS is part of the spec (see netcode.lerp's docstring
    and the 5a run-2 fix): at alpha=0/1 the render must sit exactly on the
    snapshot, not 1 ulp off. So the oracle returns the raw snapshot value
    by identity at the endpoints and only uses the lerp formula in
    between — computing pp + (cp - pp) * 1.0 here would reintroduce the
    double-rounding artifact the implementation is required to avoid."""
    a = min(1.0, max(0.0, alpha))

    def mix(pp, cp):
        if a == 0.0:
            return cp if pp is None else pp
        if a == 1.0:
            return cp
        if pp is None:
            return cp
        return (pp[0] + (cp[0] - pp[0]) * a, pp[1] + (cp[1] - pp[1]) * a)

    def mix_pose(pp, cp):
        # (x, y, angle): same endpoint rules as mix; the angle takes the
        # wrapped delta da in [-pi, pi] and lerps pp[2] -> pp[2] + da*a.
        if a == 0.0:
            return cp if pp is None else pp
        if a == 1.0:
            return cp
        if pp is None:
            return cp
        da = (cp[2] - pp[2] + math.pi) % (2 * math.pi) - math.pi
        return (pp[0] + (cp[0] - pp[0]) * a,
                pp[1] + (cp[1] - pp[1]) * a,
                pp[2] + da * a)

    prev_e = {_eid(t[1]): (t[1][0][0], t[1][0][1]) for t in prev_s[1]}
    prev_r = {_akey(t): (t[1], t[2]) for t in prev_s[5]}
    # Player ships are matched by INDEX (Session 6.1) — ships don't turn
    # over, so slot i of curr_s[0] is the same ship as slot i of prev_s[0].
    # Pose = (x, y, raw angle) (Session 6.8).
    prev_ships = {i: (p[0], p[1], p[4]) for i, p in enumerate(prev_s[0])}

    return {
        'ships': [mix_pose(prev_ships.get(i), (c[0], c[1], c[4]))
                  for i, c in enumerate(curr_s[0])],
        'enemies': [mix(prev_e.get(_eid(t[1])), (t[1][0][0], t[1][0][1]))
                    for t in curr_s[1]],
        'asteroids': [mix(prev_r.get(_akey(t)), (t[1], t[2]))
                      for t in curr_s[5]],
    }


def id_sets(s):
    """The identity sets a snapshot carries: enemy ship ids, rock ids."""
    return (frozenset(_eid(t[1]) for t in s[1]),
            frozenset(_akey(t) for t in s[5]))


def check_positions(label, got, want):
    """Compare interp_positions output against the oracle. Returns True if
    identical; on mismatch prints which entity/field differs."""
    ok = True
    for key in ('ships', 'enemies', 'asteroids'):
        if len(got[key]) != len(want[key]):
            print(f"FAIL: {label} — {key}: {len(got[key])} entries, "
                  f"want {len(want[key])}")
            ok = False
            continue
        for i, (gp, wp) in enumerate(zip(got[key], want[key])):
            if gp != wp:
                print(f"FAIL: {label} — {key}[{i}] differs: "
                      f"got {gp}, want {wp}")
                ok = False
    return ok


def check_synced(label, sa, sb):
    """Compare two synced_state() oracles; name the differing field."""
    ok = True
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            name = SYNCED_NAMES[i] if i < len(SYNCED_NAMES) else f"field{i}"
            print(f"FAIL: {label} — {name} differs")
            if i == 0:
                print(f"  A: {x}\n  B: {y}")
            else:
                print(f"  A: {x[:3]}...\n  B: {y[:3]}...")
            ok = False
    return ok


def main():
    pygame.init()
    global screen, font, big_font, light_tex, fog_surf, light_surf
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ok = True

    # --- (a) interpolation: snap_prev at tick WARMUP, snap_curr at WARMUP+K
    AIEnemy._next_id = 1   # deterministic construction ids
    Asteroid._next_id = 1
    a = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    for t in range(WARMUP):
        a.update(STEP, script_input(t))
    snap_prev = a.snapshot()
    for t in range(WARMUP, WARMUP + K):
        a.update(STEP, script_input(t))
    snap_curr = a.snapshot()

    # Capture the FULL "before" state first — the purity check at the end
    # compares against this, so it must not be truncated.
    prev_full = snap_positions(snap_prev)
    curr_full = snap_positions(snap_curr)

    # Session 5b-1: the oracles are ID-BASED and independent of netcode's
    # matching (see id_oracle above). Membership follows the CURRENT
    # snapshot, so no index truncation is needed — a count change inside
    # the window is exactly what the matching is supposed to survive.
    interp_cases = [
        (0.0, id_oracle(snap_prev, snap_curr, 0.0),
         "alpha=0 -> prev positions (survivors)"),
        (1.0, id_oracle(snap_prev, snap_curr, 1.0),
         "alpha=1 -> curr positions"),
        (0.5, id_oracle(snap_prev, snap_curr, 0.5),
         "alpha=0.5 -> exact midpoint (linear)"),
        (-1.0, id_oracle(snap_prev, snap_curr, -1.0),
         "alpha=-1 clamps to prev (no extrapolation)"),
        (2.0, id_oracle(snap_prev, snap_curr, 2.0),
         "alpha=2 clamps to curr (no extrapolation)"),
    ]
    for alpha, want, desc in interp_cases:
        got = interp_positions(snap_prev, snap_curr, alpha)
        if check_positions(f"ASSERT interp ({desc})", got, want):
            print(f"PASS: interp {desc}")
        else:
            ok = False

    # interp_positions must be PURE: a second call must not have mutated
    # either snapshot. Compare against the FULL "before" state captured
    # above (prev_full / curr_full), not the truncated oracles.
    if (snap_positions(snap_prev) == prev_full
            and snap_positions(snap_curr) == curr_full):
        print("PASS: interp_positions is pure (snapshots unmutated)")
    else:
        ok = False
        print("FAIL: interp_positions mutated a snapshot")

    # --- (b) cadence sanity: synced state is sufficient at the coarser
    # cadence. Every SNAPSHOT_INTERVAL ticks, apply A's snapshot to a
    # FRESH Game B and compare the SYNCED-ONLY state.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    a2 = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed=SEED)
    b = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    boundaries = 0
    cadence_ok = True
    for t in range(RUN_TICKS):
        a2.update(STEP, script_input(t))
        if (t + 1) % SNAPSHOT_INTERVAL == 0:
            b.apply_snapshot(a2.snapshot())
            boundaries += 1
            sa = synced_state(a2)
            sb = synced_state(b)
            if not check_synced(f"ASSERT cadence (tick {t + 1})", sa, sb):
                cadence_ok = False
                ok = False
    if cadence_ok:
        print(f"PASS: cadence — synced state matches at all {boundaries} "
              f"snapshot boundaries (every {SNAPSHOT_INTERVAL} ticks, "
              f"{RUN_TICKS} ticks total)")

    # --- (c) turnover stress: id-based matching survives entity churn.
    # Snapshots W ticks apart (10x the snapshot interval) span rock
    # splits, sector top-ups/culls, and enemy respawns. First assert the
    # window ACTUALLY contains turnover — otherwise the battery below is
    # vacuous (the ids never diverged, so index matching would have
    # passed too). Then re-run the alpha battery against the independent
    # id-based oracles.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    c = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    for t in range(WARMUP):
        c.update(STEP, script_input(t))
    t_prev = c.snapshot()
    for t in range(WARMUP, WARMUP + W):
        c.update(STEP, script_input(t))
    t_curr = c.snapshot()

    (prev_eids, prev_rkeys), (curr_eids, curr_rkeys) = (
        id_sets(t_prev), id_sets(t_curr))
    rock_turnover = len(prev_rkeys ^ curr_rkeys)
    enemy_turnover = len(prev_eids ^ curr_eids)
    if rock_turnover == 0 and enemy_turnover == 0:
        ok = False
        print(f"FAIL: turnover stress — NO turnover in the {W}-tick window "
              f"(seed {SEED}); the id-matching check would be vacuous. "
              f"Pick a seed/window that churns.")
    else:
        print(f"PASS: turnover stress — {W}-tick window churned "
              f"{rock_turnover} rock id(s), {enemy_turnover} enemy id(s)")
        for alpha, want, desc in [
            (0.0, id_oracle(t_prev, t_curr, 0.0),
             "alpha=0 -> prev positions (survivors)"),
            (1.0, id_oracle(t_prev, t_curr, 1.0),
             "alpha=1 -> curr positions"),
            (0.5, id_oracle(t_prev, t_curr, 0.5),
             "alpha=0.5 -> exact midpoint (linear)"),
            (-1.0, id_oracle(t_prev, t_curr, -1.0),
             "alpha=-1 clamps to prev (no extrapolation)"),
            (2.0, id_oracle(t_prev, t_curr, 2.0),
             "alpha=2 clamps to curr (no extrapolation)"),
        ]:
            got = interp_positions(t_prev, t_curr, alpha)
            if check_positions(f"ASSERT turnover ({desc})", got, want):
                print(f"PASS: turnover {desc}")
            else:
                ok = False

    # --- (d) no-teleport: the remote buffer never moves the ship more
    # than the sim's OWN peak per-frame displacement. The authoritative
    # Game runs RUN_TICKS ticks, pushing a snapshot into a fresh
    # SnapshotBuffer every SNAPSHOT_INTERVAL ticks; the sweep then reads
    # positions_at(INTERP_DELAY + i*STEP) and compares consecutive-frame
    # ship displacement against the sim's measured peak. Within a window
    # the buffer's per-frame disp = dist(endpoints)/SNAPSHOT_INTERVAL <=
    # the sim's peak (triangle inequality), so a matching/clamping/
    # extrapolation bug shows up as a jump far beyond the sim's real
    # motion. The ship is the probe: always present, no id matching
    # needed, a clean signal while rocks/enemies churn.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    d = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    # max_snapshots=200: the default cap of 8 would truncate the buffer to
    # the last 8 snaps (sim time 9.2-9.9) by the time the sweep runs, and
    # the sweep would then sit on the "before the first snapshot" clamp
    # for most of its frames — a 0.0px constant, not a real interpolation
    # check. 200 holds all 100 snaps so the sweep spans the full 10 s.
    buf = SnapshotBuffer(max_snapshots=200)
    ship_track = []
    snap_times = []
    for t in range(RUN_TICKS):
        if t % SNAPSHOT_INTERVAL == 0:
            buf.push(t * STEP, d.snapshot())
            snap_times.append(t * STEP)
        ship_track.append((d.ship.pos.x, d.ship.pos.y))
        if t < RUN_TICKS - 1:
            d.update(STEP, script_input(t))

    def _disp_track(track):
        """Max per-frame displacement along a position track."""
        return max(math.hypot(track[i + 1][0] - track[i][0],
                              track[i + 1][1] - track[i][1])
                   for i in range(len(track) - 1))

    max_auth = _disp_track(ship_track)
    n = round((snap_times[-1] - INTERP_DELAY) / STEP)
    render_track = []
    for i in range(n + 1):
        P = buf.positions_at(INTERP_DELAY + i * STEP)
        if P is None:
            ok = False
            print(f"FAIL: no-teleport — positions_at returned None at "
                  f"render_t={INTERP_DELAY + i * STEP}")
            break
        # Position only (Session 6.8: the ships entry is (x, y, angle)).
        render_track.append(P['ships'][0][:2])
    else:
        max_buf = _disp_track(render_track)
        if len(render_track) < 30:
            ok = False
            print(f"FAIL: no-teleport — only {len(render_track)} render "
                  f"frames in the sweep; the check would be vacuous")
        elif max_buf <= max_auth + 1e-3:
            print(f"PASS: no-teleport — {len(render_track)} render frames, "
                  f"max ship disp {max_buf:.3f}px <= sim peak "
                  f"{max_auth:.3f}px (+1e-3)")
        else:
            ok = False
            print(f"FAIL: no-teleport — buffer ship disp {max_buf:.3f}px "
                  f"exceeds the sim's own peak {max_auth:.3f}px (+1e-3): "
                  f"the remote render teleported")

    # --- (e) prediction: the local-ship ghost (netcode.PredictedShip,
    # wired into Game in Session 5b.4b) tracks the authoritative ship with
    # bounded drift. The authoritative Game runs RUN_TICKS ticks with
    # script_input(t); every SNAPSHOT_INTERVAL ticks the ghost is
    # reconciled to the sim's snapshot (via the 5b.4b push_snapshot seam,
    # which also feeds the interpolation buffer), and between reconciles
    # the ghost is stepped with the SAME input as the sim. Two bounds:
    #   1. NO-TELEPORT — the ghost's per-frame ship displacement never
    #      exceeds the sim's OWN peak (+1e-3). Flagged risk: the ghost is
    #      never browned out by weapon power draw, so it can move slightly
    #      faster than the sim; the bound is against the sim's PEAK, not
    #      the per-tick value, so a small excess is expected.
    #   2. DRIFT — after each reconcile, the ghost is within
    #      sim_peak * SNAPSHOT_INTERVAL + eps of the authoritative ship.
    # The "ship actually moved" guard (max_auth above a floor) prevents
    # the vacuous never-moved case. The max drift is REPORTED, not
    # required to be non-zero (a zero-drift result is a legitimate pass).
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    e = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    ship_track = []
    ghost_step_disp = []
    max_drift = 0.0
    for t in range(RUN_TICKS):
        if t % SNAPSHOT_INTERVAL == 0:
            # Drift at this reconcile: how far the ghost is from the
            # authoritative ship BEFORE the reconcile snaps it back.
            # (At t=0 the ghost is not seeded yet — nothing to measure.)
            if e.ghost.seeded:
                max_drift = max(max_drift, math.hypot(
                    e.ghost.ship.pos.x - e.ship.pos.x,
                    e.ghost.ship.pos.y - e.ship.pos.y))
            e.push_snapshot(t * STEP, e.snapshot())
        ship_track.append((e.ship.pos.x, e.ship.pos.y))
        if t < RUN_TICKS - 1:
            keys = script_input(t)
            e.update(STEP, keys)
            # The ghost takes a ShipInput (not raw keys) — Game.update and
            # predicted_view both derive one via ShipInput.from_keys, so do
            # the same here to give the ghost the SAME input as the sim.
            # Measure the ghost's INTEGRATION displacement (the step), not
            # the reconcile snap: the snap is a correction (bounded by the
            # drift check below), not the ghost's motion. Measuring the
            # recorded-position difference would count the snap as a
            # "per-frame displacement" and false-positive the no-teleport
            # bound.
            before = (e.ghost.ship.pos.x, e.ghost.ship.pos.y)
            e.ghost.step(STEP, ShipInput.from_keys(keys))
            after = (e.ghost.ship.pos.x, e.ghost.ship.pos.y)
            ghost_step_disp.append(math.hypot(after[0] - before[0],
                                              after[1] - before[1]))

    max_auth = _disp_track(ship_track)
    if max_auth < 1.0:
        ok = False
        print(f"FAIL: prediction — the ship barely moved (max per-frame "
              f"disp {max_auth:.3f}px < 1.0px); the drift/no-teleport "
              f"checks would be vacuous. Pick a seed/input that moves.")
    else:
        max_ghost = max(ghost_step_disp)
        drift_bound = max_auth * SNAPSHOT_INTERVAL + 1e-3
        if max_ghost <= max_auth + 1e-3:
            print(f"PASS: prediction no-teleport — {len(ghost_step_disp)} "
                  f"steps, max ghost disp {max_ghost:.3f}px <= sim peak "
                  f"{max_auth:.3f}px (+1e-3)")
        else:
            ok = False
            print(f"FAIL: prediction no-teleport — ghost ship disp "
                  f"{max_ghost:.3f}px exceeds the sim's own peak "
                  f"{max_auth:.3f}px (+1e-3): the predicted ship "
                  f"teleported (flagged risk — loosen the bound if the "
                  f"ghost's full-thrust allocation is the cause)")
        if max_drift <= drift_bound:
            print(f"PASS: prediction drift — max ghost drift "
                  f"{max_drift:.3f}px <= sim peak x interval "
                  f"{drift_bound:.3f}px (reported, not required non-zero)")
        else:
            ok = False
            print(f"FAIL: prediction drift — max ghost drift "
                  f"{max_drift:.3f}px exceeds sim peak x interval "
                  f"{drift_bound:.3f}px: the reconcile is not bounding "
                  f"the prediction")

    # --- (f) ship angle (Session 6.8): the buffer carries the ship's
    # angle as well as its position. The alpha/turnover batteries above
    # already check the ANGLE against the independent oracle (wrapped-
    # delta lerp, endpoint exactness); this part adds the no-teleport
    # bound for the angle itself: the buffer's per-frame angle change
    # must never exceed the sim's OWN peak per-frame angle change
    # (+1e-3 rad). A bug that lerps the raw angles without wrapping (or
    # drops the angle and falls back to a constant) would spin the
    # rendered hull wildly and fail here. The scripted input turns in
    # 0.5 s bursts (Q/E cycles), so the ship does spin — a max sim turn
    # of 0 would make the check vacuous, so assert a floor.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    f = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    buf = SnapshotBuffer(max_snapshots=200)
    ship_track = []
    angle_track = []
    snap_times = []
    for t in range(RUN_TICKS):
        if t % SNAPSHOT_INTERVAL == 0:
            buf.push(t * STEP, f.snapshot())
            snap_times.append(t * STEP)
        ship_track.append((f.ship.pos.x, f.ship.pos.y))
        angle_track.append(f.ship.angle)
        if t < RUN_TICKS - 1:
            f.update(STEP, script_input(t))

    def _turn_track(track):
        """Max per-frame (wrapped) angle change along an angle track."""
        return max(abs((track[i + 1] - track[i] + math.pi)
                       % (2 * math.pi) - math.pi)
                   for i in range(len(track) - 1))

    max_auth_turn = _turn_track(angle_track)
    if max_auth_turn < 0.01:
        ok = False
        print(f"FAIL: ship angle — the ship barely turned (max per-frame "
              f"turn {max_auth_turn:.4f} rad < 0.01); the angle "
              f"no-teleport check would be vacuous. Pick a seed/input "
              f"that turns.")
    else:
        n = round((snap_times[-1] - INTERP_DELAY) / STEP)
        render_angles = []
        for i in range(n + 1):
            P = buf.positions_at(INTERP_DELAY + i * STEP)
            if P is None:
                ok = False
                print(f"FAIL: ship angle — positions_at returned None at "
                      f"render_t={INTERP_DELAY + i * STEP}")
                break
            render_angles.append(P['ships'][0][2])
        else:
            max_buf_turn = _turn_track(render_angles)
            if len(render_angles) < 30:
                ok = False
                print(f"FAIL: ship angle — only {len(render_angles)} "
                      f"render frames in the sweep; the check would be "
                      f"vacuous")
            elif max_buf_turn <= max_auth_turn + 1e-3:
                print(f"PASS: ship angle no-teleport — "
                      f"{len(render_angles)} render frames, max buffer "
                      f"turn {max_buf_turn:.4f} rad <= sim peak "
                      f"{max_auth_turn:.4f} rad (+1e-3)")
            else:
                ok = False
                print(f"FAIL: ship angle — buffer turn "
                      f"{max_buf_turn:.4f} rad exceeds the sim's own peak "
                      f"{max_auth_turn:.4f} rad (+1e-3): the remote hull "
                      f"orientation teleported")

    # --- (g) ghost clock (Session 7.1): the prediction ghost must step at
    # the SIM's fixed rate, not the display's. Drive advance() with a mixed
    # frame-dt sequence (120/60/30 Hz monitors interleaved) and compare
    # against a reference ghost stepped at exactly the sim's rate over the
    # same total time. The pre-7.1 behavior (one step per display frame)
    # would take 375 steps here instead of 222 and diverge from the
    # reference — this is the D1 proof.
    frame_dts = [0.008, 0.016, 0.033] * 125      # 375 frames, 3.7 s total
    total = sum(frame_dts)
    g1 = PredictedShip()
    g2 = PredictedShip()
    g1.seed(a.snapshot()[0][0])
    g2.seed(a.snapshot()[0][0])
    inp = ShipInput.from_keys(script_input(0))
    n_steps = 0
    for dt in frame_dts:
        n_steps += g1.advance(dt, inp)
    want_steps = int(total / STEP + 1e-9)        # floor: whole STEP steps
    for _ in range(want_steps):
        g2.step(STEP, inp)
    ghost_dist = math.hypot(g1.ship.pos.x - g2.ship.pos.x,
                            g1.ship.pos.y - g2.ship.pos.y)
    if n_steps == want_steps and ghost_dist < 1e-6:
        print(f"PASS: ghost clock — {len(frame_dts)} mixed frames "
              f"({total:.2f}s) -> {n_steps} fixed steps "
              f"(= floor(total/STEP)), ghost matches the sim-rate "
              f"reference to {ghost_dist:.2e}px")
    else:
        ok = False
        print(f"FAIL: ghost clock — {n_steps} steps, want {want_steps}; "
              f"ghost-vs-reference distance {ghost_dist:.3e}px "
              f"(want < 1e-6): the ghost is not stepping at the sim's "
              f"fixed rate")

    # Hiccup clamp: a 0.5 s frame (a full freeze) must not trigger a
    # catch-up spiral — advance() clamps dt to MAX_FRAME_DT, so the ghost
    # steps at most ~MAX_FRAME_DT/STEP times for it.
    g3 = PredictedShip()
    g3.seed(a.snapshot()[0][0])
    before = (g3.ship.pos.x, g3.ship.pos.y)
    n_hiccup = g3.advance(0.5, inp)
    after = (g3.ship.pos.x, g3.ship.pos.y)
    hiccup_disp = math.hypot(after[0] - before[0], after[1] - before[1])
    max_hiccup_steps = round(MAX_FRAME_DT / STEP) + 1
    if n_hiccup <= max_hiccup_steps and hiccup_disp <= max_auth * (
            max_hiccup_steps + 1):
        print(f"PASS: ghost clock hiccup — a 0.5 s frame clamps to "
              f"{n_hiccup} steps (<= {max_hiccup_steps}), disp "
              f"{hiccup_disp:.1f}px (no catch-up spiral)")
    else:
        ok = False
        print(f"FAIL: ghost clock hiccup — a 0.5 s frame took {n_hiccup} "
              f"steps (want <= {max_hiccup_steps}), disp {hiccup_disp:.1f}px: "
              f"the accumulator did not clamp")

    # HostTimeEstimator: the client's estimate of the host's sim clock.
    # (a) Steady 10 Hz samples, zero latency: the estimate must converge to
    # the true offset (here 1.0 s) — within 5 ms after 2 s of samples.
    est = HostTimeEstimator()
    for i in range(20):
        est.record(i * 0.1, 1.0 + i * 0.1)
    err_steady = abs(est.now(2.0) - 3.0)
    if err_steady < 0.005:
        print(f"PASS: host time estimator — steady 10 Hz samples, "
              f"estimate error {err_steady * 1000:.2f} ms < 5 ms")
    else:
        ok = False
        print(f"FAIL: host time estimator — steady samples, error "
              f"{err_steady * 1000:.2f} ms (want < 5 ms)")

    # (b) A host that simulates at 0.8x real time (a struggling loop): the
    # estimate must FOLLOW the host's rate, not assume 1x — within 20 ms
    # after 2 s of samples. A wall-clock clock (the 6.6 behavior) would be
    # 400 ms off here.
    est2 = HostTimeEstimator()
    for i in range(20):
        est2.record(i * 0.1, 0.8 * i * 0.1)
    err_slow = abs(est2.now(2.0) - 1.6)
    if err_slow < 0.02:
        print(f"PASS: host time estimator — 0.8x-rate host, estimate "
              f"error {err_slow * 1000:.2f} ms < 20 ms (rate followed)")
    else:
        ok = False
        print(f"FAIL: host time estimator — 0.8x-rate host, error "
              f"{err_slow * 1000:.2f} ms (want < 20 ms): the estimate "
              f"assumes a 1x host")

    # (c) A STARVED host (contended loop, 0.1x real time): samples arrive
    # a full second apart, and the estimate must stay anchored on the DATA
    # — never more than one snapshot interval ahead of the newest stamp.
    # Without the cap, extrapolating at the clamped rate runs the estimate
    # far ahead of the data and the render point clamp-stutters (the 7.1
    # e2e regression: a contended host starved to 0.1x and the uncapped
    # estimate ran 0.4-0.6 s ahead).
    est3 = HostTimeEstimator()
    for i in range(3):
        est3.record(i * 1.0, 0.1 * i * 1.0)
    est3_val = est3.now(3.5)
    newest3 = 0.1 * 2.0
    lead3 = est3_val - newest3
    if lead3 <= 0.1 + 1e-9 and est3_val >= newest3:
        print(f"PASS: host time estimator — starved 0.1x host, estimate "
              f"leads the newest stamp by {lead3 * 1000:.0f} ms "
              f"(<= one snapshot interval, anchored on the data)")
    else:
        ok = False
        print(f"FAIL: host time estimator — starved host, estimate "
              f"{est3_val:.3f} vs newest stamp {newest3:.3f} (lead "
              f"{lead3 * 1000:.0f} ms): the estimate outran the data")

    print(f"PASS: cadence knobs — SNAPSHOT_INTERVAL={SNAPSHOT_INTERVAL} "
          f"ticks, INTERP_DELAY={INTERP_DELAY}s")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()