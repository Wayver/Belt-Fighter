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

  h. FULL REMOTE RENDERING (Session 7.2) — the buffer now carries what the
     client needs to draw REAL remote enemies and ALL projectiles (the D3
     + D4 defects: the client rendered no bullets and 10 px enemy dots).
     The alpha/turnover batteries (a)/(c) are extended: enemies carry
     (tag, x, y, angle, vx, vy) — the angle lerp'd with the SAME
     wrapped-delta rule as player ships (6.8) — and a new 'bullets' key
     carries (x, y, vx, vy, kind, owner, boost) for all three projectile
     kinds, matched by PREDICTED POSITION (not index — the sim's bullet
     lists slide when a bullet is culled from the front, so same-index
     pairs are not the same bullet; a probe over 20 windows found ~11%
     mismatches). This part adds the dedicated checks:
       1. ENEMY ANGLE NO-TELEPORT — the buffer's per-frame enemy angle
          change never exceeds the sim's OWN peak per-frame enemy angle
          change (+1e-3 rad), the same pattern as the 6.8 ship-angle part
          (f). A bug that lerps the raw enemy angle without wrapping (or
          drops it) would spin the rendered hull and fail here.
       2. BULLET POP-IN — a bullet fired mid-window (in curr, not prev)
          appears at its curr position (no earlier position to lerp from).
       3. BULLET STRAIGHT LINE — a bullet present in both snapshots
          tracks the straight line between its prev and curr positions:
          |got - line| < 1e-6 at several alphas (bullets don't steer).
       4. BULLET DROP-OUT — a bullet that expires within the window (in
          prev, not curr) is absent from the output (membership follows
          curr, the same rule as enemies/asteroids).

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

  j. ADAPTIVE DELAY COMPUTE (Session 7.5a) — the adaptive interpolation
     delay (netcode.LatencyTracker) is a PURE FUNCTION OF THE ARRIVAL
     PATTERN: delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter),
     INTERP_DELAY_MIN, INTERP_DELAY_MAX), where the jitter samples are
     the HostTimeEstimator's per-arrival offset deviations
     (estimator.last_jitter). 7.5a computes the value only — the render
     path is untouched (7.5b re-anchors the render point to it). Two
     batteries:
       1. TRACKER LAW — driving the tracker with a constant jitter
          sample: the delay rises monotonically toward the clamped
          target (BASE + k*jitter, capped at MAX), moving at most 1/60 s
          per frame (the render point it will drive can never jump);
          with zero jitter it falls monotonically back to BASE.
       2. SYNTHETIC ARRIVAL PATTERN — the estimator + tracker are fed a
          steady 100 ms stream, a 250 ms gap (one late snapshot), and a
          50 ms burst: the delay stays in [MIN, MAX] and non-decreasing
          while steady, RESPONDS to the gap (rises well above BASE), and
          RECOVERS to BASE within 40 frames after the burst settles.
          The same pattern fed to a fresh tracker reproduces the delay
          trajectory bit-for-bit (determinism).

  k. RENDER POINT (Session 7.5b) — the render point is re-anchored on
     the DATA: render_t = newest ARRIVED snapshot stamp - adaptive
     delay, chased at a bounded per-frame rate (netcode.RenderPoint).
     This replaces the 7.1 host-time-estimate render clock for the
     render point (the estimator still runs — its jitter sample feeds
     the tracker). Two batteries:
       1. STEADY STREAM — a steady 10 Hz snapshot stream (the sim's own
          cadence): the render point is CONTINUOUS (per-frame advance
          <= STEP * 1.5 — the plan's bound; the raw `newest - delay`
          anchor would jump 0.1 s per arrival and fail it) and NEVER
          EXCEEDS the newest snapshot (delay >= MIN > 0).
       2. STALL + RECOVERY — the same stream with a 250 ms stall (the
          host stamps stay steady; the local arrivals are what the wifi
          ragged — the 7.5a pattern): the point stays continuous
          through the stall and the burst, never outruns the newest
          snapshot (the 7.1 failure mode — the estimate extrapolated
          forward through the no-arrival gap and the render
          clamp-stuttered), HOLDS the last window on the static anchor
          during the no-arrival gap, and resumes tracking the host's
          sim clock after the stream settles.
"""
import math
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT, SNAPSHOT_INTERVAL, INTERP_DELAY, \
    INTERP_DELAY_MIN, INTERP_DELAY_MAX, ADAPT_K, MAX_FRAME_DT, MAX_BULLETS
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .netcode import (interp_positions, SnapshotBuffer, PredictedShip,
                      HostTimeEstimator, LatencyTracker, RenderPoint)
from .intent import ShipInput
from .bullets import Bullet

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
    """The pose a Game snapshot implies for each player ship (x, y, angle),
    each enemy (tag, x, y, angle, vx, vy — Session 7.2), each asteroid
    (x, y), and each projectile (x, y, vx, vy, kind, owner, boost —
    Session 7.2) — the oracle interp_positions is checked against.
    Mirrors the layout documented in netcode.py (Session 6.1: index 0 is a
    tuple of per-player ship snapshots; Session 6.8: ships carry their raw
    angle, ship_s[4]; Session 7.2: enemies carry angle/tag/vel, and
    indices 2/3/4 are the player/enemy/missile projectile lists)."""
    return {
        'ships': [(p[0], p[1], p[4]) for p in s[0]],
        'enemies': [(tag, e_s[0][0], e_s[0][1], e_s[0][4],
                     e_s[0][2], e_s[0][3], e_s[2]) for tag, e_s in s[1]],
        'asteroids': [(a_s[1], a_s[2]) for a_s in s[5]],
        'bullets': _snap_bullets(s),
    }


def _snap_bullets(s):
    """The projectile entries a snapshot implies, in interp_positions'
    output order (player, enemy, missile) and entry shape (x, y, vx, vy,
    kind, owner, boost). At the snapshot itself (alpha 0/1) this is
    exactly what interp_positions must return for the 'bullets' key."""
    out = []
    for kind, idx, boost_field in (("player", 2, None),
                                   ("enemy", 3, None),
                                   ("missile", 4, 6)):
        for b in s[idx]:
            boost = b[boost_field] if boost_field is not None else 0.0
            out.append((b[0], b[1], b[2], b[3], kind, b[4], boost))
    return out


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


def id_oracle(prev_s, curr_s, alpha, dt=None):
    """What interp_positions SHOULD return: membership from curr_s, each
    entity lerped from its prev_s twin when one exists, else its curr
    position (a new entity pops in). Written independently of netcode.

    Ships carry their pose (x, y, angle) (Session 6.8): the position is
    lerped linearly and the angle is lerped by its WRAPPED delta — the
    same rule Ship.sync_render and netcode.lerp_angle use, written
    independently here (the sim's raw angle is unbounded, so a plain lerp
    of the raw values would swing the wrong way around the circle).

    Enemies carry (tag, x, y, angle, vx, vy) (Session 7.2): the position
    is lerped linearly, the angle by its WRAPPED delta (the same rule as
    ships — the enemy's raw angle is unbounded too), and the tag/velocity
    come from the CURRENT snapshot (membership follows curr; velocity is
    presentation data).

    Projectiles (Session 7.2): each curr bullet is matched to its prev
    twin by PREDICTED POSITION (prev.pos + prev.vel * dt, nearest within
    a radius) — written independently of netcode._match_bullets — and
    lerp'd from the twin's position; a bullet with no twin pops in at its
    curr position. dt is the snapshot window (None -> the nominal
    cadence, as in netcode).

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

    prev_e = {}
    for _tag, p in prev_s[1]:
        es = p[0]
        prev_e[_eid(p)] = (es[0], es[1], es[4])
    prev_r = {_akey(t): (t[1], t[2]) for t in prev_s[5]}

    def mix_enemy_angle(pp, cp):
        # pp = prev angle (or None), cp = curr angle. Same endpoint rules
        # as mix/mix_pose; the angle takes the wrapped delta.
        if a == 0.0:
            return cp if pp is None else pp
        if a == 1.0:
            return cp
        if pp is None:
            return cp
        da = (cp - pp + math.pi) % (2 * math.pi) - math.pi
        return pp + da * a

    enemies = []
    for tag, t in curr_s[1]:
        pe = prev_e.get(_eid(t))
        # mix takes a (x, y) pair only — at the endpoints it returns its
        # argument BY IDENTITY, so feeding it the 3-tuple (x, y, angle)
        # would duplicate the angle into the output (7-tuple -> 8-tuple).
        enemies.append((tag,
                        *mix((pe[0], pe[1]) if pe is not None else None,
                             (t[0][0], t[0][1])),
                        mix_enemy_angle(None if pe is None else pe[2],
                                        t[0][4]),
                        t[0][2], t[0][3], _eid(t)))
    # Player ships are matched by INDEX (Session 6.1) — ships don't turn
    # over, so slot i of curr_s[0] is the same ship as slot i of prev_s[0].
    # Pose = (x, y, raw angle) (Session 6.8).
    prev_ships = {i: (p[0], p[1], p[4]) for i, p in enumerate(prev_s[0])}

    # Projectiles: predicted-position matching, written independently of
    # netcode._match_bullets (same rule, different code — so a bug in
    # netcode's matching shows up as a mismatch, not a tautology).
    if dt is None:
        dt = SNAPSHOT_INTERVAL * (1.0 / 60.0)
    bullets = []
    for kind, idx, boost_field in (("player", 2, None),
                                   ("enemy", 3, None),
                                   ("missile", 4, 6)):
        prev_list, curr_list = prev_s[idx], curr_s[idx]
        preds = [((p[0] + p[2] * dt, p[1] + p[3] * dt), i)
                 for i, p in enumerate(prev_list)]
        used = set()
        for c in curr_list:
            best, best_d = None, 30.0
            for (px, py), i in preds:
                if i in used:
                    continue
                d = math.hypot(c[0] - px, c[1] - py)
                if d <= best_d:
                    best, best_d = i, d
            if best is None:
                x, y = c[0], c[1]
            else:
                used.add(best)
                pp = prev_list[best]
                x = pp[0] + (c[0] - pp[0]) * a if 0.0 < a < 1.0 \
                    else (pp[0] if a == 0.0 else c[0])
                y = pp[1] + (c[1] - pp[1]) * a if 0.0 < a < 1.0 \
                    else (pp[1] if a == 0.0 else c[1])
            boost = c[boost_field] if boost_field is not None else 0.0
            bullets.append((x, y, c[2], c[3], kind, c[4], boost))

    return {
        'ships': [mix_pose(prev_ships.get(i), (c[0], c[1], c[4]))
                  for i, c in enumerate(curr_s[0])],
        'enemies': enemies,
        'asteroids': [mix(prev_r.get(_akey(t)), (t[1], t[2]))
                      for t in curr_s[5]],
        'bullets': bullets,
    }


def id_sets(s):
    """The identity sets a snapshot carries: enemy ship ids, rock ids."""
    return (frozenset(_eid(t[1]) for t in s[1]),
            frozenset(_akey(t) for t in s[5]))


def check_positions(label, got, want):
    """Compare interp_positions output against the oracle. Returns True if
    identical; on mismatch prints which entity/field differs. Session 7.2:
    covers all four keys (ships, enemies, asteroids, bullets)."""
    ok = True
    for key in ('ships', 'enemies', 'asteroids', 'bullets'):
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
    # The window between snap_prev and snap_curr is K ticks; the projectile
    # predicted-position matching needs that real dt (Session 7.2).
    dt_a = K * STEP
    interp_cases = [
        (0.0, id_oracle(snap_prev, snap_curr, 0.0, dt_a),
         "alpha=0 -> prev positions (survivors)"),
        (1.0, id_oracle(snap_prev, snap_curr, 1.0, dt_a),
         "alpha=1 -> curr positions"),
        (0.5, id_oracle(snap_prev, snap_curr, 0.5, dt_a),
         "alpha=0.5 -> exact midpoint (linear)"),
        (-1.0, id_oracle(snap_prev, snap_curr, -1.0, dt_a),
         "alpha=-1 clamps to prev (no extrapolation)"),
        (2.0, id_oracle(snap_prev, snap_curr, 2.0, dt_a),
         "alpha=2 clamps to curr (no extrapolation)"),
    ]
    for alpha, want, desc in interp_cases:
        got = interp_positions(snap_prev, snap_curr, alpha, dt_a)
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
        # The turnover window is W ticks (10x the snapshot interval); the
        # projectile matching needs that real dt (Session 7.2).
        dt_c = W * STEP
        for alpha, want, desc in [
            (0.0, id_oracle(t_prev, t_curr, 0.0, dt_c),
             "alpha=0 -> prev positions (survivors)"),
            (1.0, id_oracle(t_prev, t_curr, 1.0, dt_c),
             "alpha=1 -> curr positions"),
            (0.5, id_oracle(t_prev, t_curr, 0.5, dt_c),
             "alpha=0.5 -> exact midpoint (linear)"),
            (-1.0, id_oracle(t_prev, t_curr, -1.0, dt_c),
             "alpha=-1 clamps to prev (no extrapolation)"),
            (2.0, id_oracle(t_prev, t_curr, 2.0, dt_c),
             "alpha=2 clamps to curr (no extrapolation)"),
        ]:
            got = interp_positions(t_prev, t_curr, alpha, dt_c)
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

    # --- (h) full remote rendering (Session 7.2): the buffer now carries
    # what the client needs to draw REAL remote enemies (angle, tag, vel,
    # id) and ALL projectiles (the D3 + D4 defects). The alpha/turnover
    # batteries (a)/(c) already check the new enemy/bullet shapes against
    # the independent oracle; this part adds the dedicated checks.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    h = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    buf = SnapshotBuffer(max_snapshots=200)
    snap_times = []
    # The sim's OWN peak per-tick enemy angle turn (the no-teleport bound
    # for the buffer's per-frame enemy turn, same pattern as part f).
    enemy_last_angle = {}
    max_auth_enemy_turn = 0.0
    for t in range(RUN_TICKS):
        if t % SNAPSHOT_INTERVAL == 0:
            buf.push(t * STEP, h.snapshot())
            snap_times.append(t * STEP)
        for e in h.enemies:
            eid = e.ship.id
            ang = e.ship.angle
            if eid in enemy_last_angle:
                d = abs((ang - enemy_last_angle[eid] + math.pi)
                        % (2 * math.pi) - math.pi)
                max_auth_enemy_turn = max(max_auth_enemy_turn, d)
            enemy_last_angle[eid] = ang
        if t < RUN_TICKS - 1:
            h.update(STEP, script_input(t))

    # (h.1) ENEMY ANGLE NO-TELEPORT — the buffer's per-frame enemy angle
    # change (tracked by id across render frames) never exceeds the sim's
    # OWN peak per-tick enemy turn (+1e-3 rad). A bug that lerps the raw
    # enemy angle without wrapping (or drops it) would spin the rendered
    # hull and fail here.
    n = round((snap_times[-1] - INTERP_DELAY) / STEP)
    prev_enemy_angles = {}
    max_buf_enemy_turn = 0.0
    enemy_frame_pairs = 0
    for i in range(n + 1):
        P = buf.positions_at(INTERP_DELAY + i * STEP)
        if P is None:
            ok = False
            print(f"FAIL: enemy angle — positions_at returned None at "
                  f"render_t={INTERP_DELAY + i * STEP}")
            break
        cur = {eid: ang for (_tag, _x, _y, ang, _vx, _vy, eid)
               in P['enemies']}
        for eid, ang in cur.items():
            if eid in prev_enemy_angles:
                d = abs((ang - prev_enemy_angles[eid] + math.pi)
                        % (2 * math.pi) - math.pi)
                max_buf_enemy_turn = max(max_buf_enemy_turn, d)
                enemy_frame_pairs += 1
        prev_enemy_angles = cur
    else:
        if max_auth_enemy_turn < 0.01 or enemy_frame_pairs == 0:
            ok = False
            print(f"FAIL: enemy angle — vacuous (sim peak turn "
                  f"{max_auth_enemy_turn:.4f} rad, {enemy_frame_pairs} "
                  f"tracked enemy-frame pairs); the check would pass "
                  f"trivially")
        elif max_buf_enemy_turn <= max_auth_enemy_turn + 1e-3:
            print(f"PASS: enemy angle no-teleport — "
                  f"{enemy_frame_pairs} tracked enemy-frame pairs, max "
                  f"buffer turn {max_buf_enemy_turn:.4f} rad <= sim peak "
                  f"{max_auth_enemy_turn:.4f} rad (+1e-3)")
        else:
            ok = False
            print(f"FAIL: enemy angle — buffer turn "
                  f"{max_buf_enemy_turn:.4f} rad exceeds the sim's own "
                  f"peak {max_auth_enemy_turn:.4f} rad (+1e-3): the "
                  f"remote enemy hull orientation teleported")

    # (h.2-h.4) BULLETS — pop-in / straight-line / drop-out. Scan a run of
    # snapshot windows and, for each, classify every bullet by an
    # INDEPENDENT predicted-position match (prev.pos + prev.vel * dt,
    # nearest within a radius — the same rule netcode uses, written
    # separately so a matching bug shows up as a mismatch):
    #   pop-in    — in curr, no prev twin  -> must render at its CURR pos
    #               for every alpha (no earlier position to lerp from)
    #   persistent— in both, matched       -> must track the straight line
    #               prev->curr: |got - (prev + (curr-prev)*alpha)| < 1e-6
    #   drop-out  — in prev, no curr twin  -> must be ABSENT (membership
    #               follows curr, the same rule as enemies/asteroids)
    def _bullet_match(prev_list, curr_list, dt, radius=30.0):
        """Independent predicted-position match. Returns (curr_matches,
        prev_matched_flags): curr_matches[i] = prev index or None;
        prev_matched_flags[j] = True if prev bullet j was matched."""
        preds = [((p[0] + p[2] * dt, p[1] + p[3] * dt), j)
                 for j, p in enumerate(prev_list)]
        used = set()
        curr_matches = []
        for c in curr_list:
            best, best_d = None, radius
            for (px, py), j in preds:
                if j in used:
                    continue
                d = math.hypot(c[0] - px, c[1] - py)
                if d <= best_d:
                    best, best_d = j, d
            if best is None:
                curr_matches.append(None)
            else:
                used.add(best)
                curr_matches.append(best)
        prev_matched = [False] * len(prev_list)
        for j in used:
            prev_matched[j] = True
        return curr_matches, prev_matched

    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    hb = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed=SEED)
    for t in range(WARMUP):
        hb.update(STEP, script_input(t))
    dt_h = SNAPSHOT_INTERVAL * STEP
    n_popin = n_persist = n_dropout = 0
    bullet_ok = True
    for w in range(20):
        prev_s = hb.snapshot()
        for t in range(SNAPSHOT_INTERVAL):
            hb.update(STEP, script_input(WARMUP + w * SNAPSHOT_INTERVAL + t))
        curr_s = hb.snapshot()
        curr_matches, prev_matched = _bullet_match(
            prev_s[2], curr_s[2], dt_h)
        got = interp_positions(prev_s, curr_s, 0.5, dt_h)
        # Index the got bullets by (kind, owner, vel) is not unique; match
        # got entries back to curr entries by curr position (a persistent
        # bullet's got pos is between prev and curr, a pop-in's got pos IS
        # its curr pos). Simpler: check the property per curr bullet by
        # finding the got entry whose curr endpoint is this bullet.
        # Build got lookup by curr bullet identity: a got entry's curr
        # bullet is the one it was lerp'd from; for a pop-in it equals the
        # curr pos, for a persistent it is the midpoint. Instead, verify
        # the property directly against the got list by re-deriving which
        # curr bullet each got entry corresponds to (nearest curr pos).
        # Membership (drop-out + pop-in count): the output bullet count
        # must equal the curr total across all three kinds. An expired
        # bullet that was kept would inflate it; a missed pop-in would
        # deflate it. (This is the exact, non-flaky drop-out proof — a
        # kept expired bullet has no curr slot, so the count gives it
        # away even if a live bullet were swapped in.)
        want_count = (len(curr_s[2]) + len(curr_s[3]) + len(curr_s[4]))
        if len(got['bullets']) != want_count:
            bullet_ok = False
            print(f"FAIL: bullet membership — window {w}: "
                  f"{len(got['bullets'])} output bullets, want "
                  f"{want_count} (curr total); an expired bullet was "
                  f"kept or a pop-in was dropped")
        for ci, c in enumerate(curr_s[2]):
            pm = curr_matches[ci]
            if pm is None:
                # pop-in: must render at its curr pos at alpha=0.5
                n_popin += 1
                if not any(math.hypot(g[0] - c[0], g[1] - c[1]) < 1e-6
                           for g in got['bullets']):
                    bullet_ok = False
                    print(f"FAIL: bullet pop-in — curr bullet {c[:2]} "
                          f"not at its curr pos in the interp output")
            else:
                # persistent: must track the straight line prev->curr
                n_persist += 1
                p = prev_s[2][pm]
                want_x = p[0] + (c[0] - p[0]) * 0.5
                want_y = p[1] + (c[1] - p[1]) * 0.5
                if not any(math.hypot(g[0] - want_x, g[1] - want_y)
                           < 1e-6 for g in got['bullets']):
                    bullet_ok = False
                    print(f"FAIL: bullet straight-line — curr bullet "
                          f"{c[:2]} (prev {p[:2]}) not on the line at "
                          f"alpha=0.5 (want {want_x:.3f},{want_y:.3f})")
        for j, matched in enumerate(prev_matched):
            if not matched:
                n_dropout += 1
    if n_popin == 0 or n_persist == 0 or n_dropout == 0:
        ok = False
        print(f"FAIL: bullets — vacuous (pop-in={n_popin}, "
              f"persistent={n_persist}, drop-out={n_dropout}); the "
              f"window/seed does not exercise all three cases")
    elif bullet_ok:
        print(f"PASS: bullets — {n_popin} pop-in at curr pos, "
              f"{n_persist} persistent on the straight line, "
              f"{n_dropout} drop-out absent (20 windows)")
    else:
        ok = False

    # --- (i) local bullet prediction (Session 7.3): the ghost keeps its OWN
    # gun shots as presentation Bullets so the player sees their fire
    # immediately instead of waiting ~100 ms for the host's next snapshot.
    # The ghost is stepped with the SAME scripted input the sim uses
    # (script_input holds SPACE in 30-tick bursts), and its local bullets
    # are advanced + culled each step. Three checks:
    #   1. FIRE TIMING — the ghost produces a local bullet within 1 step
    #      of the first tick its fire input is held (the gun's cooldown
    #      starts at 0, so the first held-fire tick fires).
    #   2. CAP — the list never exceeds MAX_BULLETS (the host's world cap).
    #   3. STRAIGHT LINE — a tracked bullet moves at exactly its speed per
    #      step (bullets don't steer), so its per-step displacement equals
    #      speed * STEP (within float tolerance).
    #   4. CULL — over the run at least one bullet expires (life < run),
    #      proving the cull path runs (a bullet that never dies would mean
    #      the list only ever grows).
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    g = PredictedShip()
    first_fire = None
    first_bullet = None
    max_list = 0
    culls = 0
    max_step_err = 0.0
    for t in range(RUN_TICKS):
        keys = script_input(t)
        inp = ShipInput.from_keys(keys)
        fire = inp.fire
        if fire and first_fire is None:
            first_fire = t
        n_before = len(g.local_bullets)
        g.step(STEP, inp)
        # Snapshot each bullet's position (and speed) BEFORE the bullet
        # step, so the straight-line check measures the displacement the
        # ghost's step_local_bullets actually applied (not a tautology on
        # Bullet.update's own prev_pos).
        prev_pos = {id(b): (b.pos.x, b.pos.y,
                            math.hypot(b.vel.x, b.vel.y))
                    for b in g.local_bullets}
        g.step_local_bullets(STEP)
        n_after = len(g.local_bullets)
        max_list = max(max_list, n_after)
        if n_after > n_before and first_bullet is None:
            first_bullet = t
        # Straight line: every surviving bullet moved exactly speed*STEP
        # (bullets don't steer — only their position changes per step).
        for b in g.local_bullets:
            pp = prev_pos.get(id(b))
            if pp is not None:
                moved = math.hypot(b.pos.x - pp[0], b.pos.y - pp[1])
                max_step_err = max(max_step_err,
                                   abs(moved - pp[2] * STEP))
        # Count a cull when the list shrank (a bullet expired).
        if n_after < n_before:
            culls += 1
    if first_fire is None:
        ok = False
        print("FAIL: local bullets — the scripted input never fired; "
              "the test would be vacuous")
    else:
        # 1. fire timing: a bullet within 1 step of the first held-fire tick
        if first_bullet is not None and first_bullet <= first_fire + 1:
            print(f"PASS: local bullet fire timing — first bullet at tick "
                  f"{first_bullet} (first fire {first_fire}, within 1 step)")
        else:
            ok = False
            print(f"FAIL: local bullet fire timing — first bullet at tick "
                  f"{first_bullet}, first fire {first_fire} (want <= "
                  f"{first_fire + 1})")
        # 2. cap
        if max_list <= MAX_BULLETS:
            print(f"PASS: local bullet cap — max list {max_list} <= "
                  f"MAX_BULLETS {MAX_BULLETS}")
        else:
            ok = False
            print(f"FAIL: local bullet cap — max list {max_list} > "
                  f"MAX_BULLETS {MAX_BULLETS}")
        # 3. straight line: per-step displacement == speed * STEP
        if max_step_err < 1e-6:
            print(f"PASS: local bullet straight line — max per-step "
                  f"displacement error {max_step_err:.2e} < 1e-6")
        else:
            ok = False
            print(f"FAIL: local bullet straight line — max per-step "
                  f"displacement error {max_step_err:.2e} >= 1e-6")
        # 4. cull ran
        if culls > 0:
            print(f"PASS: local bullet cull — {culls} bullet(s) expired "
                  f"over {RUN_TICKS} ticks")
        else:
            ok = False
            print(f"FAIL: local bullet cull — no bullet expired over "
                  f"{RUN_TICKS} ticks (the cull path never ran)")

    # --- (j) adaptive delay compute (Session 7.5a): the adaptive
    # interpolation delay is a PURE FUNCTION OF THE ARRIVAL PATTERN —
    # delay = clamp(INTERP_DELAY + ADAPT_K * EMA(jitter), MIN, MAX),
    # jitter = the estimator's per-arrival offset deviation. 7.5a
    # computes the value only; the render path is untouched (7.5b
    # re-anchors the render point to it).
    #
    # (j.1) TRACKER LAW — a constant jitter sample: the delay rises
    # monotonically toward the clamped target (BASE + k*jitter, capped
    # at MAX), moving at most 1/60 s per frame; zero jitter pulls it
    # monotonically back to BASE.
    jt = LatencyTracker()
    if jt.delay != INTERP_DELAY:
        ok = False
        print(f"FAIL: adaptive delay law — fresh tracker delay "
              f"{jt.delay}, want BASE {INTERP_DELAY}")
    max_step_rise = 0.0
    prev_d = jt.delay
    for _ in range(300):
        jt.update(0.1)          # constant 100 ms jitter sample
        jt.tick(1.0 / 60.0)
        max_step_rise = max(max_step_rise, abs(jt.delay - prev_d))
        prev_d = jt.delay
    target_hi = min(INTERP_DELAY_MAX, INTERP_DELAY + ADAPT_K * 0.1)
    # 300 frames at 1/60 s per frame = 5 s of travel — far more than the
    # ~0.1 s the target is above BASE, so the delay must have landed on
    # the clamped target.
    rise_ok = (abs(jt.delay - target_hi) < 1e-9
               and max_step_rise <= 1.0 / 60.0 + 1e-12)
    if rise_ok:
        print(f"PASS: adaptive delay law — constant 100 ms jitter -> "
              f"delay {jt.delay:.4f}s == clamp(BASE + k*jitter, MIN, MAX) "
              f"= {target_hi:.4f}s, max per-frame move "
              f"{max_step_rise * 1000:.2f} ms <= 1/60 s")
    else:
        ok = False
        print(f"FAIL: adaptive delay law — delay {jt.delay:.4f}s, want "
              f"{target_hi:.4f}s; max per-frame move "
              f"{max_step_rise * 1000:.3f} ms (want <= 1/60 s)")
    max_step_fall = 0.0
    prev_d = jt.delay
    for _ in range(300):
        jt.update(0.0)          # jitter back to zero
        jt.tick(1.0 / 60.0)
        max_step_fall = max(max_step_fall, abs(jt.delay - prev_d))
        prev_d = jt.delay
    fall_ok = (abs(jt.delay - INTERP_DELAY) < 1e-9
               and max_step_fall <= 1.0 / 60.0 + 1e-12)
    if fall_ok:
        print(f"PASS: adaptive delay recovery — zero jitter -> delay "
              f"back to BASE {jt.delay:.4f}s, max per-frame move "
              f"{max_step_fall * 1000:.2f} ms <= 1/60 s")
    else:
        ok = False
        print(f"FAIL: adaptive delay recovery — delay {jt.delay:.4f}s, "
              f"want BASE {INTERP_DELAY:.4f}s; max per-frame move "
              f"{max_step_fall * 1000:.3f} ms")

    # (j.2) SYNTHETIC ARRIVAL PATTERN — steady 100 ms, a 250 ms gap
    # (one late snapshot), a 10 ms burst of the packets queued during
    # the stall. The estimator + tracker are fed the pattern exactly as
    # the 7.5b wiring will feed them (estimator.record on each arrival,
    # tracker.update on the jitter sample, tracker.tick every frame).
    # The delay must stay in [MIN, MAX], be non-decreasing while the
    # stream is steady, RESPOND to the gap (rise well above BASE), and
    # RECOVER to BASE after the stream settles.
    class _T:
        """A (local_time, host_stamp) arrival pair."""
        __slots__ = ("lt", "ht")

        def __init__(self, lt, ht):
            self.lt, self.ht = lt, ht

    # The HOST stamps at its own steady 0.1 s cadence (its sim clock
    # keeps running through a wifi stall); the LOCAL arrival times are
    # what the wifi ragged. Both must be strictly increasing (the
    # estimator drops out-of-order stamps, and the frame loop processes
    # arrivals in list order). Steady: local == stamp (a constant 0
    # offset). A 250 ms stall delays the stamped-3.0 packet to local
    # 3.25, and the packets queued during the stall (stamps 3.1-3.3)
    # burst out 10 ms apart.
    arrivals = []
    for i in range(20):                       # steady 100 ms, 1.0-2.9
        s = 1.0 + i * 0.1
        arrivals.append(_T(s, s))
    gap_start = 3.25
    arrivals.append(_T(3.25, 3.0))            # the late packet (250 ms)
    for i in range(3):                        # the queued burst
        s = 3.1 + i * 0.1
        arrivals.append(_T(3.26 + i * 0.01, s))
    # 40 steady arrivals (3.4-7.3): long enough that the estimator's
    # offset + rate re-converge to the new steady state (the 7.1 EMAs
    # settle in ~1.5 s after the burst) and the delay returns to BASE —
    # the recovery check below is against BASE + 15 ms, 40 frames after
    # the last arrival.
    for i in range(40):
        s = 3.4 + i * 0.1
        arrivals.append(_T(s, s))

    def _run_pattern():
        est = HostTimeEstimator()
        trk = LatencyTracker()
        delays = []
        ai = 0
        # Integer frame arithmetic (no accumulated float drift in t):
        # frame i is at local time i/60. Run to 490 frames (8.17 s) —
        # past the last arrival (7.3 s) + the 40-frame recovery window.
        for i in range(490):
            t = i / 60.0
            while ai < len(arrivals) and arrivals[ai].lt <= t + 1e-12:
                a = arrivals[ai]
                if est.record(a.lt, a.ht):
                    trk.update(est.last_jitter)
                ai += 1
            trk.tick(1.0 / 60.0)
            delays.append(trk.delay)
        return delays

    delays = _run_pattern()
    delays2 = _run_pattern()          # determinism: fresh tracker, same
                                      # pattern -> identical trajectory
    in_bounds = all(INTERP_DELAY_MIN - 1e-12 <= d <= INTERP_DELAY_MAX + 1e-12
                    for d in delays)
    # Non-decreasing while steady: every frame up to the gap (the first
    # 120 frames = 2.0 s at 60 fps) may only grow (jitter is ~0, so the
    # delay holds at BASE — growth here would be a bug).
    steady_prefix = delays[:120]
    monotone_steady = all(b >= a - 1e-12 for a, b in
                          zip(steady_prefix, steady_prefix[1:]))
    # Response: the gap's jitter sample spikes the EMA; the delay must
    # rise well above BASE (>= BASE + 0.02 s) at some point after the
    # gap and before the recovery window.
    gap_frame = round(gap_start * 60)
    response = max(delays[gap_frame:gap_frame + 60])
    responded = response >= INTERP_DELAY + 0.02
    # Recovery: 40 frames (0.67 s) after the last arrival the delay must
    # be back at BASE (the jitter EMA has decayed and the estimator's
    # offset has re-converged to the steady-state value).
    last_arrival_frame = round(arrivals[-1].lt * 60)
    recovered = delays[last_arrival_frame + 40] <= INTERP_DELAY + 0.015
    if (in_bounds and monotone_steady and responded and recovered
            and delays == delays2):
        print(f"PASS: adaptive delay pattern — steady 100 ms + 250 ms gap "
              f"+ 10 ms burst: delay in [{INTERP_DELAY_MIN}, "
              f"{INTERP_DELAY_MAX}] every frame, steady prefix "
              f"non-decreasing, gap response peak {response:.4f}s "
              f"(>= BASE + 0.02), recovered to "
              f"{delays[last_arrival_frame + 40]:.4f}s (<= BASE + 0.015) "
              f"40 frames after the burst; trajectory deterministic")
    else:
        ok = False
        print(f"FAIL: adaptive delay pattern — in_bounds={in_bounds}, "
              f"steady non-decreasing={monotone_steady}, gap response "
              f"peak {response:.4f}s (want >= {INTERP_DELAY + 0.02}), "
              f"recovered={recovered} "
              f"({delays[last_arrival_frame + 40]:.4f}s), deterministic="
              f"{delays == delays2}")

    # --- (k) render point (Session 7.5b): the render point is re-anchored
    # on the DATA — render_t = newest ARRIVED snapshot stamp - adaptive
    # delay, chased at a bounded per-frame rate (netcode.RenderPoint).
    # This replaces the 7.1 host-time-estimate render clock for the
    # render point: a model of the host clock and the data disagree
    # under jitter/loss, and the render must not wander by the
    # disagreement. The estimator still runs (its jitter sample feeds
    # the tracker), but the render reads the buffer, not the model.
    #
    # (k.1) STEADY STREAM — a steady 10 Hz snapshot stream (the sim's
    # own cadence). The plan's two invariants:
    #   1. CONTINUITY — the per-frame advance is <= STEP * 1.5. The raw
    #      `newest - delay` anchor would jump by one snapshot interval
    #      (0.1 s = 6 x STEP) at every arrival and fail this — the
    #      chase is what makes the point continuous.
    #   2. NO LOOK-AHEAD — the point never exceeds the newest snapshot
    #      (delay >= INTERP_DELAY_MIN > 0).
    # The feed mirrors the 7.5b wiring exactly: estimator.record on each
    # arrival (tracker.update on the jitter sample when it records),
    # tracker.tick + render_point.advance once per frame.
    buf_k = SnapshotBuffer(max_snapshots=200)
    est_k = HostTimeEstimator()
    trk_k = LatencyTracker()
    rp_k = RenderPoint(trk_k)
    render_track = []
    max_adv = 0.0
    never_ahead = True
    steady_ok = True
    for i in range(600):                     # 10 s at 60 fps
        t = i / 60.0
        if i % 6 == 0:                       # 10 Hz arrivals, 0-latency
            s = i * STEP
            buf_k.push(s, a.snapshot())      # a's state is static here —
                                             # the buffer's DATA is the
                                             # stamps; the pose content
                                             # is irrelevant to the
                                             # render-point law
            if est_k.record(t, s):
                trk_k.update(est_k.last_jitter)
        trk_k.tick(1.0 / 60.0)
        rp = rp_k.advance(1.0 / 60.0, buf_k.newest_time())
        if rp is None:
            continue
        render_track.append(rp)
        newest = buf_k.newest_time()
        if rp > newest + 1e-12:
            never_ahead = False
        if len(render_track) >= 2:
            adv = render_track[-1] - render_track[-2]
            max_adv = max(max_adv, adv)
            if adv > STEP * 1.5 + 1e-12:
                steady_ok = False
    if (steady_ok and never_ahead and len(render_track) > 500):
        print(f"PASS: render point steady — {len(render_track)} frames "
              f"over a steady 10 Hz stream: max per-frame advance "
              f"{max_adv * 1000:.2f} ms <= STEP x 1.5 "
              f"({STEP * 1.5 * 1000:.2f} ms), point never exceeds the "
              f"newest snapshot")
    else:
        ok = False
        print(f"FAIL: render point steady — steady_ok={steady_ok} "
              f"(max advance {max_adv * 1000:.2f} ms, want <= "
              f"{STEP * 1.5 * 1000:.2f} ms), never_ahead={never_ahead}, "
              f"frames={len(render_track)}")

    # (k.2) STALL + RECOVERY — the 7.5a pattern (steady 100 ms, a 250 ms
    # stall, a 10 ms burst of the queued packets, a long steady tail)
    # played through the REAL buffer + estimator + tracker + render
    # point, fed exactly as the 7.5b wiring feeds them. The host stamps
    # stay at the steady 0.1 s cadence (its sim clock runs through the
    # stall); the LOCAL arrival times are what the wifi ragged. The
    # render point must:
    #   1. stay CONTINUOUS through the stall and the burst (per-frame
    #      advance <= STEP * 1.5 — a late packet must not teleport it);
    #   2. HOLD during the stall — the stall is the NO-ARRIVAL gap
    #      (local 2.9 -> 3.25: the stamped-3.0 packet is still in
    #      flight), where the anchor (newest - delay) is STATIC. The
    #      point must sit on it within one frame step: it holds the
    #      last window instead of outrunning the data and
    #      clamp-stuttering (the 7.1 failure mode — the estimate kept
    #      extrapolating forward through the gap). The post-arrival
    #      transient (3.25 -> 3.4) is NOT a hold window: the anchor
    #      itself moves fast there (newest jumps 2.9 -> 3.3 while the
    #      delay rises), and a bounded-rate chaser legitimately lags a
    #      fast-moving anchor — continuity (1) is the invariant there;
    #   3. RESUME after the stream settles — the point advances again
    #      (the steady-state tracking rate, ~1x the host's sim rate).
    arrivals_k = []
    for i in range(20):                       # steady 100 ms, 1.0-2.9
        s = 1.0 + i * 0.1
        arrivals_k.append((s, s))
    arrivals_k.append((3.25, 3.0))            # the late packet (250 ms)
    for i in range(3):                        # the queued burst
        s = 3.1 + i * 0.1
        arrivals_k.append((3.26 + i * 0.01, s))
    for i in range(40):                       # steady tail 3.4-7.3
        s = 3.4 + i * 0.1
        arrivals_k.append((s, s))

    buf_k2 = SnapshotBuffer(max_snapshots=200)
    est_k2 = HostTimeEstimator()
    trk_k2 = LatencyTracker()
    rp_k2 = RenderPoint(trk_k2)
    track2 = []
    continuous = True
    outrun = False          # point past the newest snapshot (7.1 mode)
    hold_worst = 0.0        # |rp - anchor| on the converged hold window
    ai = 0
    for i in range(490):                      # 8.17 s at 60 fps
        t = i / 60.0
        while ai < len(arrivals_k) and arrivals_k[ai][0] <= t + 1e-12:
            lt, s = arrivals_k[ai]
            buf_k2.push(s, a.snapshot())
            if est_k2.record(lt, s):
                trk_k2.update(est_k2.last_jitter)
            ai += 1
        trk_k2.tick(1.0 / 60.0)
        rp = rp_k2.advance(1.0 / 60.0, buf_k2.newest_time())
        if rp is None:
            continue
        track2.append((t, rp))
        if len(track2) >= 2:
            adv = track2[-1][1] - track2[-2][1]
            if adv > STEP * 1.5 + 1e-12:
                continuous = False
        newest = buf_k2.newest_time()
        if rp > newest + 1e-12:
            outrun = True
        # HOLD WINDOW: the no-arrival gap (local 2.9 -> 3.25: the
        # stamped-3.0 packet is still in flight), after the point has
        # had its 6 convergence frames (t >= 3.05). There the anchor
        # (newest - delay) is STATIC (newest = 2.9, delay = 0.1 — no
        # jitter samples arrive during the gap), so the point must sit
        # on it: holding the last window instead of outrunning the data
        # (the 7.1 failure mode — the estimate kept extrapolating
        # forward through the gap and the render clamp-stuttered). The
        # post-arrival transient (3.25 -> 3.4) is NOT a hold window:
        # the anchor moves fast there (newest jumps 2.9 -> 3.3 while
        # the delay rises), and a bounded-rate chaser legitimately lags
        # a fast-moving anchor — continuity is the invariant there.
        if 3.05 <= t < 3.25:
            hold_worst = max(hold_worst,
                             abs(rp - (newest - trk_k2.delay)))
    # Resume: over [3.4, 5.0] (steady stream, well past the burst) the
    # point must track the host's sim clock again — the steady-state
    # rate is ~1x, so >= 1 s of advance over 1.6 s of frames (a frozen
    # point would advance 0).
    seg = [rp for (t, rp) in track2 if 3.4 <= t <= 5.0]
    resumed = len(seg) > 50 and (seg[-1] - seg[0]) >= 1.0
    if (continuous and not outrun and hold_worst < 1e-9 and resumed
            and len(track2) > 400):
        print(f"PASS: render point stall — 250 ms stall + 10 ms burst: "
              f"point continuous (<= STEP x 1.5 per frame), never "
              f"outruns the newest snapshot, holds the last window "
              f"(worst {hold_worst:.1e} from the static anchor on the "
              f"no-arrival gap), resumed after the burst (advanced "
              f"{seg[-1] - seg[0]:.2f}s over [3.4, 5.0])")
    else:
        ok = False
        print(f"FAIL: render point stall — continuous={continuous}, "
              f"outrun={outrun}, hold worst {hold_worst:.3e} (want "
              f"< 1e-9), resumed={resumed} (advance "
              f"{seg[-1] - seg[0] if len(seg) > 1 else -1:.2f}s over "
              f"[3.4, 5.0]), frames={len(track2)}")

    print(f"PASS: cadence knobs — SNAPSHOT_INTERVAL={SNAPSHOT_INTERVAL} "
          f"ticks, INTERP_DELAY={INTERP_DELAY}s, adaptive "
          f"[{INTERP_DELAY_MIN}, {INTERP_DELAY_MAX}] k={ADAPT_K}")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()