"""Remote-interpolation + snapshot-cadence check (Session 5a).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_interpolation

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens. The SDL dummy-driver boilerplate, Keys, and script_input are copied
from test_snapshot.py so the input pattern is identical.

Four proofs:

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
"""
import math
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT, SNAPSHOT_INTERVAL, INTERP_DELAY
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .netcode import interp_positions, SnapshotBuffer

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
    """The (x, y) positions a Game snapshot implies for the ship, each
    enemy, and each asteroid — the oracle interp_positions is checked
    against. Mirrors the layout documented in netcode.py."""
    return {
        'ship': (s[0][0], s[0][1]),
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

    prev_e = {_eid(t[1]): (t[1][0][0], t[1][0][1]) for t in prev_s[1]}
    prev_r = {_akey(t): (t[1], t[2]) for t in prev_s[5]}

    return {
        'ship': mix((prev_s[0][0], prev_s[0][1]), (curr_s[0][0], curr_s[0][1])),
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
    for key in ('ship', 'enemies', 'asteroids'):
        g_list = [got[key]] if key == 'ship' else got[key]
        w_list = [want[key]] if key == 'ship' else want[key]
        if len(g_list) != len(w_list):
            print(f"FAIL: {label} — {key}: {len(g_list)} entries, "
                  f"want {len(w_list)}")
            ok = False
            continue
        for i, (gp, wp) in enumerate(zip(g_list, w_list)):
            if gp != wp:
                name = key if key == 'ship' else f"{key}[{i}]"
                print(f"FAIL: {label} — {name} differs: got {gp}, want {wp}")
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
    buf = SnapshotBuffer()
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
        render_track.append(P['ship'])
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

    print(f"PASS: cadence knobs — SNAPSHOT_INTERVAL={SNAPSHOT_INTERVAL} "
          f"ticks, INTERP_DELAY={INTERP_DELAY}s")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()