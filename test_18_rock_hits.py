"""Session 18 (plan_18): the client's option-A hit-FX trigger now covers
ROCKS — the local player's bullets and missiles stop at an asteroid's
surface (spark + cull) instead of flying through it until the host's
authoritative cull arrives ~100-200 ms later.

The authoritative hit, damage, and rock death still come from the host
snapshot; this is presentation only. The client mirrors the host's
bullet/missile-vs-rock test EXACTLY: a CIRCLE (the host's test is
`dist < collision_radius`, not the drawn non-convex polygon), with R
derived from the wire verts as `(max|v| + min|v|) / 2`, and the impact at
the circle ENTRY point (the near surface, not the center).

This test proves it (value-level):
  1. HELPER   — `_seg_circle_entry`: entry point on the boundary, a clean
                miss, a start-inside segment, and a degenerate (zero-length)
                segment (in + out).
  2. R_EST    — a known vert set (radii {10, 20}) -> 15.0; a real
                `Asteroid._make_rock` output -> within [0.72R, 1.25R] of the
                true R.
  3. E2E BULLET — a ghost bullet aimed at a rock: a burst spawns, the spark
                is on the circle boundary (|impact - center| ~= R_est), and
                the bullet is culled (gone from local_bullets).
  4. E2E BUFFER BULLET — a buffer-drawn local bullet (id not in the ghost's
                local list) crossing a rock: burst + cull (the buffer copy
                stops at the rock too).
  5. MISS     — a bullet aimed through the gap beside the rock: no cull, no
                burst, the bullet is still drawn.
  6. MISSILE  — a ghost missile on a rock course: burst + the missile removed
                from local_missiles (so it gets no lingering fog light).
  7. TTL      — a cull entry older than BULLET_LIFE / MISSILE_LIFE is pruned
                (the sets are bounded); a fresh entry is kept.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_18_rock_hits

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import random
import time

import pygame

from .config import WIDTH, HEIGHT, BULLET_SPEED, MISSILE_SPEED, BULLET_LIFE, \
    MISSILE_LIFE
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .bullets import Bullet
from .asteroid import Asteroid
from .netcode import (PredictedShip, GhostMissile, RenderPoint,
                      LatencyTracker, HostTimeEstimator)
from .game import Game, _seg_circle_entry
from .fog import make_light_texture

# A gun-fitted test hull (the scout polygon + a 'gun' weapon slot) so the
# ghost can carry a local bullet. default_loadout maps 'gun' -> GUN_TYPE.
GUN_HULL = HullType(
    id='gun_test',
    polygon=((18, 0), (14, 3.5), (8, 6.5), (0, 8), (-8, 8), (-12, 11),
             (-12, 5), (-9, 3), (-9, -3), (-12, -5), (-12, -11), (-8, -8),
             (0, -8), (8, -6.5), (14, -3.5)),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN,
           REACTOR, COMPUTER, SHIELD, SENSOR),
    base_mass=1.0,
    collision_radius=12.0,
    nose=(18, 0),
    cockpit=(8, 0),
    max_speed_factor=1.0,
    turn_rate_factor=1.0,
    fill=(200, 200, 200),
    edge=(255, 255, 255),
)
LOADOUT = default_loadout(GUN_HULL)
LOCAL_INDEX = 1   # the client is player 1
ROCK_POS = (200.0, 0.0)   # the rock's world position (stationary)
# burst() spawns 14 particles for a non-big burst (particles.py).
BURST_N = 14


def make_client():
    """A scratch client Game wired for predicted_view (ghost + render point +
    host-time estimator + latency tracker), local player index 1."""
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=GUN_HULL, loadout=LOADOUT, seed=1234, players=2,
             local_index=LOCAL_INDEX)
    g.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    g.ghost = PredictedShip(hull=GUN_HULL, loadout=LOADOUT,
                            local_index=LOCAL_INDEX)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def make_host(rock_pos=ROCK_POS, extra_bullets=(), rock_size='medium'):
    """A scratch host Game with the local ship (index 1) parked at the origin
    facing +x (gun on cooldown so the ghost does not fire) and ONE stationary
    rock at `rock_pos`. Returns (tmp, rock). `extra_bullets` are added to the
    host's player-bullet list (so they enter the buffer as buffer-drawn
    projectiles)."""
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tmp = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, light_tex,
               fog_surf, light_surf, hull=GUN_HULL, loadout=LOADOUT,
               seed=1234, players=2)
    tmp.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    cs = tmp.players[LOCAL_INDEX]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 999.0   # the ghost does NOT fire during the frame
    rock = Asteroid(pygame.Vector2(*rock_pos), rock_size, vel=(0, 0),
                    rng=random.Random(42))
    tmp.asteroids = [rock]
    tmp.enemies.clear()
    tmp.enemy_bullets.clear()
    tmp.missiles.clear()
    tmp.bullets = list(extra_bullets)
    return tmp, rock


def push_stream(g, tmp, n=3, dt=0.1):
    """Push `n` snapshots (t = 0, dt, 2*dt, ...) so the buffer holds a window
    to interpolate, and the rock's verts are registered (first sight)."""
    for k in range(n):
        g.push_snapshot(k * dt, tmp.snapshot(), now=k * dt)


def pump(g, frames=40):
    """Run `frames` of the real client order (latency.tick + render_point.
    advance + predicted_view). Returns the frame index the hit-FX burst first
    appeared (len(_hit_fx) > 0), or None if it never fired."""
    for frame in range(frames):
        g.latency.tick(0.016)
        g.render_point.advance(0.016, g.snap_buf.newest_time())
        g.predicted_view(0.016, pygame.key.get_pressed())
        if len(g._hit_fx) > 0:
            return frame
    return None


def burst_center(g):
    """The burst's impact point: all particles share the SAME pos (the impact
    point) with different random velocities, so the mean of the positions is
    the impact point exactly."""
    return (sum(p.pos.x for p in g._hit_fx) / len(g._hit_fx),
            sum(p.pos.y for p in g._hit_fx) / len(g._hit_fx))


def main():
    pygame.init()

    # --- 1. HELPER: _seg_circle_entry -----------------------------------
    # Entry point on the boundary: segment (0,0)->(100,0) vs circle
    # (50,0) r=10 -> the first crossing is (40,0), |p - c| = 10 = r.
    hit, hx, hy = _seg_circle_entry(0.0, 0.0, 100.0, 0.0, 50.0, 0.0, 10.0)
    assert hit and abs(hx - 40.0) < 1e-6 and abs(hy - 0.0) < 1e-6, \
        "entry must be the first crossing (40,0): %r" % (hit, hx, hy)
    assert abs(math.hypot(hx - 50.0, hy - 0.0) - 10.0) < 1e-6, \
        "the entry point must be ON the circle boundary (|p-c| == r)"
    # A clean miss: the segment (0,0)->(100,0) never comes within r=10 of
    # (50,50).
    hit, hx, hy = _seg_circle_entry(0.0, 0.0, 100.0, 0.0, 50.0, 50.0, 10.0)
    assert not hit and hx is None and hy is None, \
        "a segment 50 px from the center must MISS: %r" % (hit, hx, hy)
    # Start-inside: a segment that BEGINS inside the circle returns (True,
    # p0) — the impact is where the projectile already is.
    hit, hx, hy = _seg_circle_entry(50.0, 0.0, 100.0, 0.0, 50.0, 0.0, 10.0)
    assert hit and abs(hx - 50.0) < 1e-6 and abs(hy - 0.0) < 1e-6, \
        "a start-inside segment must return (True, p0): %r" % (hit, hx, hy)
    # Degenerate (zero-length) segment: a plain point test. In -> hit at the
    # point; out -> miss.
    hit, hx, hy = _seg_circle_entry(50.0, 0.0, 50.0, 0.0, 50.0, 0.0, 10.0)
    assert hit and abs(hx - 50.0) < 1e-6, \
        "a degenerate segment at the center must HIT (point test): %r" \
        % (hit, hx, hy)
    hit, hx, hy = _seg_circle_entry(65.0, 0.0, 65.0, 0.0, 50.0, 0.0, 10.0)
    assert not hit, \
        "a degenerate segment 15 px from the center (r=10) must MISS: %r" \
        % (hit, hx, hy)
    print("PASS: HELPER — _seg_circle_entry returns the boundary entry point, "
          "misses cleanly, returns p0 when start-inside, and point-tests a "
          "degenerate segment")

    # --- 2. R_EST: the rock's collision circle derived from the verts ----
    # Known vert set (radii {10, 20}) -> (10 + 20) / 2 = 15.0.
    g = make_client()
    tmp, rock = make_host()
    rock.verts = [pygame.Vector2(10.0, 0.0), pygame.Vector2(0.0, 20.0)]
    push_stream(g, tmp)
    assert abs(g._rock_coll_r[rock.id] - 15.0) < 1e-6, \
        "R_est for radii {10, 20} must be 15.0: %r" \
        % g._rock_coll_r[rock.id]
    # A real Asteroid._make_rock output: every vertex radius is R x U(0.72,
    # 1.25), so (max + min) / 2 must land within [0.72R, 1.25R] of the true
    # R.
    rock2 = Asteroid(pygame.Vector2(0.0, 0.0), 'medium', vel=(0, 0),
                     rng=random.Random(7))
    radii = [v.length() for v in rock2.verts]
    r_est = (max(radii) + min(radii)) / 2.0
    R = rock2.radius
    assert 0.72 * R <= r_est <= 1.25 * R, \
        "R_est for a real rock must be within [0.72R, 1.25R] of the true R " \
        "(R=%.1f, R_est=%.2f)" % (R, r_est)
    print("PASS: R_EST — a known vert set (radii {10,20}) gives 15.0, and a "
          "real _make_rock output gives R_est=%.2f within [%.1f, %.1f] of R"
          % (r_est, 0.72 * R, 1.25 * R))

    # --- 3. E2E BULLET: a ghost bullet aimed at the rock -----------------
    g = make_client()
    tmp, rock = make_host()
    push_stream(g, tmp)
    R_est = g._rock_coll_r[rock.id]
    # A local bullet at (150, 0) moving +x at BULLET_SPEED, aimed straight at
    # the rock's center (200, 0).
    g.ghost.local_bullets = [
        Bullet(pygame.Vector2(150.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g.ghost.local_bullets[0].pending = False
    fired = pump(g)
    assert fired is not None, \
        "a ghost bullet aimed at the rock must fire the hit-FX trigger"
    assert len(g._hit_fx) == BURST_N, \
        "the rock hit must spawn a burst of %d particles (got %d)" \
        % (BURST_N, len(g._hit_fx))
    ix, iy = burst_center(g)
    d = math.hypot(ix - ROCK_POS[0], iy - ROCK_POS[1])
    assert abs(d - R_est) < 1.0, \
        "the spark must be on the circle boundary (|impact - center| ~= " \
        "R_est=%.2f, got %.2f)" % (R_est, d)
    assert len(g.ghost.local_bullets) == 0, \
        "the hit bullet must be CULLED (removed from local_bullets): %r" \
        % (g.ghost.local_bullets,)
    print("PASS: E2E BULLET — a ghost bullet aimed at the rock spawns a "
          "burst at the circle entry point (%.1f, %.1f) [d=%.2f ~= R_est "
          "%.2f] and is culled" % (ix, iy, d, R_est))

    # --- 4. E2E BUFFER BULLET: a buffer-drawn local bullet crossing the
    #     rock (id NOT in the ghost's local list, so the buffer draws it).
    g = make_client()
    buf_b = Bullet(pygame.Vector2(150.0, 0.0),
                   pygame.Vector2(BULLET_SPEED, 0.0),
                   owner=1, bid=(LOCAL_INDEX, 5))
    tmp, rock = make_host(extra_bullets=[buf_b])
    push_stream(g, tmp)
    R_est = g._rock_coll_r[rock.id]
    # The ghost has NO local bullets, so the buffer's (1,5) copy is a
    # buffer-drawn local bullet — the trigger must cull it at the rock too.
    fired = pump(g)
    assert fired is not None, \
        "a buffer-drawn local bullet crossing the rock must fire the trigger"
    assert len(g._hit_fx) == BURST_N, \
        "the buffer-bullet rock hit must spawn a burst of %d (got %d)" \
        % (BURST_N, len(g._hit_fx))
    assert (LOCAL_INDEX, 5) in g._hit_fx_cull_bids, \
        "the buffer bullet's id must be in the cull set: %r" \
        % (g._hit_fx_cull_bids,)
    ix, iy = burst_center(g)
    d = math.hypot(ix - ROCK_POS[0], iy - ROCK_POS[1])
    assert abs(d - R_est) < 1.0, \
        "the buffer-bullet spark must be on the circle boundary (d=%.2f, " \
        "R_est=%.2f)" % (d, R_est)
    print("PASS: E2E BUFFER BULLET — a buffer-drawn local bullet crossing the "
          "rock is culled and sparks at the entry point (d=%.2f ~= R_est "
          "%.2f)" % (d, R_est))

    # --- 5. MISS: a bullet aimed through the gap beside the rock ---------
    g = make_client()
    tmp, rock = make_host()
    push_stream(g, tmp)
    R_est = g._rock_coll_r[rock.id]
    # A bullet at y = +60 (well clear of the rock's R_est ~ 26) moving +x:
    # its closest approach to the rock center is 60 px > R_est -> a miss.
    gap = 60.0
    assert gap > R_est + 10.0, "test setup: the gap must clear the rock"
    g.ghost.local_bullets = [
        Bullet(pygame.Vector2(150.0, gap), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g.ghost.local_bullets[0].pending = False
    pump(g)
    assert len(g._hit_fx) == 0, \
        "a bullet through the gap must NOT spawn a burst (got %d)" \
        % len(g._hit_fx)
    assert (LOCAL_INDEX, 0) not in g._hit_fx_cull_bids, \
        "a missing bullet must NOT be culled: %r" % (g._hit_fx_cull_bids,)
    assert len(g.ghost.local_bullets) == 1, \
        "a missing bullet must still be drawn (kept in local_bullets): %r" \
        % (g.ghost.local_bullets,)
    print("PASS: MISS — a bullet through the gap beside the rock (y=+%.0f, "
          "clear of R_est=%.1f) spawns no burst, is not culled, and is still "
          "drawn" % (gap, R_est))

    # --- 6. MISSILE: a ghost missile on a rock course --------------------
    g = make_client()
    tmp, rock = make_host()
    push_stream(g, tmp)
    R_est = g._rock_coll_r[rock.id]
    # A ghost missile at (150, 0) coasting +x at MISSILE_SPEED (boost 0 ->
    # straight line, no homing) aimed at the rock's center.
    g.ghost.local_missiles = [
        GhostMissile(pygame.Vector2(150.0, 0.0),
                     pygame.Vector2(MISSILE_SPEED, 0.0),
                     mid=(LOCAL_INDEX, 0), target=None)]
    g.ghost.local_missiles[0].boost = 0.0   # coast straight (no ramp)
    fired = pump(g)
    assert fired is not None, \
        "a ghost missile on a rock course must fire the hit-FX trigger"
    assert len(g._hit_fx) == BURST_N, \
        "the missile rock hit must spawn a burst of %d (got %d)" \
        % (BURST_N, len(g._hit_fx))
    assert (LOCAL_INDEX, 0) in g._hit_fx_cull_mids, \
        "the missile's id must be in the missile cull set: %r" \
        % (g._hit_fx_cull_mids,)
    assert len(g.ghost.local_missiles) == 0, \
        "the hit missile must be REMOVED from local_missiles (so it gets no " \
        "lingering fog light): %r" % (g.ghost.local_missiles,)
    ix, iy = burst_center(g)
    d = math.hypot(ix - ROCK_POS[0], iy - ROCK_POS[1])
    assert abs(d - R_est) < 1.0, \
        "the missile spark must be on the circle boundary (d=%.2f, R_est=" \
        "%.2f)" % (d, R_est)
    print("PASS: MISSILE — a ghost missile on a rock course spawns a burst "
          "at the entry point (d=%.2f ~= R_est %.2f) and is removed from "
          "local_missiles (no lingering fog light)" % (d, R_est))

    # --- 7. TTL: stale cull entries are pruned, fresh ones kept ----------
    g = make_client()
    tmp, rock = make_host()
    push_stream(g, tmp)
    now = time.time()
    # Stale (older than the TTL) + fresh (within the TTL) entries in BOTH
    # cull sets.
    g._hit_fx_cull_bids[(LOCAL_INDEX, 99)] = now - (BULLET_LIFE + 0.5)
    g._hit_fx_cull_bids[(LOCAL_INDEX, 97)] = now - 0.1
    g._hit_fx_cull_mids[(LOCAL_INDEX, 98)] = now - (MISSILE_LIFE + 0.5)
    g._hit_fx_cull_mids[(LOCAL_INDEX, 96)] = now - 0.1
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())
    assert (LOCAL_INDEX, 99) not in g._hit_fx_cull_bids, \
        "a bullet cull entry older than BULLET_LIFE must be pruned: %r" \
        % (g._hit_fx_cull_bids,)
    assert (LOCAL_INDEX, 97) in g._hit_fx_cull_bids, \
        "a fresh bullet cull entry must be kept: %r" \
        % (g._hit_fx_cull_bids,)
    assert (LOCAL_INDEX, 98) not in g._hit_fx_cull_mids, \
        "a missile cull entry older than MISSILE_LIFE must be pruned: %r" \
        % (g._hit_fx_cull_mids,)
    assert (LOCAL_INDEX, 96) in g._hit_fx_cull_mids, \
        "a fresh missile cull entry must be kept: %r" \
        % (g._hit_fx_cull_mids,)
    print("PASS: TTL — cull entries older than BULLET_LIFE / MISSILE_LIFE are "
          "pruned (the sets are bounded); fresh entries are kept")

    print("ALL PASS: 18 (the client's option-A trigger covers rocks — "
          "bullets + missiles stop at the rock's circle with an entry-point "
          "spark; presentation only)")


if __name__ == "__main__":
    main()