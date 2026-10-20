"""Session 15.3: the option-A trigger must fire when a BUFFER-drawn local
bullet is visually on the enemy (long-range shots).

Root cause (live 2P: close-range/ghost shots sparked, long-range/buffer
shots never did): the trigger's swept segment for a buffer bullet was
(x - vx*D) -> (x), but the bullet is DRAWN at (x + vx*D) (the pull-forward
to the present). The segment ended vD BEHIND the draw (vD = 860 * 0.15 ~=
130 px), so the hit only fired when the drawn bullet was already ~130 px
PAST the target — by then the visual contact had long passed. The ghost
bullet's segment (b.prev_pos -> b.pos) ends exactly at its draw (b.pos),
which is why close-range shots worked.

The fix makes the buffer segment (x, y) -> (x + vx*D, y + vy*D): it ends
at the drawn position, so the trigger fires exactly when the drawn bullet
reaches the enemy — the same rule as the ghost bullet.

This test drives predicted_view headless with a buffer bullet that MOVES
consistently (pos = x0 + vx*t in each snapshot, so the buffer's
predicted-position matching interpolates it exactly — a frozen bullet is
a degenerate case the matcher mishandles). The enemy sits at the
MIDPOINT of the trigger segment (x) -> (x + vx*D), so it is a hit for any
reasonable D; the OLD segment (x - vx*D) -> (x) ended vD behind the draw,
so the enemy (ahead of x) was never on it.

  1. BUFFER HIT  — enemy at the segment midpoint (shield down): a small
                   burst spawns, the buffer bullet is culled, CSV tags it
                   src='buf'. (With the old segment this MISSED.)
  2. GHOST HIT   — a ghost bullet at the enemy center still sparks (the
                   close-range path must not regress).
  3. BUFFER MISS — a buffer bullet whose segment ends 300 px short of
                   the enemy: no spark, not culled.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_15_3_buffer_bullet_hit
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import csv

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (HostTimeEstimator, LatencyTracker, RenderPoint,
                      PredictedShip)
from .ship import Ship
from .bullets import Bullet
from .hulls import PLAYER_HULLS
from . import hit_fx_telemetry

SEED = 1234
# The bullet's muzzle x (it flies +x toward the enemy at BVX).
X0 = WIDTH / 2 - 100.0
BVX = 860.0
# The enemy sits at the midpoint of the trigger segment (x) -> (x +
# vx*D). In this harness the render point and the ghost clock settle at
# rp = 0.1, D = _last_step_time - rp = 0.1, so the buffer bullet is at
# x0 + vx*rp and the segment midpoint is x0 + vx*(rp + D/2) =
# x0 + vx*0.15. The placement is robust to small rp/D drift (the
# segment is ~86-130 px long; the enemy stays inside it).
ENEMY_X = X0 + BVX * 0.15
ENEMY_POS = (ENEMY_X, HEIGHT / 2)
# Particle counts (particles.py): shield_burst n=16; burst(radius=6) is
# not `big` -> n=14.
HULL_BURST_N = 14


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def make_client(screen, font, big_font, light_tex, fog_surf, light_surf):
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    g.ghost = PredictedShip(hull=client_hull, loadout=None, local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _scratch_game():
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=None, seed=SEED, players=2)


def make_snap(shield_charge, bid, t):
    """A 2-player snapshot with ONE AI enemy at ENEMY_POS (fixed id 999)
    and ONE player bullet (owner 1, id `bid`) MOVING consistently:
    pos = (X0 + BVX*t, ey), vel = (BVX, 0) — so the buffer's
    predicted-position matching interpolates it exactly. The OTHER
    player's ship (index 0) is parked far away (100, 100) so the
    trigger's ship-test (which runs before the enemy-test) does not
    intercept the bullet on the way to the enemy."""
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(100.0, 100.0)
    tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.enemies.clear()
    from .ai_enemy import AIEnemy
    e = AIEnemy(pygame.Vector2(ENEMY_POS), rng=tmp.rng)
    e.ship.id = 999
    e.ship.shield_charge = shield_charge
    tmp.enemies.append(e)
    tmp.bullets.append(Bullet(pygame.Vector2(X0 + BVX * t, HEIGHT / 2),
                              pygame.Vector2(BVX, 0), owner=1, bid=bid))
    return tmp.snapshot()


def push_stream(g, shield_charge, bid):
    """Reset buffer + ghost + render point, push a 3-snapshot stream at
    10 Hz with the enemy frozen and the bullet moving consistently."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(shield_charge, bid, t), now=t)


def seed_ghost_bullet(g, pos, bid):
    """A ghost local bullet as a POINT segment (vel=0, prev=pos) at
    `pos` — the same deterministic-segment trick as test_10_13."""
    b = Bullet(pygame.Vector2(pos), pygame.Vector2(0, 0), owner=1,
               bid=bid)
    b.prev_pos = pygame.Vector2(pos)
    b.pos = pygame.Vector2(pos)
    g.ghost.local_bullets = [b]
    return b


def render_frame(g):
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def read_hits():
    try:
        rows = list(csv.DictReader(open(hit_fx_telemetry._path)))
    except FileNotFoundError:
        return []
    return [r for r in rows if r["event"] == "HIT"]


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g = make_client(screen, font, big_font, light_tex, fog_surf,
                    light_surf)
    ex, ey = ENEMY_POS

    # --- 1. BUFFER HIT: the enemy sits at the midpoint of the buffer
    #     bullet's trigger segment (x) -> (x + vx*D). No ghost bullet
    #     with this id (handed off), so the trigger tests the BUFFER
    #     copy — the live long-range case. With the old segment
    #     ((x - vx*D) -> (x), ending vD behind the draw) this MISSED. ---
    push_stream(g, shield_charge=0.0, bid=(1, 7))
    g.ghost.local_bullets = []   # handed off: the buffer draws + is tested
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == HULL_BURST_N, \
        "a buffer bullet whose segment crosses the enemy must spawn a " \
        "small burst (%d particles): got %d" \
        % (HULL_BURST_N, len(g._hit_fx))
    assert (1, 7) in g._hit_fx_cull_bids, \
        "the hitting buffer bullet must be marked for cull: %r" \
        % (g._hit_fx_cull_bids,)
    print("PASS: BUFFER HIT — a buffer bullet whose segment crosses the "
          "enemy sparks and is culled")

    # --- 2. GHOST HIT: a ghost bullet at the enemy center still sparks
    #     (the close-range path must not regress). ---
    push_stream(g, shield_charge=0.0, bid=(1, 8))
    seed_ghost_bullet(g, (ex, ey), bid=(1, 8))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == HULL_BURST_N, \
        "a ghost bullet at the enemy center must spawn a small burst " \
        "(%d particles): got %d" % (HULL_BURST_N, len(g._hit_fx))
    assert (1, 8) in g._hit_fx_cull_bids, \
        "the hitting ghost bullet must be marked for cull: %r" \
        % (g._hit_fx_cull_bids,)
    print("PASS: GHOST HIT — a ghost bullet at the enemy center still "
          "sparks and is culled")

    # --- 3. BUFFER MISS: a buffer bullet whose segment ends 300 px
    #     short of the enemy: no spark, not culled. (The muzzle is 300
    #     px further back, so the whole segment is >= 300 - vD short.) ---
    push_stream(g, shield_charge=0.0, bid=(1, 9))
    g.ghost.local_bullets = []
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    # Move the enemy 300 px ahead of where the segment ends: the segment
    # (x) -> (x + vx*D) ends at ~X0 + vx*0.2; the enemy is at
    # ENEMY_X + 300, far beyond it.
    g.snap_buf = g.snap_buf.__class__()
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        tmp = _scratch_game()
        tmp.players[0].pos = pygame.Vector2(100.0, 100.0)
        tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        tmp.enemies.clear()
        from .ai_enemy import AIEnemy
        e = AIEnemy(pygame.Vector2(ENEMY_X + 300.0, HEIGHT / 2),
                    rng=tmp.rng)
        e.ship.id = 999
        e.ship.shield_charge = 0.0
        tmp.enemies.append(e)
        tmp.bullets.append(Bullet(pygame.Vector2(X0 + BVX * t, HEIGHT / 2),
                                  pygame.Vector2(BVX, 0), owner=1,
                                  bid=(1, 9)))
        g.push_snapshot(t, tmp.snapshot(), now=t)
    render_frame(g)
    assert len(g._hit_fx) == 0, \
        "a buffer bullet 300 px short of the enemy must NOT spark: " \
        "%r" % (g._hit_fx,)
    assert (1, 9) not in g._hit_fx_cull_bids, \
        "a missing buffer bullet must NOT be culled: %r" \
        % (g._hit_fx_cull_bids,)
    print("PASS: BUFFER MISS — a buffer bullet 300 px short of the "
          "enemy does not spark and is not culled")

    # --- 4. TELEMETRY: the buffer hit is tagged src='buf' in the CSV. ---
    hit_fx_telemetry.set_enabled(True)
    try:
        push_stream(g, shield_charge=0.0, bid=(1, 10))
        g.ghost.local_bullets = []
        g._hit_fx = []
        g._hit_fx_cull_bids.clear()
        render_frame(g)
        hits = read_hits()
        assert any(r["src"] == "buf" and r["bid"] == "1:10"
                   and r["tgt"] == "enemy:ai" for r in hits), \
            "the CSV must record a src='buf' HIT for the buffer bullet: " \
            "%r" % (hits,)
    finally:
        hit_fx_telemetry.set_enabled(False)
        hit_fx_telemetry.close()
    print("PASS: TELEMETRY — the CSV records a src='buf' HIT row")

    print("ALL PASS: 15.3 (the option-A trigger fires when a "
          "buffer-drawn bullet is visually on the enemy)")


if __name__ == "__main__":
    main()