"""Session 15.2: the client's option-A shield-oval hit test must not be
SINGULAR at the diagonal enemy angles.

Root cause (found via hit_fx_debug.csv: 33/46 HIT rows were far away,
d up to 1825 px, all with s_up=1): the option-A trigger's inline
shield-oval test (game.predicted_view) built the world->local transform
as

    A = dx*cos + dy*(-sin)     # first row used _rgt_x (= -sin)
    B = dx*(-sin) + dy*cos

i.e. the matrix [[cos, -sin], [-sin, cos]] whose determinant is
cos(2*theta). That is SINGULAR at theta = 45/135/225/315 deg: the
world direction (1,1) (the "null direction") maps to (0,0) at ANY
distance, so a bullet 500 px away along that direction passed the oval
test and spawned a shield spark + cull even though it was nowhere near
the target. The impact-point block right below it used the CORRECT
_fwd_y, which is why the sparks still *landed* plausibly.

The fix makes the first row use _fwd_y (= +sin), giving the orthonormal
rotation [[cos, sin], [-sin, cos]] (det 1) — identical to the host's
canonical Ship.shield_contains. This test guards that:

  1. SINGULAR FAR MISS — enemy facing 45 deg, a bullet 300 px away along
     the null direction (1,1). Must NOT spark and must NOT be culled.
     (With the bug this fired a spark — the regression.)
  2. SINGULAR CLOSE HIT — enemy facing 45 deg, a bullet at the enemy
     center (distance 0, inside the oval). Must spark + cull, proving the
     fix did not break genuine hits at the singular angle.
  3. ALL FOUR DIAGONALS — the far-miss case repeated at 135/225/315 deg.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_15_2_shield_oval_singular
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (HostTimeEstimator, LatencyTracker, RenderPoint,
                      PredictedShip)
from .ship import Ship
from .bullets import Bullet
from .hulls import PLAYER_HULLS

SEED = 1234
ENEMY_POS = (WIDTH / 2 + 300.0, HEIGHT / 2)
# A bullet this far from the enemy can never be a real hit (the shield
# oval's largest semi-axis is ~30 px). 300 px is well outside it.
FAR_DIST = 300.0
# Particle counts (particles.py): shield_burst n=16.
SHIELD_BURST_N = 16


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


def make_snap(enemy_pos, enemy_angle, shield_charge):
    """A 2-player snapshot with ONE AI enemy at `enemy_pos` facing
    `enemy_angle` (radians) with `shield_charge`. A FIXED ship id (999)
    so all snapshots carry the SAME enemy (the buffer matches by id)."""
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.enemies.clear()
    from .ai_enemy import AIEnemy
    e = AIEnemy(pygame.Vector2(enemy_pos), rng=tmp.rng)
    e.ship.id = 999
    e.ship.angle = enemy_angle
    e.ship.shield_charge = shield_charge
    tmp.enemies.append(e)
    return tmp.snapshot()


def push_stream(g, enemy_pos, enemy_angle, shield_charge):
    """Reset buffer + ghost + render point, push a 3-snapshot stream at
    10 Hz with the enemy FROZEN at `enemy_pos` facing `enemy_angle` (the
    same pose in every snapshot, so the buffer lerps it to exactly that
    pose)."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(enemy_pos, enemy_angle, shield_charge),
                        now=t)


def seed_bullet(g, pos, bid):
    """Give the ghost a single local bullet as a POINT segment (vel=0,
    prev=pos) at `pos`. predicted_view advances the ghost before the
    trigger; with vel=0 the bullet stays exactly where it is, so the
    segment the trigger sees is deterministic (it degenerates to `pos`)."""
    b = Bullet(pygame.Vector2(pos), pygame.Vector2(0, 0), owner=1, bid=bid)
    b.prev_pos = pygame.Vector2(pos)
    b.pos = pygame.Vector2(pos)
    g.ghost.local_bullets = [b]
    return b


def render_frame(g):
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    ex, ey = ENEMY_POS

    # --- 1. SINGULAR FAR MISS: enemy facing 45 deg, a bullet 300 px away
    #     along the null direction (1,1). With the singular transform this
    #     mapped to (0,0) and fired a spark; the fixed orthonormal transform
    #     must reject it (300 px >> the ~30 px oval). ---
    for facing_deg in (45, 135, 225, 315):
        facing = math.radians(facing_deg)
        # The null direction of the buggy matrix at facing `facing` is the
        # world direction (cos facing, sin facing) — i.e. straight along the
        # enemy's own facing. Place the bullet FAR along it.
        bx = ex + FAR_DIST * math.cos(facing)
        by = ey + FAR_DIST * math.sin(facing)
        push_stream(g, ENEMY_POS, facing, shield_charge=3.0)
        seed_bullet(g, (bx, by), bid=(1, facing_deg))
        g._hit_fx = []
        g._hit_fx_cull_bids.clear()
        render_frame(g)
        assert len(g._hit_fx) == 0, \
            "enemy facing %d deg: a bullet %.0f px away along the null " \
            "direction must NOT spark (singular transform): %r" \
            % (facing_deg, FAR_DIST, g._hit_fx)
        assert (1, facing_deg) not in g._hit_fx_cull_bids, \
            "enemy facing %d deg: a far bullet must NOT be culled: %r" \
            % (facing_deg, g._hit_fx_cull_bids)
        remaining = [tuple(b.id) for b in g.ghost.local_bullets
                     if b.id is not None]
        assert (1, facing_deg) in remaining, \
            "enemy facing %d deg: a far bullet must stay in " \
            "local_bullets: %r" % (facing_deg, remaining)
    print("PASS: SINGULAR FAR MISS — a bullet 300 px away along the null "
          "direction does not spark at 45/135/225/315 deg")

    # --- 2. SINGULAR CLOSE HIT: enemy facing 45 deg, a bullet at the enemy
    #     center (distance 0, inside the oval). Must still spark + cull —
    #     the fix must not break genuine hits at the singular angle. ---
    facing = math.radians(45)
    push_stream(g, ENEMY_POS, facing, shield_charge=3.0)
    seed_bullet(g, (ex, ey), bid=(1, 900))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a bullet at the enemy center (shield up) must spawn a " \
        "shield_burst (%d particles): got %d" \
        % (SHIELD_BURST_N, len(g._hit_fx))
    assert (1, 900) in g._hit_fx_cull_bids, \
        "a hitting bullet must be marked for cull: %r" \
        % (g._hit_fx_cull_bids,)
    remaining = [tuple(b.id) for b in g.ghost.local_bullets
                 if b.id is not None]
    assert (1, 900) not in remaining, \
        "a hitting bullet must be removed from local_bullets: %r" \
        % (remaining,)
    print("PASS: SINGULAR CLOSE HIT — a bullet at the enemy center still "
          "sparks + is culled at the singular angle")

    # --- 3. NON-SINGULAR CONTROL: enemy facing 0 deg (det=1, not
    #     singular). A far bullet along (1,0) must miss; a center bullet
    #     must hit. Confirms the fix behaves normally off the diagonals. ---
    facing = 0.0
    push_stream(g, ENEMY_POS, facing, shield_charge=3.0)
    seed_bullet(g, (ex + FAR_DIST, ey), bid=(1, 901))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == 0, \
        "facing 0 deg: a far bullet must not spark: %r" % (g._hit_fx,)
    push_stream(g, ENEMY_POS, facing, shield_charge=3.0)
    seed_bullet(g, (ex, ey), bid=(1, 902))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "facing 0 deg: a center bullet must spark: %r" % (g._hit_fx,)
    print("PASS: NON-SINGULAR CONTROL — facing 0 deg behaves normally "
          "(far misses, center hits)")

    print("ALL PASS: 15.2 (option-A shield-oval test is orthonormal, not "
          "singular, at the diagonal enemy angles)")


if __name__ == "__main__":
    main()