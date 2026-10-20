"""Session 15.4: the client must spawn a shield_burst when an INCOMING
(enemy) bullet hits the LOCAL ship.

The client never runs the sim, so it never sees the host's `register_hit` /
`shield_burst` for a hit on the local ship. The flash the player already
sees is the authoritative `shield_dump` re-anchored from the snapshot at
each reconcile (and decayed between by the ghost's own `_update_shield`).
A jump UP in the ghost's `shield_dump` is therefore the hit signal: it fires
in the same frame the flash appears, so the burst is on time with it.

The impact point is the nearest incoming buffer bullet (kind 'enemy', NOT
the local index) projected onto the shield oval — the same
`shield_impact_point` math the host's `_handle_ship_hit` uses.

  1. HIT        — the local ship's shield_dump spikes 0 -> power_hit with an
                  enemy bullet in front: a shield_burst (16 particles)
                  spawns at the oval front (the projected impact point).
  2. NO HIT     — the dump stays 0: no burst.
  3. SUSTAINED  — the dump is re-anchored at power_hit on the next snapshot
                  (a sustained hit, not a new one): NO second burst.

Run from the repo root (the directory that CONTAINS workspace/):

    python -m workspace.test_15_4_incoming_hit_fx
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (HostTimeEstimator, LatencyTracker, RenderPoint,
                      PredictedShip)
from .ship import Ship
from .bullets import EnemyBullet
from .hulls import PLAYER_HULLS

SEED = 1234
# The local ship (index 1) sits here, facing +x (angle 0).
SHIP_POS = (WIDTH / 2, HEIGHT / 2)
# The enemy bullet flies +x toward the ship. It is placed so that, at the
# hit frame, it is just in front of the ship (inside the oval's front), so
# `shield_impact_point` projects it onto the oval front.
EBVX = 200.0
# shield_burst spawns n=16 particles (particles.py).
SHIELD_BURST_N = 16
# The oval front point (angle 0, oval a=30, b=22, center (0,0)): the ship
# pos + (a, 0). A bullet in front projects here.
IMPACT_FRONT = (SHIP_POS[0] + 30.0, SHIP_POS[1])


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


def make_snap(dump, t, ebullet=True):
    """A 2-player snapshot with the LOCAL ship (index 1) at SHIP_POS facing
    +x and `shield_dump = dump`, the other player (index 0) parked far away,
    and (optionally) ONE enemy bullet (owner 0) moving +x consistently
    (pos = x0 + EBVX*t) so the buffer's predicted-position matching
    interpolates it exactly. No AI enemies (this test is about the local
    ship, not the outgoing trigger)."""
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(100.0, 100.0)
    tmp.players[1].pos = pygame.Vector2(*SHIP_POS)
    tmp.players[1].angle = 0.0
    tmp.players[1].shield_dump = dump
    tmp.enemies.clear()
    tmp.bullets.clear()
    tmp.enemy_bullets.clear()
    if ebullet:
        # Muzzle so the bullet is just in front of the ship at t.
        x0 = SHIP_POS[0] + 10.0 - EBVX * t
        tmp.enemy_bullets.append(
            EnemyBullet(pygame.Vector2(x0, SHIP_POS[1]),
                        pygame.Vector2(EBVX, 0), owner=0))
    return tmp.snapshot()


def reset_client(g):
    """Reset buffer + ghost + render point + the dump tracker."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    g._last_ghost_shield_dump = 0.0
    g._hit_fx = []


def push_and_render(g, dump, t):
    """One snapshot arriving (the local ship's shield_dump = `dump`), then
    the frame renders — the real game's order (push_snapshot in the poll
    loop, then predicted_view)."""
    g.push_snapshot(t, make_snap(dump, t), now=t)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    power_hit = g.ghost.ship.shield_comp.power_hit

    # --- 1. HIT: dump spikes 0 -> power_hit, enemy bullet in front. -------
    # Snapshots 1-2 have dump=0 (no hit); snapshot 3 spikes to power_hit.
    reset_client(g)
    push_and_render(g, 0.0, 0.0)    # dump=0
    push_and_render(g, 0.0, 0.1)    # dump=0
    assert len(g._hit_fx) == 0, \
        "no burst before the dump spikes (got %d)" % len(g._hit_fx)
    push_and_render(g, power_hit, 0.2)   # dump=power_hit -> SPIKE -> burst
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a dump spike (0 -> power_hit) must spawn a shield_burst of %d " \
        "particles (got %d)" % (SHIELD_BURST_N, len(g._hit_fx))
    # The burst must be at the projected impact point (the oval front, since
    # the bullet is in front of the ship).
    bx = sum(p.pos.x for p in g._hit_fx) / len(g._hit_fx)
    by = sum(p.pos.y for p in g._hit_fx) / len(g._hit_fx)
    err = ((bx - IMPACT_FRONT[0]) ** 2 + (by - IMPACT_FRONT[1]) ** 2) ** 0.5
    assert err < 8.0, \
        "the burst must land on the shield oval front %s (got (%.1f, %.1f), " \
        "err %.1f)" % (IMPACT_FRONT, bx, by, err)
    print("PASS: HIT — a shield_dump spike (0 -> power_hit) with an enemy "
          "bullet in front spawns a shield_burst at the oval impact point")

    # --- 3. SUSTAINED: the dump is re-anchored at power_hit on the next ---
    # snapshots (a sustained hit, not a new one). No second burst.
    push_and_render(g, power_hit, 0.3)   # re-anchored -> no new spike
    push_and_render(g, power_hit, 0.4)   # re-anchored -> no new spike
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a re-anchored (sustained) dump must NOT spawn a second burst " \
        "(got %d)" % len(g._hit_fx)
    print("PASS: SUSTAINED — a re-anchored dump (power_hit -> power_hit) "
          "does not re-fire the burst")

    # --- 2. NO HIT: the dump stays 0. -------------------------------------
    reset_client(g)
    for k in range(5):
        push_and_render(g, 0.0, k * 0.1)
    assert len(g._hit_fx) == 0, \
        "no dump spike -> no burst (got %d)" % len(g._hit_fx)
    print("PASS: NO HIT — a flat dump (0 -> 0) spawns no burst")

    print("ALL PASS: 15.4 (the client spawns a shield_burst when an "
          "incoming bullet hits the local ship, on the flash frame)")


if __name__ == "__main__":
    main()