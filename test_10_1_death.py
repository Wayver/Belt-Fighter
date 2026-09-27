"""Session 10.1 Step 1: per-player death in the sim.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_1_death

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.1 makes death PER-PLAYER, not global. Before 10.1, `_handle_ship_hit`
set `self.game_over = True` for ANY player's death, which froze the ENTIRE
sim (all ships, the field, the collisions) — so in 2P, one player's death
froze the other player's ship too (the reported "freezing and snapping" on
the client, and the host's game-over screen when the client died).

This test proves the sim-level half of the fix (the client-side ghost
half is Step 3):

  1. PER-PLAYER  — in 2P, killing player 0 leaves player 1 still stepping
                   (moving under input) and the game NOT over.
  2. FROZEN      — the dead ship is frozen: its pose is constant and it
                   takes no hits (a ram does not re-trigger the death
                   path).
  3. ENEMIES     — enemies do not target the corpse (they target the
                   nearest ALIVE player).
  4. ROUND-TRIP  — the `dead` flag survives a snapshot -> apply_snapshot
                   round-trip (the wire carries it, so the client's ghost
                   can read it — Step 3).
  5. SP-UNCHANGED— single player: a death still sets game_over (the
                   existing behavior, which test_determinism also pins).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import WIDTH, HEIGHT, SPAWN_PROTECT
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .intent import ShipInput

SEED = 1234
IDLE = ShipInput()


def make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed, players=1):
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                seed=seed, players=players)


def pose(p):
    return (round(p.pos.x, 6), round(p.pos.y, 6), round(p.angle, 6),
            round(p.vel.x, 6), round(p.vel.y, 6))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # --- 1. PER-PLAYER: kill player 0 in 2P; player 1 keeps stepping ---
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED, players=2)
    g.players[1].pos = pygame.Vector2(WIDTH / 2 + 200, HEIGHT / 2)
    g.players[1].angle = 0.0
    g.protect_timer = 0.0
    # Kill player 0 through the REAL death path (shield drained ->
    # register_hit returns False -> _handle_ship_hit marks it dead).
    p0, p1 = g.players
    p0.shield_charge = 1.0
    assert g._handle_ship_hit(p0, p0.pos) is True    # shield absorbs
    assert g._handle_ship_hit(p0, p0.pos) is False   # lethal
    assert p0.dead is True
    assert g.game_over is False, "2P death must NOT set the global flag"
    # Player 1 (alive) still steps: it reads self.remote_input (index 1);
    # give it thrust and confirm its pose changes while the corpse's
    # (player 0's) input slot stays idle.
    thrust = ShipInput(thrust_fwd=1.0)
    g.remote_input = thrust
    p1_mid = pose(p1)
    for _ in range(30):
        g._step(STEP, IDLE)   # local slot (player 0, dead) stays idle
    p1_after = pose(p1)
    assert p1_after != p1_mid, "alive player 1 must keep stepping"
    # The dead ship is FROZEN: constant pose across steps.
    p0_a = pose(p0)
    for _ in range(30):
        g._step(STEP, IDLE)
    assert pose(p0) == p0_a, "dead ship must be frozen"
    print("PASS: PER-PLAYER — 2P death leaves the other player stepping")

    # --- 2. FROZEN: the corpse takes no hits (a ram does not re-trigger
    #     the death path) ---
    g.protect_timer = 0.0
    g.enemies.clear()
    g.asteroids.clear()
    # An enemy ramming the corpse: _collisions must skip the dead ship.
    e = AIEnemy(pygame.Vector2(p0.pos.x + 10, p0.pos.y), rng=g.rng)
    e.ship.pos = p0.pos.copy()   # co-located -> overlap is True
    g.enemies.append(e)
    g._collisions()
    assert p0.dead is True and g.game_over is False
    # And the corpse's pose is unchanged by the collision pass.
    assert pose(p0) == p0_a, "corpse must not move on a collision pass"
    print("PASS: FROZEN — the corpse takes no hits")

    # --- 3. ENEMIES: they target the nearest ALIVE player ---
    target = g._nearest_player(e.pos)
    assert target is p1, "enemies must target the alive player, not the corpse"
    print("PASS: ENEMIES — the corpse is not targeted")

    # --- 4. ROUND-TRIP: the dead flag survives snapshot -> apply ---
    snap = g.snapshot()
    g2 = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                   SEED, players=2)
    g2.apply_snapshot(snap)
    assert g2.players[0].dead is True, "dead flag must round-trip"
    assert g2.players[1].dead is False
    assert g2.game_over is False
    print("PASS: ROUND-TRIP — the dead flag survives snapshot -> apply")

    # --- 5. SP-UNCHANGED: single player death still sets game_over ---
    g3 = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                   SEED, players=1)
    s = g3.players[0]
    s.shield_charge = 1.0
    assert g3._handle_ship_hit(s, s.pos) is True
    assert g3._handle_ship_hit(s, s.pos) is False
    assert s.dead is True
    assert g3.game_over is True, "single player death must still end the game"
    print("PASS: SP-UNCHANGED — single player death still sets game_over")

    print("ALL PASS: 10.1 Step 1 (per-player death in the sim)")


if __name__ == "__main__":
    main()