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

    # --- 6. RESPAWN: respawn_player(i) revives ONE ship, leaves the rest ---
    g4 = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                   SEED, players=2)
    g4.players[1].pos = pygame.Vector2(WIDTH / 2 + 300, HEIGHT / 2)
    g4.protect_timer = 0.0
    q0, q1 = g4.players
    q0.shield_charge = 1.0
    g4._handle_ship_hit(q0, q0.pos)   # absorb
    g4._handle_ship_hit(q0, q0.pos)   # lethal
    assert q0.dead is True
    # Record the world + the other player before the respawn.
    p1_before = pose(q1)
    rocks_before = tuple(sorted((round(a.pos.x, 3), round(a.pos.y, 3), a.size)
                                for a in g4.asteroids))
    enemies_before = len(g4.enemies)
    g4.respawn_player(0)
    assert q0.dead is False, "respawn must clear the dead flag"
    assert round(q0.pos.x, 3) == WIDTH / 2 and round(q0.pos.y, 3) == HEIGHT / 2
    assert q0.shield_charge == q0.shield_comp.shield_max_charge
    assert g4.protect_timer == SPAWN_PROTECT
    assert pose(q1) == p1_before, "respawn must not touch the other player"
    rocks_after = tuple(sorted((round(a.pos.x, 3), round(a.pos.y, 3), a.size)
                               for a in g4.asteroids))
    assert rocks_after == rocks_before, "respawn must not touch the world"
    assert len(g4.enemies) == enemies_before
    # No-op on a live ship (a stray R is harmless).
    p1_live = pose(q1)
    g4.respawn_player(1)
    assert pose(q1) == p1_live and q1.dead is False
    print("PASS: RESPAWN — one ship revives, the world + other player untouched")

    # --- 7. SIM-THREAD: the 'respawn' command routes to respawn_player ---
    from .sim_thread import SimThread
    g5 = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                   SEED, players=2)
    g5.players[0].shield_charge = 1.0
    g5._handle_ship_hit(g5.players[0], g5.players[0].pos)
    g5._handle_ship_hit(g5.players[0], g5.players[0].pos)
    assert g5.players[0].dead is True
    st = SimThread(g5, worker=None)
    st.start()
    st.command("respawn", 0)
    # The sim thread applies commands at the top of its next iteration
    # (~1 ms); poll for it.
    import time
    deadline = time.monotonic() + 2.0
    while g5.players[0].dead and time.monotonic() < deadline:
        time.sleep(0.005)
    st.stop()
    assert g5.players[0].dead is False, "the sim thread must apply the command"
    print("PASS: SIM-THREAD — the 'respawn' command revives the ship")

    # --- 8. GHOST: a dead ghost does not advance (no prediction -> no
    #     snapping). The host freezes a dead ship, so the authoritative
    #     snapshot is static; predicting it with local input would race
    #     ahead and every reconcile yank it back. The ghost must freeze. ---
    from .netcode import PredictedShip
    from .ship import Ship
    ghost = PredictedShip()
    ghost.seed(Ship().snapshot())
    # Baseline: an ALIVE ghost advances under thrust.
    ghost._acc = 0.0
    ghost.ship.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    ghost.ship.vel = pygame.Vector2(0, 0)
    thrust = ShipInput(thrust_fwd=1.0)
    for _ in range(30):
        ghost.advance(STEP, thrust)
    alive_moved = ghost.ship.pos.distance_to(
        pygame.Vector2(WIDTH / 2, HEIGHT / 2))
    assert alive_moved > 1.0, "alive ghost must advance under thrust"
    # Now dead: the ghost must NOT advance (no prediction, no snap).
    ghost.ship.dead = True
    ghost._acc = 0.0
    frozen = (ghost.ship.pos.x, ghost.ship.pos.y, ghost.ship.angle)
    steps = 0
    for _ in range(60):
        steps += ghost.advance(STEP, thrust)
    assert steps == 0, "a dead ghost must take 0 steps"
    assert (ghost.ship.pos.x, ghost.ship.pos.y, ghost.ship.angle) == frozen, \
        "a dead ghost must be frozen (no prediction -> no snapping)"
    # Respawn: dead cleared + a fresh authoritative snapshot -> the ghost
    # advances again from the new pose.
    ghost.ship.dead = False
    ghost.seed(Ship().snapshot())   # fresh authoritative pose
    ghost._acc = 0.0
    ghost.ship.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    ghost.ship.vel = pygame.Vector2(0, 0)
    for _ in range(30):
        ghost.advance(STEP, thrust)
    assert ghost.ship.pos.distance_to(
        pygame.Vector2(WIDTH / 2, HEIGHT / 2)) > 1.0, \
        "a respawned ghost must advance again"
    print("PASS: GHOST — a dead ghost is frozen (no prediction -> no snap)")

    # --- 9. MODEL + DRAW: the dead flag reaches the render model, and
    #     draw() shows the respawn screen ONLY to the dead player ---
    def center_pixels(s):
        r = s.get_rect()
        cx, cy = r.w // 2, r.h // 2
        return pygame.image.tostring(
            s.subsurface(cx - 120, cy - 60, 240, 120), "RGB")
    g6 = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                   SEED, players=2)
    for _ in range(10):
        g6._step(STEP, IDLE)
    g6.protect_timer = 0.0
    # Baseline: both alive -> no screen.
    g6.players[0].dead = False
    g6.players[1].dead = False
    g6.cam.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    m_alive = g6.render_model()
    assert m_alive["players"][0]["dead"] is False
    g6.draw(0.016, model=m_alive)
    center_alive = center_pixels(screen)
    # Local (player 0) dead -> the respawn screen is drawn.
    g6.players[0].dead = True
    g6.cam.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    m_dead = g6.render_model()
    assert m_dead["players"][0]["dead"] is True
    g6.draw(0.016, model=m_dead)
    center_dead = center_pixels(screen)
    assert center_dead != center_alive, \
        "respawn screen must be drawn for the dead local player"
    # Remote (player 1) dead, local alive -> NO screen (the other player
    # keeps playing). The center is back to the alive baseline.
    g6.players[0].dead = False
    g6.players[1].dead = True
    g6.cam.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    m_remote = g6.render_model()
    assert m_remote["players"][1]["dead"] is True
    g6.draw(0.016, model=m_remote)
    center_remote = center_pixels(screen)
    assert center_remote == center_alive, \
        "no screen for the alive local player (the other keeps playing)"
    print("PASS: MODEL+DRAW — the respawn screen shows only to the dead player")

    print("ALL PASS: 10.1 Steps 1-4 (per-player death + respawn + ghost freeze + screen)")


if __name__ == "__main__":
    main()