"""Session 8.x backlog: profile the RENDER cost (fog/lighting) to find why
the host is capped at ~27 FPS.

Headless. Builds a real Game, warms the sim so the world is populated
(enemies/asteroids/bullets/particles), then:
  1. Times the WHOLE draw() over many frames (wall clock).
  2. Times the individual components (fog, HUD, world, local ship) in
     isolation over many frames.
  3. Runs cProfile on draw() for a per-function breakdown.

Run from /mnt:  python -m ship5.profile_render
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time

import pygame

from .config import WIDTH, HEIGHT, BG
from .fog import make_light_texture, draw_fog
from .game import Game, STEP, _build_lights_model
from .hud import draw_hud
from .intent import ShipInput
from .ai_enemy import AIEnemy
from .asteroid import Asteroid


class Keys:
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """Combat-ish input: thrust, turn, fire (bullets + missiles)."""
    return Keys(
        {pygame.K_q: 1 if (t // 30) % 3 == 0 else 0,
         pygame.K_e: 1 if (t // 30) % 3 == 2 else 0,
         pygame.K_w: 1 if (t // 60) % 2 == 0 else 0,
         pygame.K_a: 1 if (t // 15) % 2 == 0 else 0,
         pygame.K_d: 1 if (t // 15) % 2 == 1 else 0,
         pygame.K_SPACE: 1 if (t // 30) % 2 == 0 else 0,
         pygame.K_2: 1 if (t % 60) < 3 else 0,
         pygame.K_b: 1 if (t % 90) < 5 else 0})


def setup():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                seed=1234, players=1)
    return screen, font, big_font, light_tex, fog_surf, light_surf, game


def warm(game, ticks=600):
    """Step the sim so the world is populated (enemies/asteroids/bullets)."""
    for t in range(ticks):
        game._step(STEP, ShipInput.from_keys(script_input(t)))


def bench(label, fn, frames=200):
    """Time fn() over `frames` calls; return (mean_ms, min_ms, max_ms)."""
    # warm the caches / page faults
    fn()
    t0 = time.perf_counter()
    for _ in range(frames):
        fn()
    t1 = time.perf_counter()
    total = (t1 - t0) * 1000.0
    mean = total / frames
    return mean, total / frames, total / frames


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf, game = setup()
    warm(game)
    dt = 0.016

    # A representative published model to draw.
    model = game.render_model()
    local_pack = model["players"][game.local_index]
    standin = game._get_standin(game.local_index)
    standin.pos = pygame.Vector2(local_pack["pos"])
    standin.vel = pygame.Vector2(local_pack["vel"])
    standin.angle = local_pack["angle"]
    from .game import _sync_local_ship
    _sync_local_ship(standin, local_pack)
    lights = _build_lights_model(model, standin)

    print("=== entity counts (warmed world) ===")
    print("  asteroids   :", len(model["asteroids"]))
    print("  enemies     :", len(model["enemies"]))
    print("  bullets     :", len(model["bullets"]))
    print("  enemy_bullets:", len(model["enemy_bullets"]))
    print("  missiles    :", len(model["missiles"]))
    print("  particles   :", len(model["particles"]))
    print("  stars       :", len(model["stars"]))
    print("  fog lights  :", len(lights))
    print()

    # --- 1. whole draw() ---
    def full_draw():
        game.draw(dt, model)
    m, _, _ = bench("full draw", full_draw, frames=150)
    print("=== whole draw() ===")
    print("  mean per frame : %.2f ms  (=> %.1f FPS ceiling)" % (m, 1000.0 / m))
    print()

    # --- 2. components in isolation ---
    print("=== components (isolated, per frame) ===")

    def fog_only():
        draw_fog(screen, standin, game.cam, light_tex, fog_surf,
                 light_surf, lights)
    m, _, _ = bench("fog", fog_only, frames=150)
    print("  fog (draw_fog)   : %.2f ms" % m)

    def hud_only():
        draw_hud(screen, font, model["enemies"], standin)
    m, _, _ = bench("hud", hud_only, frames=150)
    print("  hud (draw_hud)   : %.2f ms" % m)

    def fill_only():
        screen.fill(BG)
    m, _, _ = bench("fill", fill_only, frames=150)
    print("  screen.fill(BG)  : %.2f ms" % m)

    # world entities only (no fog, no hud, no ship)
    def world_only():
        screen.fill(BG)
        from .game import (_draw_world_stars, _draw_world_asteroid,
                           _draw_world_enemy, _draw_world_bullet,
                           _draw_world_missile, _draw_world_particle)
        _draw_world_stars(screen, game.cam, model["stars"])
        standins = game._get_remote_enemies()
        for pos, angle, verts in model["asteroids"]:
            _draw_world_asteroid(screen, game.cam, pos, angle, verts)
        for (tag, _id, pos, angle, _vel, _acc, _cr, _poly,
             s_dump, s_clock) in model["enemies"]:
            _draw_world_enemy(screen, game.cam, tag, pos, angle, standins,
                              s_dump, s_clock)
        for pos, vel in model["bullets"]:
            _draw_world_bullet(screen, game.cam, pos, (255, 255, 255))
        for pos, vel, boost, life in model["missiles"]:
            _draw_world_missile(screen, game.cam, pos, vel, boost, life)
        for pos, vel, color, life, max_life in model["particles"]:
            _draw_world_particle(screen, game.cam, pos, vel, color, life,
                                 max_life)
    m, _, _ = bench("world", world_only, frames=150)
    print("  world entities   : %.2f ms" % m)

    # local ship only
    def ship_only():
        from .game import _draw_local_ship
        _draw_local_ship(screen, game.cam, standin, local_pack, 0.5)
    m, _, _ = bench("ship", ship_only, frames=150)
    print("  local ship       : %.2f ms" % m)

    print()
    print("  (fog + hud + world + ship + fill ~= the full draw cost)")
    print()

    # --- 3. combat case: force bullets into the world so the fog has
    #     point lights (the per-radius + intensity cached-tex path). ---
    print("=== combat case (bullets -> fog point lights) ===")
    # Nudge the sim forward with sustained fire so bullets are live.
    for t in range(120):
        game._step(STEP, ShipInput.from_keys(Keys(
            {pygame.K_w: 1, pygame.K_SPACE: 1, pygame.K_2: 1})))
    cmodel = game.render_model()
    clights = _build_lights_model(cmodel, standin)
    print("  bullets     :", len(cmodel["bullets"]))
    print("  enemy_bullets:", len(cmodel["enemy_bullets"]))
    print("  missiles    :", len(cmodel["missiles"]))
    print("  fog lights  :", len(clights))

    def combat_fog():
        draw_fog(screen, standin, game.cam, light_tex, fog_surf,
                 light_surf, clights)
    m, _, _ = bench("fog (combat, %d lights)" % len(clights), combat_fog,
                    frames=150)
    print("  fog per frame  : %.2f ms" % m)

    def combat_draw():
        game.draw(dt, cmodel)
    m, _, _ = bench("full draw (combat)", combat_draw, frames=150)
    print("  full draw      : %.2f ms  (=> %.1f FPS ceiling)"
          % (m, 1000.0 / m))


if __name__ == "__main__":
    main()