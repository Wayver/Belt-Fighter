"""Phase 5: profile the SIM TICK cost (Game._step) at the chosen burst rate.

The §8 risk was that more live bullets -> more bullet-vs-asteroid/ship
collision checks per tick. This measures the actual tick cost under
sustained fire (the burst pattern) and reports it against the 16.67 ms
(60 Hz) tick budget.

Headless. Builds a real 2P Game, warms the world, then times _step() over
many ticks of sustained fire and reports mean/max ms per tick + the live
bullet count.

Run from /mnt:  python -m ship5.profile_tick
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time

import pygame

from .config import WIDTH, HEIGHT, TICK, BULLET_LIFE, BULLET_POOL_MAX
from .fog import make_light_texture
from .game import Game, STEP
from .intent import ShipInput
from .ai_enemy import AIEnemy
from .asteroid import Asteroid


class Keys:
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def sustained_input(t):
    """Hold fire + a little motion so the world stays populated."""
    return Keys({pygame.K_w: 1, pygame.K_SPACE: 1,
                 pygame.K_a: 1 if (t // 30) % 2 == 0 else 0,
                 pygame.K_d: 1 if (t // 30) % 2 == 1 else 0})


def setup(players=2):
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
                seed=1234, players=players)
    return game


def warm(game, ticks=600):
    for t in range(ticks):
        game._step(STEP, ShipInput.from_keys(sustained_input(t)))


def main():
    game = setup(players=2)
    warm(game)

    # Warm the timing caches / page faults.
    for _ in range(20):
        game._step(STEP, ShipInput.from_keys(sustained_input(0)))

    n = 600
    samples = []
    max_bullets = 0
    max_asteroids = 0
    t0 = time.perf_counter()
    for i in range(n):
        # Drive BOTH players to fire (the §8 worst case: 26 world bullets).
        game.set_remote_input(ShipInput.from_keys(sustained_input(i)))
        t_a = time.perf_counter()
        game._step(STEP, ShipInput.from_keys(sustained_input(i)))
        t_b = time.perf_counter()
        samples.append((t_b - t_a) * 1000.0)
        max_bullets = max(max_bullets, len(game.bullets))
        max_asteroids = max(max_asteroids, len(game.asteroids))
    wall = (time.perf_counter() - t0) * 1000.0

    samples.sort()
    mean = sum(samples) / len(samples)
    p50 = samples[len(samples) // 2]
    p99 = samples[int(len(samples) * 0.99)]
    mx = samples[-1]

    print("=== SIM TICK profile (2P, sustained fire, burst rate) ===")
    print("  ticks measured : %d" % n)
    print("  live bullets   : max %d (pool max %d, natural ~26 world)"
          % (max_bullets, BULLET_POOL_MAX))
    print("  asteroids      : max %d" % max_asteroids)
    print("  tick budget    : %.2f ms (60 Hz)" % (1000.0 / 60.0))
    print()
    print("  mean per tick  : %.3f ms" % mean)
    print("  p50 per tick   : %.3f ms" % p50)
    print("  p99 per tick   : %.3f ms" % p99)
    print("  max per tick   : %.3f ms" % mx)
    print("  wall (n ticks) : %.1f ms  (=> %.1f ticks/s sustained)"
          % (wall, n * 1000.0 / wall))
    print()
    if mean < 1000.0 / 60.0:
        print("  WITHIN BUDGET — mean tick %.3f ms < %.2f ms budget"
              % (mean, 1000.0 / 60.0))
    else:
        print("  OVER BUDGET — mean tick %.3f ms >= %.2f ms budget"
              % (mean, 1000.0 / 60.0))


if __name__ == "__main__":
    main()