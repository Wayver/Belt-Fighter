"""Determinism check: same seed -> identical sim state after N ticks.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_determinism

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens. Three runs:
  A: seed 1234
  B: seed 1234   (must be bit-identical to A, including rng.getstate())
  C: seed 9999   (must differ from A)

The snapshot covers ship, asteroids (pos/vel/size/angle/verts), enemies,
all projectile lists, and the rng state itself — so a mismatch pinpoints
which subsystem diverged.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid

TICKS = 600   # 10 simulated seconds at 60 Hz


class Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """A canned input pattern that exercises every fire path:
    turn cycles, thrust bursts, bullets, lasers, missiles, stop."""
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


def run(seed):
    # reset the class-level enemy id counter so runs A/B are truly
    # identical (ids are cosmetic, but let's not rely on that)
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=seed)
    for t in range(TICKS):
        g.update(STEP, script_input(t))
    return snapshot(g)


def main():
    pygame.init()
    global screen, font, big_font, light_tex, fog_surf, light_surf
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    a = run(1234)
    b = run(1234)
    c = run(9999)

    ok = True
    if a == b:
        print(f"PASS: seed 1234 reproducible "
              f"({len(a[1])} rocks, {len(a[2])} enemies)")
    else:
        ok = False
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                names = ["ship", "asteroids", "enemies", "bullets",
                         "enemy_bullets", "missiles", "particles",
                         "rng_state"]
                print(f"FAIL: field {names[i]} differs between runs")
                if i in (0, 7):
                    print(f"  A: {x}\n  B: {y}")
                else:
                    print(f"  A: {x[:3]}...\n  B: {y[:3]}...")
    if a != c:
        print("PASS: seed 9999 diverges from seed 1234")
    else:
        ok = False
        print("FAIL: different seeds produced identical state")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
