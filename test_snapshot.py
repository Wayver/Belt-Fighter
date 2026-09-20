"""Round-trip snapshot check: snapshot -> apply -> continue is bit-identical.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_snapshot

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens. This is the real verification that Sessions 1-3 (Ship / Asteroid /
AIEnemy / projectile / Game snapshot + apply_snapshot) actually work:

  1. Build Game A (seed 1234), run N ticks with script_input.
  2. snap = A.snapshot().
  3. Build a FRESH Game B (same seed 1234 — a clean baseline), then
     B.apply_snapshot(snap).
  4. ASSERT #1: snapshot(A) == snapshot(B) immediately after restore —
     the round-trip is lossless.
  5. Run BOTH A and B for M more ticks with the SAME input.
  6. ASSERT #2: snapshot(A) == snapshot(B) after the M ticks — the
     restored state CONTINUES to stay identical (no slow divergence).

The snapshot(g) comparison function is copied verbatim from
test_determinism.py — it is the "are two Games identical" oracle.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy

TICKS = 600   # 10 simulated seconds at 60 Hz (same as test_determinism)
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
    Copied verbatim from test_determinism.py so the input pattern is
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
    """Copied verbatim from test_determinism.py — the "are two Games
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


FIELD_NAMES = ["ship", "asteroids", "enemies", "bullets",
               "enemy_bullets", "missiles", "particles", "rng_state"]


def report_diff(label, a, b):
    """Print every field of the snapshot tuple that differs, plus the
    first few differing elements, so a mismatch pinpoints the subsystem."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            name = FIELD_NAMES[i] if i < len(FIELD_NAMES) else f"field{i}"
            print(f"FAIL: {label} — field {name} differs")
            if i in (0, 7):
                print(f"  A: {x}\n  B: {y}")
            else:
                print(f"  A: {x[:3]}...\n  B: {y[:3]}...")


def main():
    pygame.init()
    global screen, font, big_font, light_tex, fog_surf, light_surf
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # --- Game A: run N ticks, then snapshot ---
    AIEnemy._next_id = 1   # deterministic construction ids
    a = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    for t in range(TICKS):
        a.update(STEP, script_input(t))
    snap = a.snapshot()

    # --- Game B: FRESH, same seed (clean baseline), then restore ---
    AIEnemy._next_id = 1   # keep B's construction clean (apply_snapshot
                           # restores the counter anyway)
    b = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    b.apply_snapshot(snap)

    ok = True

    # ASSERT #1: the round-trip is lossless
    sa = snapshot(a)
    sb = snapshot(b)
    if sa == sb:
        print(f"PASS: ASSERT #1 — snapshot->apply is lossless "
              f"({len(sa[1])} rocks, {len(sa[2])} enemies)")
    else:
        ok = False
        report_diff("ASSERT #1 (immediately after restore)", sa, sb)

    # --- Run BOTH for M more ticks with the SAME input ---
    for t in range(TICKS, TICKS + TICKS):
        a.update(STEP, script_input(t))
        b.update(STEP, script_input(t))

    # ASSERT #2: the restored state CONTINUES to stay identical
    sa2 = snapshot(a)
    sb2 = snapshot(b)
    if sa2 == sb2:
        print(f"PASS: ASSERT #2 — A and B stay identical after "
              f"{TICKS} more ticks")
    else:
        ok = False
        report_diff("ASSERT #2 (after continuing)", sa2, sb2)

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()