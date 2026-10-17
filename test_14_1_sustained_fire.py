"""Plan 14 Phase 1: single-player sustained fire (the host fire path).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_14_1_sustained_fire

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

Phase 1 removed the host's WORLD fire gates (len(self.bullets) >=
MAX_BULLETS) from Game._step. The fire decision is now INPUT-LOCAL
(cooldown + power only), so holding fire produces a STEADY stream at the
tick-capped rate (60/s) instead of the old 12-shot-burst-then-1.3s-dead-zone
cycle. This test proves the host side of that (single-player, so no ghost /
no lockstep concern — that is Phase 2):

  1. SUSTAINED RATE — holding fire for N ticks fires EXACTLY one shot per
     tick (the ship's bullet_seq delta == 1 every tick): no burst (delta > 1)
     and no dead zone (delta == 0). Total fired == N == the 60/s tick-capped
     rate (min(1/FIRE_COOLDOWN, 60)).
  2. POOL GROWS TO NATURAL — the live bullet count climbs to the natural
     ~90 (60/s x BULLET_LIFE 1.5 s), NOT capped at the old MAX_BULLETS 12.
  3. POOL BOUNDED — the live count never exceeds BULLET_POOL_MAX (the
     cull-oldest safety net; it never binds at the natural count).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import (WIDTH, HEIGHT, TICK, FIRE_COOLDOWN, BULLET_LIFE,
                     MAX_BULLETS, BULLET_POOL_MAX)
from .fog import make_light_texture
from .game import Game, STEP
from .intent import ShipInput

N_TICKS = 240   # 4 s of sustained fire; steady state is reached in ~90 ticks
EPS = 1e-9


class Keys:
    """A minimal stand-in for pygame.key.get_pressed() — idle (all keys
    released). Player 0's input is driven by this (idle); we fire by setting
    g.players[0]'s input directly via a ShipInput, not through keys."""
    def __getitem__(self, k):
        return 0


IDLE_KEYS = Keys()


def make_res():
    """The pygame res tuple a Game needs (headless)."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return (screen, font, big_font, light_tex, fog_surf, light_surf)


def make_single(res):
    """A single-player host Game (default scout hull = one ballistic gun),
    ISOLATED (no enemies/asteroids, update_field no-op'd) so the only thing
    that removes a bullet is its own life — collisions can't skew the pool
    count. Mirrors the test_10_10 / test_interpolation isolation."""
    import ship5.game as _gm
    g = Game(*res, seed=1234, players=1, local_index=0)
    g.enemies.clear()
    g.asteroids.clear()
    _gm.update_field = lambda *a, **k: None
    return g


def main():
    pygame.init()
    ok = True
    res = make_res()

    g = make_single(res)
    ship = g.players[0]
    fire_inp = ShipInput(fire=True)

    # Drive the sim at exactly the fixed step, holding fire every tick.
    # g.update(STEP, keys) samples input from `keys` (idle) for player 0, so
    # we instead step the fixed loop directly with an explicit ShipInput to
    # hold fire. We call g._step(STEP, fire_inp) — the same fixed step
    # g.update would run — to keep the measurement exact (one step per call).
    min_delta = None
    max_delta = None
    max_live = 0
    live = 0
    for _ in range(N_TICKS):
        seq_before = ship.bullet_seq
        g._step(STEP, fire_inp)
        delta = ship.bullet_seq - seq_before
        min_delta = delta if min_delta is None else min(min_delta, delta)
        max_delta = delta if max_delta is None else max(max_delta, delta)
        live = len(g.bullets)
        max_live = max(max_live, live)

    total_fired = ship.bullet_seq

    # --- 1. SUSTAINED RATE: exactly one shot per tick (no burst, no dead
    # zone). The tick grid caps the rate at 60/s (FIRE_COOLDOWN 0.01 < TICK
    # 0.0167), so holding fire fires once per tick = N shots over N ticks.
    rate_ok = (min_delta == 1 and max_delta == 1 and total_fired == N_TICKS)
    if rate_ok:
        print(f"PASS: sustained rate — {total_fired} shots over {N_TICKS} "
              f"ticks, exactly 1/tick (no burst, no dead zone) = "
              f"{total_fired / (N_TICKS * TICK):.0f}/s "
              f"(min(1/FIRE_COOLDOWN, 60) = "
              f"{min(round(1.0 / FIRE_COOLDOWN), 60)}/s)")
    else:
        ok = False
        print(f"FAIL: sustained rate — min_delta={min_delta} "
              f"max_delta={max_delta} total_fired={total_fired} "
              f"(want exactly 1/tick, total == {N_TICKS})")

    # --- 2. POOL GROWS TO NATURAL: the live count climbs to ~90 (60/s x
    # BULLET_LIFE 1.5 s), NOT capped at the old MAX_BULLETS 12.
    natural = round(60 * BULLET_LIFE)   # 90
    pool_ok = (max_live > MAX_BULLETS and 85 <= live <= 95)
    if pool_ok:
        print(f"PASS: pool grows to natural — live count reached {live} "
              f"(steady state), max {max_live} > old cap {MAX_BULLETS} "
              f"(natural ~{natural})")
    else:
        ok = False
        print(f"FAIL: pool grows to natural — live={live} max_live={max_live} "
              f"(want steady ~{natural}, and > old cap {MAX_BULLETS})")

    # --- 3. POOL BOUNDED: the live count never exceeds the cull-oldest
    # safety net (it never binds at the natural count, but must hold).
    bound_ok = max_live <= BULLET_POOL_MAX
    if bound_ok:
        print(f"PASS: pool bounded — max live {max_live} <= "
              f"BULLET_POOL_MAX {BULLET_POOL_MAX}")
    else:
        ok = False
        print(f"FAIL: pool bounded — max live {max_live} > BULLET_POOL_MAX "
              f"{BULLET_POOL_MAX}")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()