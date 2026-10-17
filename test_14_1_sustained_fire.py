"""Plan 14 Phase 1: single-player sustained fire (the host fire path).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_14_1_sustained_fire

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

Phase 1 removed the host's WORLD fire gates (len(self.bullets) >=
MAX_BULLETS) from Game._step. The fire decision is now INPUT-LOCAL
(cooldown + power only). Phase 4 added the max-auto-burst gate: holding
fire fires GUN_BURST_SIZE shots at the within-burst rate (20/s), then the
gun rests GUN_BURST_REST seconds before the next burst. This test proves
the host side of that (single-player, so no ghost / no lockstep concern —
that is Phase 2):

  1. BURST PATTERN — holding fire for N ticks fires in bursts of exactly
     GUN_BURST_SIZE shots (per-tick bullet_seq delta is 0 or 1, never > 1),
     with a GUN_BURST_REST gap after each burst, and the total fired matches
     the expected burst count for N ticks.
  2. POOL GROWS TO NATURAL — the live bullet count climbs to the natural
     ~13 (the SUSTAINED rate 8.6/s x BULLET_LIFE 1.5 s), NOT capped at the
     old MAX_BULLETS 12.
  3. POOL BOUNDED — the live count never exceeds BULLET_POOL_MAX (the
     cull-oldest safety net; it never binds at the natural count).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import (WIDTH, HEIGHT, TICK, FIRE_COOLDOWN, BULLET_LIFE,
                     MAX_BULLETS, BULLET_POOL_MAX, GUN_BURST_SIZE,
                     GUN_BURST_REST)
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
    deltas = []
    max_live = 0
    live = 0
    for _ in range(N_TICKS):
        seq_before = ship.bullet_seq
        g._step(STEP, fire_inp)
        deltas.append(ship.bullet_seq - seq_before)
        live = len(g.bullets)
        max_live = max(max_live, live)

    total_fired = ship.bullet_seq

    # The expected total: simulate the SAME float cooldown + burst gate the
    # ship runs (cd -= TICK; fire when cd <= 0; burst_count gates the rest).
    # This is the exact reference — if the burst logic is wrong (size, gap,
    # or within-burst rate), the total diverges.
    def expected_total(n_ticks):
        cd, bc, total = 0.0, 0, 0
        for _ in range(n_ticks):
            cd -= TICK
            if cd <= 0:
                total += 1
                bc += 1
                if bc >= GUN_BURST_SIZE:
                    bc = 0
                    cd = FIRE_COOLDOWN + GUN_BURST_REST
                else:
                    cd = FIRE_COOLDOWN
        return total

    exp = expected_total(N_TICKS)

    # Group shots into bursts: a new burst starts when the gap since the
    # previous shot exceeds the within-burst spacing (3 ticks at 20/s). The
    # between-burst gap is ~27 ticks, so this cleanly separates them.
    burst_sizes = []
    last_shot_tick = None
    for t, d in enumerate(deltas):
        if d:
            if last_shot_tick is None or (t - last_shot_tick) > 3:
                burst_sizes.append(1)
            else:
                burst_sizes[-1] += 1
            last_shot_tick = t

    # --- 1. BURST PATTERN: at most one shot per tick (the within-burst rate
    # is 1 shot / 3 ticks), the total matches the exact reference, and every
    # COMPLETE burst is exactly GUN_BURST_SIZE shots (the last may be
    # truncated by N_TICKS).
    pattern_ok = (max(deltas) <= 1 and total_fired == exp
                  and all(s == GUN_BURST_SIZE for s in burst_sizes[:-1])
                  and (not burst_sizes or burst_sizes[-1] <= GUN_BURST_SIZE))
    if pattern_ok:
        print(f"PASS: burst pattern — {total_fired} shots over {N_TICKS} "
              f"ticks in {len(burst_sizes)} bursts of "
              f"{burst_sizes[0] if burst_sizes else 0} "
              f"(= {total_fired / (N_TICKS * TICK):.1f}/s sustained, "
              f"{1.0 / FIRE_COOLDOWN:.0f}/s within-burst); "
              f"max 1 shot/tick, total == expected {exp}")
    else:
        ok = False
        print(f"FAIL: burst pattern — max_delta={max(deltas)} "
              f"total_fired={total_fired} (expected {exp}) "
              f"burst_sizes={burst_sizes} (want all complete bursts == "
              f"{GUN_BURST_SIZE})")

    # --- 2. POOL GROWS TO NATURAL: the live count climbs to ~13 (the
    # SUSTAINED rate 8.6/s x BULLET_LIFE 1.5 s), NOT capped at the old
    # MAX_BULLETS 12. The burst gate keeps the sustained rate at ~8.6/s,
    # so the live count tracks THAT (a burst's 6 shots are in flight
    # alongside the tail of the previous burst's).
    sustained = (GUN_BURST_SIZE /
                 (GUN_BURST_SIZE * FIRE_COOLDOWN + GUN_BURST_REST))
    natural = round(sustained * BULLET_LIFE)   # ~13
    pool_ok = (max_live > MAX_BULLETS and 10 <= live <= 16)
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