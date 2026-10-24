"""Plan 21 S3: measure the 2P snapshot wire size, three configs.

Throwaway probe (not part of the test suite). Runs a real 2P Game with the
two players PINNED at a fixed separation (so the field reaches steady state
without reflow), then measures the host's real wire path per tick:
snapshot -> serialize_snapshot(cull) -> json.dumps.

Configs (same world, same seed, same pin distance):
  baseline : 5x5 sectors, keep 2600 (centroid), no wire cull  (pre-plan_21)
  s1       : 3x3 per-player, keep 2200, no wire cull          (after S1)
  s1s2     : 3x3 per-player, keep 2200, wire cull 640 px      (after S2)

Baseline/s1 are reproduced by monkeypatching the config knobs the same way
the production code reads them (spawning reads them at call time; net.py
reads WIRE_CULL_MARGIN at call time).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

import pygame

from ship5 import config as C
from ship5.game import Game, STEP
from ship5.intent import ShipInput
from ship5.ai_enemy import AIEnemy
from ship5.asteroid import Asteroid
from ship5.fog import make_light_texture
from ship5.net import serialize_snapshot, RockShapeTracker


class Keys:
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


_NOKEYS = ShipInput()


def setup(seed=1234):
    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((C.WIDTH, C.HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((C.WIDTH, C.HEIGHT), pygame.SRCALPHA)
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                seed=seed, players=2)
    return game


def measure(game, sep, cull, warm=600, sample=300):
    """Pin the two players `sep` px apart, warm the field, then sample the
    real wire path for `sample` ticks. Returns (mean bytes/snap, mean rock
    count on the wire, mean total rock count)."""
    p0, p1 = game.players
    a, b = pygame.Vector2(0, 0), pygame.Vector2(sep, 0)
    tracker = RockShapeTracker()
    for _ in range(warm):
        p0.pos, p1.pos = a, b
        game._step(STEP, _NOKEYS)
        game.set_remote_input(_NOKEYS)
    sizes, wire_rocks, total_rocks = [], [], []
    for _ in range(sample):
        p0.pos, p1.pos = a, b
        game._step(STEP, _NOKEYS)
        game.set_remote_input(_NOKEYS)
        snap = game.snapshot()
        cc = _remote_pos(snap) if cull is not None else None
        wire = serialize_snapshot(snap, tracker, cull_center=cc)
        sizes.append(len(json.dumps(wire)))
        wire_rocks.append(len(wire[5]))
        total_rocks.append(len(snap[5]))
    m = lambda xs: sum(xs) / len(xs)
    return m(sizes), m(wire_rocks), m(total_rocks)


def _remote_pos(snap):
    """Same extraction as net_process._remote_player_pos (defensive)."""
    try:
        pl = snap[0]
        if len(pl) < 2 or not isinstance(pl[1], (tuple, list)):
            return None
        return (pl[1][0], pl[1][1])
    except Exception:
        return None


def run_case(name, sep, sectors, keep, cull):
    # spawning.py and net.py both `from .config import ...` the knobs by
    # name, so patch the module attributes where they are actually read.
    import ship5.spawning as sp
    import ship5.net as net
    sp.ACTIVE_SECTORS = sectors
    sp.FIELD_KEEP_RADIUS = keep
    if cull is not None:
        net.WIRE_CULL_MARGIN = cull
    game = setup()
    b, wr, tr = measure(game, sep, cull)
    print("%-8s sep=%5dpx  total=%5.1f rocks  wire=%5.1f rocks  "
          "%6.0f B/snap  %6.1f KB/s @60Hz"
          % (name, sep, tr, wr, b, b * 60 / 1024))
    return b, wr, tr


def main():
    sep = 1500
    base = run_case("baseline", sep, 2, 2600, None)
    s1 = run_case("s1", sep, 1, 2200, None)
    s1s2 = run_case("s1s2", sep, 1, 2200, 640)
    print()
    print("reduction vs baseline: s1 = %.0f%%, s1+s2 = %.0f%%"
          % (100 * (1 - s1[0] / base[0]), 100 * (1 - s1s2[0] / base[0])))
    # Wide-separation sanity (plan table: 4000 px apart).
    print()
    run_case("s1s2", 4000, 1, 2200, 640)


if __name__ == "__main__":
    main()