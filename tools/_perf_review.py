"""One-off perf review (v2): client-side hot paths + real wire cost.
Throwaway — not part of the test suite.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cProfile
import pstats
import io
import json
import time

import pygame

from ship5.config import WIDTH, HEIGHT
from ship5.fog import make_light_texture
from ship5.game import Game, STEP
from ship5.intent import ShipInput
from ship5.ai_enemy import AIEnemy
from ship5.asteroid import Asteroid
from ship5.net import serialize_snapshot


class Keys:
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def sustained(t):
    return Keys({pygame.K_w: 1, pygame.K_SPACE: 1, pygame.K_2: 1,
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
    return screen, font, big_font, light_tex, fog_surf, light_surf, game


def warm(game, ticks=600):
    for t in range(ticks):
        game._step(STEP, ShipInput.from_keys(sustained(t)))
        game.set_remote_input(ShipInput.from_keys(sustained(t)))


def top(pr, n=25, sortkey="tottime"):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sortkey)
    ps.print_stats(n)
    return s.getvalue()


def main():
    _NOKEYS = Keys({})
    screen, font, big_font, light_tex, fog_surf, light_surf, game = setup()
    warm(game)

    # ---- real wire path: snapshot -> serialize(prune) -> json.dumps ----
    snap = game.snapshot()
    wire = serialize_snapshot(snap)
    for _ in range(10):
        json.dumps(wire)
    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        s = game.snapshot()
        w = serialize_snapshot(s)
        j = json.dumps(w)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    for _ in range(N):
        json.loads(j)
    t3 = time.perf_counter()
    print("=== REAL wire path (warmed world, ~100 rocks) ===")
    print("  snapshot+serialize+dumps : %.3f ms each" % ((t1 - t0) / N * 1000))
    print("  json.loads (client)      : %.3f ms each" % ((t3 - t2) / N * 1000))
    print("  wire bytes/snapshot      : %d  (at 60 Hz: %.0f KB/s)"
          % (len(j), len(j) * 60 / 1024))

    # ---- client: push_snapshot + interp_positions + predicted_view ----
    client = Game(pygame.display.set_mode((WIDTH, HEIGHT)), font, big_font,
                  light_tex, fog_surf, light_surf, seed=99, players=2)
    # Wire up the client's feel-layer the way run_client does.
    from ship5.netcode import (PredictedShip, HostTimeEstimator,
                               LatencyTracker, RenderPoint)
    client.local_index = 1
    _p1 = client.players[1]
    client.ghost = PredictedShip(hull=_p1.hull,
                                 loadout=_p1.components,
                                 local_index=1)
    client.host_time = HostTimeEstimator()
    client.latency = LatencyTracker()
    client._rtt_ow = 0.02
    client.render_point = RenderPoint(client.latency)
    # Prime with a few snapshots at increasing sim_time so the buffer has
    # a window to interpolate across. (Client takes the PRUNED wire form.)
    base_t = 100.0
    for i in range(6):
        client.push_snapshot(base_t + i * STEP, wire)
    # Advance the client's render point machinery a bit.
    client.latency.tick(0.016)
    client.render_point.advance(0.016, client.snap_buf.newest_time())

    dt = 0.016
    for _ in range(10):
        client.predicted_view(dt, _NOKEYS)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(120):
        client.predicted_view(dt, _NOKEYS)
    pr.disable()
    print("=== cProfile: client predicted_view() x120 (top by tottime) ===")
    print(top(pr, 30, "tottime"))

    # push_snapshot cost in isolation
    for _ in range(10):
        client.push_snapshot(base_t + 10 * STEP, wire)
    t0 = time.perf_counter()
    for i in range(N):
        client.push_snapshot(base_t + (10 + i) * STEP, wire)
    t1 = time.perf_counter()
    print("=== client push_snapshot (incl. ghost reconcile) ===")
    print("  %.3f ms each" % ((t1 - t0) / N * 1000))


if __name__ == "__main__":
    main()