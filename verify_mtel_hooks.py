"""10.3b telemetry self-check: drive the REAL ghost fire -> reconcile ->
handback -> render paths with telemetry enabled and confirm each hook
(FIRE / RECONCILE / HANDBACK / DEDUP) emits a row. Proves the hooks are
wired to the actual call sites, not just that mt.log works.

Run from the repo root:  python -m ship5.verify_mtel_hooks
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import TICK, WIDTH, HEIGHT
from .hulls import (HullType, FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT,
                    RCS_RIGHT, GUN, MISSILE, REACTOR, COMPUTER, SHIELD,
                    SENSOR, default_loadout)
from .ship import Ship
from .intent import ShipInput
from .netcode import (PredictedShip, GhostMissile, RenderPoint,
                      LatencyTracker, HostTimeEstimator)
from .game import Game, _GhostEnemyProxy
from .bullets import Missile
from .fog import make_light_texture
from . import missile_telemetry as mt

MISSILE_HULL = HullType(
    id='mtel', polygon=((18, 0), (14, 3.5), (8, 6.5), (0, 8), (-8, 8),
                        (-12, 11), (-12, 5), (-9, 3), (-9, -3), (-12, -5),
                        (-12, -11), (-8, -8), (0, -8), (8, -6.5), (14, -3.5)),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN, MISSILE,
           REACTOR, COMPUTER, SHIELD, SENSOR),
    base_mass=1.0, collision_radius=12.0, nose=(18, 0), cockpit=(8, 0),
    max_speed_factor=1.0, turn_rate_factor=1.0,
    fill=(200, 200, 200), edge=(255, 255, 255))
LOADOUT = default_loadout(MISSILE_HULL)
LI = 1


def seed_s():
    s = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    s.pos = pygame.Vector2(0.0, 0.0)
    s.vel = pygame.Vector2(0.0, 0.0)
    s.angle = 0.0
    for w in s.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 1.0
    return s.snapshot()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big = pygame.font.SysFont("consolas,menlo,monospace", 40)
    lt = make_light_texture()
    fog = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    lgt = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    g = Game(screen, font, big, lt, fog, lgt, hull=MISSILE_HULL,
             loadout=LOADOUT, seed=1, players=2, local_index=LI)
    g.set_player_ship(LI, Ship(hull=MISSILE_HULL, loadout=LOADOUT))
    g.ghost = PredictedShip(hull=MISSILE_HULL, loadout=LOADOUT, local_index=LI)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)

    mt.set_enabled(True)

    # 1) FIRE: the ghost fires a missile via advance (real step() path).
    proxy = _GhostEnemyProxy(pygame.Vector2(100.0, 0.0), pygame.Vector2(0, 0),
                             0.0, 1, 0.0, 0.0)
    g.ghost.seed(seed_s())
    g.ghost.record_input(0.0, ShipInput())
    g.ghost.advance(TICK, ShipInput(), enemies=[proxy])
    g.ghost.record_input(TICK, ShipInput(missile_fire=True))
    g.ghost.advance(TICK, ShipInput(missile_fire=True), enemies=[proxy])
    assert len(g.ghost.local_missiles) == 1, g.ghost.local_missiles

    # 2) RECONCILE + HANDBACK: push a PRE-fire snapshot (seq 0) -> the rewind
    #    re-runs the fire tick (re-fire guard) and handback runs. Build a FULL
    #    game snapshot by setting g's own ship state (push_snapshot reads
    #    snap[0][local_index] + snap[4], so a bare Ship.snapshot() won't do).
    g.players[LI].missile_seq = 0
    g.missiles = []
    g.push_snapshot(0.0, g.snapshot(), now=0.0)

    # 3) DEDUP: push a POST-fire snapshot that CARRIES the missile (id (1,0))
    #    so the buffer has a copy, then render a frame (predicted_view).
    g.players[LI].missile_seq = 1
    g.missiles = [Missile(pygame.Vector2(100.0, 0.0),
                          pygame.Vector2(460.0, 0.0), owner=1, mid=(LI, 0))]
    for k in range(1, 4):
        g.push_snapshot(k * 0.1, g.snapshot(), now=k * 0.1)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())

    mt.close()

    # Parse the CSV and confirm every event fired.
    import csv
    rows = list(csv.DictReader(open(mt._path)))
    events = [r["event"] for r in rows]
    print("events logged:", events)
    for want in ("FIRE", "RECONCILE", "HANDBACK", "DEDUP"):
        assert want in events, "MISSING hook: %s (got %s)" % (want, events)
    # Spot-check the FIRE row carries the id + seq move.
    fire = next(r for r in rows if r["event"] == "FIRE")
    assert fire["f_mid"] == "1:0" and fire["f_seq_before"] == "0" \
        and fire["f_seq_after"] == "1", fire
    # Spot-check the DEDUP row exists and carries the missile id somewhere
    # (ghost in-flight, or buffer drawn/skipped). After handback the buffer's
    # copy is the one DRAWN in steady state; the SKIP path is a narrow edge
    # case (missile just died, render point still showing it) — either is a
    # valid DEDUP row here.
    dedup = next(r for r in rows if r["event"] == "DEDUP")
    assert ("1:0" in dedup["d_ghost_ids"] or "1:0" in dedup["d_buf_drawn"]
            or "1:0" in dedup["d_buf_skipped"]), dedup
    print("\nALL HOOKS FIRED through the real code paths.")
    print("FIRE   :", {k: v for k, v in fire.items() if v})
    print("DEDUP  :", {k: v for k, v in dedup.items() if v})
    print("\n--- full missile_debug.csv ---")
    print(open(mt._path).read())


if __name__ == "__main__":
    main()