"""Direct probe of the 10.3c ghost sensor-contact path:
- _ghost_enemy_proxies power fields
- PredictedShip.step with sensor_on + proxies -> contacts
- _update_contacts passive + scan branches
- predicted_view full frame with contacts (drawing path)
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import traceback
import pygame

from ship5.config import WIDTH, HEIGHT, SENSOR_SIGNATURE_THRESHOLD
from ship5.fog import make_light_texture
from ship5.game import Game, _GhostEnemyProxy
from ship5.hulls import PLAYER_HULLS, default_loadout, enemy_loadout
from ship5.netcode import PredictedShip
from ship5.sound import SoundBank
from ship5.intent import ShipInput

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("consolas,menlo,monospace", 18)
big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
light_tex = make_light_texture()
fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
sfx = SoundBank()
try:
    sfx.init()
except Exception:
    pass

hull = PLAYER_HULLS[1]  # blackbird (has BB_SENSOR)
game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
            hull=hull, loadout=default_loadout(hull),
            test_mode=True, seed=7, sound=sfx, players=2, local_index=1)
game.set_player_ship(1, __import__("ship5.ship", fromlist=["Ship"]).Ship(
    hull=hull, loadout=default_loadout(hull)))
game.ghost = PredictedShip(hull=hull, loadout=default_loadout(hull),
                           local_index=1)

try:
    # 1. Proxy power fields: build a fake buffer pos with 10-tuple enemies.
    pos = {
        'ships': [(100.0, 100.0, 0.0, False, 0.0, 0.0),
                  (200.0, 200.0, 0.0, False, 0.0, 0.0)],
        'enemies': [
            ('ai', 300.0, 300.0, 0.0, 10.0, 0.0, 5, 0.0, 0.0, 30.0),
            ('mote', 400.0, 400.0, 1.0, 0.0, 5.0, 6, 0.0, 0.0, 5.0),
        ],
        'asteroids': [],
        'bullets': [],
    }
    proxies = game._ghost_enemy_proxies(pos)
    assert len(proxies) == 2, "expected 2 proxies"
    p0, p1 = proxies
    print("proxy0: power_used=%.1f idle=%.1f sig=%.1f" %
          (p0.ship.power_used, p0.ship.power_idle_total,
           p0.ship.power_used - p0.ship.power_idle_total))
    print("proxy1: power_used=%.1f idle=%.1f sig=%.1f" %
          (p1.ship.power_used, p1.ship.power_idle_total,
           p1.ship.power_used - p1.ship.power_idle_total))
    assert p0.ship.power_used == 30.0
    assert p1.ship.power_used == 5.0
    assert p0.ship.power_idle_total > 0, "idle should be a real loadout constant"

    # 2. Ghost step with sensor on -> passive contacts.
    s = game.ghost.ship
    assert s.sensor_comp is not None, "blackbird default loadout has a sensor"
    s.sensor_on = True
    s.scan_reveal = 0.0
    s.scan_cd = 0.0
    # Place the ghost near the enemies so they are in sensor_range.
    s.pos = pygame.Vector2(250.0, 250.0)
    s.vel = pygame.Vector2(0, 0)
    s.angle = 0.0
    # Seed the ghost so advance() works.
    if not game.ghost.seeded:
        game.ghost.seed(s.snapshot())
    game.ghost.step(1 / 60.0, ShipInput(), enemies=proxies)
    print("contacts after step (passive): %r" % s.contacts)
    # proxy0 sig = 30 - idle. If that crosses the threshold, expect a contact.
    sig0 = 30.0 - p0.ship.power_idle_total
    expected = 1 if sig0 >= SENSOR_SIGNATURE_THRESHOLD else 0
    assert len(s.contacts) == expected, \
        "expected %d passive contact(s), got %r" % (expected, s.contacts)

    # 3. Scan branch: reveal everything in scan_range.
    s.scan_reveal = 2.0
    game.ghost.step(1 / 60.0, ShipInput(), enemies=proxies)
    print("contacts after step (scan): %r" % s.contacts)
    assert all(c[3] for c in s.contacts), "scan contacts must be confirmed"

    # 4. Full predicted_view frame with contacts live (drawing path).
    # Need a render window: push two snapshots through the buffer.
    from ship5.netcode import RenderPoint, LatencyTracker, HostTimeEstimator
    game.render_point = RenderPoint(game.latency)
    import time as _t
    snap1 = game.snapshot()
    snap2 = game.snapshot()
    game.snap_buf.push(0.0, snap1)
    game.snap_buf.push(0.1, snap2)
    rp = game.render_point.advance(0.016, game.snap_buf.newest_time())
    print("render point:", rp)
    # Give the render point time to reach a window.
    for _ in range(120):
        game.render_point.advance(0.016, game.snap_buf.newest_time())
    s.sensor_on = True
    s.scan_reveal = 2.0
    out = game.predicted_view(0.016, pygame.key.get_pressed(),
                              host_time=0.2)
    print("predicted_view returned:", None if out is None else "pos dict")
    print("ghost contacts during predicted_view: %r" % s.contacts)
    print("ALL PROBES PASSED")
except Exception:
    traceback.print_exc()
finally:
    pygame.quit()