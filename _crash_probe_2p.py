"""2P loopback probe that presses V/G/T on the CLIENT (the new sensor path).
The committed E2E never presses V/G, so this exercises the ghost's
_update_contacts + the (missing) command-sink path in the real client loop.
Prints any traceback from either side."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import socket, threading, time, traceback
import pygame

import ship5.__main__ as M
from ship5.config import WIDTH, HEIGHT, FPS
from ship5.fog import make_light_texture
from ship5.game import Game
from ship5.hulls import PLAYER_HULLS, default_loadout
from ship5.sound import SoundBank
from ship5.netcode import RenderPoint, PredictedShip

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
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

# Hold W on both peers (thrust forward) — like the E2E.
class _Keys(dict):
    def __getitem__(self, k):
        return 1 if k == pygame.K_w else 0
pygame.key.get_pressed = lambda: _Keys()

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class _FakeMenu:
    def __init__(self, hull, ip=None, port=None):
        self.hull = hull
        self.loadout = default_loadout(hull)
        self.mode = "host"
        self.host_ip = ip or "127.0.0.1"
        self.host_port = port or 0

port = _free_port()
M.NET_PORT = port
M._lan_ip = lambda: "127.0.0.1"
host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
host_menu = _FakeMenu(host_hull)
client_menu = _FakeMenu(client_hull, "127.0.0.1", port)

errors = []
def host_side():
    try:
        M.run_host(screen, font, big_font, clock, sfx, host_menu, None,
                   light_tex, fog_surf, light_surf)
    except Exception:
        errors.append("host:\n" + traceback.format_exc())
def client_side():
    try:
        M.run_client(screen, font, big_font, clock, sfx, client_menu, None,
                     light_tex, fog_surf, light_surf)
    except Exception:
        errors.append("client:\n" + traceback.format_exc())

th_host = threading.Thread(target=host_side, daemon=True)
th_client = threading.Thread(target=client_side, daemon=True)
th_host.start()
time.sleep(0.2)
th_client.start()

# Run ~5 s, pressing V/G/T on the client's event queue periodically so the
# sensor-contact path runs in the real client loop.
t0 = time.time()
last = 0
while time.time() - t0 < 5.0:
    now = time.time()
    if now - last > 0.5:
        last = now
        for k in (pygame.K_v, pygame.K_g, pygame.K_t):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k))
    time.sleep(0.016)

stop = time.time() + 8.0
while time.time() < stop and (th_host.is_alive() or th_client.is_alive()):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    time.sleep(0.05)
th_host.join(timeout=5)
th_client.join(timeout=5)

if errors:
    print("=== CRASH DETECTED ===")
    for e in errors:
        print(e)
else:
    print("NO CRASH over 5s with V/G/T pressed on the client")
pygame.quit()