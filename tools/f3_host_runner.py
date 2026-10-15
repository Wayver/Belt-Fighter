"""Subprocess entry point: run the REAL run_host with debug_net on.

Used by dbg_f3_capture2 to put the host in its OWN process (separate GIL),
matching the real 2-machine topology where the host's render process is not
starved by a client loop. The host writes host_debug.csv.

M4 REDO: this subprocess now runs the FULL host stack — the render process
(run_host + the _NetPump thread) AND its own spawned sim process. That is
exactly the production topology, so the client's newest_stamp delta in
dbg_f3_capture2 measures the snapshot send path end to end (sim process ->
net Queue -> _NetPump -> NetWorker -> socket).

Run directly:  python -m ship5.tools.f3_host_runner <port>
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys

import pygame

from ship5.config import WIDTH, HEIGHT
from ship5.fog import make_light_texture
from ship5.game import Game
from ship5.hulls import PLAYER_HULLS, default_loadout
from ship5.sound import SoundBank
from ship5 import __main__ as M

_ORIG_GAME_INIT = Game.__init__


class _Keys:
    def __getitem__(self, k):
        return 1 if k == pygame.K_w else 0


def main(port):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    sfx = SoundBank()
    sfx.init()
    clock = pygame.time.Clock()

    M.NET_PORT = port
    M._lan_ip = lambda: "127.0.0.1"
    pygame.key.get_pressed = lambda: _Keys()

    def patched_init(self, *a, **k):
        _ORIG_GAME_INIT(self, *a, **k)
        self.debug_net = True
    Game.__init__ = patched_init

    class _Menu:
        def __init__(self, hull):
            self.hull = hull
            self.loadout = default_loadout(hull)
            self.mode = "host"
            self.host_ip = "127.0.0.1"
            self.host_port = port

    M.run_host(screen, font, big_font, clock, sfx,
               _Menu(PLAYER_HULLS[0]), None, light_tex, fog_surf, light_surf)
    Game.__init__ = _ORIG_GAME_INIT


if __name__ == "__main__":
    main(int(sys.argv[1]))