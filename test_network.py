"""Loopback end-to-end network test (Session 6.7).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_network

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens. This is the PERMANENT end-to-end test for the 2P host/client wiring
(Sessions 6.5 + 6.6): it runs the REAL `run_host` and the REAL `run_client`
in one process over a loopback TCP connection, with scripted input, and
verifies the whole client path end to end.

Why threads: the two game loops each own a non-blocking socket and pump
pygame events in their own loop (pinned decision #1: no shared state, no
threads in the GAME LOOP). The test is the one place two loops must run
concurrently, so it runs them in two threads — the same shape the 6.5/6.6
smoke runs used. The connection setup (connect/accept + the join/welcome
handshake) is still a blocking phase that happens BEFORE each game loop, so
the only concurrency is the two non-blocking game loops.

What it proves (the 6.6 client wiring, end to end):
  * connect + the join/welcome handshake succeed over a real socket.
  * the WIRE MAPPING: the client's player 0 is the HOST's hull (from the
    welcome) and the client's player 1 is the CLIENT's own hull — the two
    peers run DIFFERENT hulls, so a wrong mapping shows up as a mismatch.
  * the prediction ghost is rebuilt with the CLIENT's hull (the 6.6 fix —
    Game builds it with the default hull otherwise).
  * the client RECEIVES snapshots (the interpolation buffer fills a window)
    and its render clock advances (the 6.6 render-clock fix).
  * the client's PREDICTION integrates its own input (the ghost ship moves).
  * the host APPLIED the client's input (the host's player 1 — the client's
    ship — moved), proving input flowed client -> host -> sim.
  * the client's render path works (predicted_view returns a real frame).

The scripted input holds W (thrust forward) on BOTH peers: the client sends
it every frame and the host applies it to player 1, so both the client's
predicted ship and the host's authoritative player-1 ship move. The
"moved" checks use a generous floor (50 px) so the test is not flaky on a
slow machine, while still far above the "never moved" zero case.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import socket
import threading
import time

import pygame

from .config import WIDTH, HEIGHT, FPS
from .fog import make_light_texture
from .game import Game
from .hulls import PLAYER_HULLS, default_loadout
from .sound import SoundBank
from . import __main__ as M


# --- scripted input: hold W (thrust forward) on both peers ----------------
class _Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


_KEYS = _Keys({pygame.K_w: 1})


def _free_port():
    """Ask the OS for a free loopback TCP port (bind :0, read it back)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeMenu:
    """Just enough of a Menu for run_host/run_client (hull/loadout + target)."""
    def __init__(self, hull, ip=None, port=None):
        self.hull = hull
        self.loadout = default_loadout(hull)
        self.mode = "host"
        self.host_ip = ip or "127.0.0.1"
        self.host_port = port or 0


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    sfx = SoundBank(); sfx.init()

    # --- capture every Game constructed (host's + client's) ---------------
    created = []
    orig_init = Game.__init__
    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        created.append(self)
    Game.__init__ = patched_init

    # --- deterministic loopback setup -------------------------------------
    port = _free_port()
    M.NET_PORT = port            # run_host binds this (module-level import)
    M._lan_ip = lambda: "127.0.0.1"   # no real network lookup in the test
    pygame.key.get_pressed = lambda: _KEYS   # both peers thrust forward

    # Host on hull[0], client on hull[1] (DIFFERENT hulls -> exercises the
    # wire mapping, like the 6.5/6.6 smoke runs).
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    host_menu = _FakeMenu(host_hull)
    client_menu = _FakeMenu(client_hull, "127.0.0.1", port)

    errors = []
    def host_side():
        try:
            M.run_host(screen, font, big_font, clock, sfx, host_menu, None,
                       light_tex, fog_surf, light_surf)
        except Exception as e:
            errors.append("host: %r" % e)
    def client_side():
        try:
            M.run_client(screen, font, big_font, clock, sfx, client_menu, None,
                         light_tex, fog_surf, light_surf)
        except Exception as e:
            errors.append("client: %r" % e)

    # Start the host first and give it a moment to bind+listen before the
    # client connects (avoids a connection-refused race on loopback).
    th_host = threading.Thread(target=host_side, daemon=True)
    th_client = threading.Thread(target=client_side, daemon=True)
    th_host.start()
    time.sleep(0.2)
    th_client.start()

    # Let them run ~3.5 s (the host broadcasts 10 Hz -> ~35 snaps), then stop
    # both loops by posting QUIT events. One per 50 ms: each thread polls
    # pygame.event.get() every frame (~16 ms), so a posted QUIT is drained by
    # exactly one still-alive thread before the next is posted.
    time.sleep(3.5)
    stop = time.time() + 8.0
    while time.time() < stop and (th_host.is_alive() or th_client.is_alive()):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        time.sleep(0.05)
    th_host.join(timeout=5)
    th_client.join(timeout=5)

    # --- verify -----------------------------------------------------------
    ok = True
    def check(label, cond, extra=""):
        nonlocal ok
        print(("PASS: " if cond else "FAIL: ") + label
              + (("  " + extra) if extra else ""))
        if not cond:
            ok = False

    check("no exceptions in host/client", not errors, repr(errors))
    check("both loops exited (no hang)",
          not th_host.is_alive() and not th_client.is_alive())

    client = next((g for g in created if g.local_index == 1), None)
    host = next((g for g in created if g.local_index == 0), None)
    check("client Game built (local_index=1)", client is not None)
    check("host Game built (local_index=0)", host is not None)

    if client is not None:
        check("client player 0 = host's hull (welcome)",
              client.players[0].hull is host_hull,
              "got %r" % (client.players[0].hull.id,))
        check("client player 1 = client's hull",
              client.players[1].hull is client_hull,
              "got %r" % (client.players[1].hull.id,))
        check("ghost rebuilt with client's hull (6.6 fix)",
              client.ghost.ship.hull is client_hull,
              "got %r" % (client.ghost.ship.hull.id,))
        check("ghost seeded (a snapshot arrived)", client.ghost.seeded)
        check("client received >=2 snapshots (buffer window)",
              len(client.snap_buf) >= 2, "n=%d" % len(client.snap_buf))
        check("client sim_time advanced (render-clock fix)",
              client.sim_time > 0.5, "sim_time=%.3f" % client.sim_time)
        d = client.ghost.ship.pos.distance_to(
            pygame.Vector2(WIDTH / 2, HEIGHT / 2))
        check("client ghost ship moved (prediction integrates input)",
              d > 50.0, "moved %.1f px from spawn" % d)
        # The render path: predicted_view must return a real frame (not None)
        # now that the ghost is seeded and the buffer holds a window.
        check("predicted_view renders a frame (non-None)",
              client.predicted_view(0.016, _KEYS) is not None)

    if host is not None:
        d = host.players[1].pos.distance_to(
            pygame.Vector2(WIDTH / 2, HEIGHT / 2))
        check("host's player 1 (client's ship) moved (input applied)",
              d > 50.0, "moved %.1f px from spawn" % d)

    pygame.quit()
    print("NETWORK E2E:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()