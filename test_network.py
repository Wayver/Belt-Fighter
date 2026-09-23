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
    and its render clock advances (the 6.6 render-clock fix) and TRACKS
    THE HOST'S SIM CLOCK (Session 7.1: the client's sim_time is a
    HostTimeEstimator — an affine fit of the snapshot stamps vs local
    arrival times — not the display's wall clock).
  * the client's PREDICTION integrates its own input (the ghost ship moves).
  * the host APPLIED the client's input (the host's player 1 — the client's
    ship — moved), proving input flowed client -> host -> sim.
  * the client's render path works (predicted_view returns a real frame).
  * the interpolation buffer carries the remote ship's ANGLE (Session 6.8):
    the 'ships' entries are (x, y, angle) and the angle tracks the
    authoritative sim's angle (within the interpolation window's own
    angular span) — this is what lets the client draw the remote hull at
    its interpolated orientation instead of a dot.
  * the buffer carries FULL REMOTE RENDERING data (Session 7.2):
    'enemies' entries are (tag, x, y, angle, vx, vy, id) with the angle
    tracking the authoritative sim (the D4 fix — real enemy hulls, not
    dots), and a 'bullets' key holds (x, y, vx, vy, kind, owner, boost)
    entries for every projectile (the D3 fix — the client rendered no
    bullets at all).

The scripted input holds W (thrust forward) on BOTH peers: the client sends
it every frame and the host applies it to player 1, so both the client's
predicted ship and the host's authoritative player-1 ship move. The
"moved" checks use a generous floor (50 px) so the test is not flaky on a
slow machine, while still far above the "never moved" zero case.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import socket
import threading
import time

import pygame

from .config import WIDTH, HEIGHT, FPS, INTERP_DELAY, ROT_SPEED
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
        check("client sim_time advanced (render clock running)",
              client.sim_time > 0.5, "sim_time=%.3f" % client.sim_time)
        # Session 7.1: the client's render clock is an ESTIMATE OF THE
        # HOST'S SIM CLOCK (HostTimeEstimator), not the display's wall
        # clock. It must track the host's sim_time closely — on loopback
        # the two loops run under thread contention, so the host may
        # simulate slower than real time, and the estimator's rate term
        # must follow that (a wall-clock clock would drift away).
        if host is not None:
            d = abs(client.sim_time - host.sim_time)
            check("client sim_time tracks the host's sim clock (7.1)",
                  d < 0.25,
                  "client=%.3f host=%.3f (d=%.3f s)"
                  % (client.sim_time, host.sim_time, d))
        d = client.ghost.ship.pos.distance_to(
            pygame.Vector2(WIDTH / 2, HEIGHT / 2))
        check("client ghost ship moved (prediction integrates input)",
              d > 50.0, "moved %.1f px from spawn" % d)
        # The render path: predicted_view must return a real frame (not None)
        # now that the ghost is seeded and the buffer holds a window.
        check("predicted_view renders a frame (non-None)",
              client.predicted_view(0.016, _KEYS) is not None)

        # Session 6.8: the buffer carries the remote ship's ANGLE as well as
        # its position, so the client can draw the remote hull at its
        # interpolated orientation. The 'ships' entries must be (x, y, angle)
        # 3-tuples, and the interpolated angle must track the authoritative
        # sim's angle. The scripted input holds W only (no Q/E), so the host
        # ship does not turn — its angle stays at spawn (-pi/2) — which makes
        # "the client's buffer angle equals the host's angle" a clean,
        # non-vacuous check that the angle actually flowed through the
        # snapshot -> buffer -> interpolation path (a constant/garbage angle
        # or a dropped field would fail).
        P = client.snap_buf.positions_at(client.sim_time - INTERP_DELAY)
        if P is None or not P['ships']:
            check("buffer ships carry an angle (6.8)", False,
                  "no window/ship to inspect")
        else:
            entry = P['ships'][0]
            check("buffer ships entry is (x, y, angle)",
                  isinstance(entry, tuple) and len(entry) == 3,
                  "got %r" % (entry,))
            if isinstance(entry, tuple) and len(entry) == 3:
                da = abs((entry[2] - host.players[0].angle + math.pi)
                         % (2 * math.pi) - math.pi)
                check("buffer ship angle tracks the authoritative angle (6.8)",
                      da < 0.05,
                      "client=%.3f host=%.3f (d=%.4f rad)"
                      % (entry[2], host.players[0].angle, da))

        # Session 7.2: the buffer carries what the client needs to draw
        # REAL remote enemies (tag, angle, vel, id — the D4 defect) and
        # ALL projectiles (the D3 defect — the client rendered no bullets
        # at all). The scripted input holds W only, so neither player
        # fires; the host's AI enemies do, so 'enemy'-kind bullets are the
        # ones that must appear (a non-vacuous check that the snapshot's
        # projectile indices 2/3/4 actually flow through the buffer).
        if P is not None:
            check("buffer has a 'bullets' key (7.2)",
                  'bullets' in P, "keys=%r" % (sorted(P.keys()),))
            kinds = set()
            shapes_ok = True
            for b in P.get('bullets', ()):
                if not (isinstance(b, tuple) and len(b) == 7):
                    shapes_ok = False
                    break
                kinds.add(b[4])
            check("buffer bullet entries are (x, y, vx, vy, kind, owner, boost)",
                  shapes_ok, "n=%d" % len(P.get('bullets', ())))
            check("buffer carries >=1 enemy bullet (7.2, D3)",
                  'enemy' in kinds, "kinds=%r" % (sorted(kinds),))
            e_shape_ok = True
            tags = set()
            for e in P.get('enemies', ()):
                if not (isinstance(e, tuple) and len(e) == 7):
                    e_shape_ok = False
                    break
                tags.add(e[0])
            check("buffer enemy entries are (tag, x, y, angle, vx, vy, id) (7.2)",
                  e_shape_ok, "n=%d" % len(P.get('enemies', ())))
            check("buffer enemy tags are real hull tags (7.2, D4)",
                  tags <= {'ai', 'mote'}, "tags=%r" % (sorted(tags),))
            # The enemy ANGLE must track the authoritative sim's angle. Two
            # deterministic checks (a raw "buffer angle vs host NOW" bound
            # is flaky: the AI steers at the ship's ROT_SPEED = 3.6 rad/s, so
            # the ~0.2 s between the render point and the host's current
            # state is up to ~0.7 rad of legitimate lag):
            #   (1) FRESHNESS — the NEWEST snapshot's enemy angle is
            #       fresh against the host's FINAL angle. The host keeps
            #       stepping after its last snapshot until the QUIT event
            #       is processed (up to SNAPSHOT_INTERVAL-1 ticks plus a
            #       frame's worth — more under thread contention), so the
            #       bound is EXACT: the measured gap (host.sim_time -
            #       newest stamp) times the AI's max turn rate
            #       (ENEMY_ROT_SPEED). A dropped/garbage angle (O(1) rad
            #       off) still fails.
            #   (2) IN-SPAN — the buffer's INTERPOLATED angle lies on the
            #       wrapped arc between its bracketing snapshots' angles
            #       (positions_at clamps to the newest snapshot in steady
            #       state, so the span is 0 there and the buffer angle
            #       must equal it exactly). A dropped/garbage angle or a
            #       raw (unwrapped) lerp would leave the arc.
            def _wda(a, b):
                return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
            snaps = client.snap_buf._snaps
            render_t = client.sim_time - INTERP_DELAY
            if len(snaps) >= 2:
                tN, sN = snaps[-1]
                if render_t >= tN:
                    s_prev = s_curr = sN
                else:
                    s_prev = s_curr = snaps[0][1]
                    for i in range(len(snaps) - 1):
                        ti, si = snaps[i]
                        tj, sj = snaps[i + 1]
                        if ti <= render_t <= tj:
                            s_prev, s_curr = si, sj
                            break
                # (1) freshness of the newest snapshot's angles
                if host is not None:
                    live = {e.ship.id: e for e in host.enemies}
                    worst = None
                    for _tag, es in sN[1]:
                        e = live.get(es[2])
                        if e is None:
                            continue
                        d = _wda(es[0][4], e.ship.angle)
                        if worst is None or d > worst:
                            worst = d
                    # Exact bound: the host stepped (host.sim_time - tN)
                    # after the newest snapshot; the AI steers by feeding
                    # the ship a full turn input, so the ship rotates at
                    # ROT_SPEED (3.6 rad/s — the AI's max turn rate), and
                    # the angle can have moved at most that much. (Under
                    # thread contention the host may run several ticks
                    # past the last snapshot before the QUIT is processed,
                    # so a fixed constant is flaky.)
                    gap = max(0.0, host.sim_time - tN)
                    bound = ROT_SPEED * gap + 1e-3
                    check("newest snapshot enemy angle is fresh vs host (7.2)",
                          worst is not None and worst < bound,
                          "worst d=%.4f rad (bound %.4f, gap %.3f s)"
                          % (worst if worst is not None else -1,
                             bound, gap))
                # (2) the buffer angle lies on the bracketing arc
                prev_ang = {es[2]: es[0][4] for _tag, es in s_prev[1]}
                curr_ang = {es[2]: es[0][4] for _tag, es in s_curr[1]}
                in_span = True
                checked = 0
                for (_tag, _x, _y, ang, _vx, _vy, eid) in P.get('enemies', ()):
                    if eid not in prev_ang or eid not in curr_ang:
                        continue
                    checked += 1
                    span = _wda(prev_ang[eid], curr_ang[eid])
                    if (_wda(ang, prev_ang[eid]) > span + 1e-6
                            or _wda(ang, curr_ang[eid]) > span + 1e-6):
                        in_span = False
                check("buffer enemy angle is on the bracketing arc (7.2)",
                      checked > 0 and in_span,
                      "checked %d enemies" % checked)

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