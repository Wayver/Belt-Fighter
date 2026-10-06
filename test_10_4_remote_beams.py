"""Session 10.4: the client sees the REMOTE player's laser beams.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_4_remote_beams

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.4 closes the gap found in a live 2P test: the client never saw the
HOST player's laser beams. (10.3a was the ghost's OWN beams; 10.4 is the
remote ship's.) A laser beam is a 0.15 s flash (BEAM_TTL), but the client
renders INTERP_DELAY (0.1-0.35 s) in the PAST, so a beam whose age rode
the snapshot would be expired (or a flicker) by the time the render point
reached it. The fix is EVENT-based: the host sends a discrete T_BEAM
message the moment it fires (world-space muzzle -> endpoint + sim_time);
the client draws it immediately on receipt, fading over BEAM_TTL — the
same way the ghost draws its OWN beams (10.3a).

  * HOST (step 1): Game._resolve_beam / _beam_hit_asteroid emit
    (sx, sy, ex, ey, sim_time) into Game._beam_events; the SimThread
    drains it after each step and sends T_BEAM via the worker.
  * CLIENT (steps 2-4): run_client's poll loop pushes each T_BEAM into
    game.remote_beams as [start, end, age, ttl] (world space);
    predicted_view ages it each frame (_step_remote_beams) and draws it
    BEFORE the ships (LASER_COLOR line, width 2, fading over the ttl).

This test proves it:
  1. WIRE-SHAPE  — T_BEAM == "beam"; the host's real SimThread drain
                   emits a JSON-serializable message carrying
                   type/sim_time/sx/sy/ex/ey (the host->wire contract).
  2. RECEIVE     — a T_BEAM pushed into game.remote_beams (the exact
                   run_client append) is [Vector2, Vector2, 0.0, BEAM_TTL]
                   in world space.
  3. AGE-CULL    — _step_remote_beams advances the age by dt each frame
                   and culls the beam once age >= ttl (mirrors the ghost's
                   step_local_beams + the host's beam aging).
  4. PIXEL       — predicted_view renders the remote beam (beam-colored
                   pixels appear along the line; the no-beam baseline has
                   none).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import time

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (PredictedShip, HostTimeEstimator, LatencyTracker,
                      RenderPoint, BEAM_TTL)
from .ship import Ship
from .intent import ShipInput
from .hulls import SILAS_HULL, default_loadout
from .ai_enemy import AIEnemy
from .sim_thread import SimThread
from .net import T_BEAM

SEED = 1234
TICK = 1 / 60


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def make_client(screen, font, big_font, light_tex, fog_surf, light_surf):
    """A client Game (local_index=1) with the silas hull on BOTH players."""
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL),
             seed=SEED, players=2, local_index=1)
    g.set_player_ship(1, Ship(hull=SILAS_HULL,
                              loadout=default_loadout(SILAS_HULL)))
    g.ghost = PredictedShip(hull=SILAS_HULL,
                            loadout=default_loadout(SILAS_HULL),
                            local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless)."""
    AIEnemy._next_id = 1
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    tmp = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
               hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL),
               seed=SEED, players=2)
    tmp.set_player_ship(1, Ship(hull=SILAS_HULL,
                                loadout=default_loadout(SILAS_HULL)))
    return tmp


def make_snap(charge=1.0):
    """A valid 2-player Game snapshot with the CLIENT ship (index 1) at the
    origin facing +x, its lasers at `charge`. The client ship's pose is
    fixed so the ghost (seeded from this snapshot) sits at the origin."""
    tmp = _scratch_game()
    cs = tmp.players[1]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    snap = tmp.snapshot()
    cs_s = list(snap[0][1])
    cs_s[18] = tuple((0.0, charge, 0.0) for _ in cs.weapons)
    ships = (snap[0][0], tuple(cs_s))
    return (ships,) + snap[1:]


def push_stream(g, charge=1.0):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) so predicted_view has a
    render window."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=SILAS_HULL,
                            loadout=default_loadout(SILAS_HULL),
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(charge), now=t)


def _advance_render_point(g):
    """Advance the render point so now() returns a value (mirrors the
    10.3a/10.2b render-frame helpers)."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())


class FakeWorker:
    """Captures what the sim thread would send (send) + the snapshot
    hand-off (set_latest_snapshot), so we can assert on the T_BEAM
    messages without a real socket."""
    def __init__(self):
        self.sent = []
        self.snaps = 0
    def send(self, msg):
        self.sent.append(msg)
    def set_latest_snapshot(self, sim_time, snap):
        self.snaps += 1


def beam_pixels(screen, cam, beams):
    """Count beam-colored pixels sampled along the beam lines.

    The beam is drawn BEFORE the fog (mirrors the host's draw() order), so
    near the light source (the ghost) the fog dims LASER_COLOR
    (140, 255, 190) but it stays green-dominant (g > r and g > b). For
    each beam, sample a small box around a few points along the line
    (t = 0.3, 0.5, 0.7 — avoiding the endpoints) and count the
    beam-colored pixels (g > 100 and g > r and g > b). Baseline (no beams)
    -> 0; a fired beam -> > 0."""
    n = 0
    for (start, end, _age, _ttl) in beams:
        for t in (0.3, 0.5, 0.7):
            p = start + (end - start) * t
            s = cam.to_screen(p)
            cx, cy = int(s.x), int(s.y)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px = screen.get_at((cx + dx, cy + dy))
                    r, gg, b = px[0], px[1], px[2]
                    if gg > 100 and gg > r and gg > b:
                        n += 1
    return n


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()

    # --- 1. WIRE-SHAPE: the host's real SimThread drain emits a
    #     JSON-serializable T_BEAM carrying type/sim_time/sx/sy/ex/ey. ---
    assert T_BEAM == "beam", "T_BEAM must be the 'beam' tag: %r" % T_BEAM
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL),
             seed=SEED, players=2)
    g.set_player_ship(1, Ship(hull=SILAS_HULL,
                              loadout=default_loadout(SILAS_HULL)))
    # Pre-seed a known beam event; the sim (empty input) won't fire its
    # own, so this is the only one the drain should send.
    g._beam_events.append((111.0, 222.0, 333.0, 444.0, 0.5))
    worker = FakeWorker()
    st = SimThread(g, worker=worker)
    st.start()
    deadline = time.monotonic() + 2.0
    got = None
    while time.monotonic() < deadline:
        for m in worker.sent:
            if m.get("type") == T_BEAM:
                got = m
                break
        if got is not None:
            break
        time.sleep(0.01)
    st.stop()
    assert got is not None, "the SimThread must drain _beam_events -> T_BEAM"
    for k in ("type", "sim_time", "sx", "sy", "ex", "ey"):
        assert k in got, "T_BEAM missing field %r: %r" % (k, got)
    assert got["sim_time"] == 0.5 and got["sx"] == 111.0 \
        and got["sy"] == 222.0 and got["ex"] == 333.0 and got["ey"] == 444.0, \
        "T_BEAM fields not carried: %r" % (got,)
    # JSON-serializable (it rides the wire as a JSON frame).
    json.dumps(got)
    assert g._beam_events == [], "the buffer must be drained (cleared)"
    print("PASS: WIRE-SHAPE — the host's SimThread drain emits a "
          "JSON-safe T_BEAM %r" % (got,))

    # --- 2. RECEIVE: a T_BEAM pushed into game.remote_beams (the exact
    #     run_client append) is [Vector2, Vector2, 0.0, BEAM_TTL]. ---
    g2 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    assert g2.remote_beams == [], "remote_beams must start empty"
    # The exact append run_client performs on a T_BEAM message.
    msg = {"type": T_BEAM, "sim_time": 0.5,
           "sx": 50.0, "sy": 0.0, "ex": 250.0, "ey": 0.0}
    g2.remote_beams.append([
        pygame.Vector2(msg["sx"], msg["sy"]),
        pygame.Vector2(msg["ex"], msg["ey"]),
        0.0, BEAM_TTL])
    assert len(g2.remote_beams) == 1
    start, end, age, ttl = g2.remote_beams[0]
    assert isinstance(start, pygame.Vector2) and isinstance(end, pygame.Vector2), \
        "remote beam endpoints must be Vector2: %r" % (g2.remote_beams[0],)
    assert start.x == 50.0 and start.y == 0.0, "start = muzzle (sx, sy)"
    assert end.x == 250.0 and end.y == 0.0, "end = endpoint (ex, ey)"
    assert age == 0.0, "a freshly received beam starts at age 0"
    assert ttl == BEAM_TTL, "the beam's ttl is BEAM_TTL (0.15 s)"
    print("PASS: RECEIVE — a T_BEAM becomes [Vector2, Vector2, 0.0, "
          "BEAM_TTL] in world space")

    # --- 3. AGE-CULL: _step_remote_beams advances the age by dt each
    #     frame and culls the beam once age >= ttl. ---
    g3 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    g3.remote_beams.append([pygame.Vector2(50.0, 0.0),
                            pygame.Vector2(250.0, 0.0), 0.0, BEAM_TTL])
    # One frame: age advances by dt.
    g3._step_remote_beams(TICK)
    assert len(g3.remote_beams) == 1, "the beam must survive one frame"
    assert abs(g3.remote_beams[0][2] - TICK) < 1e-9, \
        "the beam's age must advance by dt each frame: %r" \
        % (g3.remote_beams[0][2],)
    # Age it past the ttl (0.15 s = 9 ticks): it is culled.
    for _ in range(12):
        g3._step_remote_beams(TICK)
    assert not g3.remote_beams, \
        "the beam must be culled after the ttl (0.15 s): %r" \
        % (g3.remote_beams,)
    # An empty buffer is a no-op (no crash).
    g3._step_remote_beams(TICK)
    assert not g3.remote_beams
    print("PASS: AGE-CULL — the remote beam ages by dt each frame and is "
          "culled after the 0.15 s ttl")

    # --- 4. PIXEL: predicted_view renders the remote beam (beam-colored
    #     pixels appear along the line; the no-beam baseline has none). ---
    g4 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    # Baseline: no remote beams -> 0 beam pixels.
    push_stream(g4, 1.0)
    _advance_render_point(g4)
    g4.predicted_view(0.05, pygame.key.get_pressed())
    assert not g4.remote_beams, "baseline must have no remote beams"
    base = beam_pixels(g4.screen, g4.cam, g4.remote_beams)
    assert base == 0, "baseline must have no beam pixels: %d" % base
    # Fire: push a remote beam on-screen (a horizontal beam to the right
    # of the ghost, away from its hull) and render.
    push_stream(g4, 1.0)
    _advance_render_point(g4)
    g4.remote_beams.append([pygame.Vector2(50.0, 0.0),
                            pygame.Vector2(250.0, 0.0), 0.0, BEAM_TTL])
    g4.predicted_view(0.05, pygame.key.get_pressed())
    hit = beam_pixels(g4.screen, g4.cam, g4.remote_beams)
    assert hit > 0, ("predicted_view must render the remote beam "
                     "(beam-colored pixels along the line): hit=%d" % hit)
    print("PASS: PIXEL — predicted_view renders the remote beam (%d beam "
          "pixels vs %d baseline)" % (hit, base))

    print("ALL PASS: 10.4 (remote player's laser beams)")


if __name__ == "__main__":
    main()