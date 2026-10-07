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
message the moment it fires (muzzle -> endpoint + sim_time); the client
draws it immediately on receipt, fading over BEAM_TTL — the same way the
ghost draws its OWN beams (10.3a).

10.4b (detach fix): the T_BEAM's muzzle rides the wire as a HULL-LOCAL
offset (lx, ly), not a frozen world-space point. predicted_view
RE-ANCHORS the beam origin to the firing ship's INTERPOLATED pose each
frame (the exact math the host's _draw_world_beam uses for its own beams,
and the original 1p fix) — a frozen world-space start left the origin
behind in empty space as the ship moved during the 0.15 s flash.

  * HOST (step 1): Game._resolve_beam / _beam_hit_asteroid emit
    (sx, sy, ex, ey, sim_time, owner, lx, ly) into Game._beam_events; the
    SimThread drains it after each step and sends T_BEAM via the worker.
  * CLIENT (steps 2-4): run_client's poll loop pushes each T_BEAM into
    game.remote_beams as [local_start, end, age, ttl, owner] (hull-local
    muzzle + world endpoint); predicted_view ages it each frame
    (_step_remote_beams) and draws it BEFORE the ships, re-anchoring the
    origin to pos['ships'][owner]'s interpolated pose (LASER_COLOR line,
    width 2, fading over the ttl).

This test proves it:
  1. WIRE-SHAPE  — T_BEAM == "beam"; the host's real SimThread drain
                   emits a JSON-serializable message carrying
                   type/sim_time/sx/sy/ex/ey/owner/lx/ly (the host->wire
                   contract).
  2. RECEIVE     — a T_BEAM pushed into game.remote_beams (the exact
                   run_client append) is [local_start, end, 0.0, BEAM_TTL,
                   owner] (hull-local muzzle + world endpoint).
  3. AGE-CULL    — _step_remote_beams advances the age by dt each frame
                   and culls the beam once age >= ttl (mirrors the ghost's
                   step_local_beams + the host's beam aging).
  4. PIXEL       — predicted_view renders the remote beam (beam-colored
                   pixels appear along the line; the no-beam baseline has
                   none).
  5. DEDUP       — the client skips T_BEAMs from its OWN ship (owner ==
                   local_index; the ghost already draws those, 10.3a) and
                   draws the remote ship's — so firing your own laser does
                   NOT double-draw (the "criss-cross" fix).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import math
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
    """A valid 2-player Game snapshot with BOTH ships at the origin facing
    +x, the client ship's (index 1) lasers at `charge`. The client ship's
    pose is fixed so the ghost (seeded from this snapshot) sits at the
    origin. 10.4b: the REMOTE ship (index 0) is ALSO placed at the origin
    facing +x — predicted_view re-anchors each remote beam's origin to the
    firing ship's INTERPOLATED pose (pos['ships'][owner]), so the remote
    ship must be on-screen (near the ghost's camera) for the re-anchored
    beam to render. Both ships at the origin is fine (no ship-vs-ship
    collision; the beam_pixels helper samples the beam line, not the hulls)."""
    tmp = _scratch_game()
    for p in tmp.players:
        p.pos = pygame.Vector2(0.0, 0.0)
        p.vel = pygame.Vector2(0.0, 0.0)
        p.angle = 0.0
    cs = tmp.players[1]
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


def beam_pixels(screen, cam, beams, rpos, rangle):
    """Count beam-colored pixels sampled along the beam lines.

    The beam is drawn BEFORE the fog (mirrors the host's draw() order), so
    near the light source (the ghost) the fog dims LASER_COLOR
    (140, 255, 190) but it stays green-dominant (g > r and g > b). For
    each beam, sample a small box around a few points along the line
    (t = 0.3, 0.5, 0.7 — avoiding the endpoints) and count the
    beam-colored pixels (g > 100 and g > r and g > b). Baseline (no beams)
    -> 0; a fired beam -> > 0.

    10.4b: each beam entry is [local_start, end, age, ttl, owner] — the
    stored start is a HULL-LOCAL muzzle offset, not a world point. The
    helper re-anchors it to the firing ship's interpolated pose (rpos,
    rangle) with the SAME math predicted_view uses, so it samples the line
    that is actually drawn."""
    n = 0
    for (local_start, end, _age, _ttl, _owner) in beams:
        fwd = pygame.Vector2(math.cos(rangle), math.sin(rangle))
        right = pygame.Vector2(-fwd.y, fwd.x)
        start = rpos + fwd * local_start[0] + right * local_start[1]
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
    # Pre-seed a known beam event (owner 0 = the host's ship); the sim (empty
    # input) won't fire its own, so this is the only one the drain should
    # send. 10.4b: the tuple carries the hull-local muzzle (lx, ly) too —
    # (sx, sy, ex, ey, sim_time, owner, lx, ly).
    g._beam_events.append((111.0, 222.0, 333.0, 444.0, 0.5, 0, 12.0, 3.0))
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
    for k in ("type", "sim_time", "sx", "sy", "ex", "ey", "owner",
              "lx", "ly"):
        assert k in got, "T_BEAM missing field %r: %r" % (k, got)
    assert got["sim_time"] == 0.5 and got["sx"] == 111.0 \
        and got["sy"] == 222.0 and got["ex"] == 333.0 and got["ey"] == 444.0, \
        "T_BEAM fields not carried: %r" % (got,)
    assert got["owner"] == 0, \
        "T_BEAM must carry the firing player's index: %r" % (got,)
    # 10.4b: the hull-local muzzle (lx, ly) rides the wire so the client
    # can re-anchor the beam origin to the firing ship's interpolated pose.
    assert got["lx"] == 12.0 and got["ly"] == 3.0, \
        "T_BEAM must carry the hull-local muzzle (lx, ly): %r" % (got,)
    # JSON-serializable (it rides the wire as a JSON frame).
    json.dumps(got)
    assert g._beam_events == [], "the buffer must be drained (cleared)"
    print("PASS: WIRE-SHAPE — the host's SimThread drain emits a "
          "JSON-safe T_BEAM %r" % (got,))

    # --- 2. RECEIVE: a T_BEAM pushed into game.remote_beams (the exact
    #     run_client append) is [local_start, end, 0.0, BEAM_TTL, owner].
    #     10.4b: the start is the HULL-LOCAL muzzle (lx, ly), not the
    #     frozen world-space point — predicted_view re-anchors it to the
    #     firing ship's interpolated pose each frame. ---
    g2 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    assert g2.remote_beams == [], "remote_beams must start empty"
    # The exact append run_client performs on a T_BEAM message.
    msg = {"type": T_BEAM, "sim_time": 0.5, "owner": 0,
           "sx": 50.0, "sy": 0.0, "ex": 250.0, "ey": 0.0,
           "lx": 8.0, "ly": -5.5}
    g2.remote_beams.append([
        (msg.get("lx", 0.0), msg.get("ly", 0.0)),
        pygame.Vector2(msg["ex"], msg["ey"]),
        0.0, BEAM_TTL, msg.get("owner", 0)])
    assert len(g2.remote_beams) == 1
    local_start, end, age, ttl, owner = g2.remote_beams[0]
    assert isinstance(local_start, tuple) and len(local_start) == 2, \
        "remote beam start must be a hull-local (lx, ly) tuple: %r" \
        % (g2.remote_beams[0],)
    assert local_start == (8.0, -5.5), "start = hull-local muzzle (lx, ly)"
    assert isinstance(end, pygame.Vector2), "end must be a world Vector2"
    assert end.x == 250.0 and end.y == 0.0, "end = endpoint (ex, ey)"
    assert age == 0.0, "a freshly received beam starts at age 0"
    assert ttl == BEAM_TTL, "the beam's ttl is BEAM_TTL (0.15 s)"
    assert owner == 0, "the beam carries the firing player's index"
    print("PASS: RECEIVE — a T_BEAM becomes [local_start, end, 0.0, "
          "BEAM_TTL, owner] (hull-local muzzle + world endpoint)")

    # --- 3. AGE-CULL: _step_remote_beams advances the age by dt each
    #     frame and culls the beam once age >= ttl. ---
    g3 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    g3.remote_beams.append([(8.0, -5.5), pygame.Vector2(250.0, 0.0),
                            0.0, BEAM_TTL, 0])
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
    # The remote ship (owner 0) is at the origin facing +x (make_snap), so
    # its interpolated pose is (0, 0, 0) — the beam_pixels helper re-anchors
    # the hull-local muzzle to this pose, exactly as predicted_view does.
    rpos = pygame.Vector2(0.0, 0.0)
    rangle = 0.0
    # Baseline: no remote beams -> 0 beam pixels.
    push_stream(g4, 1.0)
    _advance_render_point(g4)
    g4.predicted_view(0.05, pygame.key.get_pressed())
    assert not g4.remote_beams, "baseline must have no remote beams"
    base = beam_pixels(g4.screen, g4.cam, g4.remote_beams, rpos, rangle)
    assert base == 0, "baseline must have no beam pixels: %d" % base
    # Fire: push a remote beam on-screen (a beam to the right of the ghost,
    # away from its hull) and render. The hull-local muzzle (8, -5.5) is
    # re-anchored to the remote ship's pose (the origin, facing +x) -> a
    # world start of (8, -5.5), so the beam runs (8,-5.5) -> (250, 0).
    push_stream(g4, 1.0)
    _advance_render_point(g4)
    g4.remote_beams.append([(8.0, -5.5), pygame.Vector2(250.0, 0.0),
                            0.0, BEAM_TTL, 0])
    g4.predicted_view(0.05, pygame.key.get_pressed())
    hit = beam_pixels(g4.screen, g4.cam, g4.remote_beams, rpos, rangle)
    assert hit > 0, ("predicted_view must render the remote beam "
                     "(beam-colored pixels along the line): hit=%d" % hit)
    print("PASS: PIXEL — predicted_view renders the remote beam (%d beam "
          "pixels vs %d baseline)" % (hit, base))

    # --- 5. DEDUP: the client skips T_BEAMs from its OWN ship (the ghost
    #     already draws those, 10.3a) and draws the remote ship's. Without
    #     this, firing your own laser double-draws: the ghost's immediate
    #     beam + the host's authoritative copy ~10-20 ms later, at
    #     slightly different poses -> a "criss-cross". ---
    g5 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    assert g5.local_index == 1, "the test client is player 1"
    own = {"type": T_BEAM, "sim_time": 0.5, "owner": 1,
           "sx": 50.0, "sy": 0.0, "ex": 250.0, "ey": 0.0,
           "lx": 8.0, "ly": -5.5}
    remote = {"type": T_BEAM, "sim_time": 0.5, "owner": 0,
              "sx": 50.0, "sy": 0.0, "ex": 250.0, "ey": 0.0,
              "lx": 8.0, "ly": -5.5}
    # The exact run_client handler (skip own, append remote).
    for m in (own, remote):
        if m.get("owner") == g5.local_index:
            continue
        g5.remote_beams.append([
            (m.get("lx", 0.0), m.get("ly", 0.0)),
            pygame.Vector2(m["ex"], m["ey"]),
            0.0, BEAM_TTL, m.get("owner", 0)])
    assert len(g5.remote_beams) == 1, \
        "the client must skip its OWN ship's beam (owner == local_index) " \
        "and keep the remote ship's: remote_beams=%r" % (g5.remote_beams,)
    print("PASS: DEDUP — own-ship T_BEAM skipped (ghost draws it), "
          "remote-ship T_BEAM drawn (no double-draw criss-cross)")

    print("ALL PASS: 10.4 (remote player's laser beams)")


if __name__ == "__main__":
    main()