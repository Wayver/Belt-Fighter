"""Plan 17 — client-side asteroid polygons: wire + interp + render parity.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_17_rock_shapes

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens.

Session 1 (wire + buffer + render) is implemented and verified; this is the
Session-2 gate that locks the behavior in. Three independent proofs:

  1. WIRE — the host's RockShapeTracker delta-compresses rock shapes:
       a. FIRST-SIGHT  — a fresh tracker sends a rock's full verts the first
                         time its id appears (and records it).
       b. REPEAT       — a second snapshot of the SAME rocks (same ids)
                         carries verts=None for every rock (already sent).
       c. FRESH ID     — a rock with a fresh id (a split child) carries full
                         verts alongside the None repeats.
       d. RESET        — tracker.reset() re-sends full verts for ids the
                         tracker had already sent (a game reset restarts rock
                         ids at 1, so the host must re-send; the client's
                         registry is simply overwritten).
       e. JSON         — the pruned snapshot survives a real JSON round-trip
                         (the wire is JSON) and keeps the 11-tuple shape with
                         the rng slot as the None placeholder.
     The snapshot is a REAL Game.snapshot() (a real 9-tuple rock), so the
     test exercises the production serialize path, not a hand-built fake.

  2. INTERP — interp_positions lerp's the rock's ANGLE across the window
     (x/y are lerp'd exactly as before — already covered by the
     test_interpolation alpha/turnover batteries, which now emit the
     (id, x, y, angle) entry). Two cases:
       a. PLAIN  — angle 0.0 -> 1.0, no wrap: the lerp is the plain midpoint
                  at alpha=0.5, and the ENDPOINTS are exact (alpha=0 -> prev,
                  alpha=1 -> curr).
       b. WRAP   — angle 3.0 -> -3.0 (a ~0.28 rad CCW turn across the
                  +/-pi seam): the lerp takes the SHORT way around (the
                  wrapped delta), not the long way (a plain lerp of the raw
                  angles would swing ~5.7 rad the wrong way). A bug that
                  drops the angle or lerps it raw fails here.
     The rock is read in the WIRE 5-tuple form (what the network feeds
     interp_positions) AND the SIM 9-tuple form (loopback/tests) — both must
     agree, which is the format-tolerance contract Session 1 added.

  3. PIXEL PARITY — the client draws the rock PIXEL-IDENTICALLY to the host.
     The host's model path draws a rock via `_draw_world_asteroid(screen,
     cam, pos, angle, verts)`; the client's `_draw_remote_rock` calls the
     SAME plain-data helper, so parity is by construction. This proof makes
     that concrete: one static rock (vel=0, spin=0) at a locked camera point,
     drawn once by the host's helper and once by the client's
     `_draw_remote_rock` on a clean (fog-free) screen — the pixel diff is
     exactly 0. A full-frame host-vs-client diff is NOT used: the fog of war
     is applied in both draw() and predicted_view() with peer-specific light
     sources (the host lights from its live ship, the client from its ghost),
     so a full frame differs in the fog/ship/HUD regions regardless of the
     rock. The rock itself is what this plan changes, so the rock's pixels
     are the parity subject.
     A second, end-to-end check feeds a REAL wire snapshot (serialized with a
     fresh tracker -> first-sight verts) through the client's push_snapshot
     seam and confirms the rock-shape registry is populated with the rock's
     exact verts — the seam that makes the registry lookup in the render
     path work.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import math
import random

import pygame

from .config import WIDTH, HEIGHT, ROCK_SIZES
from .fog import make_light_texture
from .game import Game, STEP, _draw_world_asteroid
from .net import RockShapeTracker, serialize_snapshot, deserialize_snapshot
from .netcode import interp_positions
from .asteroid import Asteroid
from .hulls import PLAYER_HULLS
from .ship import Ship

SEED = 1234
# A controlled, on-screen world point (the camera is locked here, so it is
# the screen center).
ROCK_X, ROCK_Y = 500.0, 400.0
ROCK_SIZE = "large"
ROCK_ANGLE = 0.7          # a non-trivial orientation (not 0, not a multiple
                          # of pi/2) so a rotation bug would show.
# A 10-vertex rock shape (the real _make_rock produces 10 verts).
ROCK_VERTS = tuple(
    (ROCK_SIZES[ROCK_SIZE]["radius"] * 0.9 * math.cos(2 * math.pi * i / 10),
     ROCK_SIZES[ROCK_SIZE]["radius"] * 0.9 * math.sin(2 * math.pi * i / 10))
    for i in range(10))


def check(label, cond, extra=""):
    print(("PASS: " if cond else "FAIL: ") + label
          + (("  " + extra) if extra else ""))
    return bool(cond)


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def _sim_rock(rid, x, y, angle, verts):
    """A SIM-form 9-tuple rock (id, x, y, vx, vy, size, angle, spin, verts)
    — the shape serialize_snapshot consumes."""
    return (rid, x, y, 0.0, 0.0, ROCK_SIZE, angle, 0.0, verts)


def _wire_rock(rid, x, y, angle, verts):
    """A WIRE-form 5-tuple rock [id, x, y, angle, verts] — the shape the
    network feeds interp_positions (verts None for a repeat id)."""
    return [rid, x, y, angle, verts]


def _snap11(rock):
    """A minimal 11-tuple snapshot with a single rock at index 5 (enough for
    serialize_snapshot: list(snap); prune snap[5]; snap[6]=None) AND for
    interp_positions (which iterates the entity slots [0]-[4] — all empty
    here, so only the rock is interpolated)."""
    return [[], [], [], [], [], [rock], None, False, 0.0, 0, 0]


# --- 1. WIRE: RockShapeTracker delta-compression ---------------------------

def test_wire(res):
    screen, font, big_font, light_tex, fog_surf, light_surf = res
    ok = True
    # A REAL Game.snapshot() (a real 9-tuple rock) so the production
    # serialize path is exercised, not a hand-built fake.
    Asteroid._next_id = 1
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED)
    g.asteroids = [Asteroid((ROCK_X, ROCK_Y), ROCK_SIZE, vel=(0, 0),
                            rng=random.Random(0))]
    snap = g.snapshot()
    rid = snap[5][0][0]
    src_verts = snap[5][0][8]

    # (a) FIRST-SIGHT: a fresh tracker sends full verts for a new id.
    tr = RockShapeTracker()
    w1 = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap, tr))))
    r1 = w1[5][0]
    first_ok = (len(r1) == 5 and r1[0] == rid
                and isinstance(r1[4], list) and len(r1[4]) == 10
                and [(vx, vy) for (vx, vy) in r1[4]]
                    == [(vx, vy) for (vx, vy) in src_verts])
    ok &= check("wire first-sight — a fresh tracker sends the rock's full "
                "verts (10 pairs, exact) on first sight of its id",
                first_ok, "rock=%r" % (r1[:4],))

    # (b) REPEAT: a second snapshot of the SAME rocks (same ids) carries
    # verts=None for every rock (the tracker already sent them).
    w2 = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap, tr))))
    r2 = w2[5][0]
    repeat_ok = (r2[0] == rid and r2[4] is None
                 and r2[1] == r1[1] and r2[2] == r1[2] and r2[3] == r1[3])
    ok &= check("wire repeat — a second snapshot of the same id carries "
                "verts=None (x/y/angle unchanged)",
                repeat_ok, "rock=%r" % (r2,))

    # (c) FRESH ID: a rock with a fresh id (a split child) carries full
    # verts alongside the None repeat.
    child = _sim_rock(rid + 1000, 10.0, 20.0, 0.5, ROCK_VERTS)
    snap_c = list(snap)
    snap_c[5] = tuple(snap[5]) + (child,)
    w3 = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap_c, tr))))
    by_id = {r[0]: r for r in w3[5]}
    fresh_ok = (by_id[rid][4] is None
                and isinstance(by_id[rid + 1000][4], list)
                and len(by_id[rid + 1000][4]) == 10)
    ok &= check("wire fresh id — a split child (fresh id) carries full "
                "verts while the repeat id stays None",
                fresh_ok, "ids=%r" % sorted(by_id))

    # (d) RESET: tracker.reset() re-sends full verts for ids the tracker had
    # already sent (a game reset restarts rock ids at 1 -> the host re-sends;
    # the client's registry is overwritten).
    tr.reset()
    w4 = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap, tr))))
    reset_ok = (isinstance(w4[5][0][4], list) and len(w4[5][0][4]) == 10
                and len(tr) == 1)
    ok &= check("wire reset — after tracker.reset() the same id re-sends "
                "full verts (a new run re-sends shapes)",
                reset_ok, "rock=%r" % (w4[5][0][:4],))

    # (e) JSON: the pruned snapshot survives a real JSON round-trip and keeps
    # the 11-tuple shape with the rng slot as the None placeholder.
    shape_ok = (len(w1) == 11 and w1[6] is None
                and isinstance(w1[5][0][0], int))
    ok &= check("wire json — the pruned snapshot survives a JSON round-trip "
                "(11-tuple intact, rng slot None, id an int)",
                shape_ok, "len=%d rng=%r" % (len(w1), w1[6]))
    return ok


# --- 2. INTERP: the rock's angle is lerp'd across the window ---------------

def _interp_rock(prev, curr, alpha):
    """The single rock's (id, x, y, angle) from interp_positions."""
    out = interp_positions(prev, curr, alpha, dt=0.0)["asteroids"]
    assert len(out) == 1, "expected exactly one rock, got %r" % (out,)
    return out[0]


def test_interp():
    ok = True
    # (a) PLAIN — angle 0.0 -> 1.0 (no wrap). x/y are lerp'd exactly as
    # before; the angle is the plain midpoint at alpha=0.5 and EXACT at the
    # endpoints (lerp/lerp_angle return the endpoint by identity).
    plain_prev = _snap11(_wire_rock(1, 100.0, 200.0, 0.0, None))
    plain_curr = _snap11(_wire_rock(1, 110.0, 210.0, 1.0, None))
    a0 = _interp_rock(plain_prev, plain_curr, 0.0)
    a1 = _interp_rock(plain_prev, plain_curr, 1.0)
    a5 = _interp_rock(plain_prev, plain_curr, 0.5)
    plain_ok = (
        a0 == (1, 100.0, 200.0, 0.0)
        and a1 == (1, 110.0, 210.0, 1.0)
        and a5[0] == 1
        and abs(a5[1] - 105.0) < 1e-9 and abs(a5[2] - 205.0) < 1e-9
        and abs(a5[3] - 0.5) < 1e-9)
    ok &= check("interp plain — angle 0.0->1.0 lerps to the midpoint "
                "(0.5) at alpha=0.5, exact at the endpoints; x/y lerp as "
                "before", plain_ok, "a0=%r a1=%r a5=%r" % (a0, a1, a5))

    # (b) WRAP — angle 3.0 -> -3.0: the wrapped delta is ~0.283 rad CCW
    # (the short way across the +/-pi seam). The lerp must take the SHORT
    # way: at alpha=0.5 the angle is ~3.1416 (pi), NOT the long-way value
    # (~-2.5) a plain lerp of the raw angles would give.
    wrap_prev = _snap11(_wire_rock(1, 100.0, 200.0, 3.0, None))
    wrap_curr = _snap11(_wire_rock(1, 110.0, 210.0, -3.0, None))
    w5 = _interp_rock(wrap_prev, wrap_curr, 0.5)
    da = (-3.0 - 3.0 + math.pi) % (2 * math.pi) - math.pi   # ~0.2832
    want_mid = 3.0 + da * 0.5                                # ~3.1416
    wrap_ok = (abs(w5[3] - want_mid) < 1e-9
               and abs(w5[3] - math.pi) < 1e-3
               and abs(w5[3] - (-2.5)) > 1.0)   # not the long-way value
    ok &= check("interp wrap — angle 3.0->-3.0 takes the SHORT way around "
                "(~pi at alpha=0.5), not the long way (~-2.5)",
                wrap_ok, "got=%r want=%r" % (w5[3], want_mid))

    # (c) FORMAT TOLERANCE — the same rock read in the SIM 9-tuple form
    # (loopback/tests) and the WIRE 5-tuple form (the network) must give the
    # SAME interpolated (id, x, y, angle). This is the contract Session 1
    # added (the buffer is fed either form); a regression that hard-codes one
    # form's angle index (sim [6] vs wire [3]) fails here.
    sim_prev = _snap11(_sim_rock(1, 100.0, 200.0, 3.0, ROCK_VERTS))
    sim_curr = _snap11(_sim_rock(1, 110.0, 210.0, -3.0, ROCK_VERTS))
    s5 = _interp_rock(sim_prev, sim_curr, 0.5)
    same_ok = (s5[0] == w5[0] and abs(s5[1] - w5[1]) < 1e-9
               and abs(s5[2] - w5[2]) < 1e-9
               and abs(s5[3] - w5[3]) < 1e-9)
    ok &= check("interp format — the SIM 9-tuple and WIRE 5-tuple forms "
                "interpolate to the same (id, x, y, angle)",
                same_ok, "sim=%r wire=%r" % (s5, w5))
    return ok


# --- 3. PIXEL PARITY: the client draws the rock identically to the host ----

def _lock_cam(g):
    """Lock the camera at (ROCK_X, ROCK_Y) with zero lead so the
    world->screen mapping is exact: world (ROCK_X, ROCK_Y) -> screen center.
    _draw_world_asteroid / _draw_remote_rock do NOT call cam.update, so the
    lock is stable for the draw."""
    g.cam.pos = pygame.Vector2(ROCK_X, ROCK_Y)
    g.cam.lead = 0.0


def _grab(screen):
    return pygame.image.tostring(screen, "RGB")


def _blank(screen):
    screen.fill((0, 0, 0))


def test_pixel_parity(res):
    screen, font, big_font, light_tex, fog_surf, light_surf = res
    ok = True
    # A bare client Game (the _draw_remote_rock path). It is NEVER stepped —
    # only its _draw_remote_rock + cam are used, so the construction state
    # (stars, ship, etc.) is irrelevant.
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=client_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    _lock_cam(g)

    # The host's helper (the model path) draws the rock on a clean screen.
    _blank(screen)
    _draw_world_asteroid(screen, g.cam, (ROCK_X, ROCK_Y), ROCK_ANGLE,
                         ROCK_VERTS)
    host_px = _grab(screen)

    # The client's _draw_remote_rock (the predicted_view/remote_view path)
    # draws the SAME rock from the SAME verts on a clean screen.
    _blank(screen)
    g._draw_remote_rock(screen, ROCK_X, ROCK_Y, ROCK_ANGLE, ROCK_VERTS)
    client_px = _grab(screen)

    diff = sum(1 for a, b in zip(host_px, client_px) if a != b)
    ok &= check("pixel parity — the client's _draw_remote_rock draws the "
                "rock PIXEL-IDENTICALLY to the host's _draw_world_asteroid "
                "(0 bytes differ on a clean screen)",
                diff == 0, "%d bytes differ" % diff)

    # The fallback: a registry miss (verts=None) must still draw something
    # (the coarse circle) and never crash — graceful degradation.
    _blank(screen)
    try:
        g._draw_remote_rock(screen, ROCK_X, ROCK_Y, ROCK_ANGLE, None)
        fallback_px = _grab(screen)
        fallback_ok = any(b != 0 for b in fallback_px)
    except Exception as e:
        fallback_ok = False
        fallback_px = None
    ok &= check("pixel fallback — a registry miss (verts=None) draws the "
                "coarse circle without crashing",
                fallback_ok, "changed=%s"
                % (fallback_px is not None and any(b != 0 for b in fallback_px)))
    return ok


def test_registry_seam(res):
    """End-to-end: a REAL wire snapshot (fresh tracker -> first-sight verts)
    fed through the client's push_snapshot seam populates the rock-shape
    registry with the rock's exact verts — the seam that makes the render
    path's registry lookup work."""
    screen, font, big_font, light_tex, fog_surf, light_surf = res
    ok = True
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=client_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))

    # A real 2P Game -> a real snapshot (real 9-tuple rocks) -> serialize it
    # with a FRESH tracker (first sight -> full verts) -> JSON round-trip.
    Asteroid._next_id = 1
    tmp = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
               seed=SEED, players=2)
    tmp.asteroids = [Asteroid((ROCK_X, ROCK_Y), ROCK_SIZE, vel=(0, 0),
                              rng=random.Random(0))]
    tmp.players[1].pos = pygame.Vector2(ROCK_X, ROCK_Y)
    tmp.players[1].vel = pygame.Vector2(0, 0)
    snap = tmp.snapshot()
    rid = snap[5][0][0]
    src_verts = [(vx, vy) for (vx, vy) in snap[5][0][8]]
    wire = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap, RockShapeTracker()))))

    # Feed the wire snapshot through the client's seam (now=None -> a pure
    # full-snap seed of the ghost; the registry is populated from snap[5]).
    g.push_snapshot(0.0, wire, now=None)
    got = g.rock_shapes.get(rid)
    seam_ok = (got is not None
               and [tuple(v) for v in got] == src_verts)
    ok &= check("registry seam — push_snapshot(wire snap) populates "
                "rock_shapes[id] with the rock's exact verts",
                seam_ok, "got=%r" % (got,))

    # A REPEAT wire snapshot (verts=None) must NOT clobber the registry.
    # Re-serialize with a tracker that has ALREADY seen the id -> None.
    tr = RockShapeTracker()
    serialize_snapshot(snap, tr)          # first sight (records the id)
    wire2 = deserialize_snapshot(
        json.loads(json.dumps(serialize_snapshot(snap, tr))))
    assert wire2[5][0][4] is None, "expected a repeat (verts=None)"
    g.push_snapshot(0.1, wire2, now=None)
    kept_ok = [tuple(v) for v in g.rock_shapes.get(rid, [])] == src_verts
    ok &= check("registry seam — a repeat wire snap (verts=None) does not "
                "clobber the stored shape", kept_ok,
                "got=%r" % (g.rock_shapes.get(rid),))
    return ok


def main():
    res = make_resources()
    ok = True
    print("PLAN 17 — client-side asteroid polygons (wire + interp + parity)")
    ok &= test_wire(res)
    ok &= test_interp()
    ok &= test_pixel_parity(res)
    ok &= test_registry_seam(res)
    pygame.quit()
    print("\nROCK SHAPES (plan 17):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()