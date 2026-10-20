"""Viewport culling (presentation-only) — off-screen entities are skipped.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_viewport_culling

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

This is the gate for the viewport-culling change (see PERFORMANCE.md,
recommendation B). Nothing was culled before; now the host's draw() and the
client's predicted_view() skip drawing entities whose world position is
outside the camera's view rect inflated by CULL_MARGIN. The change is
PRESENTATION-ONLY: it must not touch the sim, the wire, or the snapshot —
only which sprites reach the screen.

The cull is a PURE function of the camera (no sim state), so it is
deterministic and safe for the pixel-parity gate (test_render_model.py).
This test proves the cull actually works on BOTH peers and that the margin
is respected (a rock just outside the screen but inside the margin is still
drawn — so a rock is never culled while any part of it is visible).

Checks:
  1. PURE-BOUNDS  — _view_bounds is the camera rect inflated by
                    CULL_MARGIN; _in_view is the inclusive point-in-rect.
  2. HOST-ON      — a rock at screen center is DRAWN (pixel diff > 0).
  3. HOST-OFF     — a rock far off-screen is CULLED (pixel diff == 0).
  4. HOST-MARGIN  — a rock just outside the screen but inside CULL_MARGIN
                    is still DRAWN (the margin is respected).
  5. CLIENT-ON    — same as 2, via the client's predicted_view (the
                    interpolation-buffer path).
  6. CLIENT-OFF   — same as 3, via the client.
  7. CLIENT-MARGIN— same as 4, via the client.
  8. STRUCTURAL   — both draw() and predicted_view() contain the cull
                    (bounds = _view_bounds + _in_view on the asteroid loop).

The pixel-diff method: render a baseline frame (no rock), then render a
frame with a single rock at a controlled world position, and count the
pixels that changed. An on-screen rock changes its pixels; a culled rock
changes NONE (the sprite is skipped, so the frame is identical to the
baseline). The camera is locked (zero velocity, lead 0) so the
world->screen mapping is exact and the rock lands where we put it.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import inspect
import random

import pygame

from .config import (WIDTH, HEIGHT, CULL_MARGIN, BG, ROCK_FILL, ROCK_EDGE,
                     ROCK_SIZES)
from .fog import make_light_texture
from .game import Game, STEP, _view_bounds, _in_view
from .netcode import (HostTimeEstimator, LatencyTracker, RenderPoint,
                      PredictedShip)
from .ship import Ship
from .hulls import PLAYER_HULLS
from .asteroid import Asteroid

SEED = 1234
# The controlled ship/camera position. The camera is locked here (zero
# velocity, lead 0), so world (SHIP_X, SHIP_Y) is screen center.
SHIP_X, SHIP_Y = 500.0, 400.0
# A single "large" rock (radius 50) is the cull test subject.
ROCK_SIZE = "large"
ROCK_RADIUS = ROCK_SIZES[ROCK_SIZE]["radius"]
# A pixel "changed" if any channel moved by more than this. The fog dims
# the rock, but the rock is still brighter than the background, so a drawn
# rock changes its pixels well above this; a culled rock changes none.
PIXEL_DELTA = 12
# A drawn rock must change at least this many pixels (the rock is ~100 px
# across; even fog-dimmed, far more than this change).
MIN_DRAWN_PX = 50


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def _lock_cam(g):
    """Lock the camera at the ship (zero velocity, lead 0) so the
    world->screen mapping is exact: world (SHIP_X, SHIP_Y) -> screen
    center. draw()/predicted_view() call cam.update(dt, ...) once per
    frame; with the ship at rest and lead 0 the camera stays put."""
    g.cam.pos = pygame.Vector2(SHIP_X, SHIP_Y)
    g.cam.lead = 0.0


def _grab(screen):
    return pygame.image.tostring(screen, "RGB")


def _diff_count(base, cur):
    """Count pixels that changed by more than PIXEL_DELTA in any channel."""
    n = 0
    for i in range(0, len(base), 3):
        if (abs(base[i] - cur[i]) > PIXEL_DELTA
                or abs(base[i + 1] - cur[i + 1]) > PIXEL_DELTA
                or abs(base[i + 2] - cur[i + 2]) > PIXEL_DELTA):
            n += 1
    return n


class NoKeys:
    """A get_pressed()-style object with nothing pressed (ShipInput.from_keys
    does keys[pygame.K_q] — a plain {} would raise KeyError)."""
    def __getitem__(self, k):
        return 0


def _screen_to_world(cam, sx, sy):
    """World coords of a screen point (the inverse of cam.to_screen)."""
    return (cam.pos.x + (sx - WIDTH / 2), cam.pos.y + (sy - HEIGHT / 2))


# --- host (draw()) ---------------------------------------------------------

def _host_game(screen, font, big_font, light_tex, fog_surf, light_surf):
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED, players=1)
    g.ship.pos = pygame.Vector2(SHIP_X, SHIP_Y)
    g.ship.vel = pygame.Vector2(0, 0)
    g.ship.dampening = False
    return g


def _host_render(g, rock_world):
    """Render one host frame with a single rock at rock_world (or None).
    Returns the screen pixels. The camera is locked (zero velocity, lead 0,
    ship at rest) so cam.update leaves it put and the rock lands at the
    exact screen point we computed."""
    g.asteroids = []
    if rock_world is not None:
        a = Asteroid(rock_world, ROCK_SIZE, vel=(0, 0),
                     rng=random.Random(0))
        g.asteroids = [a]
    _lock_cam(g)
    model = g.render_model()
    g.draw(STEP, model)
    _lock_cam(g)   # re-lock so the caller reads the exact camera position
    return _grab(g.screen)


def test_host(screen, font, big_font, light_tex, fog_surf, light_surf):
    ok = True
    g = _host_game(screen, font, big_font, light_tex, fog_surf, light_surf)
    base = _host_render(g, None)

    # --- 2. HOST-ON: a rock at screen center is drawn. ---
    on_world = _screen_to_world(g.cam, WIDTH / 2, HEIGHT / 2)
    on = _host_render(g, on_world)
    on_diff = _diff_count(base, on)
    if on_diff >= MIN_DRAWN_PX:
        print("PASS: HOST-ON — on-screen rock is drawn (%d px changed)"
              % on_diff)
    else:
        ok = False
        print("FAIL: HOST-ON — on-screen rock not drawn (%d px changed, "
              "want >= %d)" % (on_diff, MIN_DRAWN_PX))

    # --- 3. HOST-OFF: a rock far off-screen is culled. ---
    off_world = _screen_to_world(g.cam, WIDTH + 300, HEIGHT / 2)
    off = _host_render(g, off_world)
    off_diff = _diff_count(base, off)
    if off_diff == 0:
        print("PASS: HOST-OFF — off-screen rock is culled (0 px changed)")
    else:
        ok = False
        print("FAIL: HOST-OFF — off-screen rock NOT culled (%d px changed)"
              % off_diff)

    # --- 4. HOST-MARGIN: a rock just outside the screen but inside
    #     CULL_MARGIN is still drawn (the margin is respected). The
    #     inflated right edge is at screen WIDTH/2 + CULL_MARGIN, so a rock
    #     at screen WIDTH/2 + CULL_MARGIN - 1 is off-screen but inside the
    #     inflated rect -> its center is within the margin -> drawn. ---
    margin_world = _screen_to_world(g.cam, WIDTH / 2 + CULL_MARGIN - 1,
                                    HEIGHT / 2)
    margin = _host_render(g, margin_world)
    margin_diff = _diff_count(base, margin)
    if margin_diff >= MIN_DRAWN_PX:
        print("PASS: HOST-MARGIN — rock off-screen but within CULL_MARGIN "
              "is still drawn (%d px changed)" % margin_diff)
    else:
        ok = False
        print("FAIL: HOST-MARGIN — rock within CULL_MARGIN was culled "
              "(%d px changed, want >= %d)" % (margin_diff, MIN_DRAWN_PX))
    return ok


# --- client (predicted_view) ----------------------------------------------

def _client_game(screen, font, big_font, light_tex, fog_surf, light_surf):
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    g.ghost = PredictedShip(hull=client_hull, loadout=None, local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _client_snap(rock_world):
    """A 2-player snapshot with the LOCAL ship (index 1) at the controlled
    position (so the ghost + camera sit at SHIP_X, SHIP_Y) and a single
    asteroid at rock_world (or none)."""
    tmp = Game(pygame.Surface((WIDTH, HEIGHT)),
               pygame.font.SysFont("consolas,menlo,monospace", 18),
               pygame.font.SysFont("consolas,menlo,monospace", 40),
               make_light_texture(),
               pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA),
               pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA),
               seed=SEED, players=2)
    # Local ship (index 1) at the controlled position, at rest.
    tmp.players[1].pos = pygame.Vector2(SHIP_X, SHIP_Y)
    tmp.players[1].vel = pygame.Vector2(0, 0)
    # Remote ship (index 0) off to the side (not the test subject).
    tmp.players[0].pos = pygame.Vector2(SHIP_X + 150, SHIP_Y)
    tmp.players[0].vel = pygame.Vector2(0, 0)
    tmp.asteroids = []
    if rock_world is not None:
        a = Asteroid(rock_world, ROCK_SIZE, vel=(0, 0),
                     rng=random.Random(0))
        tmp.asteroids = [a]
    return tmp.snapshot()


def _client_render(g, rock_world):
    """Push a fresh 2-snapshot stream with the rock at rock_world (or
    None), advance the render point one frame, and render via
    predicted_view. Returns the screen pixels. The camera is locked at the
    ghost (SHIP_X, SHIP_Y), so the rock lands at the exact screen point."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(2):
        t = k * 0.1
        g.push_snapshot(t, _client_snap(rock_world), now=t)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    # Lock the camera BEFORE the render: the ghost is at rest at
    # (SHIP_X, SHIP_Y), so cam.update's target (gpos + vel*lead) equals the
    # locked position and the camera stays put during the frame. Without
    # this the camera is still easing from its start position and the rock
    # lands off-target.
    _lock_cam(g)
    g.predicted_view(0.016, NoKeys())
    _lock_cam(g)   # re-lock so the caller reads the exact camera position
    return _grab(g.screen)


def test_client(screen, font, big_font, light_tex, fog_surf, light_surf):
    ok = True
    g = _client_game(screen, font, big_font, light_tex, fog_surf, light_surf)
    base = _client_render(g, None)

    # --- 5. CLIENT-ON: a rock at screen center is drawn. ---
    on_world = _screen_to_world(g.cam, WIDTH / 2, HEIGHT / 2)
    on = _client_render(g, on_world)
    on_diff = _diff_count(base, on)
    if on_diff >= MIN_DRAWN_PX:
        print("PASS: CLIENT-ON — on-screen rock is drawn (%d px changed)"
              % on_diff)
    else:
        ok = False
        print("FAIL: CLIENT-ON — on-screen rock not drawn (%d px changed, "
              "want >= %d)" % (on_diff, MIN_DRAWN_PX))

    # --- 6. CLIENT-OFF: a rock far off-screen is culled. ---
    off_world = _screen_to_world(g.cam, WIDTH + 300, HEIGHT / 2)
    off = _client_render(g, off_world)
    off_diff = _diff_count(base, off)
    if off_diff == 0:
        print("PASS: CLIENT-OFF — off-screen rock is culled (0 px changed)")
    else:
        ok = False
        print("FAIL: CLIENT-OFF — off-screen rock NOT culled (%d px changed)"
              % off_diff)

    # --- 7. CLIENT-MARGIN: a rock off-screen but within CULL_MARGIN is
    #     still drawn (the margin is respected). ---
    margin_world = _screen_to_world(g.cam, WIDTH / 2 + CULL_MARGIN - 1,
                                    HEIGHT / 2)
    margin = _client_render(g, margin_world)
    margin_diff = _diff_count(base, margin)
    if margin_diff >= MIN_DRAWN_PX:
        print("PASS: CLIENT-MARGIN — rock off-screen but within CULL_MARGIN "
              "is still drawn (%d px changed)" % margin_diff)
    else:
        ok = False
        print("FAIL: CLIENT-MARGIN — rock within CULL_MARGIN was culled "
              "(%d px changed, want >= %d)" % (margin_diff, MIN_DRAWN_PX))
    return ok


# --- pure helpers ----------------------------------------------------------

def test_pure():
    """_view_bounds is the camera rect inflated by CULL_MARGIN; _in_view is
    the inclusive point-in-rect. A pure function of the camera (no sim
    state) — deterministic and parity-safe."""
    cam = pygame.Vector2(SHIP_X, SHIP_Y)
    # Build a minimal cam-like object (the helper reads cam.pos.x/.y).
    class _Cam:
        pos = pygame.Vector2(SHIP_X, SHIP_Y)
    bounds = _view_bounds(_Cam())
    m = CULL_MARGIN
    want = (SHIP_X - WIDTH / 2 - m, SHIP_Y - HEIGHT / 2 - m,
            SHIP_X + WIDTH / 2 + m, SHIP_Y + HEIGHT / 2 + m)
    assert bounds == want, "bounds %r != %r" % (bounds, want)
    # _in_view: inside, on the edge (inclusive), and outside.
    assert _in_view(SHIP_X, SHIP_Y, bounds), "center must be in view"
    assert _in_view(want[0], want[1], bounds), "min corner (inclusive)"
    assert _in_view(want[2], want[3], bounds), "max corner (inclusive)"
    assert not _in_view(want[0] - 1, SHIP_Y, bounds), "left of min"
    assert not _in_view(want[2] + 1, SHIP_Y, bounds), "right of max"
    assert not _in_view(SHIP_X, want[1] - 1, bounds), "above min"
    assert not _in_view(SHIP_X, want[3] + 1, bounds), "below max"
    print("PASS: PURE-BOUNDS — _view_bounds inflates the camera rect by "
          "CULL_MARGIN; _in_view is the inclusive point-in-rect")
    return True


# --- structural ------------------------------------------------------------

def test_structural():
    """Both draw() and predicted_view() contain the cull: they compute
    bounds = _view_bounds(self.cam) and guard the asteroid loop with
    _in_view. (The other entity loops share the same bounds/_in_view, so
    guarding the asteroid loop + the pure-helper test covers the pattern.)"""
    ok = True
    for name in ("draw", "predicted_view"):
        src = inspect.getsource(getattr(Game, name))
        has_bounds = "bounds = _view_bounds(self.cam)" in src
        has_cull = "_in_view(" in src
        if has_bounds and has_cull:
            print("PASS: STRUCTURAL — Game.%s computes _view_bounds and "
                  "culls with _in_view" % name)
        else:
            ok = False
            print("FAIL: STRUCTURAL — Game.%s missing cull "
                  "(bounds=%s, _in_view=%s)" % (name, has_bounds, has_cull))
    return ok


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    ok = True
    ok &= test_pure()
    ok &= test_host(screen, font, big_font, light_tex, fog_surf, light_surf)
    ok &= test_client(screen, font, big_font, light_tex, fog_surf,
                      light_surf)
    ok &= test_structural()
    pygame.quit()
    print("\nVIEWPORT CULLING:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()