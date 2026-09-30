"""Session 10.2: the remote player's shield-impact flash.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_2_remote_shield

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.2 makes the REMOTE ship's shield-impact flash render on the client, the
same blue->white glow the host already sees when a ship's shield is struck.
Today the client's own ship (the prediction ghost) shows its flash (a real
Ship, restored on reconcile), but the REMOTE ship (self.players[i], drawn
via Ship.draw) never did — its shield_dump / shield_clock were never fed
from the interpolation buffer, so Ship._draw_shield saw a zero dump and
drew nothing.

The fix (no wire change — both fields are already in the ship snapshot):
  * netcode.interp_positions carries shield_dump (ship_s[6]) +
    shield_clock (ship_s[7]) through the buffer's ships entry, LERP'd
    across the window like the pose. The entry is now
    (x, y, angle, dead, shield_dump, shield_clock).
  * game.predicted_view feeds those two fields to the remote ship before
    drawing it, so Ship._draw_shield renders the impact flash identically
    to the host.

This test proves it:
  1. BUFFER-CARRIES — the buffer's ships entry is a 7-tuple with
                     shield_dump at [4] and shield_clock at [5].
  2. LERP           — interp_positions lerps shield_dump/clock between two
                     snapshots (midpoint), endpoint-exact at alpha 0/1.
  3. FEEDS-SHIP     — after predicted_view, the remote ship's
                     shield_dump/shield_clock match the buffer's entry.
  4. PIXEL-FLASH    — the remote ship's impact flash renders: a fresh hit
                     (dump=power_hit) produces bluish-white flash pixels
                     around the ship; no hit (dump=0) produces none.
  5. DECAY          — a decaying dump (a partial hit) flashes dimmer than
                     a fresh hit (fewer flash pixels).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (interp_positions, HostTimeEstimator, LatencyTracker,
                      RenderPoint, PredictedShip)
from .ship import Ship
from .hulls import PLAYER_HULLS

SEED = 1234
# The remote ship (player 0) sits this far to the right of the local ship
# (player 1, the ghost) so the two are distinct on screen. 150 px is close
# enough that the fog (the local ship is the light source) does not dim the
# remote ship's flash below the flashy-pixel threshold — a distant ship's
# flash IS dimmer (correct fog behavior), but the test needs the flash to
# survive the fog to measure it. At 150 px the values are clean + monotonic
# (base=0, dump=15 -> ~118 px, dump=25 -> ~298 px).
REMOTE_OFFSET = 150.0


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
    """A client Game (local_index=1) with the host's hull on player 0 and
    the client's hull on player 1 — the same layout run_client builds."""
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    g.ghost = PredictedShip(hull=client_hull, loadout=None, local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g, host_hull, client_hull


def make_snap(remote_dump, remote_clock, t):
    """A valid 2-player Game snapshot with the REMOTE ship (index 0) at a
    fixed pose and a controllable shield_dump / shield_clock. The local
    ship (index 1) sits at center with no flash. `t` is unused (the
    snapshot carries no clock — the caller stamps it on push)."""
    # Build a throwaway real Game just to get a valid snapshot shape, then
    # patch the remote ship's shield state.
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(WIDTH / 2 + REMOTE_OFFSET,
                                        HEIGHT / 2)
    tmp.players[0].angle = 0.0
    tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.players[0].shield_dump = remote_dump
    tmp.players[0].shield_clock = remote_clock
    tmp.players[0].shield_charge = 3.0
    tmp.players[1].shield_charge = 3.0
    return tmp.snapshot()


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless)."""
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=None, seed=SEED, players=2)


def push_stream(g, remote_dump, remote_clock):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the remote ship's
    shield state held at (remote_dump, remote_clock)."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(remote_dump, remote_clock, t), now=t)


def render_frame(g):
    """Advance the render point one frame and render via predicted_view."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def flashy_pixels(g):
    """Count bluish-white 'flash' pixels in a box around the remote ship's
    screen position. The shield-impact flash (Ship._draw_shield) lerps
    toward SHIELD_COLOR_BRIGHT (120, 200, 255) — a saturated blue that is
    far brighter/bluer than the dim hull or the fog, so 'b > 150 and
    b > r + 20 and g > 120' isolates it."""
    rpos = pygame.Vector2(WIDTH / 2 + REMOTE_OFFSET, HEIGHT / 2)
    s = g.cam.to_screen(rpos)
    cx, cy = int(s.x), int(s.y)
    box = g.screen.subsurface(cx - 40, cy - 40, 80, 80)
    data = pygame.image.tostring(box, "RGB")
    n = 0
    for i in range(0, len(data), 3):
        r, gg, b = data[i], data[i + 1], data[i + 2]
        if b > 150 and b > r + 20 and gg > 120:
            n += 1
    return n


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g, host_hull, client_hull = make_client(screen, font, big_font,
                                            light_tex, fog_surf, light_surf)

    # --- 1. BUFFER-CARRIES: the ships entry is a 7-tuple with the shield
    #     state at [4] (dump) and [5] (clock). 10.6 grew it to a 7-tuple
    #     (+ flame_mags at [6]); the shield state is unchanged. ---
    prev = make_snap(0.0, 0.0, 0.0)
    curr = make_snap(25.0, 0.05, 0.1)
    P = interp_positions(prev, curr, 1.0, dt=0.1)
    entry = P['ships'][0]
    assert isinstance(entry, tuple) and len(entry) == 7, \
        "ships entry must be a 7-tuple, got %r" % (entry,)
    # At alpha=1 the entry sits exactly on curr (endpoint-exact): the
    # remote ship's dump/clock are curr's (25.0, 0.05).
    assert entry[4] == 25.0 and entry[5] == 0.05, \
        "shield_dump/clock must be at [4]/[5], endpoint-exact: %r" % (entry,)
    print("PASS: BUFFER-CARRIES — ships entry is "
          "(x,y,angle,dead,dump,clock,flame_mags)")

    # --- 2. LERP: interp_positions lerps the shield state between two
    #     snapshots (midpoint), endpoint-exact at alpha 0/1. ---
    # prev dump=0, curr dump=25 -> midpoint alpha=0.5 gives 12.5.
    Pm = interp_positions(prev, curr, 0.5, dt=0.1)
    assert abs(Pm['ships'][0][4] - 12.5) < 1e-9, \
        "midpoint shield_dump must lerp to 12.5, got %r" % (Pm['ships'][0][4],)
    # Endpoint-exact: alpha=0 -> prev's dump (0.0), alpha=1 -> curr's (25.0).
    assert interp_positions(prev, curr, 0.0, dt=0.1)['ships'][0][4] == 0.0
    assert interp_positions(prev, curr, 1.0, dt=0.1)['ships'][0][4] == 25.0
    print("PASS: LERP — shield_dump lerps across the window (midpoint 12.5)")

    # --- 3. FEEDS-SHIP: after predicted_view, the remote ship's
    #     shield_dump/shield_clock match the buffer's ships[0] entry. ---
    push_stream(g, 25.0, 0.05)
    render_frame(g)
    P = g.snap_buf.positions_at(g.render_point.now())
    want_dump, want_clock = P['ships'][0][4], P['ships'][0][5]
    assert abs(g.players[0].shield_dump - want_dump) < 1e-9, \
        "remote ship's shield_dump must be fed from the buffer: " \
        "ship=%r buffer=%r" % (g.players[0].shield_dump, want_dump)
    assert abs(g.players[0].shield_clock - want_clock) < 1e-9, \
        "remote ship's shield_clock must be fed from the buffer: " \
        "ship=%r buffer=%r" % (g.players[0].shield_clock, want_clock)
    print("PASS: FEEDS-SHIP — predicted_view feeds the remote ship's shield state")

    # --- 4. PIXEL-FLASH: a fresh hit (dump=power_hit=25) renders the
    #     bluish-white flash around the remote ship; no hit (dump=0) does
    #     not. ---
    push_stream(g, 0.0, 0.0)
    render_frame(g)
    base = flashy_pixels(g)
    push_stream(g, 25.0, 0.05)
    render_frame(g)
    hit = flashy_pixels(g)
    assert hit > base + 50, \
        "a fresh shield hit must render flash pixels: hit=%d base=%d" \
        % (hit, base)
    print("PASS: PIXEL-FLASH — the remote ship's impact flash renders "
          "(%d vs %d baseline pixels)" % (hit, base))

    # --- 5. DECAY: a partial hit (dump=15, a decaying flash) flashes
    #     dimmer than a fresh hit (dump=25) — fewer flash pixels. The
    #     flash has a per-frame random flicker jitter (Ship._draw_shield),
    #     so the margins are generous; the measured gap is large
    #     (~118 vs ~298 px at 150 px offset). ---
    push_stream(g, 15.0, 0.2)
    render_frame(g)
    partial = flashy_pixels(g)
    assert partial < hit - 20, \
        "a decaying dump must flash dimmer than a fresh hit: " \
        "partial=%d fresh=%d" % (partial, hit)
    assert partial > base + 20, \
        "a partial hit must still flash (above the no-hit baseline): " \
        "partial=%d base=%d" % (partial, base)
    print("PASS: DECAY — a decaying dump flashes dimmer (%d vs %d fresh)"
          % (partial, hit))

    print("ALL PASS: 10.2 (remote player's shield-impact flash)")


if __name__ == "__main__":
    main()