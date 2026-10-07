"""Session 10.5: brownout zaps (lightning arcs) on the remote player's ship.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_5_brownout_arcs

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.5 makes the REMOTE ship's brownout crackle render on the client, the same
blue lightning the host already sees when a ship's power sags. The host's
model path already shows the remote player's arcs (render_model packs
p.arcs; _sync_local_ship feeds them; Ship.draw renders them via
_draw_arcs), so this is a CLIENT-only gap: the client's remote-ship
stand-in (self.players[i]) was never fed brownout / power_factor, and the
client never runs Ship.update for it, so Ship._update_arcs (which spawns +
ages the arcs) never ran for the remote ship.

The fix (no wire change — both fields are already in the ship snapshot):
  * netcode.interp_positions carries brownout (ship_s[16], the latched flag,
    from the CURRENT snapshot like `dead`) + power_factor (ship_s[17], the
    0..1 allocation scale, LERP'd across the window like the shield state)
    through the buffer's ships entry. The entry is now a 9-tuple:
    (x, y, angle, dead, shield_dump, shield_clock, flame_mags, brownout,
    power_factor).
  * game.predicted_view (and the legacy remote_view mirror) feed both fields
    to the remote stand-in and drive standin._update_arcs(dt) each frame —
    the client must advance the arcs itself because it never runs
    Ship.update. The arcs are random presentation (never synced — the
    snapshot has no arcs field), so the client generates its OWN bolts,
    scaled by (1 - power_factor), exactly like the ghost's own arcs.

This test proves it:
  1. BUFFER-CARRIES — the buffer's ships entry is a 9-tuple with brownout
                     at [7] and power_factor at [8].
  2. LERP           — interp_positions lerps power_factor across the window
                     (midpoint), endpoint-exact at alpha 0/1; brownout is
                     taken from the CURRENT snapshot (a latch, not lerp'd).
  3. FEEDS-SHIP     — after predicted_view, the remote ship's brownout /
                     power_factor match the buffer's entry.
  4. ARCS-SPAWN     — with brownout active (power_factor < 1) the remote
                     stand-in spawns arcs over a run of frames (the client
                     drives _update_arcs(dt)). Arcs are brief flashes
                     (ttl 0.05-0.12 s) on a ~0.345 s spawn interval at
                     pf=0.3, so the gate tracks the MAX arc count over the
                     run (a single-frame count is 0 most of the time).
  5. ARCS-NONE      — with brownout off the stand-in never spawns (max arc
                     count over a run of frames stays 0).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

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
# (player 1, the ghost) so the two are distinct on screen (same offset the
# 10.2 test uses).
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


def make_snap(remote_brownout, remote_pf, t):
    """A valid 2-player Game snapshot with the REMOTE ship (index 0) at a
    fixed pose and a controllable brownout / power_factor. The local ship
    (index 1) sits at center, not browned out. `t` is unused (the snapshot
    carries no clock — the caller stamps it on push)."""
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(WIDTH / 2 + REMOTE_OFFSET,
                                        HEIGHT / 2)
    tmp.players[0].angle = 0.0
    tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.players[0].brownout = remote_brownout
    tmp.players[0].power_factor = remote_pf
    tmp.players[1].brownout = False
    tmp.players[1].power_factor = 1.0
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


def push_stream(g, remote_brownout, remote_pf):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the remote ship's power
    state held at (remote_brownout, remote_pf). The 3 snapshots give the
    render point a 2-snapshot window to interpolate from. No further
    snapshots are pushed during the render loop, so the remote stand-in's
    arc_clock accumulates freely (reconciles only touch the ghost, never
    players[0])."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    # Start the remote stand-in's arc state clean (no leftover arcs/clock
    # from a previous gate).
    g.players[0].arcs = []
    g.players[0].arc_clock = 0.0
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(remote_brownout, remote_pf, t), now=t)


def render_frame(g):
    """Advance the render point one frame and render via predicted_view."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g, host_hull, client_hull = make_client(screen, font, big_font,
                                            light_tex, fog_surf, light_surf)

    # --- 1. BUFFER-CARRIES: the ships entry is a 9-tuple with brownout at
    #     [7] and power_factor at [8]. ---
    prev = make_snap(False, 1.0, 0.0)
    curr = make_snap(True, 0.3, 0.1)
    P = interp_positions(prev, curr, 1.0, dt=0.1)
    entry = P['ships'][0]
    assert isinstance(entry, tuple) and len(entry) == 9, \
        "ships entry must be a 9-tuple, got %r" % (entry,)
    # At alpha=1 the entry sits exactly on curr (endpoint-exact): brownout
    # is curr's latch (True), power_factor is curr's (0.3).
    assert entry[7] is True and abs(entry[8] - 0.3) < 1e-9, \
        "brownout/power_factor must be at [7]/[8], endpoint-exact: %r" \
        % (entry,)
    print("PASS: BUFFER-CARRIES — ships entry is "
          "(x,y,angle,dead,dump,clock,flame_mags,brownout,power_factor)")

    # --- 2. LERP: power_factor lerps across the window (midpoint),
    #     endpoint-exact at alpha 0/1; brownout is taken from the CURRENT
    #     snapshot (a latch — not lerp'd). ---
    # prev pf=0.2, curr pf=0.8 -> midpoint alpha=0.5 gives 0.5.
    prev2 = make_snap(False, 0.2, 0.0)
    curr2 = make_snap(True, 0.8, 0.1)
    Pm = interp_positions(prev2, curr2, 0.5, dt=0.1)
    assert abs(Pm['ships'][0][8] - 0.5) < 1e-9, \
        "midpoint power_factor must lerp to 0.5, got %r" \
        % (Pm['ships'][0][8],)
    # Endpoint-exact: alpha=0 -> prev's pf (0.2), alpha=1 -> curr's (0.8).
    assert abs(interp_positions(prev2, curr2, 0.0, dt=0.1)['ships'][0][8]
               - 0.2) < 1e-9
    assert abs(interp_positions(prev2, curr2, 1.0, dt=0.1)['ships'][0][8]
               - 0.8) < 1e-9
    # brownout is from the CURRENT snapshot at BOTH endpoints (a latch):
    # curr brownout=True shows at alpha=0 even though prev was False.
    assert interp_positions(prev2, curr2, 0.0, dt=0.1)['ships'][0][7] is True
    assert interp_positions(prev2, curr2, 1.0, dt=0.1)['ships'][0][7] is True
    print("PASS: LERP — power_factor lerps (midpoint 0.5); brownout from curr")

    # --- 3. FEEDS-SHIP: after predicted_view, the remote ship's brownout /
    #     power_factor match the buffer's ships[0] entry. ---
    push_stream(g, True, 0.3)
    render_frame(g)
    P = g.snap_buf.positions_at(g.render_point.now())
    want_bo, want_pf = P['ships'][0][7], P['ships'][0][8]
    assert g.players[0].brownout is want_bo, \
        "remote ship's brownout must be fed from the buffer: " \
        "ship=%r buffer=%r" % (g.players[0].brownout, want_bo)
    assert abs(g.players[0].power_factor - want_pf) < 1e-9, \
        "remote ship's power_factor must be fed from the buffer: " \
        "ship=%r buffer=%r" % (g.players[0].power_factor, want_pf)
    print("PASS: FEEDS-SHIP — predicted_view feeds the remote ship's power state")

    # --- 4. ARCS-SPAWN: with brownout active (power_factor < 1) the remote
    #     stand-in SPAWNS arcs over a run of frames (the client drives
    #     _update_arcs(dt)). Arcs are brief flashes — Ship._make_arc gives
    #     each a ttl of 0.05-0.12 s, while the spawn interval is
    #     0.45 - 0.15*severity s (Ship._update_arcs); at pf=0.3 that's
    #     ~0.345 s, so an arc is present for only ~3-7 frames every ~21.
    #     Checking the count at ONE frame is therefore unreliable (it's 0
    #     most of the time). Instead we track the MAX arc count observed
    #     over the run: a browned-out ship must reach >= 1 at some frame.
    #     (The remote stand-in's arc_clock accumulates freely — reconciles
    #     only touch the ghost, never players[0] — so the default 3-
    #     snapshot seed is fine; no further pushes happen during the render
    #     loop, so the clock is not reset.)
    push_stream(g, True, 0.3)
    max_arcs = 0
    for _ in range(60):
        render_frame(g)
        max_arcs = max(max_arcs, len(g.players[0].arcs))
    assert max_arcs >= 1, \
        "a browned-out remote ship must spawn arcs (client drives " \
        "_update_arcs): max over 60 frames was %d" % max_arcs
    print("PASS: ARCS-SPAWN — the remote ship spawns arcs while browned "
          "out (max %d live at once)" % max_arcs)

    # --- 5. ARCS-NONE: with brownout off the stand-in never spawns (a
    #     non-browned-out _update_arcs returns before the spawn), so the
    #     max arc count over a run of frames stays 0.
    push_stream(g, False, 1.0)
    max_arcs_off = 0
    for _ in range(40):
        render_frame(g)
        max_arcs_off = max(max_arcs_off, len(g.players[0].arcs))
    assert max_arcs_off == 0, \
        "a non-browned-out remote ship must not spawn arcs: max over 40 " \
        "frames was %d" % max_arcs_off
    assert g.players[0].brownout is False, \
        "brownout must be fed False when the buffer says so: %r" \
        % (g.players[0].brownout,)
    print("PASS: ARCS-NONE — no arcs spawn once the brownout ends")

    print("ALL PASS: 10.5 (brownout zaps on the remote ship)")


if __name__ == "__main__":
    main()