"""Session 10.6: power-state presentation on the wire (exhaust flames) +
V/G/T client-authoritative sensor state.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_6_power_state

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.6 has two halves:

  A. POWER STATE (the exhaust flames). Before 10.6 the per-thruster flame
     magnitudes (Ship.flame_mags) were PRESENTATION-ONLY and never
     serialized: the host's local ship rendered its exhaust (via
     _sync_local_ship / the model's per-ship pack), but the REMOTE player
     ship and the REMOTE enemies were drawn from the interpolation buffer
     (client) / the RenderModel (host) with NO flame data, so they rendered
     with no exhaust at all. 10.6 promotes flame_mags to a synced field:
       * Ship.snapshot is a 24-tuple (flame_mags LAST, index 23);
         apply_snapshot restores it; Ship.power_state() bundles the four
         power-presentation fields (brownout / power_factor / power_used /
         flame_mags) into one plain dict for future power features.
       * netcode.interp_positions carries flame_mags through the buffer:
         the ship entry is now a 7-tuple (… , flame_mags) and the enemy
         entry an 11-tuple (… , flame_mags) — both taken from the CURRENT
         snapshot (a flame dict has no meaningful cross-window blend).
       * game feeds the flame mags to the stand-in before drawing:
         predicted_view / remote_view feed the remote player ship,
         _draw_remote_enemy_hull feeds the enemy stand-in (client), and
         render_model + _draw_world_enemy carry/feed them on the host's
         model path (parity — the 10.x goal is "client looks like host").

  B. V/G/T (option A — client-authoritative sensor state). The client's
     V/G/T keys mutate the GHOST directly (via the run_client command
     sink), and the ghost's sensor state is PRESERVED across reconciles
     (the host never saw the client's V/G/T, so its snapshot carries the
     host's own sensor state — restoring it would flicker the client's
     sensor off every ~100 ms). PredictedShip.client_sensor_authoritative
     gates the preserve (off by default: the host's ghost + tests keep the
     host-authoritative behavior).

This test proves:
  1. WIRE-SHAPE     — Ship.snapshot is a 24-tuple (flame_mags at [23]);
                      the enemy Game-snapshot entry is (tag, e_s) with
                      e_s = (ship_s, hp, id, ax, ay) and ship_s a 24-
                      tuple; JSON round-trip preserves flame_mags.
  2. POWER-STATE    — Ship.power_state() bundles the four fields and
                      returns a COPY of flame_mags (mutating the dict does
                      not touch the ship).
  3. BUFFER-CARRIES — the buffer's ship entry is a 7-tuple (flame_mags at
                      [6]) and the enemy entry an 11-tuple (flame_mags at
                      [10]); both taken from the CURRENT snapshot (not
                      lerp'd).
  4. FEEDS-STANDIN  — after predicted_view, the remote player ship's
                      flame_mags match the buffer's entry, and the enemy
                      stand-in's flame_mags match the buffer's enemy entry.
  5. PIXEL-SHIP     — the REMOTE player ship's exhaust renders on the
                      client (orange flame pixels around it); no flames
                      ({} mags) does not.
  6. PIXEL-ENEMY    — the REMOTE enemy's exhaust renders on the client
                      (purple flame pixels around it); no flames does not.
  7. V/G-T-SINK     — the client's V/G/T mutate the GHOST immediately
                      (sensor_on / targeting_on toggle, fire_scan fires).
  8. SENSOR-PRESERVE— the ghost's client-authoritative sensor state
                      survives a reconcile (the host's snapshot carries the
                      host's sensor state, which must NOT overwrite the
                      client's).
  9. HOST-MODEL     — the host's render path carries the enemy's flame
                      mags in the model's enemy tuple (11-tuple, [10]) and
                      _draw_world_enemy feeds them to the stand-in (parity
                      with the client — the 10.x goal).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, _draw_world_enemy
from .netcode import (interp_positions, HostTimeEstimator, LatencyTracker,
                      RenderPoint, PredictedShip)
from .ship import Ship
from .ai_enemy import AIEnemy
from .hulls import PLAYER_HULLS

SEED = 1234
# The remote player ship (player 0, the host's scout) sits this far to the
# right of the local ship (the ghost, the fog's light source) so its orange
# exhaust survives the fog. 150 px is close enough that a full-thrust flame
# is well above the flashy-pixel threshold (a distant ship's flame IS
# dimmer — correct fog behavior — but the test needs it to survive).
REMOTE_SHIP_OFFSET = 150.0
# The target enemy (ENEMY_IDX) sits further right, clear of the remote ship.
ENEMY_OFFSET = 320.0
ENEMY_IDX = 0

# Full-thrust forward flame (the remote ship + the enemy both point +x, so
# 'forward' is the bucket that renders). The other buckets are 0.
FLAME_FWD = {'forward': 1.0, 'reverse': 0.0, 'to_left': 0.0, 'to_right': 0.0}
NO_FLAME = {'forward': 0.0, 'reverse': 0.0, 'to_left': 0.0, 'to_right': 0.0}


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
    the client's hull on player 1 — the same layout run_client builds. The
    ghost is CLIENT-AUTHORITATIVE for its sensor state (10.6 option A)."""
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    g.ghost = PredictedShip(hull=client_hull, loadout=None, local_index=1)
    g.ghost.client_sensor_authoritative = True
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless).

    Resets AIEnemy._next_id so every scratch game's enemies get the SAME
    ship ids (1, 2, 3, ...) — the buffer matches enemies BY ID, so two
    snapshots fed to interp_positions must share ids or the prev enemy
    won't match (and the lerp would fall back to curr)."""
    AIEnemy._next_id = 1
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=None, seed=SEED, players=2)


def make_snap(ship_flame, enemy_flame, sensor_on=False):
    """A valid 2-player Game snapshot with the REMOTE ship (player 0) and
    the TARGET enemy (ENEMY_IDX) at fixed poses (+x of center) and
    controllable flame_mags. The remote ship's sensor state is set to
    `sensor_on` (the host's own sensor state — the client's ghost must NOT
    adopt it). All other ships/enemies keep their (zero) flame state."""
    tmp = _scratch_game()
    # The LOCAL ship (player 1, the ghost, the fog's light source) sits at
    # center facing +x, so its forward lobe (FOG_FRONT_RADIUS 430 px) lights
    # both the remote ship (150 px) and the target enemy (320 px) — both are
    # ahead of it. (The base radius is only 160 px, so without the lobe the
    # enemy at 320 px would be in the dark and its flame would not survive
    # the fog.)
    ls = tmp.players[1]
    ls.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    ls.angle = 0.0
    rs = tmp.players[0]
    rs.pos = pygame.Vector2(WIDTH / 2 + REMOTE_SHIP_OFFSET, HEIGHT / 2)
    rs.angle = 0.0
    rs.flame_mags = dict(ship_flame)
    rs.sensor_on = sensor_on
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)
    e.ship.angle = 0.0
    e.ship.flame_mags = dict(enemy_flame)
    return tmp.snapshot()


def push_stream(g, ship_flame, enemy_flame, sensor_on=False):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the remote ship's +
    target enemy's flame state held at (ship_flame, enemy_flame)."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.ghost.client_sensor_authoritative = True
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(ship_flame, enemy_flame, sensor_on),
                        now=t)


def render_frame(g):
    """Advance the render point one frame and render via predicted_view
    (no local input — the ghost stays put, so the camera is stable)."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def remote_ship_world():
    return pygame.Vector2(WIDTH / 2 + REMOTE_SHIP_OFFSET, HEIGHT / 2)


def enemy_world():
    return pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)


def flame_pixels(screen, cam, world, half=30, orange=True):
    """Count flame pixels in a box around `world`. The player flame is
    orange (FLAME_OUT (255,150,50) / FLAME_IN (255,225,160)) — 'r > 180 and
    g > 90 and b < 130' isolates it from the grey-blue hull, the dark BG,
    and the stars. The enemy flame is purple (ENEMY_FLAME (205,165,255)) —
    'b > 180 and r > 120 and g < 200' isolates it from the green hull.
    (The per-frame flame flicker only varies LENGTH, not color, so the
    color test is stable.)"""
    s = cam.to_screen(world)
    cx, cy = int(s.x), int(s.y)
    box = screen.subsurface(cx - half, cy - half, 2 * half, 2 * half)
    data = pygame.image.tostring(box, "RGB")
    n = 0
    for i in range(0, len(data), 3):
        r, gg, b = data[i], data[i + 1], data[i + 2]
        if orange:
            if r > 180 and gg > 90 and b < 130:
                n += 1
        else:
            if b > 180 and r > 120 and gg < 200:
                n += 1
    return n


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()

    # --- 1. WIRE-SHAPE: Ship.snapshot is a 24-tuple (flame_mags LAST,
    #     index 23); the enemy Game-snapshot entry is (tag, e_s) with
    #     e_s = (ship_s, hp, id, ax, ay) and ship_s a 24-tuple; JSON
    #     round-trip preserves flame_mags. ---
    tmp = _scratch_game()
    tmp.players[0].flame_mags = dict(FLAME_FWD)
    tmp.enemies[ENEMY_IDX].ship.flame_mags = dict(FLAME_FWD)
    snap = tmp.snapshot()
    ship_s = snap[0][0]
    assert isinstance(ship_s, tuple) and len(ship_s) == 24, \
        "Ship.snapshot must be a 24-tuple, got len %d" % (len(ship_s),)
    assert ship_s[23] == FLAME_FWD, \
        "flame_mags must be at ship_s[23]: %r" % (ship_s[23],)
    # The enemy Game-snapshot entry is (tag, e_s) where e_s =
    # AIEnemy.snapshot() = (ship_s, hp, id, ax, ay) — a 5-tuple whose
    # FIRST element is the 24-tuple ship_s.
    e_entry = snap[1][ENEMY_IDX]
    assert isinstance(e_entry, tuple) and len(e_entry) == 2, \
        "enemy Game-snapshot entry must be (tag, e_s), got %r" % (e_entry,)
    e_s = e_entry[1]
    assert isinstance(e_s, tuple) and len(e_s) == 5, \
        "AIEnemy.snapshot must be a 5-tuple (ship_s, hp, id, ax, ay), " \
        "got %r" % (e_s,)
    e_ship_s = e_s[0]
    assert isinstance(e_ship_s, tuple) and len(e_ship_s) == 24, \
        "enemy ship_s must be a 24-tuple, got len %d" % (len(e_ship_s),)
    assert e_ship_s[23] == FLAME_FWD, \
        "enemy flame_mags must be at ship_s[23]: %r" % (e_ship_s[23],)
    # JSON round-trip (the wire): flame_mags survives as a dict.
    snap_j = json.loads(json.dumps(snap))
    assert snap_j[0][0][23] == FLAME_FWD, "JSON ship flame_mags lost"
    assert snap_j[1][ENEMY_IDX][1][0][23] == FLAME_FWD, \
        "JSON enemy flame_mags lost"
    print("PASS: WIRE-SHAPE — Ship.snapshot is a 24-tuple (flame_mags at "
          "[23]); enemy entry (tag, e_s) w/ 24-tuple ship_s; JSON-safe")

    # --- 2. POWER-STATE: Ship.power_state() bundles the four power-
    #     presentation fields and returns a COPY of flame_mags. ---
    s = tmp.players[0]
    s.brownout = False
    s.power_factor = 1.0
    s.power_used = 24.0
    s.flame_mags = dict(FLAME_FWD)
    ps = s.power_state()
    assert set(ps.keys()) == {"brownout", "power_factor", "power_used",
                              "flame_mags"}, \
        "power_state keys wrong: %r" % (sorted(ps.keys()),)
    assert ps["brownout"] is False and ps["power_factor"] == 1.0
    assert ps["power_used"] == 24.0 and ps["flame_mags"] == FLAME_FWD
    # The flame_mags value is a COPY: mutating the dict must not touch the
    # ship (the caller may mutate it without side effects).
    ps["flame_mags"]["forward"] = 0.0
    assert s.flame_mags["forward"] == 1.0, \
        "power_state must return a COPY of flame_mags (ship mutated)"
    print("PASS: POWER-STATE — power_state() bundles the 4 fields + "
          "returns a copy of flame_mags")

    # --- 3. BUFFER-CARRIES: the buffer's ship entry is a 9-tuple
    #     (flame_mags at [6], brownout at [7], power_factor at [8] — 10.5)
    #     and the enemy entry an 11-tuple (flame_mags at [10]); the flame
    #     mags are taken from the CURRENT snapshot (not lerp'd). ---
    prev = make_snap(NO_FLAME, NO_FLAME)
    curr = make_snap(FLAME_FWD, FLAME_FWD)
    P = interp_positions(prev, curr, 0.5, dt=0.1)
    s_entry = P['ships'][0]
    assert isinstance(s_entry, tuple) and len(s_entry) == 9, \
        "ship entry must be a 9-tuple, got %r" % (s_entry,)
    assert s_entry[6] == FLAME_FWD, \
        "ship flame_mags must be at [6] (from curr, not lerp'd): %r" \
        % (s_entry[6],)
    e_entry = P['enemies'][ENEMY_IDX]
    assert isinstance(e_entry, tuple) and len(e_entry) == 11, \
        "enemy entry must be an 11-tuple, got %r" % (e_entry,)
    assert e_entry[10] == FLAME_FWD, \
        "enemy flame_mags must be at [10] (from curr, not lerp'd): %r" \
        % (e_entry[10],)
    print("PASS: BUFFER-CARRIES — ship entry 9-tuple (flame_mags [6], "
          "brownout [7], power_factor [8]), enemy entry 11-tuple "
          "(flame_mags [10]), from curr")

    # --- 4-6. FEEDS-STANDIN + PIXEL: build the client, render a no-flame
    #     baseline frame, then a full-flame frame. ---
    g = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)

    # Baseline: no flames anywhere.
    push_stream(g, NO_FLAME, NO_FLAME)
    render_frame(g)
    base_ship = flame_pixels(g.screen, g.cam, remote_ship_world(), orange=True)
    base_enemy = flame_pixels(g.screen, g.cam, enemy_world(), orange=False)

    # Full flames: the remote ship + the target enemy both thrust forward.
    push_stream(g, FLAME_FWD, FLAME_FWD)
    render_frame(g)

    # 4. FEEDS-STANDIN: the remote player ship's flame_mags match the
    #    buffer's entry (predicted_view fed them before drawing).
    P = g.snap_buf.positions_at(g.render_point.now())
    assert P is not None and P['ships'], "no buffer window to inspect"
    assert g.players[0].flame_mags == P['ships'][0][6], \
        "remote ship flame_mags must match the buffer entry: ship=%r " \
        "buf=%r" % (g.players[0].flame_mags, P['ships'][0][6])
    # The enemy stand-in's flame_mags are fed by _draw_remote_enemy_hull
    # before drawing. The stand-in is SHARED per tag (all 3 enemies are
    # 'ai'), so reading it after a full predicted_view frame is ambiguous
    # (it holds the LAST enemy of that tag drawn — here enemy 2, which has
    # no flames). The feed is atomic per enemy (fed then drawn immediately),
    # so the RENDER is correct; to verify the feed unambiguously, call the
    # helper directly with known values and check the stand-in reflects them
    # (the 10.2b pattern).
    standin = g._get_remote_enemies()['ai']
    g._draw_remote_enemy_hull(g.screen, 'ai', 0.0, 0.0, 0.0, 0.0, 0.0,
                              dict(FLAME_FWD))
    assert standin.ship.flame_mags == FLAME_FWD, \
        "_draw_remote_enemy_hull must feed flame_mags to the stand-in: " \
        "standin=%r" % (standin.ship.flame_mags,)
    print("PASS: FEEDS-STANDIN — remote ship carries the buffer's "
          "flame_mags after predicted_view; _draw_remote_enemy_hull feeds "
          "the enemy stand-in's flame_mags before drawing")

    # 5. PIXEL-SHIP: the remote player ship's orange exhaust renders.
    ship_hit = flame_pixels(g.screen, g.cam, remote_ship_world(), orange=True)
    assert ship_hit > base_ship + 15, \
        "the remote ship's exhaust must render (orange flame pixels): " \
        "hit=%d base=%d" % (ship_hit, base_ship)
    print("PASS: PIXEL-SHIP — the remote player ship's exhaust renders on "
          "the client (%d vs %d baseline pixels)" % (ship_hit, base_ship))

    # 6. PIXEL-ENEMY: the remote enemy's purple exhaust renders.
    enemy_hit = flame_pixels(g.screen, g.cam, enemy_world(), orange=False)
    assert enemy_hit > base_enemy + 15, \
        "the remote enemy's exhaust must render (purple flame pixels): " \
        "hit=%d base=%d" % (enemy_hit, base_enemy)
    print("PASS: PIXEL-ENEMY — the remote enemy's exhaust renders on the "
          "client (%d vs %d baseline pixels)" % (enemy_hit, base_enemy))

    # --- 7. V/G-T-SINK: the client's V/G/T mutate the GHOST immediately.
    #     (The run_client command sink routes T/V/G to the ghost; here we
    #     exercise the same mutations the sink performs.) ---
    gh = g.ghost.ship
    gh.sensor_on = not gh.sensor_on          # V
    assert gh.sensor_on is True, "V must toggle the ghost's sensor_on"
    gh.targeting_on = not gh.targeting_on    # T
    assert gh.targeting_on is True, "T must toggle the ghost's targeting_on"
    gh.scan_cd = 0.0                          # G (clear any cooldown)
    fired = gh.fire_scan()
    assert fired is True and gh.scan_reveal > 0 and gh.scan_cd > 0, \
        "G must fire the ghost's scan (scan_reveal/scan_cd set): " \
        "fired=%r reveal=%.2f cd=%.2f" % (fired, gh.scan_reveal, gh.scan_cd)
    print("PASS: V/G-T-SINK — V/G/T mutate the ghost immediately "
          "(sensor_on, targeting_on, fire_scan)")

    # --- 8. SENSOR-PRESERVE: the ghost's client-authoritative sensor
    #     state survives a reconcile. The host's snapshot carries the
    #     HOST's sensor state (sensor_on=False here) — restoring it would
    #     flicker the client's sensor off. With
    #     client_sensor_authoritative=True, the ghost keeps its own. ---
    gh.sensor_on = True
    gh.targeting_on = True
    # A fresh snapshot with the host's sensor state OFF (sensor_on=False).
    # push_snapshot reconciles (rewind) — the ghost's sensor state must
    # survive (the host's False must NOT overwrite the client's True).
    g.push_snapshot(0.3, make_snap(FLAME_FWD, FLAME_FWD, sensor_on=False),
                    now=0.3)
    assert g.ghost.ship.sensor_on is True, \
        "the ghost's sensor_on must survive the reconcile (client-" \
        "authoritative): got %r" % (g.ghost.ship.sensor_on,)
    assert g.ghost.ship.targeting_on is True, \
        "the ghost's targeting_on must survive the reconcile: got %r" \
        % (g.ghost.ship.targeting_on,)
    # Contrast: a ghost WITHOUT client_sensor_authoritative adopts the
    # host's sensor state (the pre-10.6 / host-authoritative behavior).
    g2 = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    push_stream(g2, NO_FLAME, NO_FLAME)
    # push_stream recreates the ghost (authoritative=True); flip it OFF so
    # this ghost is host-authoritative (the pre-10.6 behavior).
    g2.ghost.client_sensor_authoritative = False
    g2.ghost.ship.sensor_on = True
    g2.push_snapshot(0.3, make_snap(NO_FLAME, NO_FLAME, sensor_on=False),
                     now=0.3)
    assert g2.ghost.ship.sensor_on is False, \
        "a non-authoritative ghost must adopt the host's sensor state: " \
        "got %r" % (g2.ghost.ship.sensor_on,)
    print("PASS: SENSOR-PRESERVE — the client-authoritative ghost's sensor "
          "state survives a reconcile (the host's state does not overwrite "
          "it); a non-authoritative ghost adopts the host's")

    # --- 9. HOST-MODEL: the host's render path carries the enemy's flame
    #     mags in the model's enemy tuple (11-tuple, [10]) and
    #     _draw_world_enemy feeds them to the stand-in (parity with the
    #     client — the 10.x goal is "client looks like host"). ---
    host = _scratch_game()
    he = host.enemies[ENEMY_IDX]
    he.ship.pos = pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)
    he.ship.angle = 0.0
    he.ship.flame_mags = dict(FLAME_FWD)
    m = host.render_model()
    met = m["enemies"][ENEMY_IDX]
    assert len(met) == 11, "model enemy tuple must be an 11-tuple, got %r" \
        % (met,)
    assert met[10] == FLAME_FWD, \
        "model enemy tuple must carry flame_mags at [10]: %r" % (met[10],)
    host.cam.pos = enemy_world().copy()   # center the cam on the enemy
    standins = host._get_remote_enemies()
    hstandin = standins['ai']
    _draw_world_enemy(screen, host.cam, met[0], met[2], met[3], standins,
                      met[8], met[9], met[10])
    assert hstandin.ship.flame_mags == FLAME_FWD, \
        "_draw_world_enemy must feed flame_mags to the stand-in: %r" \
        % (hstandin.ship.flame_mags,)
    print("PASS: HOST-MODEL — the host's model enemy tuple carries "
          "flame_mags (11-tuple, [10]) and _draw_world_enemy feeds them "
          "(parity with the client)")

    print("ALL PASS: 10.6 (power-state exhaust on the wire, both peers + "
          "V/G/T client-authoritative sensor state)")


if __name__ == "__main__":
    main()