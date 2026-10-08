"""Session 10.3c: the prediction ghost emits sensor contacts.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3c_ghost_contacts

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3c makes the CLIENT's local ship (the prediction ghost) show the SAME
sensor contacts (passive blips/arrows for enemies whose active power draw
crosses the signature threshold; confirmed contacts during a scan reveal)
that the host's local ship shows — drawn above the fog, like the host.

The contact signature is `power_used - power_idle_total` (the host's
`_update_contacts` reads it off the LIVE enemy). `power_idle_total` is a
LOADOUT CONSTANT the client derives (the enemy hull+loadout is fixed on
both peers — the same stand-in `_build_remote_enemy_standins` builds);
`power_used` is the LIVE total power demand, recomputed every tick by
`Ship._allocate` and NOT derivable client-side (the enemy's thruster state
is not synced) — so it is carried on the wire:
  * Ship.snapshot is a 25-tuple (10.7: bullet_seq LAST, index 24;
         power_used at index 22, before the
    10.6 flame_mags at 23); apply_snapshot restores it.
  * netcode.interp_positions carries power_used through the buffer's
    enemy entry (11-tuple, index 9), LERP'd across the window like the
    shield state (the enemy's power demand eases as it throttles/fires).
  * game._ghost_enemy_proxies fills the proxy's .ship stand-in:
    power_used from the buffer, power_idle_total derived from the tag's
    loadout. The ghost's `_update_contacts` (netcode.PredictedShip)
    mirrors the host's `Game._update_contacts` EXACTLY, so given the same
    sensor state + the same enemy positions the ghost's contacts match
    the host's.
  * game.predicted_view draws the ghost's scan pulse ring + sensor
    contacts ABOVE the fog (the SAME two calls the host's model path
    makes: fog -> scan pulse -> contacts -> HUD).

The ghost's sensor state (sensor_on / scan_reveal / scan_cd / scan_pulse)
is CLIENT-AUTHORITATIVE (10.6 option A): the client's V/G command sink
mutates the ghost directly, and a reconcile does NOT flicker it off with
the host's state (the host never saw the client's V/G/T).

This test proves it:
  1. WIRE-SHAPE   — Ship.snapshot is a 25-tuple (power_used at [22],
                       bullet_seq at [24]); the
                    enemy's nested ship snapshot carries it; JSON
                    round-trip preserves it.
  2. BUFFER-CARRIES — the buffer's enemy entry is an 11-tuple with
                    power_used at [9], lerp'd across the window
                    (midpoint = average, endpoint-exact at alpha 0/1).
  3. PROXY-POWER  — _ghost_enemy_proxies fills power_used (from the
                    buffer) + power_idle_total (derived from the tag's
                    loadout) on the proxy's .ship.
  4. CONTACT-PASSIVE — a ghost with sensor_on=True + a proxy whose
                    (power_used - power_idle_total) >=
                    SENSOR_SIGNATURE_THRESHOLD within sensor_range -> a
                    passive contact (confirmed=False, strength =
                    sig/SIG_FULL clamped). Below threshold -> no contact.
                    sensor_on=False -> no contact.
  5. CONTACT-SCAN — a ghost with scan_reveal>0 + a proxy within
                    scan_range -> a confirmed contact (confirmed=True,
                    strength=1.0) that REPLACES a passive one for the
                    same enemy. G (fire_scan) takes effect immediately.
  6. CONTACT-PARITY — the ghost's contacts (same sensor state + same
                    enemy positions) match the host's _update_contacts
                    output (same enemy, same dist/strength/confirmed) —
                    proves the math is identical.
  7. PIXEL-BLIP   — predicted_view draws the ghost's passive contact
                    blip above the fog (SENSOR_COLOR pixels at the
                    enemy's screen pos; the sensor-off baseline has
                    none).
  8. PIXEL-PULSE  — predicted_view draws the ghost's scan pulse ring
                    (the expanding circle on a G press) above the fog.
  9. PIXEL-ARROW  — an off-screen enemy shows an edge arrow (the
                    on-screen blip is absent).
"""
import json
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import (WIDTH, HEIGHT, SENSOR_COLOR, SENSOR_SCAN_COLOR,
                     SENSOR_SIGNATURE_THRESHOLD, SENSOR_SIG_FULL,
                     SENSOR_ARROW_MARGIN)
from .fog import make_light_texture
from .game import Game, _GhostEnemyProxy
from .netcode import PredictedShip, HostTimeEstimator, LatencyTracker, RenderPoint
from .ship import Ship
from .intent import ShipInput
from .hulls import (PLAYER_HULLS, ENEMY_HULL, enemy_loadout,
                    default_loadout, PASSIVE_SENSOR, SENSOR_ARRAY)
from .ai_enemy import AIEnemy

SEED = 1234
TICK = 1 / 60
ENEMY_IDX = 0
# The target enemy sits this far to the right of the local ship (the
# ghost, the fog's light source) — inside the forward lobe (430 px) so
# the blip survives the fog, and on-screen (the local ship is at screen
# center, so world = screen + center).
ENEMY_OFFSET = 320.0
# The enemy's live power demand: signature = power_used - idle (11.0 for
# the enemy loadout) = 14.0 >= SENSOR_SIGNATURE_THRESHOLD (12.0) ->
# passively detectable. power_used = 11.0 -> signature 0.0 -> invisible.
ENEMY_POWER_USED = 25.0
ENEMY_POWER_QUIET = 11.0
# An off-screen enemy (the edge-arrow case): far enough right that its
# screen pos is past WIDTH + 20.
ENEMY_OFFSET_FAR = 1000.0


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def _swap_sensor(comp):
    """The client hull's default loadout with the sensor slot swapped to
    `comp`. The default (ACTIVE_SCANNER) has sensor_range=0 — passive
    sensing is OFF — so the passive-contact gates need a fit with a
    passive range."""
    out = dict(default_loadout(PLAYER_HULLS[1]))
    for name, c in out.items():
        if c.slot_types == ('sensor',):
            out[name] = comp
    return out


def passive_loadout():
    """Default loadout with a PASSIVE_SENSOR (sensor_range 1800,
    scan_range 0). Used by the passive-contact gates (4, 6, 7, 9) —
    passive sensing is OFF on the default ACTIVE_SCANNER."""
    return _swap_sensor(PASSIVE_SENSOR)


def array_loadout():
    """Default loadout with a SENSOR_ARRAY (sensor_range 1500 AND
    scan_range 2000). The ONLY fit with both a passive range and a scan
    range, so it is the one that can show a scan reveal REPLACING a
    passive contact for the same enemy (gate 5's ghost4)."""
    return _swap_sensor(SENSOR_ARRAY)


def make_client(screen, font, big_font, light_tex, fog_surf, light_surf,
                loadout=None):
    """A client Game (local_index=1) with the host's hull on player 0 and
    the client's hull on player 1 — the same layout run_client builds.
    The ghost is CLIENT-AUTHORITATIVE for its sensor state (10.6
    option A), like the real client. `loadout` (None = the hull's
    default) is the CLIENT's fit — the ghost + the client's ship share
    it, so the snapshot's weapon/sensor state matches the ghost's."""
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=loadout))
    g.ghost = PredictedShip(hull=client_hull, loadout=loadout,
                            local_index=1)
    g.ghost.client_sensor_authoritative = True
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless).

    Resets AIEnemy._next_id so every scratch game's enemies get the SAME
    ship ids (1, 2, 3, ...) — the buffer matches enemies BY ID, so two
    snapshots fed to the buffer must share ids."""
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


def make_snap(power_used, enemy_offset=ENEMY_OFFSET):
    """A valid 2-player Game snapshot with the LOCAL ship (player 1, the
    ghost, the fog's light source) at screen center facing +x and the
    TARGET enemy (ENEMY_IDX) at (center + enemy_offset, center) with the
    given live power demand. The local ship's sensor state is OFF (the
    client's ghost sets its own — 10.6 option A)."""
    tmp = _scratch_game()
    ls = tmp.players[1]
    ls.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    ls.vel = pygame.Vector2(0.0, 0.0)
    ls.angle = 0.0
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(WIDTH / 2 + enemy_offset, HEIGHT / 2)
    e.ship.angle = 0.0
    e.ship.power_used = power_used
    return tmp.snapshot()


def push_stream(g, power_used, enemy_offset=ENEMY_OFFSET, loadout=None):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the target enemy's
    power held at `power_used`. `loadout` (None = the client ship's
    current components) rebuilds the ghost with the client's fit."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=loadout if loadout is not None
                            else g.players[1].components,
                            local_index=1)
    g.ghost.client_sensor_authoritative = True
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(power_used, enemy_offset), now=t)


def render_frame(g):
    """Advance the render point one frame and render via predicted_view
    (no local input — the ghost stays put, so the camera is stable)."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def enemy_world(offset=ENEMY_OFFSET):
    return pygame.Vector2(WIDTH / 2 + offset, HEIGHT / 2)


def blip_screen(offset=ENEMY_OFFSET):
    """The enemy's screen pos. The local ship (the ghost) sits at screen
    center and never moves (vel 0, no input), so the camera eases to
    center and world->screen is a constant +center offset."""
    return pygame.Vector2(WIDTH / 2 + offset, HEIGHT / 2)


def sensor_pixels(screen, sx, sy, half=25, warm=False):
    """Count sensor-colored pixels in a box around screen pos (sx, sy).

    The scan pulse ring + confirmed contacts are SENSOR_SCAN_COLOR
    (255, 220, 120) — warm (r > 150 and g > 120 and b < 180). The
    colors are drawn ABOVE the fog (un-dimmed), so the raw-color test
    is stable. The box is large enough to absorb the camera's
    sub-pixel easing drift.

    NOTE: the PASSIVE blip (SENSOR_COLOR, green) is NOT counted this
    way — the enemy hull is ALSO green (ENEMY_FILL/ENEMY_EDGE), so a
    raw green count matches the hull, not the blip. The passive gates
    use `box_diff` (sensor-off vs sensor-on) instead."""
    cx, cy = int(sx), int(sy)
    box = screen.subsurface(cx - half, cy - half, 2 * half, 2 * half)
    data = pygame.image.tostring(box, "RGB")
    n = 0
    for i in range(0, len(data), 3):
        r, gg, b = data[i], data[i + 1], data[i + 2]
        if warm:
            if r > 150 and gg > 120 and b < 180:
                n += 1
        else:
            if gg > 150 and gg > r and gg > b:
                n += 1
    return n


def box_bytes(g, sx, sy, half=25):
    """The raw RGB bytes of a box around screen pos (sx, sy). Used by
    `box_diff` to compare two frames (sensor-off vs sensor-on)."""
    cx, cy = int(sx), int(sy)
    box = g.screen.subsurface(cx - half, cy - half, 2 * half, 2 * half)
    return pygame.image.tostring(box, "RGB")


def box_diff(a, b):
    """Count the pixels that DIFFER between two same-sized box byte
    strings. The passive blip is green — the SAME color family as the
    enemy hull (ENEMY_FILL/ENEMY_EDGE) — so a raw green count can't
    tell the blip from the hull. But the blip is only drawn when the
    sensor is ON, so diffing the sensor-off frame against the
    sensor-on frame isolates exactly the pixels the blip added."""
    n = 0
    for i in range(0, len(a), 3):
        if a[i:i + 3] != b[i:i + 3]:
            n += 1
    return n


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()

    # --- 1. WIRE-SHAPE: Ship.snapshot is a 25-tuple (10.7: bullet_seq
    #     LAST, index 24; power_used at [22]); the enemy's nested ship
    #     snapshot carries it; JSON round-trip preserves it. ---
    tmp = _scratch_game()
    ls = tmp.players[1]
    ls.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)
    e.ship.power_used = ENEMY_POWER_USED
    snap = tmp.snapshot()
    ls_s = snap[0][1]
    assert len(ls_s) == 25, "Ship.snapshot must be a 25-tuple, got %d" \
        % len(ls_s)
    # The enemy's Game-snapshot entry is (tag, e.snapshot()) where
    # e.snapshot() = (ship_s, hp, id, ax, ay) — the ship snapshot is at
    # [1][0].
    es_s = snap[1][ENEMY_IDX][1][0]
    assert len(es_s) == 25, "enemy ship snapshot must be a 25-tuple, " \
        "got %d" % len(es_s)
    assert abs(es_s[22] - ENEMY_POWER_USED) < 1e-9, \
        "enemy ship snapshot must carry power_used at [22]: %r" \
        % (es_s[22],)
    rt = json.loads(json.dumps(snap))
    assert abs(rt[1][ENEMY_IDX][1][0][22] - ENEMY_POWER_USED) < 1e-9, \
        "JSON round-trip must preserve power_used: %r" \
        % (rt[1][ENEMY_IDX][1][0][22],)
    print("PASS: WIRE-SHAPE — Ship.snapshot is a 25-tuple (power_used at "
          "[22], bullet_seq at [24]); the enemy's nested ship snapshot "
          "carries it; JSON round-trip preserves it")

    # --- 2. BUFFER-CARRIES: the buffer's enemy entry is an 11-tuple with
    #     power_used at [9], lerp'd across the window. ---
    g = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)
    push_stream(g, ENEMY_POWER_USED)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    pos = g.snap_buf.positions_at(g.render_point.now())
    assert pos is not None and pos['enemies'], "no buffer window"
    entry = pos['enemies'][ENEMY_IDX]
    assert len(entry) == 11, "buffer enemy entry must be an 11-tuple, " \
        "got %d" % len(entry)
    assert abs(entry[9] - ENEMY_POWER_USED) < 1e-9, \
        "buffer enemy entry must carry power_used at [9]: %r" % (entry[9],)
    # Lerp: two snapshots with different power_used, sampled at the
    # midpoint (alpha 0.5) -> the average; endpoint-exact at alpha 0/1.
    g.snap_buf = g.snap_buf.__class__()
    g.push_snapshot(0.0, make_snap(10.0), now=0.0)
    g.push_snapshot(0.1, make_snap(30.0), now=0.1)
    mid = g.snap_buf.positions_at(0.05)['enemies'][ENEMY_IDX][9]
    assert abs(mid - 20.0) < 1e-6, \
        "power_used must lerp across the window (midpoint = average): " \
        "%r" % (mid,)
    lo = g.snap_buf.positions_at(0.0)['enemies'][ENEMY_IDX][9]
    hi = g.snap_buf.positions_at(0.1)['enemies'][ENEMY_IDX][9]
    assert abs(lo - 10.0) < 1e-9 and abs(hi - 30.0) < 1e-9, \
        "power_used must be endpoint-exact at alpha 0/1: %r / %r" \
        % (lo, hi)
    print("PASS: BUFFER-CARRIES — buffer enemy entry is an 11-tuple "
          "(power_used at [9]), lerp'd across the window (midpoint = "
          "average, endpoint-exact at alpha 0/1)")

    # --- 3. PROXY-POWER: _ghost_enemy_proxies fills power_used (from the
    #     buffer) + power_idle_total (derived from the tag's loadout) on
    #     the proxy's .ship. ---
    push_stream(g, ENEMY_POWER_USED)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    pos = g.snap_buf.positions_at(g.render_point.now())
    proxies = g._ghost_enemy_proxies(pos)
    assert len(proxies) == len(pos['enemies']), \
        "one proxy per buffer enemy: %d vs %d" % (len(proxies),
                                                  len(pos['enemies']))
    p = proxies[ENEMY_IDX]
    assert abs(p.ship.power_used - ENEMY_POWER_USED) < 1e-9, \
        "proxy power_used must come from the buffer: %r" \
        % (p.ship.power_used,)
    idle = Ship(hull=ENEMY_HULL, loadout=enemy_loadout()).power_idle_total
    assert abs(p.ship.power_idle_total - idle) < 1e-9, \
        "proxy power_idle_total must be derived from the tag's loadout: " \
        "%r vs %r" % (p.ship.power_idle_total, idle)
    sig = p.ship.power_used - p.ship.power_idle_total
    assert abs(sig - (ENEMY_POWER_USED - idle)) < 1e-9, \
        "the proxy's signature must be power_used - power_idle_total: " \
        "%r" % (sig,)
    print("PASS: PROXY-POWER — proxies carry power_used (from the buffer) "
          "+ power_idle_total (derived from the tag's loadout); signature "
          "= %.1f" % sig)

    # --- 4. CONTACT-PASSIVE: a ghost with sensor_on=True + a proxy whose
    #     signature crosses the threshold within sensor_range -> a
    #     passive contact. Below threshold / sensor off -> no contact.
    #     The ghost is built with the PASSIVE fit (the default
    #     ACTIVE_SCANNER has sensor_range=0 — passive sensing off). ---
    ghost = PredictedShip(hull=PLAYER_HULLS[1], loadout=passive_loadout(),
                          local_index=1)
    ghost.seed(make_snap(ENEMY_POWER_USED)[0][1])
    ghost.ship.sensor_on = True
    proxy = _GhostEnemyProxy(enemy_world(), pygame.Vector2(0, 0), 0.0,
                             1, ENEMY_POWER_USED, idle)
    ghost.step(TICK, ShipInput(), enemies=[proxy])
    assert len(ghost.ship.contacts) == 1, \
        "a detectable enemy in range must produce a passive contact: " \
        "contacts=%r" % (ghost.ship.contacts,)
    cpos, cdist, cstr, cconf = ghost.ship.contacts[0]
    assert cconf is False, "a passive contact must be unconfirmed: %r" \
        % (cconf,)
    assert abs(cdist - ENEMY_OFFSET) < 1e-6, \
        "the contact's distance must be the enemy's range: %r" % (cdist,)
    want = min(1.0, (ENEMY_POWER_USED - idle) / SENSOR_SIG_FULL)
    assert abs(cstr - want) < 1e-9, \
        "the contact's strength must be sig/SIG_FULL clamped: %r vs %r" \
        % (cstr, want)
    assert cpos.distance_to(enemy_world()) < 1e-6, \
        "the contact's pos must be the enemy's pos: %r" % (cpos,)
    # Below threshold: no contact.
    ghost2 = PredictedShip(hull=PLAYER_HULLS[1], loadout=passive_loadout(),
                           local_index=1)
    ghost2.seed(make_snap(ENEMY_POWER_QUIET)[0][1])
    ghost2.ship.sensor_on = True
    proxy2 = _GhostEnemyProxy(enemy_world(), pygame.Vector2(0, 0), 0.0,
                              1, ENEMY_POWER_QUIET, idle)
    ghost2.step(TICK, ShipInput(), enemies=[proxy2])
    assert not ghost2.ship.contacts, \
        "an enemy below the signature threshold must NOT produce a " \
        "contact: contacts=%r" % (ghost2.ship.contacts,)
    # Sensor off: no contact (even for a detectable enemy).
    ghost3 = PredictedShip(hull=PLAYER_HULLS[1], loadout=passive_loadout(),
                           local_index=1)
    ghost3.seed(make_snap(ENEMY_POWER_USED)[0][1])
    ghost3.ship.sensor_on = False
    ghost3.step(TICK, ShipInput(), enemies=[proxy])
    assert not ghost3.ship.contacts, \
        "sensor_on=False must produce no passive contact: contacts=%r" \
        % (ghost3.ship.contacts,)
    print("PASS: CONTACT-PASSIVE — a detectable enemy in range produces a "
          "passive contact (dist=%.0f, strength=%.2f, unconfirmed); below "
          "threshold or sensor off -> no contact" % (cdist, cstr))

    # --- 5. CONTACT-SCAN: a ghost with scan_reveal>0 + a proxy within
    #     scan_range -> a confirmed contact (strength 1.0) that REPLACES
    #     a passive one for the same enemy. G (fire_scan) takes effect
    #     immediately. ghost4 uses the SENSOR_ARRAY fit — the only fit
    #     with BOTH a passive range (1500) and a scan range (2000) — so
    #     the "replaces a passive one" claim is real (the enemy is
    #     passively detectable too); a PASSIVE_SENSOR fit would have
    #     scan_range=0 and no scan contact at all. ghost5 keeps the
    #     DEFAULT fit (ACTIVE_SCANNER, scan_range 2400) to prove the
    #     scan works with either fit. ---
    ghost4 = PredictedShip(hull=PLAYER_HULLS[1], loadout=array_loadout(),
                           local_index=1)
    ghost4.seed(make_snap(ENEMY_POWER_USED)[0][1])
    ghost4.ship.sensor_on = True
    ghost4.ship.scan_reveal = 1.0   # a ping's reveal is in progress
    ghost4.step(TICK, ShipInput(), enemies=[proxy])
    assert len(ghost4.ship.contacts) == 1, \
        "a scan reveal must produce a contact: contacts=%r" \
        % (ghost4.ship.contacts,)
    cpos, cdist, cstr, cconf = ghost4.ship.contacts[0]
    assert cconf is True, "a scan-reveal contact must be confirmed: %r" \
        % (cconf,)
    assert abs(cstr - 1.0) < 1e-9, \
        "a confirmed contact must have full strength: %r" % (cstr,)
    assert cpos.distance_to(enemy_world()) < 1e-6, \
        "the confirmed contact must be at the enemy's pos: %r" % (cpos,)
    # G (fire_scan) takes effect immediately: the client's command sink
    # calls fire_scan() on the ghost; the NEXT step's contacts are
    # confirmed (scan_reveal > 0 from this tick).
    ghost5 = PredictedShip(hull=PLAYER_HULLS[1], loadout=None, local_index=1)
    ghost5.seed(make_snap(ENEMY_POWER_USED)[0][1])
    ghost5.ship.sensor_on = False   # the scan works even with the passive
    # sensor off (scan_reveal is independent of sensor_on)
    fired = ghost5.ship.fire_scan()
    assert fired is True and ghost5.ship.scan_reveal > 0, \
        "G must fire the ghost's scan (scan_reveal set): fired=%r " \
        "reveal=%.2f" % (fired, ghost5.ship.scan_reveal)
    ghost5.step(TICK, ShipInput(), enemies=[proxy])
    assert len(ghost5.ship.contacts) == 1, \
        "the scan must produce a contact on the next step: contacts=%r" \
        % (ghost5.ship.contacts,)
    assert ghost5.ship.contacts[0][3] is True, \
        "the scan's contact must be confirmed: %r" % (ghost5.ship.contacts,)
    print("PASS: CONTACT-SCAN — a scan reveal produces a confirmed contact "
          "(strength 1.0) that replaces the passive one; G (fire_scan) "
          "takes effect immediately")

    # --- 6. CONTACT-PARITY: the ghost's contacts (same sensor state +
    #     same enemy positions) match the host's _update_contacts output
    #     — proves the math is identical. BOTH sides use the passive fit
    #     (the default ACTIVE_SCANNER has sensor_range=0 -> no passive
    #     contact on either side): the host's scratch ship gets the
    #     passive sensor swapped in-place (sensor_comp is a reference
    #     into components[slot.name]), the ghost is built with it. ---
    host = _scratch_game()
    hs = host.players[1]
    hs.components[hs.sensor_slot.name] = PASSIVE_SENSOR
    hs.sensor_comp = PASSIVE_SENSOR
    hs.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    hs.sensor_on = True
    he = host.enemies[ENEMY_IDX]
    he.ship.pos = enemy_world()
    he.ship.power_used = ENEMY_POWER_USED
    host._update_contacts(hs)
    ghost6 = PredictedShip(hull=PLAYER_HULLS[1], loadout=passive_loadout(),
                           local_index=1)
    ghost6.seed(make_snap(ENEMY_POWER_USED)[0][1])
    ghost6.ship.sensor_on = True
    proxy6 = _GhostEnemyProxy(enemy_world(), pygame.Vector2(0, 0), 0.0,
                              he.ship.id, ENEMY_POWER_USED, idle)
    ghost6.step(TICK, ShipInput(), enemies=[proxy6])
    assert len(ghost6.ship.contacts) == len(hs.contacts) == 1, \
        "the ghost and the host must produce the same number of contacts: " \
        "ghost=%d host=%d" % (len(ghost6.ship.contacts), len(hs.contacts))
    gpos, gdist, gstr, gconf = ghost6.ship.contacts[0]
    hpos, hdist, hstr, hconf = hs.contacts[0]
    assert gconf == hconf, "confirmed must match: %r vs %r" % (gconf, hconf)
    assert abs(gdist - hdist) < 1e-6, \
        "distance must match: %r vs %r" % (gdist, hdist)
    assert abs(gstr - hstr) < 1e-9, \
        "strength must match: %r vs %r" % (gstr, hstr)
    assert gpos.distance_to(hpos) < 1e-6, \
        "pos must match: %r vs %r" % (gpos, hpos)
    print("PASS: CONTACT-PARITY — the ghost's contacts match the host's "
          "_update_contacts output (same enemy, dist=%.1f, strength=%.2f, "
          "confirmed=%r)" % (hdist, hstr, hconf))

    # --- 7. PIXEL-BLIP: predicted_view draws the ghost's passive contact
    #     blip above the fog (SENSOR_COLOR pixels at the enemy's screen
    #     pos; the sensor-off baseline has none). The ghost is rebuilt
    #     with the PASSIVE fit (the default ACTIVE_SCANNER has
    #     sensor_range=0 -> no passive blip). ---
    push_stream(g, ENEMY_POWER_USED, loadout=passive_loadout())
    render_frame(g)
    assert not g.ghost.ship.contacts, \
        "baseline (sensor off) must have no contacts: %r" \
        % (g.ghost.ship.contacts,)
    # The passive blip is green — the SAME color family as the enemy
    # hull (ENEMY_FILL/ENEMY_EDGE) — so a raw green count can't tell
    # the blip from the hull. Instead, diff the sensor-off frame
    # against the sensor-on frame: the blip is only drawn when the
    # sensor is on, so the diff isolates exactly the pixels it added.
    off = box_bytes(g, *blip_screen())
    # The client's V: the command sink sets the ghost's sensor_on
    # directly (client-authoritative — the host's snapshot carries
    # sensor_on=False, so this is the client's own state).
    g.ghost.ship.sensor_on = True
    render_frame(g)
    assert len(g.ghost.ship.contacts) == 1, \
        ("predicted_view must build the ghost's contacts (sensor on, "
         "detectable enemy in range): contacts=%r"
         % (g.ghost.ship.contacts,))
    on = box_bytes(g, *blip_screen())
    added = box_diff(off, on)
    assert added > 5, \
        ("predicted_view must render the ghost's passive contact blip "
         "(pixels added at the enemy's screen pos when the sensor turns "
         "on): added=%d" % (added,))
    print("PASS: PIXEL-BLIP — predicted_view renders the ghost's passive "
          "contact blip above the fog (%d pixels added at the enemy's "
          "screen pos when the sensor turns on)" % (added,))

    # --- 8. PIXEL-PULSE: predicted_view draws the ghost's scan pulse
    #     ring (the expanding circle on a G press) above the fog. ---
    push_stream(g, ENEMY_POWER_USED)
    render_frame(g)
    base_pulse = sensor_pixels(g.screen, WIDTH / 2, HEIGHT / 2,
                               half=120, warm=True)
    # The client's G: the command sink calls fire_scan() on the ghost.
    fired = g.ghost.ship.fire_scan()
    assert fired is True, "G must fire the ghost's scan: fired=%r" % (fired,)
    render_frame(g)
    assert g.ghost.ship.scan_pulse >= 0, \
        "the ghost's scan_pulse must be in flight after G: %r" \
        % (g.ghost.ship.scan_pulse,)
    # The ring's radius at render time: r = min(1, t) * scan_range,
    # t = scan_pulse / scan_duration. Sample a box around a point on the
    # ring (the ring is 2 px wide; the box absorbs the camera's
    # sub-pixel drift + the 2 px line width).
    c = g.ghost.ship.sensor_comp
    t = g.ghost.ship.scan_pulse / max(c.scan_duration, 0.01)
    r = min(1.0, t) * c.scan_range
    ring_pt = pygame.Vector2(WIDTH / 2 + r, HEIGHT / 2)
    pulse = sensor_pixels(g.screen, *ring_pt, half=12, warm=True)
    assert pulse > 0, \
        ("predicted_view must render the ghost's scan pulse ring (warm "
         "pixels on the ring at r=%.0f): pulse=%d" % (r, pulse))
    print("PASS: PIXEL-PULSE — predicted_view renders the ghost's scan "
          "pulse ring above the fog (%d warm pixels on the ring at "
          "r=%.0f)" % (pulse, r))

    # --- 9. PIXEL-ARROW: an off-screen enemy shows an edge arrow (the
    #     on-screen blip is absent). The ghost is rebuilt with the
    #     PASSIVE fit (the default ACTIVE_SCANNER has sensor_range=0 ->
    #     no passive contact -> no arrow). ---
    push_stream(g, ENEMY_POWER_USED, enemy_offset=ENEMY_OFFSET_FAR,
                loadout=passive_loadout())
    render_frame(g)
    # The passive arrow is DIMMED by strength (0.35 -> g=147, below the
    # raw green test's g>150), so — like the blip — diff the sensor-off
    # frame against the sensor-on frame to isolate the arrow's pixels.
    arrow_x = WIDTH - SENSOR_ARROW_MARGIN
    off = box_bytes(g, arrow_x, HEIGHT / 2, half=15)
    g.ghost.ship.sensor_on = True
    render_frame(g)
    assert len(g.ghost.ship.contacts) == 1, \
        "the off-screen enemy must produce a contact: contacts=%r" \
        % (g.ghost.ship.contacts,)
    # The enemy's screen pos is past WIDTH + 20 (off-screen), so the
    # blip is NOT drawn; the edge arrow is at the screen edge (the
    # enemy is straight right of the ship, so the arrow is at
    # (WIDTH - SENSOR_ARROW_MARGIN, HEIGHT / 2)).
    assert blip_screen(ENEMY_OFFSET_FAR).x > WIDTH + 20, \
        "the far enemy must be off-screen: %r" \
        % (blip_screen(ENEMY_OFFSET_FAR).x,)
    on = box_bytes(g, arrow_x, HEIGHT / 2, half=15)
    added = box_diff(off, on)
    assert added > 0, \
        ("predicted_view must render the off-screen enemy's edge arrow "
         "(pixels added at the screen edge when the sensor turns on): "
         "added=%d" % (added,))
    print("PASS: PIXEL-ARROW — an off-screen enemy shows an edge arrow "
          "(%d pixels added at the screen edge when the sensor turns "
          "on; the on-screen blip is absent)" % (added,))

    print("ALL PASS: 10.3c (ghost emits sensor contacts)")


if __name__ == "__main__":
    main()