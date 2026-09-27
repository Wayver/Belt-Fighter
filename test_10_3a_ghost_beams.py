"""Session 10.3a: the prediction ghost emits laser beams.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3a_ghost_beams

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3a makes the CLIENT's local ship (the prediction ghost) emit laser
beams, the same hitscan discharge the host's sim produces. Before 10.3a
the ghost stepped BLIND (no enemy list to target) and discarded the beams
`s.update()` returned, so the player's own beam appeared only when the
host's next snapshot arrived (~100 ms later).

The fix (NO wire change — the ghost's weapon state, charge/lock/
cooldown, is already synced via apply_snapshot on reconcile; the only
missing input was the TARGET):
  * predicted_view is REORDERED: the render point + buffer window are
    computed BEFORE the ghost advances, so the ghost can see the buffer's
    enemies while it steps. A lightweight enemy-proxy list is built from
    the buffer's enemy entries and fed to ghost.advance -> step.
  * step() picks the ghost's laser target from the proxies (the SAME
    nearest-in-range math the host's _pick_laser_target uses) and KEEPS
    the beams s.update() returns as presentation entries in
    ghost.local_beams (world-space muzzle -> target, aged at the host's
    0.15 s beam ttl).
  * predicted_view draws the ghost's beams BEFORE the local ship (mirrors
    the host's draw() order), reusing the host's beam visual (LASER_COLOR
    line, width 2, fading over the ttl).

The ghost is PRESENTATION-ONLY: it never feeds the sim. The host's draw()
+ the sim (Game._step) + the wire snapshot are UNCHANGED.

This test proves it (value-level — flames/arcs use per-frame random, so a
full-frame pixel compare is not the gate; the PIXEL gate is a
beam-pixel-count sanity check):
  1. PLUMBING     — _ghost_enemy_proxies builds one proxy per buffer
                    enemy entry (pos/vel/angle/id from the entry); the
                    ghost's laser target is picked from the proxies
                    (nearest-in-range; None when out of range / no laser).
  2. SAME-TICK    — for a scripted input that fires the laser at a buffer
                    enemy, the ghost emits a beam on the same tick the
                    host does, with the same start/end (the ghost is
                    seeded from the host's snapshot, so the muzzle matches;
                    the buffer's enemy pos ~ the host's at that instant).
  3. NO-BEAM      — no beam when the host fires none (enemy out of range,
                    or in range but laser_fire not pressed).
  4. AGE-CULL     — a fired beam ages at the host's 0.15 s ttl and is
                    culled (mirrors the host's beam aging).
  5. REWIND-CHARGE— reconcile_rewind feeds the enemy proxies to the
                    replayed steps, so the replayed laser charge matches
                    the host's (the host's replayed ticks saw the live
                    enemies in the wedge).
  6. PIXEL        — predicted_view renders the ghost's beam (greenish
                    beam pixels appear on the client; the no-beam baseline
                    has none).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT, LASER_COLOR
from .fog import make_light_texture
from .game import Game, _GhostEnemyProxy
from .netcode import PredictedShip, HostTimeEstimator, LatencyTracker, RenderPoint
from .ship import Ship
from .intent import ShipInput
from .hulls import SILAS_HULL, default_loadout
from .ai_enemy import AIEnemy

SEED = 1234
# The target enemy sits this far to the right of the local ship (the ghost,
# the fog's light source) — inside the laser's 600 px range and the
# silas 360-degree firing arc, so the ghost charges + fires at it.
ENEMY_OFFSET = 100.0
ENEMY_IDX = 0
# Silas's eyes: two LASER_360 (range 600, charge 0.5 s, 360-degree arc).
LASER_RANGE = 600.0
CHARGE_TIME = 0.5
# The beam's ttl (host's Game._step: b[5] = 0.15).
BEAM_TTL = 0.15
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
    """A client Game (local_index=1) with the silas hull on BOTH players
    (the scratch game + the client share the hull, so the snapshot's
    client-ship weapon count matches the ghost's)."""
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
    tmp = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL),
                seed=SEED, players=2)
    # Player 1 (the client's ship) is a bare placeholder by default (no
    # weapons) — give it a real silas ship so the client-ship snapshot
    # carries the weapons tuple (charge/lock) the ghost restores on seed.
    tmp.set_player_ship(1, Ship(hull=SILAS_HULL,
                                loadout=default_loadout(SILAS_HULL)))
    return tmp


def make_snap(charge, enemy_offset=ENEMY_OFFSET):
    """A valid 2-player Game snapshot with the CLIENT ship (index 1) at the
    origin facing +x, its lasers at `charge`, and the TARGET enemy
    (ENEMY_IDX) at (enemy_offset, 0). All other enemies keep their
    (zero) weapon state. The client ship's pose is fixed so the ghost's
    muzzle (seeded from this snapshot) matches the host's at the fire
    tick."""
    tmp = _scratch_game()
    # Client ship (index 1): origin, facing +x, vel 0.
    cs = tmp.players[1]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    # Target enemy: enemy_offset right of the client ship.
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(enemy_offset, 0.0)
    e.ship.angle = 0.0
    snap = tmp.snapshot()
    # Rebuild the client-ship snapshot with the lasers at `charge`.
    cs_s = list(snap[0][1])
    cs_s[18] = tuple((0.0, charge, 0.0) for _ in cs.weapons)
    ships = (snap[0][0], tuple(cs_s))
    return (ships,) + snap[1:]


def push_stream(g, charge, enemy_offset=ENEMY_OFFSET):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the client ship at
    `charge` and the target enemy at (enemy_offset, 0)."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=SILAS_HULL,
                            loadout=default_loadout(SILAS_HULL),
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(charge, enemy_offset), now=t)


def beam_pixels(screen, cam, beams):
    """Count beam-colored pixels sampled along the ghost's beam lines.

    The beam is drawn BEFORE the fog (mirrors the host's draw() order), so
    at ~100 px from the light source (the ghost) the fog dims LASER_COLOR
    (140, 255, 190) to ~(93, 170, 126) — still green-dominant (g > r and
    g > b), but no longer the raw color. The beam line runs from the
    muzzle to the enemy; the points BETWEEN them are dark background in
    the baseline (no beam) and beam-colored when the beam is drawn. So
    for each beam, sample a small box around a few points along the line
    (t = 0.3, 0.5, 0.7 — avoiding the muzzle and the enemy hull) and count
    the beam-colored pixels (g > 100 and g > r and g > b). Baseline (no
    beams) -> 0; a fired beam -> > 0. This is a value-level pixel check
    (the beam is a deterministic 2 px line; flames/arcs use per-frame
    random, so a full-frame compare is not the gate)."""
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


def _keys_with_laser_fire():
    """A key-state object with laser_fire (K_r) pressed, nothing else.
    predicted_view reads ShipInput.from_keys(keys); this drives the
    ghost's laser_fire without a real keyboard."""
    class _K:
        def __getitem__(self, k):
            return k == pygame.K_r
    return _K()


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g = make_client(screen, font, big_font, light_tex, fog_surf, light_surf)

    # --- 1. PLUMBING: _ghost_enemy_proxies builds one proxy per buffer
    #     enemy entry (pos/vel/angle/id from the entry); the ghost's laser
    #     target is picked from the proxies (nearest-in-range; None when
    #     out of range / no laser). ---
    push_stream(g, 0.0)
    # Advance the render point so now() returns a value (it is None until
    # the first advance, mirroring the 10.2b render_frame helper).
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    pos = g.snap_buf.positions_at(g.render_point.now())
    proxies = g._ghost_enemy_proxies(pos)
    assert len(proxies) == len(pos['enemies']), \
        "one proxy per buffer enemy: %d vs %d" % (len(proxies),
                                                  len(pos['enemies']))
    for p, e in zip(proxies, pos['enemies']):
        assert isinstance(p, _GhostEnemyProxy)
        assert abs(p.pos.x - e[1]) < 1e-9 and abs(p.pos.y - e[2]) < 1e-9, \
            "proxy pos from the buffer entry: %r vs %r" % (p.pos, e[1:3])
        assert abs(p.vel.x - e[4]) < 1e-9 and abs(p.vel.y - e[5]) < 1e-9, \
            "proxy vel from the buffer entry: %r vs %r" % (p.vel, e[4:6])
        assert p.id == e[6], "proxy id from the buffer entry: %r vs %r" \
            % (p.id, e[6])
    # Laser target: nearest-in-range. The target enemy (ENEMY_IDX) is at
    # (100, 0), 100 px from the ghost (origin) — in range. The other
    # enemies are far (the scratch game spawns them around the world), so
    # the target is the nearest.
    g.ghost.seed(make_snap(0.0)[0][1])
    g.ghost.step(TICK, ShipInput(), enemies=proxies)
    tgt = g.ghost.ship.laser_target
    assert tgt is not None, "the ghost's laser target must be picked from " \
        "the proxies (the target enemy is in range)"
    assert tgt.id == pos['enemies'][ENEMY_IDX][6], \
        "the laser target must be the nearest-in-range proxy (the target " \
        "enemy): got id %r want %r" % (tgt.id, pos['enemies'][ENEMY_IDX][6])
    # Out of range: no target. Build a synthetic out-of-range proxy directly
    # (the buffer holds the in-range stream, so this isolates the range
    # gate).
    far = _GhostEnemyProxy(pygame.Vector2(1000.0, 0.0), pygame.Vector2(0, 0),
                           0.0, 99, 0.0, 0.0)
    g.ghost.seed(make_snap(0.0)[0][1])
    g.ghost.step(TICK, ShipInput(), enemies=[far])
    assert g.ghost.ship.laser_target is None, \
        "no laser target when the only enemy is out of range (1000 px > " \
        "600 px): got %r" % (g.ghost.ship.laser_target,)
    # No enemies: no target.
    g.ghost.step(TICK, ShipInput(), enemies=[])
    assert g.ghost.ship.laser_target is None, \
        "no laser target with no enemies"
    print("PASS: PLUMBING — proxies carry pos/vel/angle/id; the laser "
          "target is picked from them (nearest-in-range; None out of range)")

    # --- 2. SAME-TICK: the ghost emits a beam on the same tick the host
    #     does, with the same start/end. The ghost is seeded from the
    #     host's snapshot (muzzle matches); the buffer's enemy pos ~ the
    #     host's at that instant. ---
    tmp = _scratch_game()
    cs = tmp.players[1]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        w.charge = 1.0
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(ENEMY_OFFSET, 0.0)
    e.ship.angle = 0.0
    # Mirror the host's _step: set the laser target BEFORE update() (the
    # host's _pick_laser_target picks the nearest-in-range enemy).
    cs.laser_target = tmp._pick_laser_target(cs)
    assert cs.laser_target is e, "the host's laser target must be the " \
        "target enemy (in range): got %r" % (cs.laser_target,)
    host_beam = None
    shots, beams, _m = cs.update(TICK, ShipInput(laser_fire=True))
    for b in beams:
        host_beam = b
        break
    assert host_beam is not None, "the host must fire a beam (charge 1.0, " \
        "laser_fire, enemy in range)"
    # The ghost, seeded from the SAME snapshot, fires on the same tick.
    ghost = PredictedShip(hull=SILAS_HULL,
                          loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost.seed(make_snap(1.0)[0][1])
    proxy = _GhostEnemyProxy(pygame.Vector2(ENEMY_OFFSET, 0.0),
                             pygame.Vector2(0, 0), 0.0,
                             e.ship.id, 0.0, 0.0)
    ghost.step(TICK, ShipInput(laser_fire=True), enemies=[proxy])
    ghost.step_local_beams(TICK)
    assert len(ghost.local_beams) >= 1, \
        "the ghost must emit a beam on the same tick the host does: " \
        "local_beams=%r" % (ghost.local_beams,)
    gb = ghost.local_beams[0]
    # start (muzzle): the ghost is seeded from the host's snapshot, so the
    # muzzle matches the host's beam.start.
    assert gb[0].distance_to(host_beam.start) < 1e-6, \
        "the ghost's beam start (muzzle) must match the host's: %r vs %r" \
        % (gb[0], host_beam.start)
    # end (target): the buffer's enemy pos ~ the host's at that instant
    # (the enemy is static here, so it matches exactly).
    assert gb[1].distance_to(host_beam.end) < 1e-6, \
        "the ghost's beam end (target) must match the host's: %r vs %r" \
        % (gb[1], host_beam.end)
    print("PASS: SAME-TICK — the ghost fires a beam on the same tick the "
          "host does, same start/end (muzzle %r, target %r)"
          % (tuple(round(v, 1) for v in gb[0]),
             tuple(round(v, 1) for v in gb[1])))

    # --- 3. NO-BEAM: no beam when the host fires none. ---
    # (a) Enemy out of range: the host's _pick_laser_target returns None,
    #     so the host charges nothing and fires nothing; the ghost must
    #     match (no target -> no charge -> no beam).
    tmp2 = _scratch_game()
    cs2 = tmp2.players[1]
    cs2.pos = pygame.Vector2(0.0, 0.0)
    cs2.vel = pygame.Vector2(0.0, 0.0)
    cs2.angle = 0.0
    e2 = tmp2.enemies[ENEMY_IDX]
    e2.ship.pos = pygame.Vector2(1000.0, 0.0)   # out of range
    tmp2._step(TICK, ShipInput(laser_fire=True))
    assert not tmp2.beams, "the host must fire no beam (enemy out of " \
        "range): beams=%r" % (tmp2.beams,)
    ghost2 = PredictedShip(hull=SILAS_HULL,
                           loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost2.seed(make_snap(0.0, enemy_offset=1000.0)[0][1])
    far2 = _GhostEnemyProxy(pygame.Vector2(1000.0, 0.0),
                            pygame.Vector2(0, 0), 0.0, e2.ship.id, 0.0, 0.0)
    for _ in range(40):
        ghost2.step(TICK, ShipInput(laser_fire=True), enemies=[far2])
        ghost2.step_local_beams(TICK)
    assert not ghost2.local_beams, \
        "the ghost must fire no beam when the enemy is out of range: " \
        "local_beams=%r" % (ghost2.local_beams,)
    # (b) Enemy in range but laser_fire not pressed: the host charges but
    #     does not fire; the ghost must match (charge but no beam).
    tmp3 = _scratch_game()
    cs3 = tmp3.players[1]
    cs3.pos = pygame.Vector2(0.0, 0.0)
    cs3.vel = pygame.Vector2(0.0, 0.0)
    cs3.angle = 0.0
    e3 = tmp3.enemies[ENEMY_IDX]
    e3.ship.pos = pygame.Vector2(ENEMY_OFFSET, 0.0)
    tmp3._step(TICK, ShipInput())   # no laser_fire
    assert not tmp3.beams, "the host must fire no beam (laser_fire not " \
        "pressed): beams=%r" % (tmp3.beams,)
    ghost3 = PredictedShip(hull=SILAS_HULL,
                           loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost3.seed(make_snap(0.0)[0][1])
    proxy3 = _GhostEnemyProxy(pygame.Vector2(ENEMY_OFFSET, 0.0),
                              pygame.Vector2(0, 0), 0.0, e3.ship.id, 0.0, 0.0)
    for _ in range(40):
        ghost3.step(TICK, ShipInput(), enemies=[proxy3])
        ghost3.step_local_beams(TICK)
    assert not ghost3.local_beams, \
        "the ghost must fire no beam when laser_fire is not pressed: " \
        "local_beams=%r" % (ghost3.local_beams,)
    print("PASS: NO-BEAM — no beam when the host fires none (out of range "
          "or laser_fire not pressed)")

    # --- 4. AGE-CULL: a fired beam ages at the host's 0.15 s ttl and is
    #     culled (mirrors the host's beam aging). ---
    ghost4 = PredictedShip(hull=SILAS_HULL,
                           loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost4.seed(make_snap(1.0)[0][1])
    proxy4 = _GhostEnemyProxy(pygame.Vector2(ENEMY_OFFSET, 0.0),
                              pygame.Vector2(0, 0), 0.0, 1, 0.0, 0.0)
    ghost4.step(TICK, ShipInput(laser_fire=True), enemies=[proxy4])
    ghost4.step_local_beams(TICK)
    assert len(ghost4.local_beams) >= 1, "the ghost must fire a beam"
    n_beams = len(ghost4.local_beams)
    # Age the beam one tick: age advances by TICK.
    ghost4.step(TICK, ShipInput(), enemies=[proxy4])
    ghost4.step_local_beams(TICK)
    # The beam(s) fired last tick are now aged by 2 ticks (the fire tick's
    # step_local_beams + this tick's). The charge reset to 0 on fire, so
    # no new beam this tick.
    assert len(ghost4.local_beams) == n_beams, \
        "the beam must persist for one more tick (age < ttl): %d vs %d" \
        % (len(ghost4.local_beams), n_beams)
    age = ghost4.local_beams[0][2]
    assert abs(age - 2 * TICK) < 1e-9, \
        "the beam's age must advance by TICK each step: %r vs %r" \
        % (age, 2 * TICK)
    # Age the beam past the ttl (0.15 s = 9 ticks): it is culled.
    for _ in range(12):
        ghost4.step(TICK, ShipInput(), enemies=[proxy4])
        ghost4.step_local_beams(TICK)
    assert not ghost4.local_beams, \
        "the beam must be culled after the ttl (0.15 s): local_beams=%r" \
        % (ghost4.local_beams,)
    print("PASS: AGE-CULL — a fired beam ages at the host's 0.15 s ttl "
          "and is culled")

    # --- 5. REWIND-CHARGE: reconcile_rewind feeds the enemy proxies to
    #     the replayed steps, so the replayed laser charge matches the
    #     host's (the host's replayed ticks saw the live enemies in the
    #     wedge). ---
    ghost5 = PredictedShip(hull=SILAS_HULL,
                           loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost5.seed(make_snap(0.9)[0][1])   # charge 0.9
    # Buffer 30 ticks (0.5 s) of idle input, stamped at host time 0..0.5.
    for i in range(30):
        ghost5.record_input(i * TICK, ShipInput())
    proxy5 = _GhostEnemyProxy(pygame.Vector2(ENEMY_OFFSET, 0.0),
                              pygame.Vector2(0, 0), 0.0, 1, 0.0, 0.0)
    # Rewind: replay the 30 buffered ticks with the enemy in the wedge.
    # The charge advances 0.9 -> 1.0 (capped) over the replay.
    ghost5.reconcile_rewind(make_snap(0.9)[0][1], 0.0, 30 * TICK,
                            enemies=[proxy5])
    charge = min(w.charge for w in ghost5.ship.weapons)
    assert charge >= 0.95, \
        "the replayed laser charge must advance (the host's replayed ticks " \
        "saw the enemy in the wedge): charge=%r" % (charge,)
    # Without enemies (the pre-10.3a behavior) the charge would NOT
    # advance (no target -> the laser resets to 0). Verify the difference:
    # a fresh ghost with no enemies replays to charge 0 (reset), not 1.0.
    ghost5b = PredictedShip(hull=SILAS_HULL,
                            loadout=default_loadout(SILAS_HULL), local_index=1)
    ghost5b.seed(make_snap(0.9)[0][1])
    for i in range(30):
        ghost5b.record_input(i * TICK, ShipInput())
    ghost5b.reconcile_rewind(make_snap(0.9)[0][1], 0.0, 30 * TICK,
                             enemies=None)
    charge_b = min(w.charge for w in ghost5b.ship.weapons)
    assert charge_b < 0.95, \
        "without enemies the replayed charge must NOT advance (no target " \
        "-> reset): charge=%r" % (charge_b,)
    print("PASS: REWIND-CHARGE — the rewind replay feeds the enemy "
          "proxies, so the replayed laser charge matches the host's "
          "(with-enemies charge=%r, without=%r)" % (charge, charge_b))

    # --- 6. PIXEL: predicted_view renders the ghost's beam (beam-colored
    #     pixels appear along the beam line; the no-beam baseline has
    #     none). ---
    # dt=0.05 guarantees the ghost's fixed-step accumulator takes >=1 step
    # (0.016 < TICK=0.01667 would take none, so no beam would fire). The
    # ghost is seeded at charge 1.0, so it fires on the first step.
    # Baseline: charge 1.0 but laser_fire NOT pressed -> the ghost is
    # ready but does not fire -> no beam -> local_beams empty -> 0 pixels.
    push_stream(g, 1.0)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.05, pygame.key.get_pressed())
    assert not g.ghost.local_beams, \
        "baseline (no laser_fire) must fire no beam: local_beams=%r" \
        % (g.ghost.local_beams,)
    base = beam_pixels(g.screen, g.cam, g.ghost.local_beams)
    assert base == 0, "baseline must have no beam pixels: %d" % base
    # Fire: laser_fire pressed -> the ghost fires a beam this frame.
    push_stream(g, 1.0)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.05, _keys_with_laser_fire())
    assert len(g.ghost.local_beams) >= 1, \
        "predicted_view must fire the ghost's beam (laser_fire pressed, " \
        "charge 1.0, enemy in range): local_beams=%r" \
        % (g.ghost.local_beams,)
    hit = beam_pixels(g.screen, g.cam, g.ghost.local_beams)
    assert hit > 0, \
        "predicted_view must render the ghost's beam (beam-colored pixels " \
        "along the line): hit=%d" % hit
    print("PASS: PIXEL — predicted_view renders the ghost's beam (%d beam "
          "pixels vs %d baseline)" % (hit, base))

    print("ALL PASS: 10.3a (ghost emits laser beams)")


if __name__ == "__main__":
    main()