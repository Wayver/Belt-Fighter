"""Session 10.3b: the prediction ghost's laser beam stops at an asteroid.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3b_ghost_beam_rock

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

BUG FIXED
---------
In 2P, the CLIENT's own (ghost's) laser beam was drawn as a straight line
from the muzzle to the raw target position (`beam.end`). But the host's
sim runs `_resolve_beam` -> `_beam_blocked`, which clamps the beam's
visual endpoint (`vis_end`) to the nearest asteroid that intersects the
beam segment — the host's beam STOPS at the rock's surface. The client's
ghost path (`PredictedShip.step`) stored the raw `beam.end` and never ran
the occlusion test, so the client's beam visually PUNCHED THROUGH an
asteroid that the host's beam was blocked by.

THE FIX (client-side mirror of the host's occlusion — NO wire change)
---------------------------------------------------------------------
  * game.py: `_GhostRockProxy` (pos + collision_radius) +
    `Game._ghost_rock_proxies(pos)` build one proxy per buffer asteroid
    entry. The collision radius is the client's derived estimate
    (`self._rock_coll_r`, the (max|v|+min|v|)/2 of the wire verts — the
    same estimate the plan_18 hit-fx path uses).
  * netcode.py: `_beam_blocked_end(beam, rocks)` mirrors the host's
    `_beam_blocked` + `_beam_hit_asteroid` vis_end math (nearest-by-t
    segment-vs-circle test + surface point). `PredictedShip.step` clamps
    each fired beam's stored `end` through it (fed via `advance` /
    `reconcile_rewind` / `step`'s new `rocks` kwarg).
  * The call sites (game.py predicted_view + push_snapshot, __main__.py
    _advance_ghost) pass the rock proxies built from the buffer window.

The ghost is PRESENTATION-ONLY: it never feeds the sim, so this only
changes what the client draws. The host's `_resolve_beam` / `_beam_blocked`
/ the wire snapshot are UNCHANGED.

This test proves it (value-level, deterministic):
  1. PLUMBING   — _ghost_rock_proxies builds one proxy per buffer asteroid
                  entry (pos from the entry, radius from _rock_coll_r);
                  a rock with no registered radius is skipped.
  2. BLOCKED    — host fires a beam at a target with a rock in the way:
                  the host's vis_end is at the rock's surface. The ghost
                  (seeded from the same snapshot, fed the enemy + rock
                  proxies) fires a beam whose stored END matches the
                  host's vis_end EXACTLY (same radius) — the client beam
                  stops at the rock, not the target.
  3. DERIVED-R  — the ghost's end with the client's DERIVED radius (from
                  the wire verts, what the real client uses) is within
                  ~1 px of the host's vis_end (the radius estimate is
                  (max|v|+min|v|)/2, ~+/-3%).
  4. NO-ROCK    — no rock in the way: the ghost's beam end is the raw
                  target (unchanged).
  5. NOT-IN-WAY — a rock behind the muzzle (t<0) or beyond the target
                  (t>1) does not block: the beam reaches the target.
"""
import math
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .game import Game, _GhostRockProxy, _GhostEnemyProxy
from .netcode import PredictedShip, _beam_blocked_end
from .ship import Ship
from .asteroid import Asteroid
from .intent import ShipInput
from .hulls import SILAS_HULL, default_loadout
from .ai_enemy import AIEnemy

SEED = 1234
TICK = 1 / 60
ENEMY_IDX = 0
# The deterministic scene: the local ship (the ghost) at the origin facing
# +x, the target enemy at (100, 0), and a MEDIUM rock (radius 28) at
# (50, 0) — dead in the middle of the beam, so the host's beam is blocked
# at the rock's surface (~x=28) instead of reaching the target at x=100.
ENEMY_POS = pygame.Vector2(100.0, 0.0)
ROCK_POS = pygame.Vector2(50.0, 0.0)
ROCK_SIZE = "medium"
ROCK_RADIUS = 28.0   # ROCK_SIZES['medium']['radius']
EPS = 1e-6


def make_resources():
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    from .fog import make_light_texture
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def _scratch_game():
    """A minimal 2-player Game (headless) with the silas hull on BOTH
    players. Resets the id counters so the scene is deterministic."""
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    from .fog import make_light_texture
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL),
             seed=SEED, players=2)
    # Player 1 (the client's ship) is a bare placeholder by default (no
    # weapons) — give it a real silas ship so it can fire the laser.
    g.set_player_ship(1, Ship(hull=SILAS_HULL,
                              loadout=default_loadout(SILAS_HULL)))
    return g


def _setup_scene(g, rock_pos=ROCK_POS, rock_size=ROCK_SIZE,
                 enemy_pos=ENEMY_POS):
    """Deterministic scene: player 1 at the origin facing +x (charge 1.0),
    the target enemy at `enemy_pos`, and ONE rock at `rock_pos`. Returns
    (rock, host_beam, host_vis_end, rock_radius): the host fires a beam and
    resolves it, so host_vis_end is the host's authoritative visual
    endpoint (the rock's surface when blocked, the target when not)."""
    g.asteroids.clear()
    # A STATIC rock (vel 0, spin 0) with deterministic verts (fixed rng).
    import random as _random
    rock = Asteroid(rock_pos, rock_size, vel=(0, 0), rng=_random.Random(0))
    rock.vel = pygame.Vector2(0, 0)
    rock.spin = 0.0
    g.asteroids.append(rock)
    rock_radius = rock.collision_radius
    # Target enemy in range (the silas laser's 600 px, 360-degree arc).
    e = g.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(enemy_pos)
    e.ship.angle = 0.0
    # Local ship (player 1) at the origin facing +x, lasers fully charged.
    cs = g.players[1]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        w.charge = 1.0
    cs.laser_target = g._pick_laser_target(cs)
    assert cs.laser_target is e, "the host's laser target must be the " \
        "target enemy (in range): got %r" % (cs.laser_target,)
    shots, beams, _m = cs.update(TICK, ShipInput(laser_fire=True))
    assert beams, "the host must fire a beam (charge 1.0, laser_fire, " \
        "enemy in range)"
    host_beam = beams[0]
    # Resolve it the way the host's _step does: this clamps the visual
    # endpoint to the rock's surface (and removes/splits the rock).
    g._resolve_beam(host_beam, 1)
    assert g.beams, "the host must have a beam entry after resolve"
    host_vis_end = g.beams[0][3]
    return rock, host_beam, host_vis_end, rock_radius


def _make_ghost():
    return PredictedShip(hull=SILAS_HULL,
                         loadout=default_loadout(SILAS_HULL),
                         local_index=1)


def _seed_ghost_at_origin(ghost, charge=1.0):
    """Seed the ghost from a snapshot with the client ship at the origin
    facing +x, lasers at `charge` (mirrors the 10.3a make_snap). `seed`
    takes the client-ship snapshot (snap[0][local_index])."""
    tmp = _scratch_game()
    cs = tmp.players[1]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    snap = tmp.snapshot()
    cs_s = list(snap[0][1])
    cs_s[18] = tuple((0.0, charge, 0.0, 0) for _ in cs.weapons)
    ghost.seed(tuple(cs_s))


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()

    # --- 1. PLUMBING: _ghost_rock_proxies builds one proxy per buffer
    #     asteroid entry (pos from the entry, radius from _rock_coll_r);
    #     a rock with no registered radius is skipped. ---
    g = _scratch_game()
    # A fake buffer positions dict: two asteroids, one with a registered
    # radius (in _rock_coll_r) and one without (a registry miss).
    g._rock_coll_r = {10: 28.0}
    pos = {"asteroids": [(10, 50.0, 0.0, 0.0), (11, 200.0, 0.0, 0.0)]}
    proxies = g._ghost_rock_proxies(pos)
    assert len(proxies) == 1, \
        "one proxy per REGISTERED asteroid (the unregistered one is " \
        "skipped): got %d" % len(proxies)
    p = proxies[0]
    assert isinstance(p, _GhostRockProxy)
    assert abs(p.pos.x - 50.0) < 1e-9 and abs(p.pos.y - 0.0) < 1e-9, \
        "proxy pos from the buffer entry: %r" % (p.pos,)
    assert abs(p.collision_radius - 28.0) < 1e-9, \
        "proxy radius from _rock_coll_r: %r" % (p.collision_radius,)
    print("PASS: PLUMBING — _ghost_rock_proxies builds one proxy per "
          "registered buffer asteroid (pos + derived radius); skips a "
          "registry miss")

    # --- 2. BLOCKED: the host's beam is blocked at the rock's surface;
    #     the ghost's beam END matches the host's vis_end EXACTLY (same
    #     radius) — the client beam stops at the rock, not the target. ---
    g2 = _scratch_game()
    rock, host_beam, host_vis_end, rock_radius = _setup_scene(g2)
    # Sanity: the host's beam is actually blocked (vis_end at the rock's
    # surface, NOT the raw target). The raw target is at (100, 0); the
    # rock's surface is rock_radius px from the rock center (50, 0).
    raw_target = host_beam.end
    assert abs(host_vis_end.distance_to(ROCK_POS) - rock_radius) < 1e-3, \
        "host vis_end must be on the rock surface: %r (rock %r, r=%r)" \
        % (host_vis_end, ROCK_POS, rock_radius)
    assert host_vis_end.distance_to(raw_target) > 20, \
        "host beam must be BLOCKED (vis_end far from raw target): %r vs %r" \
        % (host_vis_end, raw_target)
    # The ghost, seeded from the same pose, fires a beam and clamps its
    # END through _beam_blocked_end with the SAME radius the host used.
    ghost = _make_ghost()
    _seed_ghost_at_origin(ghost, charge=1.0)
    enemy_proxy = _GhostEnemyProxy(pygame.Vector2(ENEMY_POS),
                                   pygame.Vector2(0, 0), 0.0,
                                   1, 0.0, 0.0)
    rock_proxy = _GhostRockProxy(pygame.Vector2(ROCK_POS), rock_radius)
    ghost.step(TICK, ShipInput(laser_fire=True),
               enemies=[enemy_proxy], rocks=[rock_proxy])
    assert ghost.local_beams, \
        "the ghost must fire a beam (laser_fire, charge 1.0, enemy in " \
        "range): local_beams=%r" % (ghost.local_beams,)
    ghost_end = ghost.local_beams[0][1]
    assert ghost_end.distance_to(host_vis_end) < EPS, \
        "the ghost's beam END must match the host's vis_end EXACTLY (same " \
        "radius) — the client beam stops at the rock: ghost=%r host=%r " \
        "delta=%r" % (ghost_end, host_vis_end,
                      ghost_end.distance_to(host_vis_end))
    # And it must NOT be the raw target (the pre-fix behavior).
    assert ghost_end.distance_to(raw_target) > 20, \
        "the ghost's beam must be BLOCKED (not the raw target (100,0)): " \
        "ghost_end=%r target=%r" % (ghost_end, raw_target)
    print("PASS: BLOCKED — the ghost's beam END matches the host's "
          "vis_end exactly (ghost=%r host=%r); the client beam stops at "
          "the rock's surface, not the target (raw target %r)"
          % (tuple(round(v, 2) for v in ghost_end),
             tuple(round(v, 2) for v in host_vis_end),
             tuple(round(v, 2) for v in raw_target)))

    # --- 3. DERIVED-R: the ghost's end with the client's DERIVED radius
    #     (from the wire verts, what the real client uses) is within ~1 px
    #     of the host's vis_end. ---
    rr = [math.hypot(v.x, v.y) for v in rock.verts]
    derived = (max(rr) + min(rr)) / 2.0
    ghost3 = _make_ghost()
    _seed_ghost_at_origin(ghost3, charge=1.0)
    rock_proxy3 = _GhostRockProxy(pygame.Vector2(ROCK_POS), derived)
    ghost3.step(TICK, ShipInput(laser_fire=True),
                enemies=[enemy_proxy], rocks=[rock_proxy3])
    assert ghost3.local_beams, "the ghost must fire a beam"
    ghost3_end = ghost3.local_beams[0][1]
    d3 = ghost3_end.distance_to(host_vis_end)
    assert d3 < 1.5, \
        "the ghost's beam END with the DERIVED radius must be within ~1 px " \
        "of the host's vis_end (the radius estimate is (max|v|+min|v|)/2, " \
        "~+/-3%): ghost=%r host=%r delta=%r (derived r=%r vs exact %r)" \
        % (ghost3_end, host_vis_end, d3, derived, rock_radius)
    print("PASS: DERIVED-R — the ghost's beam END with the client's "
          "derived radius (from the wire verts) is within %r px of the "
          "host's vis_end (derived r=%r vs exact %r)"
          % (round(d3, 3), round(derived, 3), rock_radius))

    # --- 4. NO-ROCK: no rock in the way -> the ghost's beam end is the
    #     raw target (unchanged). ---
    ghost4 = _make_ghost()
    _seed_ghost_at_origin(ghost4, charge=1.0)
    ghost4.step(TICK, ShipInput(laser_fire=True),
                enemies=[enemy_proxy], rocks=[])
    assert ghost4.local_beams, "the ghost must fire a beam"
    ghost4_end = ghost4.local_beams[0][1]
    assert ghost4_end.distance_to(ENEMY_POS) < EPS, \
        "with no rock the ghost's beam END must be the raw target: " \
        "ghost=%r target=%r" % (ghost4_end, ENEMY_POS)
    # rocks=None (the pre-fix / no-window path) behaves the same.
    ghost4b = _make_ghost()
    _seed_ghost_at_origin(ghost4b, charge=1.0)
    ghost4b.step(TICK, ShipInput(laser_fire=True),
                 enemies=[enemy_proxy], rocks=None)
    assert ghost4b.local_beams[0][1].distance_to(ENEMY_POS) < EPS, \
        "rocks=None must leave the beam END at the raw target"
    print("PASS: NO-ROCK — with no rock (or rocks=None) the ghost's beam "
          "END is the raw target (unchanged)")

    # --- 5. NOT-IN-WAY: a rock behind the muzzle (t<0) or beyond the
    #     target (t>1) does not block — the beam reaches the target. ---
    # (a) Rock BEHIND the muzzle: the beam goes +x from the origin; a rock
    #     at (-50, 0) is behind the muzzle (t<0, clamped out).
    ghost5a = _make_ghost()
    _seed_ghost_at_origin(ghost5a, charge=1.0)
    behind = _GhostRockProxy(pygame.Vector2(-50.0, 0.0), ROCK_RADIUS)
    ghost5a.step(TICK, ShipInput(laser_fire=True),
                 enemies=[enemy_proxy], rocks=[behind])
    assert ghost5a.local_beams[0][1].distance_to(ENEMY_POS) < EPS, \
        "a rock BEHIND the muzzle must not block (t<0 clamped): ghost=%r " \
        "target=%r" % (ghost5a.local_beams[0][1], ENEMY_POS)
    # (b) Rock BEYOND the target: a rock at (200, 0) is past the target
    #     (t>1, clamped out) — the beam stops at the target, not the rock.
    ghost5b = _make_ghost()
    _seed_ghost_at_origin(ghost5b, charge=1.0)
    beyond = _GhostRockProxy(pygame.Vector2(200.0, 0.0), ROCK_RADIUS)
    ghost5b.step(TICK, ShipInput(laser_fire=True),
                 enemies=[enemy_proxy], rocks=[beyond])
    assert ghost5b.local_beams[0][1].distance_to(ENEMY_POS) < EPS, \
        "a rock BEYOND the target must not block (t>1 clamped): ghost=%r " \
        "target=%r" % (ghost5b.local_beams[0][1], ENEMY_POS)
    # (c) Rock OFF the beam line: a rock at (50, 200) is 200 px off the
    #     beam's y=0 line — far outside the radius, so no block.
    ghost5c = _make_ghost()
    _seed_ghost_at_origin(ghost5c, charge=1.0)
    offline = _GhostRockProxy(pygame.Vector2(50.0, 200.0), ROCK_RADIUS)
    ghost5c.step(TICK, ShipInput(laser_fire=True),
                 enemies=[enemy_proxy], rocks=[offline])
    assert ghost5c.local_beams[0][1].distance_to(ENEMY_POS) < EPS, \
        "a rock OFF the beam line must not block: ghost=%r target=%r" \
        % (ghost5c.local_beams[0][1], ENEMY_POS)
    print("PASS: NOT-IN-WAY — a rock behind the muzzle (t<0), beyond the "
          "target (t>1), or off the beam line does not block")

    print("ALL PASS: 10.3b (the ghost's laser beam stops at an asteroid, "
          "matching the host)")


if __name__ == "__main__":
    main()