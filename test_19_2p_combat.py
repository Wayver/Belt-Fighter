"""Session 19 (plan_19): 2P player-vs-player combat — team-based targeting,
host-side damage, client-side lock-on, and client-side hit FX.

plan_19 introduces a `team` field on the ship + a single `can_target`
predicate so the two human players can actually fight each other (today the
game is hardcoded to "players shoot AI enemies"). This test proves the full
feature end-to-end at the value level:

  TEAM          — `can_target` (off-team alive -> True; same team -> False;
                  dead -> False; enemy neutral -> True) and `Ship.lead_point`
                  (a converging intercept point; None when the target
                  outruns the bullet).
  HOST TARGET   — 2P, both ships in range -> each picks the OTHER player as
                  laser + missile target; a dead target is skipped;
                  single-player -> enemies only (bit-identical).
  HOST DAMAGE   — a player bullet / missile / beam each reduce the remote
                  ship's shield then hull; a kill -> `dead` + the other
                  player alive; a shooter's own bullet cannot self-hit.
  GHOST LOCK-ON — the prediction ghost picks the remote player as laser /
                  missile target (the `_GhostPlayerProxy`), and a ghost
                  missile HOMES on the proxy (closer than a coasting one).
  CLIENT HIT FX — the other player's bullet / missile / beam each trigger the
                  local shield flash + burst on the client (the plan_19
                  generalization of `_local_incoming_hit_fx` + the new
                  `_remote_beam_hit_fx`).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_19_2p_combat

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import random

import pygame

from .config import (WIDTH, HEIGHT, BULLET_SPEED, MISSILE_SPEED,
                     TARGETING_MAX_LEAD, TEAM_NEUTRAL)
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, MISSILE, REACTOR, COMPUTER, SHIELD, SENSOR,
                    MAIN_ENGINE, NOSE_THRUSTER, RCS,
                    REACTOR_TYPE, COMPUTER_TYPE, SHIELD_TYPE,
                    PASSIVE_SENSOR,
                    LASER_TYPE, MISSILE_TYPE)
from .ship import Ship
from .bullets import Bullet, Missile
from .ai_enemy import AIEnemy
from .game import (Game, can_target, _GhostPlayerProxy, _GhostEnemyProxy)
from .netcode import (PredictedShip, GhostMissile, HostTimeEstimator,
                      LatencyTracker, RenderPoint, BEAM_TTL)
from .fog import make_light_texture

SEED = 1234
TICK = 1 / 60
# The two ships, both facing +x (angle 0), 200 px apart (well inside the
# laser range 600 and the missile lock range 900).
HOST_POS = (WIDTH / 2 - 100.0, HEIGHT / 2)
CLIENT_POS = (WIDTH / 2 + 100.0, HEIGHT / 2)
# shield_burst spawns n=16 particles (particles.py).
SHIELD_BURST_N = 16
# The local ship's oval front point (angle 0, oval a=30, b=22, center (0,0)):
# pos + (a, 0). A bullet/beam in front projects here.
IMPACT_FRONT = (CLIENT_POS[0] + 30.0, CLIENT_POS[1])


# --- Test hull: gun + laser + missile ------------------------------------
# The stock DEFAULT_HULL's `gun` slot is a pulse GUN (no laser / missile), so
# the pickers would return None. A custom loadout puts a LASER in the `gun`
# slot and a MISSILE in the `missile` slot, so the host + ghost pickers have
# a laser (range 600) and a missile (lock range 900) to pick against.
TEST_HULL = HullType(
    id='test_2p',
    polygon=((18, 0), (14, 3.5), (8, 6.5), (0, 8), (-8, 8), (-12, 11),
             (-12, 5), (-9, 3), (-9, -3), (-12, -5), (-12, -11), (-8, -8),
             (0, -8), (8, -6.5), (14, -3.5)),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN,
           MISSILE, REACTOR, COMPUTER, SHIELD, SENSOR),
    base_mass=1.0,
    collision_radius=12.0,
    nose=(18, 0),
    cockpit=(8, 0),
)
# slot_name -> ComponentType (the fitted parts). The `gun` slot carries a
# LASER (range 600) and the `missile` slot a MISSILE (lock range 900), so the
# host + ghost pickers have a laser + missile to pick against.
TEST_LOADOUT = {
    'forward_s': MAIN_ENGINE,
    'forward_p': MAIN_ENGINE,
    'reverse': NOSE_THRUSTER,
    'to_left': RCS,
    'to_right': RCS,
    'gun': LASER_TYPE,
    'missile': MISSILE_TYPE,
    'reactor': REACTOR_TYPE,
    'computer': COMPUTER_TYPE,
    'shield': SHIELD_TYPE,
    'sensor': PASSIVE_SENSOR,
}


def _res():
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    return font, big_font, light_tex, fog_surf, light_surf


def make_host(players=2):
    """A scratch host Game (local_index=0). Both player ships are TEST_HULL
    (gun slot = laser, missile slot = missile) so the pickers have a laser +
    missile to pick against. Returns the Game."""
    font, big_font, lt, fog, light = _res()
    g = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, lt, fog, light,
             hull=TEST_HULL, loadout=TEST_LOADOUT, seed=SEED,
             players=players, local_index=0)
    if players == 2:
        g.set_player_ship(1, Ship(hull=TEST_HULL, loadout=TEST_LOADOUT))
    g.enemies.clear()
    g.bullets.clear()
    g.enemy_bullets.clear()
    g.missiles.clear()
    g.asteroids.clear()
    g.beams.clear()
    g._beam_events.clear()
    return g


def make_client():
    """A scratch client Game (local_index=1) wired for predicted_view (ghost
    + render point + host-time estimator + latency tracker). Both players are
    TEST_HULL (the local ship's ghost carries a laser + missile)."""
    font, big_font, lt, fog, light = _res()
    g = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, lt, fog, light,
             hull=TEST_HULL, loadout=TEST_LOADOUT, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=TEST_HULL, loadout=TEST_LOADOUT))
    g.ghost = PredictedShip(hull=TEST_HULL, loadout=TEST_LOADOUT,
                            local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def place_host(g, enemy_pos=(WIDTH / 2, 60.0)):
    """Park the two player ships at HOST_POS / CLIENT_POS facing +x and add
    ONE AI enemy at `enemy_pos` (a neutral candidate). Returns (host, other,
    enemy)."""
    h, o = g.players[0], g.players[1]
    h.pos = pygame.Vector2(*HOST_POS)
    h.vel = pygame.Vector2(0, 0)
    h.angle = 0.0
    o.pos = pygame.Vector2(*CLIENT_POS)
    o.vel = pygame.Vector2(0, 0)
    o.angle = 0.0
    e = AIEnemy(pygame.Vector2(*enemy_pos), rng=random.Random(7))
    g.enemies = [e]
    return h, o, e


# =========================================================================
# 1. TEAM — can_target + Ship.lead_point
# =========================================================================
def test_team():
    # can_target: off-team alive -> True.
    a = Ship(hull=TEST_HULL, loadout=TEST_LOADOUT)
    a.team = 0
    b = Ship(hull=TEST_HULL, loadout=TEST_LOADOUT)
    b.team = 1
    assert can_target(a, b) is True, "off-team alive must be targetable"
    # same team -> False.
    c = Ship(hull=TEST_HULL, loadout=TEST_LOADOUT)
    c.team = 0
    assert can_target(a, c) is False, "same team must NOT be targetable"
    # dead -> False (a frozen corpse is not a target).
    b.dead = True
    assert can_target(a, b) is False, "a dead ship must NOT be targetable"
    b.dead = False
    # enemy neutral -> True (no .dead -> getattr False; TEAM_NEUTRAL off-team).
    e = AIEnemy(pygame.Vector2(0, 0), rng=random.Random(1))
    assert e.team == TEAM_NEUTRAL, "an AI enemy must be TEAM_NEUTRAL"
    assert can_target(a, e) is True, "a neutral enemy must be targetable"
    print("PASS: TEAM — can_target (off-team alive True; same team False; "
          "dead False; enemy neutral True)")

    # Ship.lead_point: a converging intercept point.
    s = Ship(hull=TEST_HULL, loadout=TEST_LOADOUT)
    s.pos = pygame.Vector2(100.0, 0.0)
    s.vel = pygame.Vector2(100.0, 0.0)   # moving +x, toward the shooter
    shooter = pygame.Vector2(0.0, 0.0)
    lp = s.lead_point(shooter, BULLET_SPEED)
    assert lp is not None, "a converging target must have a lead point"
    # The intercept is AHEAD of the current pos (the ship keeps moving +x
    # while the bullet flies) but behind the naive d0/speed overshoot.
    assert lp.x > s.pos.x, \
        "the lead point must be ahead of the current pos (got %r)" % (lp,)
    # The intercept distance ~= bullet_speed * T (the bullet's flight).
    d = (lp - shooter).length()
    T = d / BULLET_SPEED
    # At the intercept the ship is at pos + vel*T; it must equal lp (within
    # the 1e-3 convergence tolerance).
    err = (lp - (s.pos + s.vel * T)).length()
    assert err < 2.0, "the lead point must equal pos + vel*T (err %.2f)" % err
    # None when the target outruns the bullet (moving away faster than the
    # bullet -> T grows past TARGETING_MAX_LEAD).
    s2 = Ship(hull=TEST_HULL, loadout=TEST_LOADOUT)
    s2.pos = pygame.Vector2(100.0, 0.0)
    s2.vel = pygame.Vector2(BULLET_SPEED * 1.5, 0.0)   # faster than the bullet
    assert s2.lead_point(shooter, BULLET_SPEED) is None, \
        "a target outrunning the bullet must have no lead point"
    print("PASS: TEAM — Ship.lead_point (converging intercept; None when "
          "the target outruns the bullet)")


# =========================================================================
# 2. HOST TARGETING — each player picks the other; single-player unchanged
# =========================================================================
def test_host_targeting():
    g = make_host(players=2)
    h, o, e = place_host(g)
    # Each ship's laser + missile target is the OTHER player (200 px, closer
    # than the enemy at 400 px).
    assert g._pick_laser_target(h) is o, \
        "the host's laser must pick the other player (got %r)" % (
            g._pick_laser_target(h),)
    assert g._pick_laser_target(o) is h, \
        "the other player's laser must pick the host (got %r)" % (
            g._pick_laser_target(o),)
    assert g._pick_missile_target(h) is o, \
        "the host's missile must lock the other player (got %r)" % (
            g._pick_missile_target(h),)
    assert g._pick_missile_target(o) is h, \
        "the other player's missile must lock the host (got %r)" % (
            g._pick_missile_target(o),)
    # A dead target is skipped: the laser falls back to the enemy.
    o.dead = True
    assert g._pick_laser_target(h) is e, \
        "a dead other player must be skipped (fall back to the enemy)"
    o.dead = False
    print("PASS: HOST TARGET — both players pick each other (laser + "
          "missile); a dead target is skipped")

    # Single-player: the other player does not exist -> enemies only
    # (bit-identical to the old "loop over self.enemies").
    g1 = make_host(players=1)
    h1 = g1.players[0]
    h1.pos = pygame.Vector2(*HOST_POS)
    e1 = AIEnemy(pygame.Vector2(WIDTH / 2, 60.0), rng=random.Random(7))
    g1.enemies = [e1]
    assert g1._pick_laser_target(h1) is e1, \
        "single-player laser must pick the enemy (got %r)" % (
            g1._pick_laser_target(h1),)
    assert g1._pick_missile_target(h1) is e1, \
        "single-player missile must lock the enemy (got %r)" % (
            g1._pick_missile_target(h1),)
    print("PASS: HOST TARGET — single-player picks enemies only "
          "(bit-identical)")


# =========================================================================
# 3. HOST DAMAGE — bullet / missile / beam reduce shield then hull; kill
# =========================================================================
def test_host_damage():
    # --- BULLET: a player bullet (owner 0) hits the other player's shield.
    g = make_host(players=2)
    h, o, _ = place_host(g)
    assert o.shield_on and o.shield_charge == 3.0
    # A bullet inside the other player's shield oval (in front, +x).
    b = Bullet(pygame.Vector2(CLIENT_POS[0] + 20.0, CLIENT_POS[1]),
               pygame.Vector2(-BULLET_SPEED, 0.0), owner=0)
    b.id = (0, 0)
    g.bullets = [b]
    g._collisions()
    assert b not in g.bullets, "a hitting bullet must be culled"
    assert o.shield_charge == 2.0, \
        "a bullet hit must drop the shield charge by 1 (got %.1f)" % (
            o.shield_charge,)
    assert not o.dead, "a shield hit must not kill"
    print("PASS: HOST DAMAGE — a player bullet reduces the remote shield")

    # --- MISSILE: a player missile (owner 0, dmg 2) drops the shield by 2.
    g = make_host(players=2)
    h, o, _ = place_host(g)
    m = Missile(pygame.Vector2(CLIENT_POS[0] + 20.0, CLIENT_POS[1]),
                pygame.Vector2(-MISSILE_SPEED, 0.0), owner=0)
    m.id = (0, 0)
    g.missiles = [m]
    g._collisions()
    assert m not in g.missiles, "a hitting missile must be culled"
    assert o.shield_charge == 1.0, \
        "a missile hit (dmg 2) must drop the shield charge by 2 (got %.1f)" % (
            o.shield_charge,)
    assert not o.dead
    print("PASS: HOST DAMAGE — a player missile reduces the remote shield "
          "(dmg loop)")

    # --- KILL: shield down + a bullet sweeping the hull -> dead, the other
    # player (the shooter) stays alive.
    g = make_host(players=2)
    h, o, _ = place_host(g)
    o.shield_charge = 0.0   # shield down -> the swept-hull test applies
    b = Bullet(pygame.Vector2(CLIENT_POS[0] + 40.0, CLIENT_POS[1]),
               pygame.Vector2(-BULLET_SPEED, 0.0), owner=0)
    b.id = (0, 0)
    b.prev_pos = pygame.Vector2(CLIENT_POS[0] + 40.0, CLIENT_POS[1])
    b.pos = pygame.Vector2(CLIENT_POS[0] + 10.0, CLIENT_POS[1])   # crosses nose
    g.bullets = [b]
    g._collisions()
    assert b not in g.bullets, "a hull-killing bullet must be culled"
    assert o.dead, "a hull hit with the shield down must kill"
    assert not h.dead, "the shooter must stay alive"
    print("PASS: HOST DAMAGE — a hull hit (shield down) kills; the shooter "
          "stays alive")

    # --- SELF-HIT SAFETY: a shooter's own bullet (owner 0) is routed to the
    # OTHER player, so it can never self-hit.
    g = make_host(players=2)
    h, o, _ = place_host(g)
    # A bullet inside the SHOOTER's own shield oval (in front of the host).
    b = Bullet(pygame.Vector2(HOST_POS[0] + 20.0, HOST_POS[1]),
               pygame.Vector2(BULLET_SPEED, 0.0), owner=0)
    b.id = (0, 0)
    g.bullets = [b]
    g._collisions()
    assert b in g.bullets, \
        "a shooter's own bullet must NOT self-hit (it is routed to the " \
        "other player, which it is not near)"
    assert h.shield_charge == 3.0, "the shooter's shield must be untouched"
    print("PASS: HOST DAMAGE — a shooter's own bullet cannot self-hit")

    # --- BEAM: a beam that hits the other player reduces the shield + the
    # T_BEAM event carries target "p<other>".
    g = make_host(players=2)
    h, o, _ = place_host(g)
    from .bullets import Beam
    # A beam from the host's muzzle, ending just in front of the other
    # player (within collision_radius + 4 of their pos).
    beam = Beam(start=pygame.Vector2(HOST_POS[0] + 18.0, HOST_POS[1]),
                end=pygame.Vector2(CLIENT_POS[0] - 10.0, CLIENT_POS[1]),
                owner=0, damage=1,
                local_start=(18.0, 0.0))
    g._resolve_beam(beam, owner=0)
    assert o.shield_charge == 2.0, \
        "a beam hit must drop the shield charge by 1 (got %.1f)" % (
            o.shield_charge,)
    assert not o.dead
    # The T_BEAM event carries target "p1" (the other player's index).
    assert len(g._beam_events) == 1, \
        "a beam hit must emit a T_BEAM event (got %d)" % len(g._beam_events)
    ev = g._beam_events[0]
    assert ev[-1] == "p1", \
        "the T_BEAM event must carry target 'p1' (got %r)" % (ev[-1],)
    # The beam entry's last field is the target id too.
    assert g.beams and g.beams[0][-1] == "p1", \
        "the beam entry must carry target 'p1' (got %r)" % (
            g.beams[0][-1] if g.beams else None,)
    print("PASS: HOST DAMAGE — a beam reduces the remote shield + the "
          "T_BEAM event carries target 'p1'")


# =========================================================================
# 4. GHOST LOCK-ON — the ghost picks the remote player; a ghost missile
#    homes on the _GhostPlayerProxy
# =========================================================================
def test_ghost_lock_on():
    ps = PredictedShip(hull=TEST_HULL, loadout=TEST_LOADOUT, local_index=1)
    s = ps._ship
    s.pos = pygame.Vector2(*CLIENT_POS)
    s.vel = pygame.Vector2(0, 0)
    s.angle = 0.0
    # The remote player (index 0) is 200 px away (in laser + missile range);
    # an enemy is 400 px away (also in range, but farther).
    proxy = _GhostPlayerProxy(pygame.Vector2(*HOST_POS),
                              pygame.Vector2(0, 0), 0.0, 0, False,
                              (30.0, 22.0, 0.0, 0.0))
    enemy = _GhostEnemyProxy(pygame.Vector2(WIDTH / 2, 60.0),
                             pygame.Vector2(0, 0), 0.0, 1, 0.0, 0.0)
    # The ghost's laser + missile target is the remote player (nearest).
    assert ps._pick_laser_target(s, [enemy], proxy) is proxy, \
        "the ghost's laser must pick the remote player (got %r)" % (
            ps._pick_laser_target(s, [enemy], proxy),)
    assert ps._pick_missile_target(s, [enemy], proxy) is proxy, \
        "the ghost's missile must lock the remote player (got %r)" % (
            ps._pick_missile_target(s, [enemy], proxy),)
    # A dead remote player is skipped (fall back to the enemy).
    proxy.dead = True
    assert ps._pick_laser_target(s, [enemy], proxy) is enemy, \
        "a dead remote player must be skipped (fall back to the enemy)"
    proxy.dead = False
    # No proxy (single-player / no window) -> the enemy only.
    assert ps._pick_laser_target(s, [enemy], None) is enemy, \
        "with no proxy the ghost must pick the enemy"
    print("PASS: GHOST LOCK-ON — the ghost picks the remote player (laser + "
          "missile); a dead proxy is skipped")

    # A ghost missile HOMES on the proxy (closer to the target than a coasting
    # one). The proxy MOVES (vel (0, 120) — drifting down) and is stepped each
    # tick, so the homing missile must turn to track it while a coasting
    # missile (no target) keeps its straight launch heading and misses. Both
    # missiles skip the boost phase (boost = 0) so they are in the seek phase
    # from the first tick (the 0.62 s boost is straight flight — no turning).
    proxy.pos = pygame.Vector2(*HOST_POS)
    proxy.vel = pygame.Vector2(0.0, 120.0)
    start = pygame.Vector2(CLIENT_POS[0] + 20.0, CLIENT_POS[1])
    vel = pygame.Vector2(-MISSILE_SPEED, 0.0)   # launched toward the proxy
    home = GhostMissile(start.copy(), vel.copy(), (1, 0), target=proxy)
    coast = GhostMissile(start.copy(), vel.copy(), (1, 1), target=None)
    home.boost = 0.0
    coast.boost = 0.0
    for _ in range(30):
        proxy.pos += proxy.vel * TICK   # the target actually moves
        home.update(TICK, proxy)        # re-resolved to the proxy each step
        coast.update(TICK, None)        # no target -> coasts straight
    d_home = (home.pos - proxy.pos).length()
    d_coast = (coast.pos - proxy.pos).length()
    assert d_home < d_coast, \
        "a homing ghost missile must be closer to the (moving) target than " \
        "a coasting one (home %.1f vs coast %.1f)" % (d_home, d_coast)
    # The homing missile must have turned toward the (descending) target —
    # its heading is no longer purely -x (it steers toward the lead point,
    # which is below the launch line, so vel.y goes positive).
    assert home.vel.y > 1.0, \
        "the homing missile must steer down toward the target (vel.y %.1f)" % (
            home.vel.y,)
    print("PASS: GHOST LOCK-ON — a ghost missile homes on the "
          "_GhostPlayerProxy (closer than a coasting one)")


# =========================================================================
# 5. CLIENT HIT FX — the other player's bullet / missile / beam trigger the
#    local shield flash + burst
# =========================================================================
def _client_snap(g, dump, t, pbullet=True):
    """A 2-player snapshot with the LOCAL ship (index 1) at CLIENT_POS facing
    +x and `shield_dump = dump`, the other player (index 0) parked far away,
    and (optionally) ONE player bullet (owner 0 = the remote player) moving
    -x toward the local ship, just in front of it. No AI enemies."""
    tmp = make_host(players=2)
    tmp.players[0].pos = pygame.Vector2(100.0, 100.0)
    tmp.players[1].pos = pygame.Vector2(*CLIENT_POS)
    tmp.players[1].angle = 0.0
    tmp.players[1].shield_dump = dump
    tmp.enemies.clear()
    tmp.bullets.clear()
    tmp.enemy_bullets.clear()
    tmp.missiles.clear()
    if pbullet:
        # Muzzle so the bullet is just in front of the ship at t (moving -x).
        x0 = CLIENT_POS[0] + 10.0 + BULLET_SPEED * t
        tmp.bullets.append(
            Bullet(pygame.Vector2(x0, CLIENT_POS[1]),
                   pygame.Vector2(-BULLET_SPEED, 0.0), owner=0))
    return tmp.snapshot()


def _push_and_render(g, dump, t, pbullet=True):
    g.push_snapshot(t, _client_snap(g, dump, t, pbullet), now=t)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def _reset_client(g):
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=TEST_HULL, loadout=TEST_LOADOUT,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    g._last_ghost_shield_dump = 0.0
    g._hit_fx = []
    g._beam_fx_fired = set()
    g.remote_beams = []
    g._sfx_events = []


def test_client_hit_fx():
    g = make_client()
    power_hit = g.ghost.ship.shield_comp.power_hit

    # --- BULLET: the other player's bullet (owner 0) + a dump spike -> a
    # shield_burst at the oval front (the projected impact point).
    _reset_client(g)
    _push_and_render(g, 0.0, 0.0)
    _push_and_render(g, 0.0, 0.1)
    assert len(g._hit_fx) == 0, "no burst before the dump spikes"
    _push_and_render(g, power_hit, 0.2)   # SPIKE -> burst
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a dump spike with the other player's bullet in front must spawn a " \
        "shield_burst of %d (got %d)" % (SHIELD_BURST_N, len(g._hit_fx))
    bx = sum(p.pos.x for p in g._hit_fx) / len(g._hit_fx)
    by = sum(p.pos.y for p in g._hit_fx) / len(g._hit_fx)
    err = ((bx - IMPACT_FRONT[0]) ** 2 + (by - IMPACT_FRONT[1]) ** 2) ** 0.5
    assert err < 8.0, \
        "the burst must land on the oval front %s (got (%.1f, %.1f), err " \
        "%.1f)" % (IMPACT_FRONT, bx, by, err)
    # A re-anchored (sustained) dump must NOT re-fire.
    _push_and_render(g, power_hit, 0.3)
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a re-anchored dump must NOT spawn a second burst (got %d)" % (
            len(g._hit_fx),)
    print("PASS: CLIENT HIT FX — the other player's bullet + a dump spike "
          "spawn a shield_burst at the oval impact point")

    # --- MISSILE: a dump spike with NO bullet in the buffer (a missile has
    # no buffer bullet to project from) -> the oval-front fallback.
    _reset_client(g)
    _push_and_render(g, 0.0, 0.0, pbullet=False)
    _push_and_render(g, 0.0, 0.1, pbullet=False)
    _push_and_render(g, power_hit, 0.2, pbullet=False)   # SPIKE, no bullet
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a dump spike with no bullet (a missile hit) must spawn a " \
        "shield_burst (got %d)" % len(g._hit_fx)
    bx = sum(p.pos.x for p in g._hit_fx) / len(g._hit_fx)
    by = sum(p.pos.y for p in g._hit_fx) / len(g._hit_fx)
    err = ((bx - IMPACT_FRONT[0]) ** 2 + (by - IMPACT_FRONT[1]) ** 2) ** 0.5
    assert err < 8.0, \
        "the missile-hit burst must fall back to the oval front %s (got " \
        "(%.1f, %.1f), err %.1f)" % (IMPACT_FRONT, bx, by, err)
    print("PASS: CLIENT HIT FX — a missile hit (dump spike, no bullet) "
          "falls back to the oval front")

    # --- BEAM: a T_BEAM whose target is the LOCAL player fires the local
    # shield_burst at the beam endpoint + the throttled shield_hit SFX, ONCE
    # (a beam stays in remote_beams for BEAM_TTL, so it must not re-fire).
    _reset_client(g)
    end = pygame.Vector2(*IMPACT_FRONT)
    g.remote_beams.append([(18.0, 0.0), end, 0.0, BEAM_TTL, 0, "p1"])
    g._remote_beam_hit_fx()
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a beam targeting the local player must spawn a shield_burst (got " \
        "%d)" % len(g._hit_fx)
    bx = sum(p.pos.x for p in g._hit_fx) / len(g._hit_fx)
    by = sum(p.pos.y for p in g._hit_fx) / len(g._hit_fx)
    err = ((bx - end.x) ** 2 + (by - end.y) ** 2) ** 0.5
    assert err < 2.0, \
        "the beam burst must be at the beam endpoint %s (got (%.1f, %.1f), " \
        "err %.1f)" % (end, bx, by, err)
    sfx = [e for e in g._sfx_events if e[0] == "shield_hit"]
    assert len(sfx) == 1, \
        "a beam hit must fire the shield_hit SFX once (got %d)" % len(sfx)
    # Re-running the FX (the beam is still in remote_beams) must NOT re-fire.
    g._remote_beam_hit_fx()
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a beam must fire its FX exactly once (got %d)" % len(g._hit_fx)
    # A beam targeting the REMOTE player (or a rock / enemy) must NOT fire.
    _reset_client(g)
    g.remote_beams.append([(18.0, 0.0), end, 0.0, BEAM_TTL, 0, "p0"])
    g._remote_beam_hit_fx()
    assert len(g._hit_fx) == 0, \
        "a beam targeting the remote player must NOT fire the local FX"
    g.remote_beams.append([(18.0, 0.0), end, 0.0, BEAM_TTL, 0, None])
    g._remote_beam_hit_fx()
    assert len(g._hit_fx) == 0, \
        "a beam targeting a rock (None) must NOT fire the local FX"
    # A culled beam's id is pruned (a later beam reusing the address is not
    # suppressed).
    _reset_client(g)
    g.remote_beams.append([(18.0, 0.0), end, 0.0, BEAM_TTL, 0, "p1"])
    g._remote_beam_hit_fx()
    assert len(g._hit_fx) == SHIELD_BURST_N
    g._step_remote_beams(BEAM_TTL + 0.01)   # cull the beam
    assert not g.remote_beams, "the beam must be culled after its ttl"
    # The next FX pass (predicted_view calls it every frame) prunes the
    # culled beam's id from the fired set.
    g._remote_beam_hit_fx()
    assert not g._beam_fx_fired, \
        "a culled beam's id must be pruned from the fired set"
    print("PASS: CLIENT HIT FX — a beam targeting the local player fires "
          "the shield_burst + SFX once; remote/rock beams do not; a culled "
          "beam's id is pruned")


def main():
    pygame.init()
    test_team()
    test_host_targeting()
    test_host_damage()
    test_ghost_lock_on()
    test_client_hit_fx()
    print("ALL PASS: 19 (2P player-vs-player combat — team targeting, host "
          "damage, ghost lock-on, client hit FX)")


if __name__ == "__main__":
    main()