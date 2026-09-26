"""Game state: entities, collisions, and per-frame update/draw.

Networked-ready:
- update() runs the whole sim on a fixed timestep (STEP) via an
  accumulator, so every entity steps on the same deterministic tick.
- Player intent (movement + fire) arrives as a ShipInput, not raw
  key state — the same type a remote player's input will be.
"""
import math
import random
from dataclasses import replace

import pygame

from .config import (WIDTH, HEIGHT, SPAWN_PROTECT, MAX_BULLETS,
                    BULLET_SPEED,
                    ROCK_SPLIT, ROCK_SIZES, BG, STAR_COLOR,
                    BULLET_COLOR, ENEMY_BULLET_COLOR,
                    TARGETING_ASSIST, TARGETING_COLOR, TARGETING_HORIZON,
                    TARGETING_STEPS, TARGETING_RANGE,
                    TARGETING_USE_ACCEL, TARGETING_MAX_LEAD,
                    TARGETING_COLOR_GREEN, TARGETING_ALIGN_TOL, LASER_COLOR,
                    BEAM_IMPACT_SPREAD, SENSOR_COLOR, SENSOR_SCAN_COLOR,
                    SENSOR_SIGNATURE_THRESHOLD, SENSOR_SIG_FULL, SENSOR_ARROW_MARGIN,
                    SCAN_DUMP_DECAY, DEBUG_COLLISION,
                    MISSILE_COLOR, MAX_MISSILES,
                    SFX_LASER_MIN_INTERVAL, SFX_ENEMY_LASER_MIN_INTERVAL,
                    SFX_SHIELD_HIT_MIN_INTERVAL)

from .bullets import Bullet, EnemyBullet, Missile, MissileShot

from .ship import Ship, wrapped_delta, _dim_color
from .intent import ShipInput
from .asteroid import Asteroid
from .bullets import Bullet, EnemyBullet
from .particles import burst, shield_burst
from .spawning import spawn_enemy, make_stars, update_field, TestTarget
from .ai_enemy import AIEnemy, MoteEnemy
from .hulls import ENEMY_HULL, MOTE_HULL, enemy_loadout, mote_loadout

from .fog import draw_fog, LightSource

from .hud import draw_hud, draw_game_over
from .camera import Camera
from .netcode import SnapshotBuffer, PredictedShip
from .config import (INTERP_DELAY, SNAPSHOT_INTERVAL, ROCK_FILL, ROCK_EDGE,
                    ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME,
                    SHIP_COLOR, SHIP_EDGE)

STEP = 1 / 60   # fixed simulation timestep


class _RemoteEnemyProxy:
    """Presentation stand-in for drawing a remote enemy's REAL hull and
    computing its targeting lead point (Session 7.2).

    The client never runs the sim, so it has no live AIEnemy objects — but
    the enemy hulls are FIXED on both peers by construction (ENEMY_HULL /
    MOTE_HULL + fixed loadouts), so the client can build a stand-in that
    owns a real Ship (to reuse Ship.draw) and exposes just enough of the
    AIEnemy surface for the remote render: `ship` (drawn at the buffer's
    interpolated pos/angle), `hull` (fill/edge), `pos` (set from the
    buffer each frame), and `lead_point` (the same intercept math the host
    uses, so the client's reticle lines up with the host's).

    It deliberately does NOT subclass AIEnemy: the AIEnemy constructor
    draws a ship id from the SHARED class counter AIEnemy._next_id, and in
    the loopback e2e the host and client run in separate threads of one
    process — a stand-in build racing the host's enemy spawns would
    perturb the host's enemy-id stream. A plain object with its own Ship
    (Ship.__init__ takes an explicit ship_id, default 0) never touches the
    counter. It is never stepped and never fed to the sim — pure
    presentation.
    """

    def __init__(self, hull, loadout):
        self.hull = hull
        self.ship = Ship(hull=hull, loadout=loadout)
        self._acc_smooth = pygame.Vector2(0, 0)

    @property
    def pos(self):
        return self.ship.pos

    def lead_point(self, shooter_pos, bullet_speed, use_accel=True):
        """Intercept solution — the same math as AIEnemy.lead_point, using
        this proxy's ship pos/vel and (zero) smoothed accel. The buffer
        does not carry the enemy's smoothed accel, so a zero accel gives a
        constant-velocity lead: the right order of accuracy for a reticle,
        and honest about what the proxy knows."""
        e0 = self.ship.pos
        v = self.ship.vel
        a = self._acc_smooth if use_accel else pygame.Vector2(0, 0)
        d0 = (e0 - shooter_pos).length()
        if d0 < 1:
            return None
        T = d0 / bullet_speed
        for _ in range(6):
            eT = e0 + v * T + 0.5 * a * (T * T)
            T_new = (eT - shooter_pos).length() / bullet_speed
            if T_new > TARGETING_MAX_LEAD:
                return None
            if abs(T_new - T) < 1e-3:
                T = T_new
                break
            T = T_new
        return e0 + v * T + 0.5 * a * (T * T)


# --- Session 9.x M2a: model-driven WORLD render helpers -------------------
#
# The 9.x goal is for the render to stop reading live mutable sim state
# (self.asteroids / self.enemies / self.bullets / ...) and instead consume
# the plain-data RenderModel (see Game.render_model). M2a moves the WORLD
# entities (stars, asteroids, enemies, bullets, enemy_bullets, missiles,
# particles) onto that path. These are module-level functions that read ONLY
# the plain-data model + the camera + per-tag presentation stand-ins — never
# a live sim object — so they are safe to call from a render thread later
# (M3) once the model is published by an atomic reference swap.
#
# They mirror the live `a.draw` / `e.draw` / `p.draw` / inline bullet-missile
# code exactly, so the world renders identically. The one deliberate loss
# (restorable in M2b): the enemy tuple carries pose + hull tag, NOT the
# enemy ship's presentation state (flame_mags / shield_impacts / arcs), so a
# stand-in draws the enemy hull WITHOUT its thruster flames / shield flash.
# The local ship (M2b) keeps its full presentation, so the host's own feel is
# unaffected. To restore enemy flames later, add flame_mags/shield_impacts to
# the model's enemy tuple and pass them into _draw_world_enemy.
#
# Deferred to M2b (still read live state in draw() for now): the laser BEAMS
# and the targeting RETICLE — both are entangled with the local ship's
# interpolated pose (self.ship.rpos/rangle) and the local ship's own
# presentation, so they move with the local-ship milestone, not the world.

def _draw_world_stars(screen, cam, stars):
    """Parallax background: one dim dot per star, offset by 0.2x the camera.
    `stars` is the model's [(x, y, r), ...] (plain data)."""
    for x, y, r in stars:
        sx = (x - cam.pos.x * 0.2) % WIDTH
        sy = (y - cam.pos.y * 0.2) % HEIGHT
        pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)


def _draw_world_asteroid(screen, cam, pos, angle, verts):
    """One asteroid as a filled+stroked polygon at (pos, angle). Mirrors
    Asteroid.draw exactly (same rotation + ROCK_FILL/ROCK_EDGE). `verts` is
    the model's plain [(vx, vy), ...] rock shape."""
    sx, sy = cam.to_screen(pygame.Vector2(pos))
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for v in verts:
        pts.append((sx + v[0] * ca - v[1] * sa, sy + v[0] * sa + v[1] * ca))
    pygame.draw.polygon(screen, ROCK_FILL, pts)
    pygame.draw.polygon(screen, ROCK_EDGE, pts, 2)


def _draw_world_enemy(screen, cam, tag, pos, angle, standins):
    """One enemy as its REAL hull at (pos, angle), via the per-tag
    presentation stand-in (the same stand-ins the client's remote render
    uses — the hull + loadout are fixed on both peers by construction).
    Mirrors AIEnemy.draw. `standins` is a {tag: _RemoteEnemyProxy} dict.
    Unknown tags (e.g. 'test') draw nothing (a test-range construct)."""
    e = standins.get(tag)
    if e is None:
        return
    e.ship.draw(screen, cam, pygame.Vector2(pos), angle,
                fill=e.hull.fill or ENEMY_FILL,
                edge=e.hull.edge or ENEMY_EDGE,
                flame_out=ENEMY_FLAME, flame_in=ENEMY_FLAME)


def _draw_world_bullet(screen, cam, pos, color):
    """One bullet as a 3px dot at (pos). Mirrors the inline bullet code in
    draw() (the host draws a plain dot, not the client's velocity streak)."""
    s = cam.to_screen(pygame.Vector2(pos))
    pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 3)


def _draw_world_missile(screen, cam, pos, vel, boost, life):
    """One missile: body line + nose + boost exhaust flicker. Mirrors the
    inline missile code in draw() exactly (the flicker is a pure function of
    `life`, so it is identical for the same model)."""
    p = pygame.Vector2(pos)
    v = pygame.Vector2(vel)
    s = cam.to_screen(p)
    fwd = v.normalize()
    tail = cam.to_screen(p - fwd * 14)
    # body: longer, thicker than a bullet
    pygame.draw.line(screen, _dim_color(MISSILE_COLOR, 0.7), tail, s, 3)
    # nose: bright tip
    pygame.draw.circle(screen, MISSILE_COLOR, (int(s.x), int(s.y)), 3)
    # exhaust: only during the boost ramp, flickering length
    if boost > 0:
        flick = 6 * (0.5 + 0.5 * math.sin(life * 40))
        flame = cam.to_screen(p - fwd * (14 + flick))
        pygame.draw.line(screen, (255, 220, 120), tail, flame, 2)


def _draw_world_particle(screen, cam, pos, vel, color, life, max_life):
    """One explosion particle: a short streak along -vel, fading with life.
    Mirrors Particle.draw exactly."""
    p = pygame.Vector2(pos)
    v = pygame.Vector2(vel)
    a = max(0.0, life / max_life)
    tail = p - v * 0.03
    s1 = cam.to_screen(p)
    s2 = cam.to_screen(tail)
    pygame.draw.line(screen, color, (s1.x, s1.y), (s2.x, s2.y),
                     max(1, int(2 * a)))


# --- Session 9.x M2b: model-driven LOCAL-SHIP render helpers --------------
#
# M2a moved the WORLD entities onto the plain-data RenderModel. M2b moves
# the rest of draw() — the local ship (hull + flames + shield + arcs +
# scan pulse), the camera, the laser beams, the targeting reticle, the fog
# lights, the sensor contacts, and the HUD — so the render path reads NO
# live sim state. That is what makes the M3 atomic reference swap safe:
# the render thread will hold a model and call these, never self.*.
#
# The local ship reuses the SAME stand-in pattern as the world enemies:
# a lazily-built presentation Ship (the player's hull + loadout are fixed
# on both peers by construction) whose PRESENTATION fields are synced from
# the model's per-ship pack each frame, then drawn via Ship.draw. The
# stand-in is never stepped and never fed to the sim — pure presentation.
# (The ship's draw code is large — hull/panels/flames/shield/impacts/arcs/
# laser-charge/missile-lock — so reusing Ship.draw on a synced stand-in is
# far cleaner than re-implementing it as a plain-data function.)


def _ship_pose(pack, alpha):
    """The interpolated render pose (rpos, rangle) from the model's
    prev/curr pose + the accumulator alpha. Mirrors Ship.sync_render
    exactly (lerp pos, shortest-arc lerp angle)."""
    p0 = pygame.Vector2(pack["prev_pos"])
    p1 = pygame.Vector2(pack["pos"])
    rpos = p0.lerp(p1, alpha)
    a0, a1 = pack["prev_angle"], pack["angle"]
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    rangle = a0 + da * alpha
    return rpos, rangle


def _sync_local_ship(standin, pack):
    """Copy the model's per-ship presentation state onto the local-ship
    stand-in so its Ship.draw renders exactly what the live ship would.
    Only presentation fields are touched (the stand-in is never stepped,
    so its synced fields are irrelevant to draw())."""
    s = standin
    s.flame_mags = dict(pack["flame_mags"])
    s.arcs = [([pygame.Vector2(pt) for pt in pts], age, ttl)
              for pts, age, ttl in pack["arcs"]]
    s.shield_impacts = [list(t) for t in pack["shield_impacts"]]
    s.scan_pulse = pack["scan_pulse"]
    s.contacts = [(pygame.Vector2(pos), dist, strength, confirmed)
                  for pos, dist, strength, confirmed in pack["contacts"]]
    s.targeting_on = pack["targeting_on"]
    s.tracked = pack["tracked"]
    s.sensor_on = pack["sensor_on"]
    s.scan_cd = pack["scan_cd"]
    s.scan_reveal = pack["scan_reveal"]
    s.shield_charge = pack["shield_charge"]
    s.shield_dump = pack["shield_dump"]
    s.shield_clock = pack["shield_clock"]
    s.brownout = pack["brownout"]
    s.power_used = pack["power_used"]
    s.power_supply = pack["power_supply"]
    s.compute_used = pack["compute_used"]
    s.compute_supply = pack["compute_supply"]
    for w, (cooldown, charge, lock_progress) in zip(s.weapons,
                                                    pack["weapons"]):
        w.cooldown = cooldown
        w.charge = charge
        w.lock_progress = lock_progress


def _draw_local_ship(screen, cam, standin, pack, alpha,
                     fill=None, edge=None):
    """Draw the local player's ship from the model: sync the stand-in's
    presentation from the pack, compute the interpolated pose, and draw.
    Mirrors the live `p.sync_render(alpha); p.draw(screen, cam, p.rpos,
    p.rangle)` exactly.

    The stand-in's CURRENT pose (pos/angle) is also set from the pack, so
    the fog (draw_fog reads ship.pos/vel/angle/axes) sees the same pose the
    live ship has — the fog is drawn AFTER the ship, mirroring the live
    order. Returns the interpolated (rpos, rangle) so the beams (drawn
    after the ship) can use it, exactly as the live code reads
    self.ship.rpos/rangle."""
    _sync_local_ship(standin, pack)
    standin.pos = pygame.Vector2(pack["pos"])
    standin.vel = pygame.Vector2(pack["vel"])
    standin.angle = pack["angle"]
    rpos, rangle = _ship_pose(pack, alpha)
    standin.draw(screen, cam, rpos, rangle,
                 fill=fill if fill is not None else (standin.hull.fill
                                                     or SHIP_COLOR),
                 edge=edge if edge is not None else (standin.hull.edge
                                                     or SHIP_EDGE))
    return rpos, rangle


def _draw_world_beam(screen, cam, rpos, rangle, beam, enemies_by_id,
                     standins):
    """One laser beam from the model. `beam` is (local_start, target_id,
    d, vis_end, age, ttl); (rpos, rangle) is the LOCAL ship's interpolated
    pose (returned by _draw_local_ship this frame); `enemies_by_id` maps
    ship_id -> model enemy tuple (tag, ship_id, pos, angle, vel, acc, cr,
    poly); `standins` is the {tag: _RemoteEnemyProxy} dict (for the
    target's shield oval). The start is computed from the local ship's
    interpolated pose; the end resolves the target by ship_id (a plain
    lookup, no live-list membership) and lands on the TARGET's shield oval
    via the same math as Ship.shield_impact_point. Mirrors the live beam
    loop in draw()."""
    local, target_id, d, vis_end, age, ttl = beam
    fade = 1.0 - age / ttl
    c = tuple(int(ch * fade) for ch in LASER_COLOR)
    fwd = pygame.Vector2(math.cos(rangle), math.sin(rangle))
    right = pygame.Vector2(-fwd.y, fwd.x)
    start = rpos + fwd * local[0] + right * local[1]
    if target_id is not None and target_id in enemies_by_id:
        e = enemies_by_id[target_id]
        epos = pygame.Vector2(e[2])
        eangle = e[3]
        proxy = standins.get(e[0])
        if proxy is not None and proxy.ship.shield_comp is not None:
            end = _shield_impact_point_pose(epos, eangle,
                                            proxy.ship.shield_oval,
                                            epos - d * e[6])
        else:
            end = epos - d * e[6]
    else:
        end = pygame.Vector2(vis_end)
    pygame.draw.line(screen, c, cam.to_screen(start),
                     cam.to_screen(end), 2)


def _shield_impact_point_pose(pos, angle, oval, world_pos):
    """Plain-data mirror of Ship.shield_impact_point: the point on the
    shield oval (a, b, cx, cy) centered at `pos` facing `angle`, in the
    direction of world_pos. The live method uses self.pos/self.angle/
    self.shield_oval; this takes them explicitly so it works on a model
    enemy (whose oval comes from its per-tag stand-in)."""
    a, b, cx, cy = oval
    d = world_pos - pos
    if d.length_squared() < 1e-6:
        return pos
    fwd = pygame.Vector2(math.cos(angle), math.sin(angle))
    right = pygame.Vector2(-fwd.y, fwd.x)
    dx, dy = d.dot(fwd) - cx, d.dot(right) - cy
    t = 1.0 / math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return pos + fwd * (cx + t * dx) + right * (cy + t * dy)


def _lead_point(pos, vel, acc, shooter_pos, bullet_speed, use_accel=True):
    """Plain-data mirror of AIEnemy.lead_point / _RemoteEnemyProxy.lead_point:
    the intercept solution for a target at (pos, vel, acc). Same 6-iteration
    math, so the model-driven reticle lines up with the live one."""
    e0 = pygame.Vector2(pos)
    v = pygame.Vector2(vel)
    a = pygame.Vector2(acc) if use_accel else pygame.Vector2(0, 0)
    d0 = (e0 - shooter_pos).length()
    if d0 < 1:
        return None
    T = d0 / bullet_speed
    for _ in range(6):
        eT = e0 + v * T + 0.5 * a * (T * T)
        T_new = (eT - shooter_pos).length() / bullet_speed
        if T_new > TARGETING_MAX_LEAD:
            return None
        if abs(T_new - T) < 1e-3:
            T = T_new
            break
        T = T_new
    return e0 + v * T + 0.5 * a * (T * T)


def _lead_aligned(e_pos, p, ship_pos, ship_angle):
    """Plain-data mirror of Game._lead_aligned: True when the player's nose
    points between the reticle and the enemy."""
    to_ret = p - ship_pos
    to_en = pygame.Vector2(e_pos) - ship_pos
    if to_ret.length() < 1 or to_en.length() < 1:
        return False
    ang_r = math.atan2(to_ret.y, to_ret.x)
    ang_e = math.atan2(to_en.y, to_en.x)
    d_f = wrapped_delta(ang_r, ship_angle, 2 * math.pi)   # reticle -> facing
    d_e = wrapped_delta(ang_r, ang_e, 2 * math.pi)        # reticle -> enemy
    if d_f * d_e < 0:
        return False   # facing on the far side of the reticle
    return abs(d_f) <= abs(d_e) + TARGETING_ALIGN_TOL


def _draw_lead_model(screen, cam, e, ship_pos, ship_angle):
    """One targeting reticle from the model. `e` is a model enemy tuple
    (tag, ship_id, pos, angle, vel, acc, cr, poly); (ship_pos, ship_angle)
    is the LOCAL ship's interpolated pose. Mirrors Game._draw_lead
    exactly (same lead math, same green-flash tick, same crosshair)."""
    p = _lead_point(e[2], e[4], e[5], ship_pos, BULLET_SPEED,
                    TARGETING_USE_ACCEL)
    if p is None or (p - ship_pos).length() > TARGETING_RANGE:
        return
    if _lead_aligned(e[2], p, ship_pos, ship_angle):
        c = (TARGETING_COLOR_GREEN if (pygame.time.get_ticks() // 100) % 2
             else TARGETING_COLOR)
    else:
        c = TARGETING_COLOR
    s = cam.to_screen(p)
    x, y = int(s.x), int(s.y)
    R, gap = 8, 3
    pygame.draw.line(screen, c, (x, y - R), (x, y - gap), 2)
    pygame.draw.line(screen, c, (x, y + R), (x, y + gap), 2)
    pygame.draw.line(screen, c, (x - R, y), (x - gap, y), 2)
    pygame.draw.line(screen, c, (x + R, y), (x + gap, y), 2)


def _build_lights_model(model, standin):
    """Whitelist of things that shine through the fog, built from the
    model (no live state). Mirrors Game._build_lights: the targeting
    reticle lights (one per lead point in range), the bullets/missiles,
    and the local ship's shield-impact flashes.

    The live _build_lights reads the ship's CURRENT pose (ship.pos /
    ship.angle / ship.axes()), so this uses the stand-in's current pose —
    which _draw_local_ship set from the pack this frame (the fog is drawn
    after the ship, mirroring the live order)."""
    lights = []
    ship_pos = standin.pos
    ship_angle = standin.angle
    # --- Targeting reticle: a small light at each predicted lead point.
    if TARGETING_ASSIST and standin.targeting_on:
        for e in model["enemies"]:
            p = _lead_point(e[2], e[4], e[5], ship_pos, BULLET_SPEED,
                            TARGETING_USE_ACCEL)
            if p is not None and (p - ship_pos).length() <= TARGETING_RANGE:
                lights.append(LightSource(p, 50, 0.6))
    # --- Weapon fire: bullets glow as they fly through the dark.
    for pos, _vel in model["bullets"]:
        lights.append(LightSource(pygame.Vector2(pos), 30, 0.5))
    for pos, _vel in model["enemy_bullets"]:
        lights.append(LightSource(pygame.Vector2(pos), 24, 0.4))
    for pos, _vel, _boost, _life in model["missiles"]:
        lights.append(LightSource(pygame.Vector2(pos), 30, 0.5))
    # --- Shield impacts: a fading flash at the hit point on the oval.
    # The synced shield_impacts are [theta, age, ttl] in local hull space;
    # convert to world the same way _draw_shield_impacts does.
    if standin.shield_impacts:
        a, b, cx, cy = standin.shield_oval
        fwd = pygame.Vector2(math.cos(ship_angle), math.sin(ship_angle))
        right = pygame.Vector2(-fwd.y, fwd.x)
        for theta, age, ttl in standin.shield_impacts:
            fade = 1.0 - age / ttl
            world = (ship_pos + fwd * (cx + a * math.cos(theta))
                             + right * (cy + b * math.sin(theta)))
            lights.append(LightSource(world, 40, 0.5 * fade))
    return lights


def _draw_sensor_contacts_model(screen, cam, font, ship_pos, contacts):
    """Sensor contacts above the fog: on-screen blips, off-screen edge
    arrows. Plain-data mirror of Game._draw_sensor_contacts — `contacts`
    is the model's [(pos, dist, strength, confirmed), ...] and ship_pos
    is the local ship's CURRENT pose (the live method reads ship.pos)."""
    if not contacts:
        return
    sp = cam.to_screen(ship_pos)
    for pos, dist, strength, confirmed in contacts:
        s = cam.to_screen(pygame.Vector2(pos))
        base = SENSOR_SCAN_COLOR if confirmed else SENSOR_COLOR
        color = _dim_color(base, 0.35 + 0.65 * strength)
        if -20 <= s.x <= WIDTH + 20 and -20 <= s.y <= HEIGHT + 20:
            pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 5, 2)
            continue
        d = pygame.Vector2(s.x - sp.x, s.y - sp.y)
        if d.length_squared() < 1:
            continue
        d.normalize_ip()
        m = SENSOR_ARROW_MARGIN
        ts = []
        if d.x > 0:
            ts.append((WIDTH - m - sp.x) / d.x)
        elif d.x < 0:
            ts.append((m - sp.x) / d.x)
        if d.y > 0:
            ts.append((HEIGHT - m - sp.y) / d.y)
        elif d.y < 0:
            ts.append((m - sp.y) / d.y)
        if not ts:
            continue
        p = sp + d * min(ts)
        ang = math.atan2(d.y, d.x)
        for off in (0.5, -0.5):
            pygame.draw.line(screen, color, (p.x, p.y),
                             (p.x - math.cos(ang + off) * 9,
                              p.y - math.sin(ang + off) * 9), 2)
        if confirmed:
            txt = font.render(f"{dist:.0f}", True, color)
            screen.blit(txt, (p.x - txt.get_width() / 2, p.y + 10))


class Game:
    def __init__(self, screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=None, loadout=None, test_mode=False, seed=None, sound=None,
                local_index=0, players=1):
        self.screen = screen
        self.font = font
        self.big_font = big_font
        self.light_tex = light_tex
        self.fog_surf = fog_surf
        self.light_surf = light_surf
        # One rng for the whole sim: same seed -> same run. None -> random
        # each launch (the pre-seed behavior).
        self.rng = random.Random(seed)
        self.stars = make_stars(rng=self.rng)

        # Player ships (Session 6.1): a list, not a single attribute.
        # Single-player is a 1-ship list; 2P is 2 ships in the same world.
        # `self.ship` (the property below) is players[0] — the backward-
        # compat alias the sim, the HUD, and the tests still read.
        # players: 1 = single-player (the local hull/loadout); 2 = 2P host
        # (Session 6.5) — player 0 is the host's ship, player 1 is a
        # PLACEHOLDER (the client's hull/loadout is unknown until the join
        # handshake). The host replaces the placeholder after the handshake
        # via `set_player_ship` (the client's shield ring is rebuilt then).
        if players == 1:
            self.players = [Ship(hull=hull, loadout=loadout)]
        else:
            self.players = [Ship(hull=hull, loadout=loadout), Ship()]
        # local_index: which player this Game controls locally (0 = host's
        # own ship, 1 = the client's). The prediction ghost tracks
        # players[local_index] (Session 6.1 layout; the host applies the
        # remote player's input in 6.2).
        self.local_index = local_index
        # Per-player shield ring (Session 6.2a): one pre-drawn Surface per
        # player, sized by that player's hull radius (hulls may differ in 2P).
        self.shields = [self._make_shield(p) for p in self.players]

        self.cam = Camera(self.ship.pos)
        self.bullets = []
        self.missiles = []
        self.beams   = []
        self.enemy_bullets = []
        self.particles = []
        self.asteroids = []
        self.enemies = []
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.test_mode = test_mode
        self.sound = sound      # SoundBank or None (silent, e.g. headless)
        # --- networking: remote-interpolation render state (Session 5b.3) ---
        # sim_time: seconds of sim this Game has advanced. The authoritative
        # peer stamps each snapshot with it; the remote peer renders
        # sim_time - INTERP_DELAY (see remote_view).
        self.sim_time = 0.0
        self._snap_tick = 0        # sim ticks since the last snapshot stamp
        self.snap_buf = SnapshotBuffer()
        self.ghost = PredictedShip(local_index=local_index)   # local-ship prediction ghost (5b.4b); local_index (7.3) carried for the ghost's own-bullet bookkeeping
        # Session 7.2: per-tag enemy-hull stand-ins for the remote render
        # (built lazily on first use — the client only, via
        # _get_remote_enemies; the host never draws remote enemies).
        self._remote_enemies = None
        # Session 9.x M2b: per-player presentation stand-ins (built lazily on
        # first draw; see _get_standin). Each is a real Ship with that
        # player's FIXED hull + loadout, whose presentation fields are
        # synced from the model's per-ship pack each frame. Never stepped,
        # never fed to the sim — pure presentation.
        self._standins = {}
        # Remote player's latest input (Session 6.2a): the host stores the
        # client's ShipInput here (set_remote_input) and _step applies it to
        # player 1. Empty default = no thrust/fire; single-player never sets
        # it (only player 0 exists), so the sim is untouched.
        self.remote_input = ShipInput()
        # --- Session 7.8: net debug overlay + CSV log state (client only) ---
        # debug_net is toggled by F3 (handle_events); the client loop reads
        # it to draw the overlay + write the log. last_snap_px / snap_count
        # are set by push_snapshot on each reconcile (the ghost's
        # displacement across the rewind) — the overlay + log read them.
        # Inert on the host (it never calls push_snapshot / draws the
        # overlay).
        self.debug_net = False
        self.last_snap_px = 0.0
        self.snap_count = 0
        self.reset()

    @property
    def ship(self):
        """Backward-compat alias: the local player's ship (players[0]).

        Session 6.1 generalized the sim to a players list (2P: two ships in
        the same world). Everything that reads self.ship — the sim, the HUD,
        the tests — keeps working; 2P code uses self.players directly.
        Read-only: assign to self.players, not self.ship."""
        return self.players[0]

    def _make_shield(self, p):
        """Pre-drawn shield ring for one player ship, sized by that hull's
        collision radius (hulls may differ in 2P)."""
        r = p.collision_radius
        sh = pygame.Surface((int(r * 2 + 10), int(r * 2 + 10)),
                            pygame.SRCALPHA)
        pygame.draw.circle(sh, (120, 200, 255, 100),
                           (sh.get_width() // 2, sh.get_height() // 2),
                           r + 5, 2)
        return sh

    def set_player_ship(self, i, ship):
        """Replace player slot `i` with a built Ship (Session 6.5).

        The host calls this once, after the join handshake, to swap the
        player-1 placeholder for the client's real hull/loadout. The
        placeholder has never been stepped (the handshake happens before
        the first tick), so replacing it in place is safe; the shield ring
        is rebuilt for the new hull's radius. The caller must build the
        Ship itself (it owns the hull/loadout from the wire).
        """
        self.players[i] = ship
        self.shields[i] = self._make_shield(ship)

    def _sfx(self, name):
        """Play a sound if a SoundBank is attached. Pure side effect:
        the sim never reads sound state, so determinism is untouched."""
        if self.sound:
            self.sound.play(name)

    def _sfx_throttled(self, name, min_interval):
        """Throttled variant for rapid-fire weapons (see SoundBank)."""
        if self.sound:
            self.sound.play_throttled(name, min_interval)

    def _sfx_thruster(self, on):
        """Engine loop on/off (see SoundBank.set_thruster)."""
        if self.sound:
            self.sound.set_thruster(on)

    def reset(self):
        # Reset ALL player ships (Session 6.2a): each to center, idle.
        # (Both players spawn at center in v1 — no ship-vs-ship collision,
        # so they pass through each other.)
        for p in self.players:
            p.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
            p.vel = pygame.Vector2(0, 0)
            p.angle = -math.pi / 2
            p.prev_pos = p.pos.copy()
            p.prev_angle = p.angle
            p.reset_shield()
            p.targeting_on = False
            p.tracked = 0
            p.reset_missiles()
            p.reset_lasers()
            p.reset_sensors()
        self.bullets.clear()
        self.enemy_bullets.clear()
        self.beams.clear()
        self.missiles.clear()
        self.particles.clear()
        self.asteroids.clear()
        self.enemies.clear()
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.cam.pos = self.ship.pos.copy()
        self._sfx_thruster(False)   # never carry the engine loop across a reset
        if self.test_mode:
            self._setup_test_scene()
            return
        update_field(self.asteroids, [p.pos for p in self.players], 0,
                     rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)

    # --- networking: whole-sim snapshot (see Ship.snapshot for the
    # ship-level classification) ---
    #
    # SYNCED (serialized, must be identical on both peers):
    #   players              index 0: tuple of one Ship.snapshot() per
    #                        player (21-field tuple each, see ship.py).
    #                        2P: both ships in the same world (6.1).
    #   enemies              per enemy: variant tag + e.snapshot()
    #                        tag = 'mote' (MoteEnemy) | 'test' (TestTarget)
    #                        | 'ai' (AIEnemy). Order matters: MoteEnemy and
    #                        TestTarget are both AIEnemy subclasses, so check
    #                        them first.
    #   bullets, enemy_bullets, missiles
    #                        per projectile: pos/vel/owner/life (+ boost for
    #                        missiles). Missile target is stored as the
    #                        enemy's ship id; apply_snapshot re-wires it.
    #   asteroids            per rock: id/pos/vel/size/angle/spin/verts
    #   rng state            the sim's single rng — the strongest check;
    #                        without it the peers diverge on the next roll
    #   game_over, protect_timer
    #   AIEnemy._next_id     class-level id counter (not instance state —
    #                        a naive snapshot misses it)
    #   Asteroid._next_id    same for rocks (Session 5b.1)
    #
    # NOT SYNCED (presentation / config / local timing — never serialized):
    #   stars                cosmetic background; regenerated locally
    #   shield               pre-drawn pygame.Surface ring (derived from
    #                        ship radius)
    #   cam                  Camera; re-derived from ship.pos on apply
    #   beams, particles     presentation (0.15 s beam ttl, cosmetic bursts)
    #   acc                  fixed-timestep accumulator — local timing, not
    #                        sim state
    #   screen, font, big_font, light_tex, fog_surf, light_surf
    #                        render resources
    #   test_mode, sound     mode flag + side-effect layer
    #   (score/wave/wave_timer no longer exist — removed in the wave/score
    #   cleanup; do not re-add them here.)

    def snapshot(self):
        """Capture the whole sim as ONE plain-data tuple: no pygame
        objects, no references — safe to pickle / send over the wire."""
        return (
            tuple(p.snapshot() for p in self.players),
            tuple((self._enemy_tag(e), e.snapshot()) for e in self.enemies),
            tuple(b.snapshot() for b in self.bullets),
            tuple(b.snapshot() for b in self.enemy_bullets),
            tuple(m.snapshot() for m in self.missiles),
            tuple(a.snapshot() for a in self.asteroids),
            self.rng.getstate(),
            self.game_over,
            self.protect_timer,
            AIEnemy._next_id,
            Asteroid._next_id,
        )

    @staticmethod
    def _enemy_tag(e):
        # Order matters: MoteEnemy and TestTarget are both AIEnemy
        # subclasses, so they must be checked before the base class.
        if isinstance(e, MoteEnemy):
            return 'mote'
        if isinstance(e, TestTarget):
            return 'test'
        return 'ai'

    def apply_snapshot(self, s):
        """Restore a snapshot() tuple into this Game. Entities are
        constructed via the same paths the sim uses (so invariants hold),
        then their synced fields are overwritten."""
        (players_s, enemies_s, bullets_s, enemy_bullets_s, missiles_s,
         asteroids_s, rng_state, game_over, protect_timer, next_id,
         rock_next_id) = s

        # Player ships (Session 6.1): index 0 is a tuple of one
        # Ship.snapshot() per player; apply each to the matching slot.
        for p, p_s in zip(self.players, players_s):
            p.apply_snapshot(p_s)

        # Enemies: build with a THROWAWAY rng so construction (which rolls
        # ship.angle and consumes _next_id) neither burns the real rng
        # stream nor matters — apply_snapshot overwrites the synced fields.
        throwaway = random.Random(0)
        origin = pygame.Vector2(0, 0)
        self.enemies.clear()
        for tag, e_s in enemies_s:
            if tag == 'mote':
                e = MoteEnemy(origin, rng=throwaway)
            elif tag == 'test':
                e = TestTarget(origin, rng=throwaway)
            else:
                e = AIEnemy(origin, rng=throwaway)
            e.apply_snapshot(e_s)
            self.enemies.append(e)

        # Bullets: construct minimally, then overwrite.
        self.bullets.clear()
        for b_s in bullets_s:
            px, py, vx, vy, owner, _life = b_s
            b = Bullet(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                       owner=owner)
            b.apply_snapshot(b_s)
            self.bullets.append(b)

        self.enemy_bullets.clear()
        for b_s in enemy_bullets_s:
            px, py, vx, vy, owner, _life = b_s
            b = EnemyBullet(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                            owner=owner)
            b.apply_snapshot(b_s)
            self.enemy_bullets.append(b)

        # Missiles: same pattern; target is re-wired below, once every
        # enemy exists.
        self.missiles.clear()
        for m_s in missiles_s:
            px, py, vx, vy, owner, _life, _boost, _tid = m_s
            m = Missile(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                        owner=owner)
            m.apply_snapshot(m_s)
            self.missiles.append(m)
        id_map = {e.ship.id: e for e in self.enemies}
        for m in self.missiles:
            m.target = id_map.get(m._target_id)

        # Asteroids: construct minimally (vel=None rolls a throwaway vel
        # from the GLOBAL random module — the sim's rng is untouched, and
        # apply_snapshot overwrites it anyway); apply_snapshot rebuilds
        # radius/collision_radius from ROCK_SIZES[size] and restores the
        # rock's id.
        self.asteroids.clear()
        for a_s in asteroids_s:
            _aid, px, py, _vx, _vy, size, _angle, _spin, _verts = a_s
            a = Asteroid(pygame.Vector2(px, py), size)
            a.apply_snapshot(a_s)
            self.asteroids.append(a)

        self.rng.setstate(rng_state)
        self.game_over = game_over
        self.protect_timer = protect_timer
        # AFTER constructing enemies/asteroids: their construction already
        # consumed ids from the class counters, so restore the snapshotted
        # values now (same pattern for both counters).
        AIEnemy._next_id = next_id
        Asteroid._next_id = rock_next_id
        # Camera is not synced — re-derive it from the restored ship.
        self.cam.pos = self.ship.pos.copy()

    # --- Session 9.x M1: the render-model (the enabler) ---------------------
    #
    # A PLAIN-DATA structure the sim publishes and the render thread consumes.
    # It is the existing 11-tuple snapshot() (the SYNCED state) PLUS the
    # presentation-only fields draw() reads, so the render path can stop
    # reading live mutable sim state (the 9.x goal: decouple the sim from the
    # frame loop).
    #
    # Why plain data: the sim thread (later, M3) will build this and publish
    # it by an atomic reference swap (the exact NetWorker _latest_snapshot
    # pattern). Because every value is a tuple/list of numbers/strings (no
    # pygame objects, no live references), a reference swap is atomic under
    # the GIL and there is NO shared mutable state -> no race, no lock.
    #
    # This method is a PURE BUILDER: it reads the same fields draw() reads and
    # packs them into plain data. It MUST NOT mutate any sim state (the parity
    # test asserts that). The wire snapshot (serialize_snapshot) is UNCHANGED —
    # the rich model is local-only (D1 = (a) rich local render-model).
    #
    # M1 is additive: draw() is untouched (it still reads self.*). M2 refactors
    # draw() to consume this model; M3 moves the sim to its own thread.

    def render_model(self):
        """Capture the whole sim as ONE plain-data render-model: no pygame
        objects, no live references — safe to publish by reference swap.

        Carries every field draw() reads (the parity test in
        test_render_model.py asserts this exhaustively). Layout:

          snapshot      the existing 11-tuple snapshot() (SYNCED state)
          sim_time      the sim clock (seconds) at this step
          step_alpha    self.acc / STEP — the fixed-step accumulator fraction
                        (the render thread's interpolation alpha; M3 moves
                        this to the render thread's own clock)
          camera        (pos, vel, dampening) — the cam.update() target
                        (the local ship's pos/vel + its dampening flag)
          stars         [(x, y, r), ...] — the parallax background
          asteroids     [(pos, angle, verts), ...]
          enemies       [(tag, ship_id, pos, angle, vel, acc_smooth,
                         collision_radius, local_poly), ...]
                         (ship_id: the enemy's ship id — M2b resolves a
                         beam's target_id back to this tuple, a plain
                         lookup with no live-list membership)
          bullets       [(pos, vel), ...]
          enemy_bullets [(pos, vel), ...]
          missiles      [(pos, vel, boost, life), ...]
          beams         [(local_start, target_id, d, vis_end, age, ttl), ...]
                        target_id is the enemy's ship id (or None for a rock
                        beam) — draw() resolves it back to the enemy so the
                        `target in self.enemies` membership test becomes a
                        plain lookup (no live-list membership at render time)
          particles     [(pos, vel, color, life, max_life), ...]
          players       [per-ship dict, ...] — see _ship_render_pack
          game_over     bool
          protect_timer float
        """
        ship = self.ship
        return {
            # --- SYNCED (the existing snapshot) ---
            "snapshot": self.snapshot(),
            "sim_time": self.sim_time,
            "step_alpha": self.acc / STEP,
            # --- camera: the cam.update(dt, ship) target ---
            "camera": ((ship.pos.x, ship.pos.y),
                       (ship.vel.x, ship.vel.y),
                       ship.dampening),
            # --- background ---
            "stars": list(self.stars),
            # --- world entities (draw() iterates these) ---
            "asteroids": [((a.pos.x, a.pos.y), a.angle,
                           tuple((v.x, v.y) for v in a.verts))
                          for a in self.asteroids],
            "enemies": [((self._enemy_tag(e),
                          e.ship.id,
                          (e.pos.x, e.pos.y),
                          e.ship.angle,
                          (e.ship.vel.x, e.ship.vel.y),
                          (e._acc_smooth.x, e._acc_smooth.y),
                          e.collision_radius,
                          tuple(e.ship.collision.local_poly)))
                        for e in self.enemies],
            "bullets": [((b.pos.x, b.pos.y), (b.vel.x, b.vel.y))
                        for b in self.bullets],
            "enemy_bullets": [((b.pos.x, b.pos.y), (b.vel.x, b.vel.y))
                              for b in self.enemy_bullets],
            "missiles": [((m.pos.x, m.pos.y), (m.vel.x, m.vel.y),
                          m.boost, m.life) for m in self.missiles],
            # --- beams: target_id replaces the live enemy reference ---
            "beams": [((beam[0],
                        beam[1].ship.id if beam[1] is not None else None,
                        (beam[2].x, beam[2].y),
                        (beam[3].x, beam[3].y),
                        beam[4], beam[5]))
                      for beam in self.beams],
            # --- particles ---
            "particles": [((p.pos.x, p.pos.y), (p.vel.x, p.vel.y),
                           tuple(p.color), p.life, p.max_life)
                          for p in self.particles],
            # --- per-ship presentation (the rich local model, D1a) ---
            "players": [self._ship_render_pack(p) for p in self.players],
            # --- flags ---
            "game_over": self.game_over,
            "protect_timer": self.protect_timer,
        }

    @staticmethod
    def _ship_render_pack(p):
        """Pack ONE player ship's render fields into a plain-data dict.

        Carries exactly what draw() + _build_lights + _draw_sensor_contacts +
        draw_hud read for the local ship, plus the prev/curr pose the render
        thread interpolates between (M3: the render thread owns the
        interpolation, so it needs prev+curr, not just the interpolated
        rpos/rangle). The synced ship state (power/shield/scan/weapon
        charge) is ALSO carried here so the HUD + presentation can read it
        from the model without touching the live ship — it is the same data
        snapshot() carries, re-exposed for the render path.
        """
        return {
            # identity / pose (the render thread interpolates prev -> curr)
            "id": p.id,
            "pos": (p.pos.x, p.pos.y),
            "vel": (p.vel.x, p.vel.y),
            "angle": p.angle,
            "prev_pos": (p.prev_pos.x, p.prev_pos.y),
            "prev_angle": p.prev_angle,
            "dampening": p.dampening,
            # presentation-only (never serialized in snapshot())
            "flame_mags": dict(p.flame_mags),
            "arcs": [([tuple(pt) for pt in pts], age, ttl)
                     for pts, age, ttl in p.arcs],
            "shield_impacts": [(theta, age, ttl)
                               for theta, age, ttl in p.shield_impacts],
            "scan_pulse": p.scan_pulse,
            "contacts": [((pos.x, pos.y), dist, strength, confirmed)
                         for pos, dist, strength, confirmed in p.contacts],
            # synced state the HUD + presentation read (== snapshot() fields)
            "targeting_on": p.targeting_on,
            "tracked": p.tracked,
            "sensor_on": p.sensor_on,
            "scan_cd": p.scan_cd,
            "scan_reveal": p.scan_reveal,
            "shield_charge": p.shield_charge,
            "shield_dump": p.shield_dump,
            "shield_clock": p.shield_clock,
            "brownout": p.brownout,
            "power_used": p.power_used,
            "power_supply": p.power_supply,
            "compute_used": p.compute_used,
            "compute_supply": p.compute_supply,
            "weapons": [(w.cooldown, w.charge, w.lock_progress)
                        for w in p.weapons],
        }

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_t and not self.game_over:
                    self.ship.targeting_on = not self.ship.targeting_on
                elif event.key == pygame.K_r and self.game_over:
                    self.reset()
                elif event.key == pygame.K_v and not self.game_over:
                    self.ship.sensor_on = not self.ship.sensor_on
                elif event.key == pygame.K_g and not self.game_over:
                    self.ship.fire_scan()
                elif event.key == pygame.K_f and (self.test_mode or self.game_over):
                    self.reset()
                elif event.key == pygame.K_F3:
                    self.debug_net = not self.debug_net   # 7.8 net debug overlay + log
        return True

    # Session 8.5: the fixed-step loop's hiccup policy. `dt` is the
    # UNCLAMPED frame time (the host loop no longer pre-clamps it), so the
    # sim clock tracks real time: a 62 ms hiccup frame advances the sim
    # 62 ms (here, or over the next frame or two) instead of dropping the
    # 12 ms the old frame-loop clamp lost. Two guards keep a pathological
    # stall (GC pause, window drag, OS suspend) from spiraling:
    #   MAX_STEPS_PER_FRAME — one frame may run at most this many fixed
    #     steps (5 = 83 ms of sim time). Covers every observed hiccup
    #     (max 77 ms — the 8.5 Step 1 diagnostic) with zero loss; a
    #     500 ms stall would otherwise fire 30 steps in one frame.
    #   ACC_BACKLOG_CAP — the accumulator itself is capped, so a
    #     pathological stall drops its excess backlog (the old
    #     min(dt, 0.25) semantics, now applied to the backlog instead of
    #     the per-frame dt). The bounded catch-up then runs at most
    #     MAX_STEPS_PER_FRAME steps/frame until the backlog drains.
    MAX_STEPS_PER_FRAME = 5
    ACC_BACKLOG_CAP = 0.25

    def update(self, dt, keys):
        # Sample input once per frame; apply it to each fixed step.
        inp = ShipInput.from_keys(keys)
        # Session 8.5: dt is the UNCLAMPED frame time — the sim clock
        # tracks real time. The old min(dt, 0.25) here + the host loop's
        # min(raw_dt, 0.05) DROPPED the excess on every hiccup frame and
        # the sim clock never caught up (8.5 Step 1 proved it: 0.77 s
        # lost over 63 s, hiccup-frame advances pinned at exactly 50 ms,
        # sim-clock rate 0.988x). The ACC_BACKLOG_CAP above is the
        # spiral-of-death guard now.
        self.acc = min(self.acc + dt, self.ACC_BACKLOG_CAP)
        for _ in range(self.MAX_STEPS_PER_FRAME):
            if self.acc < STEP:
                break
            self._step(STEP, inp)
            self.acc -= STEP
    def _step(self, dt, inp):
        self.sim_time += dt   # sim clock: every fixed step advances it
        if not self.game_over:
            # Movement + fire for EVERY player (Session 6.2a). Player 0 uses
            # the local keys' input; the remote player (index 1) uses the
            # host's latest received input (self.remote_input). Single-player
            # is a 1-ship list, so this loop runs once with the local input —
            # bit-identical to the old single-ship path.
            for i, p in enumerate(self.players):
                p_inp = inp if i == 0 else self.remote_input
                # Per-ship targeting (Session 6.2b): each player gets its own
                # tracked count / laser target / missile target / sensor
                # contacts, all set BEFORE p.update() so this tick's
                # _allocate()/_update_lasers()/_update_missiles() see them.
                # Player 0 is self.ship, so single-player is bit-identical.
                p.tracked = sum(
                    1 for e in self.enemies
                    if e.pos.distance_to(p.pos) <= TARGETING_RANGE)
                p.laser_target = self._pick_laser_target(p)
                p.missile_target = self._pick_missile_target(p)
                self._update_contacts(p)
                if p_inp.fire and len(self.bullets) >= MAX_BULLETS:
                    p_inp = replace(p_inp, fire=False)   # world cap: no room
                if p_inp.missile_fire and len(self.missiles) >= MAX_MISSILES:
                    p_inp = replace(p_inp, missile_fire=False)
                shots, beams, missiles = p.update(dt, p_inp)

                for shot in shots:
                    self.bullets.append(Bullet(shot.pos, shot.vel,
                                               owner=shot.owner))
                # S3: fire sounds. The gun fires ~100 shots/s (FIRE_COOLDOWN
                # 0.01), so the laser blip is throttled to a human rate.
                if shots:
                    self._sfx_throttled("laser", SFX_LASER_MIN_INTERVAL)

                for m in missiles:
                    self.missiles.append(Missile(m.pos, m.vel, owner=m.owner,
                                                 target=m.target))
                if missiles:
                    self._sfx("missile_launch")

                for beam in beams:
                    self._resolve_beam(beam)
                if beams:
                    # Hitscan laser: one discharge per charge cycle (~1/s),
                    # so no throttle needed — unlike the rapid-fire gun blip.
                    self._sfx("beam")
            
            self.protect_timer -= dt
            if not self.test_mode:
                update_field(self.asteroids, [p.pos for p in self.players],
                             dt, rng=self.rng)

        # Engine loop: on while the ship is alive and any fitted thruster
        # has resolved force (demand * allocation > 0). Death or a power
        # brownout cuts the sound; set_thruster(False) is a cheap no-op
        # when the loop is already off.
        self._sfx_thruster(not self.game_over
                           and any(t.force > 0.0
                                   for p in self.players
                                   for t in p.thrusters))

        enemy_fired = False
        for e in self.enemies:
            # Each enemy targets the NEAREST player (Session 6.2a). With one
            # player this is that player — bit-identical to the old path.
            shots = e.update(dt, self._nearest_player(e.pos), self.asteroids)
            for shot in shots:
                self.enemy_bullets.append(EnemyBullet(shot.pos, shot.vel, owner=shot.owner))
            if shots:
                enemy_fired = True
        if enemy_fired:
            self._sfx_throttled("enemy_laser", SFX_ENEMY_LASER_MIN_INTERVAL)

        for b in self.bullets:
            b.update(dt)
        for b in self.enemy_bullets:
            b.update(dt)
        for m in self.missiles:
            m.update(dt)
        for a in self.asteroids:
            a.update(dt)
        for p in self.particles:
            p.update(dt)

        # cull expired projectiles and particles
        self.bullets = [b for b in self.bullets if b.life > 0]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.life > 0]
        self.missiles = [m for m in self.missiles if m.life > 0]
        self.particles = [p for p in self.particles if p.life > 0]

        # collisions after movement, so this tick's motion counts
        self._collisions()

        # age the beam visuals
        for b in self.beams:
            b[4] += dt
        self.beams = [b for b in self.beams if b[4] < b[5]]

    def set_remote_input(self, inp):
        """Store the remote player's latest ShipInput (Session 6.2a). The
        host calls this when an 'input' message arrives; _step applies it to
        player 1 (index 1) each tick. Single-player never calls it."""
        self.remote_input = inp

    def _nearest_player(self, pos):
        """The player ship nearest to `pos` (Session 6.2a). Enemies target
        the nearest player; with one player this is that player."""
        return min(self.players,
                   key=lambda p: p.pos.distance_squared_to(pos))

    
    def _build_lights(self, ship):
        """Whitelist of things that shine through the fog of war.
        Returns a list of LightSource in world coords, built fresh each frame.
        (Session 6.2b: per-ship — called with the local ship.)"""
        lights = []


        # --- Targeting reticle: a small light at each predicted lead point.
        # Mirrors the guard in _draw_lead so the light and the reticle
        # appear/disappear together.
        if TARGETING_ASSIST and ship.targeting_on:
            for e in self.enemies:
                p = e.lead_point(ship.pos, BULLET_SPEED, TARGETING_USE_ACCEL)
                if p is not None and (p - ship.pos).length() <= TARGETING_RANGE:
                    lights.append(LightSource(p, 50, 0.6))

        # --- Weapon fire: bullets glow as they fly through the dark.
        for b in self.bullets:
            lights.append(LightSource(b.pos, 30, 0.5))
        for b in self.enemy_bullets:
            lights.append(LightSource(b.pos, 24, 0.4))

        for m in self.missiles:
            lights.append(LightSource(m.pos, 30, 0.5))

        # --- Shield impacts: a fading flash at the hit point on the oval.
        # shield_impacts stores [theta, age, ttl] in local hull space, so
        # convert to world the same way _draw_shield_impacts does.
        if ship.shield_impacts:
            a, b, cx, cy = ship.shield_oval
            fwd, right = ship.axes()
            for theta, age, ttl in ship.shield_impacts:
                fade = 1.0 - age / ttl
                world = (ship.pos + fwd * (cx + a * math.cos(theta))
                                 + right * (cy + b * math.sin(theta)))
                lights.append(LightSource(world, 40, 0.5 * fade))

        return lights


    def _update_contacts(self, ship):
        """Build the ship's contact list from the fitted sensor.

        Passive: enemies in sensor_range whose active power draw
        (power_used - idle) crosses the signature threshold. Active:
        while a ping's reveal lasts, everything in scan_range is
        confirmed. A confirmed contact replaces a passive one.
        (Session 6.2b: per-ship, called once per player.)
        """
        c = ship.sensor_comp
        ship.contacts = []
        if c is None:
            return
        found = {}
        if ship.sensor_on and c.sensor_range > 0:
            for e in self.enemies:
                d = e.pos.distance_to(ship.pos)
                if d <= c.sensor_range:
                    sig = e.ship.power_used - e.ship.power_idle_total
                    if sig >= SENSOR_SIGNATURE_THRESHOLD:
                        found[e] = [e.pos.copy(), d,
                                    min(1.0, sig / SENSOR_SIG_FULL), False]
        if ship.scan_reveal > 0 and c.scan_range > 0:
            for e in self.enemies:
                d = e.pos.distance_to(ship.pos)
                if d <= c.scan_range:
                    found[e] = [e.pos.copy(), d, 1.0, True]
        ship.contacts = list(found.values())

    def _draw_sensor_contacts(self):
        """Contacts above the fog: on-screen blips, off-screen edge
        arrows. Confirmed (scanned) contacts are brighter + show distance."""
        screen = self.screen
        ship = self.ship
        if not ship.contacts:
            return
        sp = self.cam.to_screen(ship.pos)
        for pos, dist, strength, confirmed in ship.contacts:
            s = self.cam.to_screen(pos)
            base = SENSOR_SCAN_COLOR if confirmed else SENSOR_COLOR
            color = _dim_color(base, 0.35 + 0.65 * strength)
            if -20 <= s.x <= WIDTH + 20 and -20 <= s.y <= HEIGHT + 20:
                pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 5, 2)
                continue
            d = pygame.Vector2(s.x - sp.x, s.y - sp.y)
            if d.length_squared() < 1:
                continue
            d.normalize_ip()
            m = SENSOR_ARROW_MARGIN
            ts = []
            if d.x > 0:
                ts.append((WIDTH - m - sp.x) / d.x)
            elif d.x < 0:
                ts.append((m - sp.x) / d.x)
            if d.y > 0:
                ts.append((HEIGHT - m - sp.y) / d.y)
            elif d.y < 0:
                ts.append((m - sp.y) / d.y)
            if not ts:
                continue
            p = sp + d * min(ts)
            ang = math.atan2(d.y, d.x)
            for off in (0.5, -0.5):
                pygame.draw.line(screen, color, (p.x, p.y),
                                  (p.x - math.cos(ang + off) * 9,
                                   p.y - math.sin(ang + off) * 9), 2)
            if confirmed:
                txt = self.font.render(f"{dist:.0f}", True, color)
                screen.blit(txt, (p.x - txt.get_width() / 2, p.y + 10))

    def _pick_laser_target(self, ship):
        """Nearest enemy within the max laser range; None if no laser fitted.
        (Session 6.2b: per-ship, called once per player.)"""
        max_range = max((w.comp.laser_range for w in ship.weapons
                         if w.comp.laser_range > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(ship.pos)
            if d <= best_d:
                best, best_d = e, d
        return best


    def _pick_missile_target(self, ship):
        """Nearest enemy within the max missile lock range; None if no
        missile fitted. (Session 6.2b: per-ship, called once per player.)"""
        max_range = max((w.comp.missile_lock_range for w in ship.weapons
                         if w.comp.missile_speed > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(ship.pos)
            if d <= best_d:
                best, best_d = e, d
        return best


    def _resolve_beam(self, beam):
        """Hitscan: hit the first enemy near the beam's end point.

        The beam *line* is drawn to the point on the target's shield/hull
        facing the muzzle, so beams from different muzzles land at different
        spots. Each damage point additionally jitters its shield flash.
        """
        rock, hit_pt = self._beam_blocked(beam)
        if rock is not None:
            self._beam_hit_asteroid(rock, hit_pt, beam)
            return

        for i, e in enumerate(self.enemies):
            if e.pos.distance_to(beam.end) < e.collision_radius + 4:
                approach = e.pos - beam.start
                if approach.length_squared() < 1e-6:
                    approach = pygame.Vector2(1, 0)
                base_ang = math.atan2(approach.y, approach.x)
                # Where the beam line visually lands: on the shield oval (or
                # hull) facing the muzzle.
                d = pygame.Vector2(math.cos(base_ang), math.sin(base_ang))
                vis_end = e.ship.shield_impact_point(e.pos - d * e.collision_radius)
                for _ in range(beam.damage):
                    ang = base_ang + self.rng.uniform(-BEAM_IMPACT_SPREAD,
                                                BEAM_IMPACT_SPREAD)
                    d2 = pygame.Vector2(math.cos(ang), math.sin(ang))
                    impact = e.pos - d2 * e.collision_radius
                    if not e.register_hit(impact):
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                        break
                    shield_burst(self.particles,
                             e.ship.shield_impact_point(impact),
                              rng=self.rng)
                self.beams.append([beam.local_start, e, d,  vis_end, 0.0, 0.15])
                return


    def _beam_blocked(self, beam):
        """Nearest asteroid intersecting the beam segment, or (None, None)."""
        seg = beam.end - beam.start
        seg_len2 = seg.length_squared()
        if seg_len2 < 1e-6:
            return None, None
        best_t, best_a, best_pt = 1.0, None, None
        for a in self.asteroids:
            t = (a.pos - beam.start).dot(seg) / seg_len2
            t = max(0.0, min(1.0, t))          # clamp: only between muzzle and target
            closest = beam.start + seg * t
            if closest.distance_to(a.pos) < a.collision_radius and t < best_t:
                best_t, best_a, best_pt = t, a, closest
        return best_a, best_pt

    def _beam_hit_asteroid(self, a, hit_pt, beam):
        # Same kill/split as a bullet hitting a rock.
        burst(self.particles, hit_pt, a.radius, rng=self.rng)
        self._sfx("small_explosion")
        child_size = ROCK_SPLIT[a.size]
        if child_size:
            for _ in range(2):
                ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                ka = self.rng.uniform(0, 2 * math.pi)
                kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                self.asteroids.append(Asteroid(a.pos, child_size,
                                           vel=a.vel * 0.5 + kick,
                                           rng=self.rng))
        self.asteroids.remove(a)
        # Beam visual: fixed endpoint on the rock's surface. target=None makes
        # draw() use vis_end (the "target already gone" path).
        d = hit_pt - a.pos
        if d.length_squared() < 1e-6:
            d = beam.end - beam.start
        d = d.normalize()
        vis_end = a.pos + d * a.collision_radius
        self.beams.append([beam.local_start, None, d, vis_end, 0.0, 0.15])

    def _draw_targeting(self, screen, e):
        pts = e.predict_path(TARGETING_HORIZON, TARGETING_STEPS)
        n = len(pts)
        for i, p in enumerate(pts):
            s = self.cam.to_screen(p)          # world -> screen [3]
            t = i / (n - 1)                    # 0 at enemy, 1 at far end
            r = max(1, int(3 * (1.0 - t)))     # dots shrink toward the far end
            c = tuple(int(ch * (1.0 - 0.6 * t)) for ch in TARGETING_COLOR)
            pygame.draw.circle(screen, c, (int(s.x), int(s.y)), r)

    def _draw_lead(self, screen, e, ship):
        # `ship` is the LOCAL player's ship: self.ship on the host, the
        # prediction ghost's ship on a client (Session 7.2 — the reticle
        # must lead from where the player IS, not from the stale
        # players[0] the client never steps).
        p = e.lead_point(ship.pos, BULLET_SPEED, TARGETING_USE_ACCEL)
        if p is None or (p - ship.pos).length() > TARGETING_RANGE:
            return
        if self._lead_aligned(e, p, ship):
            # flash green: alternate between green and the base light blue
            c = (TARGETING_COLOR_GREEN if (pygame.time.get_ticks() // 100) % 2
                 else TARGETING_COLOR)
        else:
            c = TARGETING_COLOR
        s = self.cam.to_screen(p)
        x, y = int(s.x), int(s.y)
        R, gap = 8, 3
        pygame.draw.line(screen, c, (x, y - R), (x, y - gap), 2)
        pygame.draw.line(screen, c, (x, y + R), (x, y + gap), 2)
        pygame.draw.line(screen, c, (x - R, y), (x - gap, y), 2)
        pygame.draw.line(screen, c, (x + R, y), (x + gap, y), 2)

    def _lead_aligned(self, e, p, ship):
        """True when the player's nose points between the reticle and the
        enemy: the facing direction lies in the angular span (plus tolerance)
        between the direction to the reticle and the direction to the enemy.
        `ship` is the local player's ship (host: self.ship; client: the
        ghost's ship — Session 7.2)."""
        to_ret = p - ship.pos
        to_en = e.pos - ship.pos
        if to_ret.length() < 1 or to_en.length() < 1:
            return False
        ang_r = math.atan2(to_ret.y, to_ret.x)
        ang_e = math.atan2(to_en.y, to_en.x)
        d_f = wrapped_delta(ang_r, ship.angle, 2 * math.pi)   # reticle -> facing
        d_e = wrapped_delta(ang_r, ang_e, 2 * math.pi)        # reticle -> enemy
        if d_f * d_e < 0:
            return False   # facing on the far side of the reticle
        return abs(d_f) <= abs(d_e) + TARGETING_ALIGN_TOL


    def _handle_ship_hit(self, ship, source_pos):
        """Handle a hit on a player ship (Session 6.2a: takes the ship).
        Returns True if the ship survives."""
        if ship.register_hit():
            impact = ship.shield_impact_point(source_pos)
            shield_burst(self.particles, impact, rng=self.rng)
            # Throttled: 3 enemies can hit ~20/s; the ping is 0.3 s long.
            self._sfx_throttled("shield_hit", SFX_SHIELD_HIT_MIN_INTERVAL)
            return True
        self.game_over = True
        burst(self.particles, ship.pos, 30, big=True, rng=self.rng)
        self._sfx("explosion")
        self._sfx("game_over")
        return False

    def _collisions(self):
        # player bullet vs enemy (swept segment vs hull polygon)
        for b in self.bullets[:]:
            for i, e in enumerate(self.enemies):
                hit, hit_pt = e.ship.collision_segment(b.prev_pos, b.pos)
                if hit:
                    self.bullets.remove(b)
                    impact = pygame.Vector2(hit_pt)
                    if e.register_hit(impact):
                        burst(self.particles, impact, 6, rng=self.rng)
                    else:
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                    break

        # bullet vs asteroid
        for b in self.bullets[:]:
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    self.bullets.remove(b)
                    break

        # missile vs enemy (swept segment vs hull polygon)
        for m in self.missiles[:]:
            for i, e in enumerate(self.enemies):
                hit, hit_pt = e.ship.collision_segment(m.prev_pos, m.pos)
                if hit:
                    self.missiles.remove(m)
                    impact = pygame.Vector2(hit_pt)
                    alive = True
                    for _ in range(m.dmg):
                        alive = e.register_hit(impact)
                    if alive:
                        burst(self.particles, impact, 6, rng=self.rng)
                    else:
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                    break

        # missile vs asteroid (detonate, same as a bullet)
        for m in self.missiles[:]:
            for i, a in enumerate(self.asteroids):
                if m.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    self.missiles.remove(m)
                    break

        # enemy bullet vs asteroid
        for b in self.enemy_bullets:
            if b.life <= 0:
                continue
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    b.life = 0.0          # was: self.enemy_bullets.remove(b)
                    break

        # enemy bullet vs EACH player ship (Session 6.2a: shield surface if
        # up, else swept hull). A bullet is consumed by the first ship it
        # hits; one dead ship ends the game (game_over).
        if self.protect_timer <= 0:
            for b in self.enemy_bullets:
                if b.life <= 0:
                    continue
                for p in self.players:
                    if p.shield_on:
                        if p.shield_contains(b.pos):
                            b.life = 0.0
                            if not self._handle_ship_hit(p, b.pos):
                                break
                            break   # bullet consumed by this ship
                    else:
                        hit, hit_pt = p.collision_segment(b.prev_pos, b.pos)
                        if hit:
                            b.life = 0.0
                            if not self._handle_ship_hit(p,
                                                         pygame.Vector2(hit_pt)):
                                break
                            break   # bullet consumed by this ship
                if self.game_over:
                    break


        # EACH player ship vs enemy (ram) — SAT on both hull polygons
        if self.protect_timer <= 0 and not self.game_over:
            for p in self.players:
                for e in self.enemies:
                    if p.collision_overlaps_ship(e.ship):
                        if not self._handle_ship_hit(p, e.pos):
                            break
                if self.game_over:
                    break

        # EACH player ship vs asteroid — swept hull polygon vs circle
        if self.protect_timer <= 0 and not self.game_over:
            for p in self.players:
                for a in self.asteroids:
                    if p.collision_swept_overlaps_circle(a.pos, a.collision_radius):
                        if not self._handle_ship_hit(p, a.pos):
                            break
                if self.game_over:
                    break
        
        # enemy vs asteroid (rocks are hazards for everyone)
        for i, e in enumerate(self.enemies[:]):
            for a in self.asteroids:
                if e.pos.distance_to(a.pos) < a.collision_radius + e.collision_radius:
                    burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                    self._sfx("explosion")
                    self.enemies.pop(i)
                    spawn_enemy(self.enemies, self.ship, rng=self.rng)   # instant respawn
                    break

    def draw(self, dt, model=None):
        screen = self.screen
        # Session 9.x M2b: the ENTIRE render path reads the plain-data
        # RenderModel, not live sim state. The model is built once per frame
        # (here, when the caller doesn't supply one) — M3 moves this build to
        # the sim thread and the render thread will hold the published model
        # and call this with it. No self.* live reads remain in this method
        # (the grep gate in test_m2b_local asserts that), which is what makes
        # the M3 atomic reference swap safe.
        if model is None:
            model = self.render_model()
        # Camera: the model's camera target (the local ship's pos/vel/
        # dampening) — plain values, no live ship.
        cpos, cvel, cdamp = model["camera"]
        self.cam.update(dt, cpos, cvel, cdamp)
        screen.fill(BG)
        _draw_world_stars(screen, self.cam, model["stars"])
        for pos, angle, verts in model["asteroids"]:
            _draw_world_asteroid(screen, self.cam, pos, angle, verts)
        standins = self._get_remote_enemies()
        # ship_id -> model enemy tuple (for beam-target resolution).
        enemies_by_id = {e[1]: e for e in model["enemies"]}
        for tag, _id, pos, angle, _vel, _acc, _cr, _poly in model["enemies"]:
            _draw_world_enemy(screen, self.cam, tag, pos, angle, standins)
        # The LOCAL player's model pack (fog / beams / reticle / contacts / HUD
        # all read the local ship, mirroring the live self.ship = players[0]
        # on the host, players[local_index] on a client).
        local_pack = model["players"][self.local_index]
        # Targeting reticle from the model (the local ship's CURRENT pose —
        # the live _draw_lead reads ship.pos/ship.angle, not rpos/rangle).
        ship_pos = pygame.Vector2(local_pack["pos"])
        ship_angle = local_pack["angle"]
        if TARGETING_ASSIST and local_pack["targeting_on"]:
            for e in model["enemies"]:
                _draw_lead_model(screen, self.cam, e, ship_pos, ship_angle)
        for pos, vel in model["bullets"]:
            _draw_world_bullet(screen, self.cam, pos, BULLET_COLOR)
        for pos, vel, boost, life in model["missiles"]:
            _draw_world_missile(screen, self.cam, pos, vel, boost, life)
        # Laser beams from the model — drawn BEFORE the ships, mirroring the
        # live order. The start uses the local ship's INTERPOLATED pose
        # (computed from the model pack, the same value the live code reads
        # as self.ship.rpos/rangle); the end resolves the target by ship_id.
        local_rpos, local_rangle = _ship_pose(local_pack, model["step_alpha"])
        for beam in model["beams"]:
            _draw_world_beam(screen, self.cam, local_rpos, local_rangle,
                             beam, enemies_by_id, standins)
        for pos, vel in model["enemy_bullets"]:
            _draw_world_bullet(screen, self.cam, pos, ENEMY_BULLET_COLOR)
        for pos, vel, color, life, max_life in model["particles"]:
            _draw_world_particle(screen, self.cam, pos, vel, color, life,
                                 max_life)
        # EVERY player ship (Session 6.2a), each with its own shield ring —
        # mirroring the live `if not game_over: for i, p in
        # enumerate(self.players)`. The stand-in is built lazily from that
        # player's fixed hull + loadout; its presentation is synced from the
        # model pack. (On game_over the live code skips the ships, so this
        # loop is gated the same way.)
        if not model["game_over"]:
            for i, pack in enumerate(model["players"]):
                standin = self._get_standin(i)
                rpos, rangle = _draw_local_ship(screen, self.cam, standin,
                                                pack, model["step_alpha"])
                if model["protect_timer"] > 0:
                    sh = self.shields[i]
                    ssx, ssy = self.cam.to_screen(rpos)
                    screen.blit(sh, (ssx - sh.get_width() // 2,
                                     ssy - sh.get_height() // 2))
        # Fog: the local stand-in must carry the local ship's current pose +
        # synced presentation for draw_fog + the model-built light list. The
        # ships loop sets these when it draws the local ship, but on
        # game_over the loop is skipped — so set them explicitly here
        # (idempotent when the loop already ran).
        local_standin = self._get_standin(self.local_index)
        local_standin.pos = pygame.Vector2(local_pack["pos"])
        local_standin.vel = pygame.Vector2(local_pack["vel"])
        local_standin.angle = local_pack["angle"]
        _sync_local_ship(local_standin, local_pack)
        draw_fog(screen, local_standin, self.cam, self.light_tex,
                 self.fog_surf, self.light_surf,
                 _build_lights_model(model, local_standin))
        # Scan pulse above the fog: a bright ring sweeping through the dark.
        if not model["game_over"]:
            local_standin._draw_scan_pulse(screen, self.cam, local_rpos)
        _draw_sensor_contacts_model(screen, self.cam, self.font, ship_pos,
                                    local_pack["contacts"])
        draw_hud(screen, self.font, model["enemies"], local_standin)
        if model["game_over"]:
            draw_game_over(screen, self.big_font, self.font)
        if DEBUG_COLLISION:
            local_standin.draw_collision(screen, self.cam)
            for e in model["enemies"]:
                proxy = standins.get(e[0])
                if proxy is not None:
                    # draw_collision reads self.pos/self.angle (no explicit
                    # pose arg), so set the proxy's pose to the model enemy's
                    # first. Debug-only (DEBUG_COLLISION off by default); the
                    # hull draw passes explicit pos/angle and ignores these.
                    proxy.ship.pos = pygame.Vector2(e[2])
                    proxy.ship.angle = e[3]
                    proxy.ship.draw_collision(screen, self.cam)

    # --- networking: remote-interpolation render (Session 5b.3) ---
    #
    # The remote peer does NOT run the sim for remote entities. It pushes
    # each received snapshot (stamped with the authoritative peer's sim_time)
    # into self.snap_buf, then renders INTERP_DELAY seconds in the PAST:
    # positions_at(sim_time - INTERP_DELAY) interpolates between the two
    # snapshots that bracket that time. The local player's own ship, bullets,
    # beams, particles, and fog are drawn by the normal draw() path — this
    # hook only draws the REMOTE entities (ship, enemies, asteroids) at their
    # interpolated positions, so the two paths never fight over a pixel.

    def remote_view(self, dt):
        """Draw the remote entities at their interpolated positions.

        Called by the remote peer's render loop INSTEAD of (or before) the
        local draw() for the shared entities. It reads only self.snap_buf and
        self.sim_time — it never mutates sim state, so it is safe to call
        every frame and it leaves the local path untouched.

        Returns the interpolated positions dict (or None when the buffer has
        not yet filled a window) so a caller can, e.g., skip the frame or
        draw a "waiting for snapshots" state.
        """
        screen = self.screen
        self.cam.update(dt, self.ship.pos, self.ship.vel,
                        self.ship.dampening)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)

        pos = self.snap_buf.positions_at(self.sim_time - INTERP_DELAY)
        if pos is None:
            # Not enough snapshots yet to interpolate: draw nothing for the
            # remote entities (the caller may show a waiting state).
            return None

        # Asteroids: fill + edge polygon at the interpolated center. The
        # shape (verts) and spin come from the latest snapshot's data; only
        # the center is interpolated, which is what the buffer provides.
        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        # Remote enemies as their REAL hulls (Session 7.2, the D4 fix) —
        # the buffer carries (tag, x, y, angle, vx, vy, id).
        for (tag, x, y, ang, vx, vy, _eid) in pos['enemies']:
            self._draw_remote_enemy_hull(screen, tag, x, y, ang)
        # Remote projectiles (Session 7.2, the D3 fix).
        for (x, y, vx, vy, kind, owner, boost) in pos['bullets']:
            self._draw_remote_bullet(screen, x, y, vx, vy, kind, boost)
        # All player ships (Session 6.1): the buffer's 'ships' list, by index.
        # Session 6.8: each is drawn as its REAL hull at the interpolated
        # (pos, angle) — the buffer lerps the angle with lerp_angle —
        # instead of a dot.
        for i, (x, y, ang) in enumerate(pos['ships']):
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        draw_hud(screen, self.font, self.enemies, self.ship)
        return pos

    def push_snapshot(self, sim_time, snap, now=None):
        """The single seam where a received authoritative snapshot enters
        the remote peer: record it in the interpolation buffer, then feed
        the local ship's prediction ghost — seed on the first snapshot,
        reconcile on every one after. snap[0] is the tuple of per-player
        ship snapshots (Session 6.1); the ghost is the LOCAL player's
        ship, so it takes snap[0][self.local_index]. The network layer
        (and the 5b.4c test) call this.

        Session 7.6 (dead-reckoning rewind): when `now` (the client's
        current estimate of the host's sim clock) is given, the reconcile
        is a REWIND — the ghost is set to the authoritative snapshot and
        the local inputs the host applied since the snapshot are replayed
        (the prediction is rebuilt, not snapped; pinned #4). When `now` is
        None (tests, or the v1 fallback), it defaults to `sim_time`, so
        the replay span is 0 and this is a pure full snap — the v1
        behavior the 5b.4c drift test (test_interpolation part e) still
        exercises."""
        self.snap_buf.push(sim_time, snap)
        local_s = snap[0][self.local_index]
        if not self.ghost.seeded:
            self.ghost.seed(local_s)
        else:
            # Session 7.8: measure the ghost's displacement ACROSS the
            # reconcile (the "snap size") for the net debug overlay + log.
            # With dead-reckoning rewind + identical input this is ~0; a
            # large value = the prediction diverged from authority (in-
            # flight input / clock error).
            _bx, _by = self.ghost.ship.pos.x, self.ghost.ship.pos.y
            self.ghost.reconcile_rewind(
                local_s, sim_time, sim_time if now is None else now)
            self.last_snap_px = math.hypot(
                self.ghost.ship.pos.x - _bx, self.ghost.ship.pos.y - _by)
            self.snap_count += 1

    def predicted_view(self, dt, keys, host_time=None):
        """Draw the frame with the LOCAL ship taken from the prediction
        ghost instead of the sim (client-side prediction, Session 5b.4b).

        Mirrors remote_view, except the local ship is the ghost: each frame
        the ghost is advanced with the local player's input at the sim's
        FIXED rate (Session 7.1: `ghost.advance(dt, inp)` — a fixed-step
        accumulator, not one step per display frame), and the ghost's ship
        is drawn at its own (predicted) position. Remote entities (enemies,
        asteroids) come from the interpolation buffer at the RENDER POINT
        (Session 7.5b: `self.render_point.now()` — the newest ARRIVED
        snapshot's stamp minus the adaptive delay, chased at a bounded
        per-frame rate; before 7.5b this was
        `sim_time - INTERP_DELAY` with sim_time the 7.1 host-time
        estimate). The buffer's 'ships' entry is drawn for every player
        EXCEPT the local one (Session 6.6: the remote ship in 2P;
        Session 6.8: as its real hull at the interpolated (pos, angle));
        the local ship's buffer entry is NOT drawn — it IS the local
        ship, now predicted.

        `host_time` (Session 7.1, superseded for the render point in
        7.5b): the caller's estimate of the host's sim clock. When
        given, it still becomes self.sim_time — the client's sim_time is
        the HOST's clock as carried by the wire, not the display's wall
        clock (the 6.6 `sim_time += dt` was two independent clocks
        drifting) — but the REMOTE render no longer reads it: the render
        point is anchored on the buffer's newest snapshot (see
        netcode.RenderPoint), because a model of the host clock and the
        data disagree under jitter/loss. When None (tests), the clock is
        left as-is.

        The HUD reads the ghost ship (the local player's power/shield/vel)
        and the buffer's enemy count — self.ship (players[0]) and
        self.enemies are the host's ship / the client's stale initial spawn
        on a client, so they would be wrong here.

        The ghost is PRESENTATION-ONLY: it is never fed back into the sim
        (update() still runs on self.ship). Returns the interpolated
        positions dict (or None while waiting for the first snapshot).
        """
        if not self.ghost.seeded:
            # No authoritative snapshot has arrived yet: nothing to predict
            # from. The caller may show a waiting state.
            return None

        if host_time is not None:
            self.sim_time = host_time

        inp = ShipInput.from_keys(keys)
        self.ghost.advance(dt, inp)

        screen = self.screen
        self.cam.update(dt, self.ghost.ship.pos, self.ghost.ship.vel,
                        self.ghost.ship.dampening)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)

        # Session 7.5b: the render point is anchored on the buffer's
        # newest ARRIVED snapshot (newest stamp - adaptive delay, chased
        # at a bounded per-frame rate — netcode.RenderPoint), not on the
        # 7.1 host-time estimate. None until the first snapshot arrives.
        rp = self.render_point.now()
        if rp is None:
            # No snapshot has arrived yet: nothing to render from.
            return None
        pos = self.snap_buf.positions_at(rp)
        if pos is None:
            # The render point exists (first snapshot arrived) but the
            # buffer holds no window yet (fewer than two snapshots, or
            # the point is before the first snapshot): not enough data
            # to interpolate the remote entities.
            return None

        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        # Remote enemies as their REAL hulls (Session 7.2, the D4 fix):
        # the buffer carries (tag, x, y, angle, vx, vy); the angle is
        # lerp'd with lerp_angle (the same wrapped-delta rule as player
        # ships, 6.8) and the hull is fixed on both peers by
        # construction, so the stand-in per tag draws it faithfully.
        for (tag, x, y, ang, vx, vy, _eid) in pos['enemies']:
            self._draw_remote_enemy_hull(screen, tag, x, y, ang)
        # Remote projectiles (Session 7.2, the D3 fix): every bullet and
        # missile in the buffer, at its interpolated position.
        for (x, y, vx, vy, kind, owner, boost) in pos['bullets']:
            self._draw_remote_bullet(screen, x, y, vx, vy, kind, boost)
        # Remote ships (Session 6.6, hulls drawn in 6.8): every player
        # EXCEPT the local one, from the buffer by index (Session 6.1:
        # ships matched by index). The local ship is the ghost, drawn
        # below — so skip it here. Each is drawn as its REAL hull at the
        # interpolated (pos, angle) (the buffer lerps the angle with
        # lerp_angle, Session 6.8) — the remote peer knows the remote
        # hull from the join/welcome handshake, so self.players[i] IS
        # the remote ship's hull/loadout.
        for i, (x, y, ang) in enumerate(pos['ships']):
            if i == self.local_index:
                continue
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        # The LOCAL ship, from the ghost (its own flame_mags render).
        self.ghost.ship.draw(screen, self.cam,
                             self.ghost.ship.pos, self.ghost.ship.angle)

        # The player's OWN gun shots (Session 7.3): the ghost's predicted
        # bullets, drawn immediately at their predicted positions so the
        # player sees their fire the instant they pull the trigger — not
        # ~100 ms later when the host's next snapshot arrives. Drawn the
        # same way the host draws the local player's bullets (a
        # BULLET_COLOR dot at the bullet's position). Presentation only:
        # no collision, no sim feedback. (Missiles/beams are deferred.)
        for b in self.ghost.local_bullets:
            s = self.cam.to_screen(b.pos)
            pygame.draw.circle(screen, BULLET_COLOR, (int(s.x), int(s.y)), 3)

        # Targeting reticle (Session 7.2): the client has no enemy list
        # to target (it never runs the sim), so build lightweight proxies
        # from the buffer's enemy entries (pos + vel from the
        # interpolated data) and run the SAME lead math the host uses.
        # The reticle leads from the GHOST (where the player is), not
        # from the stale players[0]. The proxies are also the fog's
        # reticle-light sources (below), so build them once.
        proxies = (self._targeting_proxies(pos)
                   if TARGETING_ASSIST and self.ghost.ship.targeting_on
                   else [])
        for e in proxies:
            self._draw_lead(screen, e, self.ghost.ship)

        # Fog of war (Session 7.2 — predicted_view previously skipped it
        # entirely, so fire did not glow through the dark like on the
        # host). The light bubble follows the ghost (the local ship);
        # the lights are the interpolated bullets/missiles plus a reticle
        # light at each lead point (mirrors _build_lights' guard so the
        # light and the reticle appear/disappear together).
        lights = self._remote_fog_lights(pos, self.ghost.ship)
        # The player's OWN gun shots glow too (Session 7.3) — mirrors the
        # host's _build_lights, which adds a LightSource per player bullet.
        for b in self.ghost.local_bullets:
            lights.append(LightSource(b.pos, 30, 0.5))
        for e in proxies:
            p = e.lead_point(self.ghost.ship.pos, BULLET_SPEED,
                             TARGETING_USE_ACCEL)
            if p is not None and (p - self.ghost.ship.pos).length() \
                    <= TARGETING_RANGE:
                lights.append(LightSource(p, 50, 0.6))
        draw_fog(screen, self.ghost.ship, self.cam, self.light_tex,
                 self.fog_surf, self.light_surf, lights)

        # HUD: the LOCAL (ghost) ship's power/shield/velocity, and the real
        # (interpolated) enemy count from the buffer. self.ship is players[0]
        # (the HOST's ship on a client) and self.enemies is the client's stale
        # initial spawn (the client never runs the sim), so both would be
        # wrong here.
        draw_hud(screen, self.font, pos['enemies'], self.ghost.ship)
        return pos

    # --- Session 7.2: full remote rendering helpers -------------------------
    #
    # The client renders REMOTE enemies as their REAL hulls (not 10 px
    # dots) and ALL projectiles (bullets + missiles) from the
    # interpolation buffer. The enemy hulls are FIXED on both peers by
    # construction (ENEMY_HULL/MOTE_HULL + fixed loadouts — the plan's
    # pinned decision), so the client can build a presentation stand-in
    # per tag once and draw it at the buffer's interpolated (pos, angle).
    # The stand-in is a real AIEnemy/MoteEnemy ONLY to reuse Ship.draw —
    # it is never stepped, never fed to the sim, and its ship state is
    # irrelevant (draw() takes explicit pos/angle).

    def _get_remote_enemies(self):
        """The per-tag presentation stand-ins, built lazily ONCE (the hull
        + loadout are identical on both peers by construction, so no hull
        data needs to cross the wire). Returns {'ai': _RemoteEnemyProxy,
        'mote': _RemoteEnemyProxy} — 'test' targets are not drawn as
        remote enemies (they are a single-player test-range construct,
        never in a 2P snapshot)."""
        if self._remote_enemies is None:
            self._remote_enemies = self._build_remote_enemy_standins()
        return self._remote_enemies

    def _build_remote_enemy_standins(self):
        """Build the per-tag stand-ins (see _get_remote_enemies).

        Uses _RemoteEnemyProxy (a plain object owning a Ship), NOT
        AIEnemy/MoteEnemy: the AIEnemy constructor draws a ship id from
        the SHARED class counter AIEnemy._next_id, and in the loopback
        e2e the host and client run in separate threads of one process —
        a stand-in build racing the host's enemy spawns would perturb the
        host's enemy-id stream. The proxy never touches the counter."""
        return {
            'ai': _RemoteEnemyProxy(ENEMY_HULL, enemy_loadout()),
            'mote': _RemoteEnemyProxy(MOTE_HULL, mote_loadout()),
        }

    def _get_standin(self, i):
        """The presentation stand-in for player `i` (Session 9.x M2b),
        built lazily ONCE per player. A real Ship with that player's
        FIXED hull + loadout (identical on both peers by construction) —
        the M2b render path syncs its presentation fields from the
        model's per-ship pack each frame, then draws it via Ship.draw.
        It is never stepped and never fed to the sim; its synced fields
        are irrelevant to draw() (which takes explicit pos/angle).

        The local player's stand-in (i == local_index) is the one the
        fog / beams / reticle / contacts / HUD read; the remote player's
        (2P host) is drawn like any other world entity."""
        s = self._standins.get(i)
        if s is None:
            p = self.players[i]
            s = Ship(hull=p.hull, loadout=p.components)
            self._standins[i] = s
        return s

    def _draw_remote_enemy_hull(self, screen, tag, x, y, ang):
        """Draw a remote enemy as its REAL hull at the interpolated
        (pos, angle) (Session 7.2 — replaces the 10 px dot, the D4
        defect). The stand-in's Ship.draw takes explicit pos/angle, so
        the stand-in's own (never-stepped) state is irrelevant."""
        e = self._get_remote_enemies().get(tag)
        if e is None:
            # Unknown tag (e.g. 'test'): fall back to the coarse dot so
            # an unexpected enemy type still renders something.
            self._draw_remote_enemy(screen, x, y)
            return
        e.ship.draw(screen, self.cam, pygame.Vector2(x, y), ang,
                    fill=e.hull.fill or ENEMY_FILL,
                    edge=e.hull.edge or ENEMY_EDGE,
                    flame_out=ENEMY_FLAME, flame_in=ENEMY_FLAME)

    def _draw_remote_bullet(self, screen, x, y, vx, vy, kind, boost):
        """Draw one remote projectile at its interpolated position
        (Session 7.2 — the D3 defect: the client rendered no bullets at
        all). Bullets are a stretched capsule along the (lerped)
        velocity: length ~ speed * 0.016 (a 860 px/s bullet is ~14 px,
        a 720 px/s enemy bullet ~11 px), so the streak reads as motion.
        Missiles are drawn as on the host's draw(): body line + nose +
        exhaust flicker from `boost` (the boost remaining, from the
        CURRENT snapshot — it drives the flicker, not the physics)."""
        s = self.cam.to_screen(pygame.Vector2(x, y))
        speed = math.hypot(vx, vy)
        if kind == 'missile':
            fwd = pygame.Vector2(vx, vy)
            if fwd.length_squared() < 1e-6:
                fwd = pygame.Vector2(1, 0)
            else:
                fwd.normalize_ip()
            tail = self.cam.to_screen(pygame.Vector2(x, y) - fwd * 14)
            pygame.draw.line(screen, _dim_color(MISSILE_COLOR, 0.7),
                             tail, s, 3)
            pygame.draw.circle(screen, MISSILE_COLOR, (int(s.x), int(s.y)), 3)
            if boost > 0:
                flick = 6 * (0.5 + 0.5 * math.sin(speed * 40))
                flame = self.cam.to_screen(pygame.Vector2(x, y)
                                           - fwd * (14 + flick))
                pygame.draw.line(screen, (255, 220, 120), tail, flame, 2)
            return
        color = BULLET_COLOR if kind == 'player' else ENEMY_BULLET_COLOR
        if speed < 1e-6:
            pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 3)
            return
        fwd = pygame.Vector2(vx, vy) / speed
        length = min(16.0, max(6.0, speed * 0.016))
        tail = pygame.Vector2(s.x - fwd.x * length, s.y - fwd.y * length)
        pygame.draw.line(screen, color, tail, (int(s.x), int(s.y)), 3)
        pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 2)

    def _remote_fog_lights(self, pos, local_ship):
        """LightSources for the client's fog of war (Session 7.2 —
        predicted_view previously skipped draw_fog entirely, so fire did
        not glow through the dark like on the host). Mirrors
        `_build_lights` for the data the client actually has: the
        interpolated bullets/missiles (the buffer's 'bullets' entries)
        and the local (ghost) ship's position. The targeting-reticle
        lights are added by the caller (they need the proxy's
        lead_point, which the caller already computes for the reticle).
        `local_ship` is the ghost's ship (the client's light bubble
        follows the player, not the stale players[0])."""
        lights = []
        for (x, y, vx, vy, kind, owner, boost) in pos['bullets']:
            if kind == 'player':
                lights.append(LightSource(pygame.Vector2(x, y), 30, 0.5))
            elif kind == 'enemy':
                lights.append(LightSource(pygame.Vector2(x, y), 24, 0.4))
            else:   # missile
                lights.append(LightSource(pygame.Vector2(x, y), 30, 0.5))
        return lights

    def _targeting_proxies(self, pos):
        """Lightweight targeting proxies from the buffer's enemy entries
        (Session 7.2): enough of an AIEnemy for `lead_point` /
        `_draw_lead` to work client-side. Each proxy is a real AIEnemy
        (to reuse lead_point's math) whose ship pos/vel are set from the
        interpolated (x, y, vx, vy) and whose _acc_smooth is zero (the
        buffer does not carry the enemy's smoothed accel — a zero accel
        gives a constant-velocity lead, which is the right order of
        accuracy for a reticle and keeps the proxy honest about what it
        knows). The proxies are rebuilt each frame from the buffer —
        they are presentation, never stepped, never fed to the sim."""
        proxies = []
        standins = self._get_remote_enemies()
        for (tag, x, y, ang, vx, vy, _eid) in pos['enemies']:
            e = standins.get(tag)
            if e is None:
                continue
            e.ship.pos = pygame.Vector2(x, y)
            e.ship.vel = pygame.Vector2(vx, vy)
            e._acc_smooth = pygame.Vector2(0, 0)
            proxies.append(e)
        return proxies

    def _draw_remote_rock(self, screen, x, y):
        """Draw a remote asteroid at an interpolated world position.

        The buffer only interpolates the CENTER (x, y); the rock's shape and
        orientation are presentation detail. We draw a simple filled circle
        sized by the latest snapshot's rock — good enough for the remote
        view, and it keeps this hook free of per-rock state it does not own.
        """
        s = self.cam.to_screen(pygame.Vector2(x, y))
        # Latest rock snapshot for the size (first rock; the remote view is
        # a coarse stand-in, not a pixel-faithful replica).
        r = 20
        pygame.draw.circle(screen, ROCK_FILL, (int(s.x), int(s.y)), r)
        pygame.draw.circle(screen, ROCK_EDGE, (int(s.x), int(s.y)), r, 2)

    def _draw_remote_enemy(self, screen, x, y):
        """Draw a remote enemy at an interpolated world position (coarse)."""
        s = self.cam.to_screen(pygame.Vector2(x, y))
        pygame.draw.circle(screen, ENEMY_FILL, (int(s.x), int(s.y)), 10)
        pygame.draw.circle(screen, ENEMY_EDGE, (int(s.x), int(s.y)), 10, 2)

# --- test range: static scene for laser occlusion testing ---
    TEST_ROCK_POS   = (960, 340)   # 300 px ahead, dead center (large)
    TEST_TARGET_POS = (960, 40)    # 600 px ahead, behind the rock

    def _setup_test_scene(self):
        self.asteroids.clear()
        self.enemies.clear()

        # Dead-center large rock (the primary occluder)
        rock = Asteroid(pygame.Vector2(self.TEST_ROCK_POS), 'large')
        rock.vel = pygame.Vector2(0, 0)
        rock.spin = 0.0
        self.asteroids.append(rock)

        # Medium rock, slightly off-axis and further out.
        # Tests nearest-t: if both are on the beam line, the closer
        # (large) rock should win.
        rock2 = Asteroid(pygame.Vector2(990, 250), 'medium')
        rock2.vel = pygame.Vector2(0, 0)
        rock2.spin = 0.0
        self.asteroids.append(rock2)

        # Small rock, off-axis and closer in. Should NOT be hit by a
        # center-line beam (verifies the segment test doesn't false-positive).
        rock3 = Asteroid(pygame.Vector2(920, 480), 'small')
        rock3.vel = pygame.Vector2(0, 0)
        rock3.spin = 0.0
        self.asteroids.append(rock3)

        # Medium rock, well off-axis to the right. Should never be hit
        # unless you deliberately turn the ship.
        rock4 = Asteroid(pygame.Vector2(1060, 380), 'medium')
        rock4.vel = pygame.Vector2(0, 0)
        rock4.spin = 0.0
        self.asteroids.append(rock4)

        self.enemies.append(TestTarget(pygame.Vector2(self.TEST_TARGET_POS)))
