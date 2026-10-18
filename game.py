"""Game state: entities, collisions, and per-frame update/draw.

Networked-ready:
- update() runs the whole sim on a fixed timestep (STEP) via an
  accumulator, so every entity steps on the same deterministic tick.
- Player intent (movement + fire) arrives as a ShipInput, not raw
  key state — the same type a remote player's input will be.
"""
import math
import random

import pygame

from .config import (WIDTH, HEIGHT, SPAWN_PROTECT, BULLET_POOL_MAX,
                    BULLET_SPEED, BULLET_LIFE,
                    ROCK_SPLIT, ROCK_SIZES, BG, STAR_COLOR,
                    BULLET_COLOR, ENEMY_BULLET_COLOR,
                    TARGETING_ASSIST, TARGETING_COLOR, TARGETING_HORIZON,
                    TARGETING_STEPS, TARGETING_RANGE,
                    TARGETING_USE_ACCEL, TARGETING_MAX_LEAD,
                    TARGETING_COLOR_GREEN, TARGETING_ALIGN_TOL, LASER_COLOR,
                    BEAM_IMPACT_SPREAD, SENSOR_COLOR, SENSOR_SCAN_COLOR,
                    SENSOR_SIGNATURE_THRESHOLD, SENSOR_SIG_FULL, SENSOR_ARROW_MARGIN,
                    SCAN_DUMP_DECAY, DEBUG_COLLISION,
                    MISSILE_COLOR, MISSILE_POOL_MAX,
                    SFX_LASER_MIN_INTERVAL, SFX_ENEMY_LASER_MIN_INTERVAL,
                    SFX_SHIELD_HIT_MIN_INTERVAL, PARTICLE_COLORS)

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

from .hud import draw_hud, draw_game_over, draw_respawn
from .camera import Camera
from .netcode import SnapshotBuffer, PredictedShip, BEAM_TTL
from . import missile_telemetry as _mtel
from . import bullet_telemetry as _btel
from .config import (INTERP_DELAY, SNAPSHOT_INTERVAL, ROCK_FILL, ROCK_EDGE,
                    ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME,
                    SHIP_COLOR, SHIP_EDGE,
                    MISSILE_HANDOFF_DECAY, MISSILE_HANDOFF_OFFSET_EPS,
                    MISSILE_HANDOFF_MAX_SHRINK)

STEP = 1 / 60   # fixed simulation timestep


class _RemoteEnemyProxy:
    """Presentation stand-in for drawing a remote enemy's REAL hull and
    computing its targeting lead point.

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


class _GhostEnemyProxy:
    """Lightweight enemy proxy for the prediction ghost's weapon targeting.

    The ghost is a real Ship stepped with the local input, but the client
    never runs the sim — it has no live AIEnemy list to target. The
    interpolation buffer's enemy entries DO carry each enemy's
    interpolated (pos, vel, angle, id, shield state), so predicted_view
    builds one of these per buffer enemy each frame and feeds the list to
    `ghost.advance` -> `step`. The ghost's laser target is picked from
    them with the SAME nearest-in-range math the host's
    `_pick_laser_target` uses, so the ghost charges + fires a beam on the
    same tick the host does (the ghost's weapon state — charge/lock/
    cooldown — is already synced via apply_snapshot on reconcile).

    It exposes only what the ship's weapon math reads: `.pos` (the
    interpolated world pos — `Ship._in_wedge` / `_laser_in_wedge` /
    `_pick_laser_target` all read `t.pos`), `.id` (matching a specific
    enemy across frames), and a `.ship` stand-in whose
    `power_used` / `power_idle_total` come from the buffer's enemy
    signature fields (the sensor-contact signature is
    `power_used - power_idle_total`, computable client-side from the
    snapshot). It is a plain object — never stepped, never fed to the
    sim, and it does not touch the AIEnemy id counter (the same
    construction reason as _RemoteEnemyProxy).
    """

    def __init__(self, pos, vel, angle, eid, power_used, power_idle_total):
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.id = eid
        self._ship = _GhostEnemyShip(power_used, power_idle_total)

    @property
    def ship(self):
        return self._ship

    def lead_point(self, shooter_pos, bullet_speed, use_accel=True):
        """Intercept solution — the same math as AIEnemy.lead_point /
        _RemoteEnemyProxy.lead_point, using this proxy's pos/vel and a
        zero accel (the buffer does not carry the enemy's smoothed
        accel — a constant-velocity lead, the right order of accuracy
        for a reticle). The targeting reticle reuses these proxies, so
        the proxy must expose lead_point too."""
        e0 = self.pos
        v = self.vel
        d0 = (e0 - shooter_pos).length()
        if d0 < 1:
            return None
        T = d0 / bullet_speed
        for _ in range(6):
            eT = e0 + v * T
            T_new = (eT - shooter_pos).length() / bullet_speed
            if T_new > TARGETING_MAX_LEAD:
                return None
            if abs(T_new - T) < 1e-3:
                T = T_new
                break
            T = T_new
        return e0 + v * T


class _GhostEnemyShip:
    """The `.ship` stand-in a _GhostEnemyProxy exposes ).

    Only the sensor-contact signature fields matter: `power_used` and
    `power_idle_total` (the host's `_update_contacts` reads
    `e.ship.power_used - e.ship.power_idle_total`). Both come from the
    buffer's enemy entry (the enemy's Ship snapshot fields 18/19 —
    power_used / power_factor... see _ghost_enemy_proxies). Never
    stepped, never fed to the sim."""

    def __init__(self, power_used, power_idle_total):
        self.power_used = power_used
        self.power_idle_total = power_idle_total


# --- model-driven WORLD render helpers ------------------------------------
# Module-level functions that read ONLY the plain-data RenderModel + camera
# + per-tag presentation stand-ins — never live sim state — so the render
# thread can call them on the atomically-published model. They mirror the
# live draw code exactly. Enemy stand-ins get shield flash + thruster flames
# fed by _draw_world_enemy (client buffer path does the same); impact sparks
# and brownout arcs are still not carried in the model.

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


def _draw_world_enemy(screen, cam, tag, pos, angle, standins,
                      shield_dump=0.0, shield_clock=0.0,
                      flame_mags=None):
    """One enemy as its REAL hull at (pos, angle), via the per-tag
    presentation stand-in (the same stand-ins the client's remote render
    uses — the hull + loadout are fixed on both peers by construction).
    Mirrors AIEnemy.draw. `standins` is a {tag: _RemoteEnemyProxy} dict.
    Unknown tags (e.g. 'test') draw nothing (a test-range construct).

    `shield_dump` / `shield_clock` (from the model's enemy tuple) are fed
    to the stand-in before the draw, so Ship._draw_shield renders the
    enemy's shield-impact flash. `flame_mags` (the CURRENT frame — not
    lerp'd) is fed the same way, so Ship.draw renders the enemy's thruster
    exhaust (the client's buffer path does the same, so both peers show
    enemy exhaust). None (or an empty dict) draws no exhaust."""
    e = standins.get(tag)
    if e is None:
        return
    e.ship.shield_dump = shield_dump
    e.ship.shield_clock = shield_clock
    e.ship.flame_mags = dict(flame_mags) if flame_mags else {}
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


# --- model-driven LOCAL-SHIP render helpers -------------------------------
# The rest of draw() (local ship, camera, beams, reticle, fog lights,
# sensor contacts, HUD) also reads only the model — no live sim state —
# which is what makes the atomic model swap safe. The local ship reuses the
# enemy stand-in pattern: a lazily-built presentation Ship (hull + loadout
# fixed on both peers) whose presentation fields are synced from the model's
# per-ship pack each frame, then drawn via Ship.draw. Never stepped, never
# fed to the sim.


def _ship_pose(pack, alpha):
    """The interpolated render pose (rpos, rangle) from the model's
    prev/curr pose + the accumulator alpha. Mirrors Ship.sync_render
    exactly (lerp pos, shortest-arc lerp angle).

    `alpha` is clamped to [0, 1]: on a hiccup frame the accumulator can
    exceed one step — update() caps the backlog at ACC_BACKLOG_CAP (0.25 s)
    but runs at most MAX_STEPS_PER_FRAME (5) steps, so a >83 ms frame leaves
    acc up to ~10 steps behind (step_alpha up to ~10). Interpolating past
    the current pose would extrapolate into the future, so we render the
    latest simulated state (alpha=1) instead. pygame's lerp raises on alpha
    outside [0, 1], so this clamp is required, not just cosmetic. (The SP
    host is light enough to never exceed the 5-step cap; the MP host runs
    two ships + the snapshot + the NetWorker, so it hiccups past it.)"""
    alpha = max(0.0, min(1.0, alpha))
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
    for w, (cooldown, charge, lock_progress, burst_count) in \
                zip(s.weapons, pack["weapons"]):
        w.cooldown = cooldown
        w.charge = charge
        w.lock_progress = lock_progress
        w.burst_count = burst_count


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
    d, vis_end, age, ttl, owner); (rpos, rangle) is the FIRING ship's
    interpolated pose (draw() picks it from the owner's model pack —
    the same pose _draw_local_ship draws that ship at this frame);
    `enemies_by_id` maps ship_id -> model enemy tuple (tag, ship_id, pos,
    angle, vel, acc, cr, poly); `standins` is the {tag: _RemoteEnemyProxy}
    dict (for the target's shield oval). The start is computed from the
    firing ship's interpolated pose; the end resolves the target by
    ship_id (a plain lookup, no live-list membership) and lands on the
    TARGET's shield oval via the same math as Ship.shield_impact_point.
    Mirrors the live beam loop in draw()."""
    local, target_id, d, vis_end, age, ttl, _owner = beam
    # `d` is a PLAIN TUPLE in the model (render_model serializes the live
    # Vector2 as (x, y)) — convert it back before the `d * e[6]` math, or a
    # beam whose target is still alive crashes (tuple * float). The
    # target-dead branch (vis_end) never touches d, which is why one-shot
    # lasers (360) masked the bug.
    d = pygame.Vector2(d)
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

        # Player ships: a list, not a single attribute. Single-player is a
        # 1-ship list; 2P is 2 ships in the same world. `self.ship` (the
        # property below) is players[0] — the backward-compat alias the
        # sim, the HUD, and the tests still read. players: 1 = single-
        # player (the local hull/loadout); 2 = 2P host — player 0 is the
        # host's ship, player 1 is a PLACEHOLDER (the client's
        # hull/loadout is unknown until the join handshake). The host
        # replaces the placeholder after the handshake via
        # `set_player_ship` (the client's shield ring is rebuilt then).
        if players == 1:
            self.players = [Ship(hull=hull, loadout=loadout)]
        else:
            self.players = [Ship(hull=hull, loadout=loadout), Ship()]
        # local_index: which player this Game controls locally (0 = host's
        # own ship, 1 = the client's). The prediction ghost tracks
        # players[local_index].
        self.local_index = local_index
        # Per-player shield ring: one pre-drawn Surface per player, sized
        # by that player's hull radius (hulls may differ in 2P).
        self.shields = [self._make_shield(p) for p in self.players]

        self.cam = Camera(self.ship.pos)
        self.bullets = []
        self.missiles = []
        self.beams   = []
        # Laser beams the host fired, queued for the sim thread to send to the
        # client as T_BEAM events (one per discharge). Each entry is
        # (sx, sy, ex, ey, sim_time) — world-space muzzle -> endpoint +
        # the sim clock at fire time. The client draws it immediately on
        # receipt (a beam is a 0.15 s flash, too short to ride the
        # snapshot's INTERP_DELAY window). Drained after each sim step
        # (sim_thread.py); cleared on reset/respawn.
        self._beam_events = []
        # Input-echo queue: (sim_time, ShipInput) per tick, remote player
        # only. _step appends the input it actually applied (the world
        # fire cap is gone — plan_14 — so this is the raw input, which is
        # exactly what the host applies). Drained by the sim thread into
        # T_ECHO; cleared on reset/respawn.
        self._remote_input_echo = []
        self.enemy_bullets = []
        self.particles = []
        self.asteroids = []
        self.enemies = []
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.test_mode = test_mode
        self.sound = sound      # SoundBank or None (silent, e.g. headless)
        # SFX event queue (plan_12 D.1). _step fires presentation sounds via
        # _sfx/_sfx_throttled/_sfx_thruster; each call APPENDS a plain
        # (name, *params) event here AND (if a SoundBank is attached) plays
        # it immediately (the in-process path). render_model() DRAINS this
        # queue into the model's "sfx" list — the presentation sound events
        # since the last publish. In the separate-process sim (sound=None)
        # nothing plays here; the render process plays the model's "sfx"
        # list via its own SoundBank. Bounded: drained every publish (one
        # per sim step), so it holds at most one step's worth of events.
        self._sfx_events = []
        # --- networking: remote-interpolation render state ---
        # sim_time: seconds of sim advanced; the authoritative peer stamps
        # each snapshot with it, the remote peer renders behind it.
        self.sim_time = 0.0
        self._snap_tick = 0        # sim ticks since the last snapshot stamp
        self.snap_buf = SnapshotBuffer()
        # Local-ship prediction ghost; local_index is its own-bullet bookkeeping.
        self.ghost = PredictedShip(local_index=local_index)
        # Per-tag enemy-hull stand-ins for the remote render (client only,
        # built lazily; the host never draws remote enemies).
        self._remote_enemies = None
        # Per-player presentation stand-ins (see _get_standin): real Ships
        # with the player's fixed hull + loadout, presentation synced from
        # the model's per-ship pack each frame. Never stepped, never sim.
        self._standins = {}
        # Remote player's latest input: the host stores the client's
        # ShipInput here (set_remote_input); _step applies it to player 1.
        # Empty default = no thrust/fire; single-player never sets it.
        self.remote_input = ShipInput()
        # --- net debug overlay + CSV log state (client only, F3) ---
        # debug_net toggled by F3; the client loop reads it to draw the
        # overlay + write the log. last_snap_px / snap_count are set by
        # push_snapshot on each reconcile. Inert on the host.
        self.debug_net = False
        self.last_snap_px = 0.0
        # Last reconcile's replay length (set by push_snapshot, CSV log).
        self.last_replay_ticks = 0
        self.last_replay_span = 0.0
        # Ghost clock captured BEFORE the reconcile (phase-drift diagnosis;
        # CSV log). Inert on the host.
        self.last_ghost_sim_time = 0.0
        # Ghost step clock + render acc captured before the reconcile.
        # snap_px ~= |v| * (snap_time - _last_step_time): ~0 when the
        # ghost's step grid aligns with the host's snapshot grid, ~1 tick
        # when it is one tick behind.
        self.last_ghost_last_step_time = 0.0
        self.last_ghost_acc = 0.0
        # Phase offset (snap_time - ghost._last_step_time) in TICK units.
        self.last_ghost_phase_ticks = 0.0
        # RENDER pose's displacement across the reconcile (what the player
        # sees); last_snap_px is the PHYSICS pose's — the two differ.
        self.last_render_snap_px = 0.0
        # Physics steps the ghost took since the previous reconcile
        # (predicted_view accumulates in _dbg_steps_since_reconcile).
        self.last_advance_steps = 0
        self._dbg_steps_since_reconcile = 0
        # Raw frame dt of the frame that ran the last predicted_view.
        self.last_frame_dt = 0.0
        self.snap_count = 0
        # Client's R key (per-player death) sets this; run_client drains it and
        # sends T_RESPAWN. Inert on the host (R routes to the sim thread).
        self._respawn_requested = False
        # Client-side death explosions: the client never runs the sim, so on
        # an alive->dead buffer flip it spawns its own presentation burst at
        # the death pose. {(ship_index): [Particle, ...]}, never synced.
        self._death_bursts = {}
        # Local ghost's dead state last frame (detects the alive->dead
        # transition so the burst spawns exactly once).
        self._local_was_dead = False
        # Remote player's laser beams, received as T_BEAM events (one per
        # discharge). [start, end, age, ttl] in WORLD space, same shape as
        # the ghost's local_beams. A beam is a 0.15 s flash (BEAM_TTL), too
        # short to ride the snapshot window, so it is drawn immediately and
        # aged here (_step_remote_beams). Presentation only.
        self.remote_beams = []
        # Missile handback offset (client-only). At handback the buffer's copy of
        # a ghost missile is INTERP_DELAY behind the ghost's predicted
        # position, so draw it with offset (ghost_pos - interp_pos) and
        # decay it (MISSILE_HANDOFF_DECAY) until it converges. Invariants:
        # - hand off on the frame the buffer copy is VISIBLE AND ADVANCING
        #   (a new buffer missile is frozen for its pop-in window; handing
        #   off earlier blinks or stalls), capturing the ghost pos then.
        # - decay is capped at MISSILE_HANDOFF_MAX_SHRINK x the buffer's
        #   per-frame displacement, so the rendered position never moves
        #   backward.
        #   _missile_handoff_ghost_pos: {mid: (gx, gy)} ghost pos at handoff.
        #   _missile_handoff_offsets:   {mid: (dx, dy)} offset, decayed.
        #   _missile_handoff_buf_pos:   {mid: (x, y)} buffer's INTERPOLATED
        #       pos last frame (diffed for the displacement cap).
        self._missile_handoff_ghost_pos = {}
        self._missile_handoff_offsets = {}
        self._missile_handoff_buf_pos = {}
        # Bullet handoff needs only the prev-frame buffer pos (the
        # pull-forward is closed-form): {bid: (x, y)}; the handoff fires on
        # the first frame the position CHANGES (the pop-in window is
        # frozen). Client-only.
        self._bullet_handoff_buf_pos = {}
        self.reset()

    @property
    def ship(self):
        """Backward-compat alias: the local player's ship (players[0]).
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
        """Replace player slot `i` with a built Ship.

        The host calls this once after the join handshake to swap the
        player-1 placeholder for the client's real hull/loadout (safe: the
        placeholder was never stepped). The shield ring is rebuilt for the
        new hull's radius; the caller builds the Ship (it owns the
        hull/loadout from the wire).
        """
        self.players[i] = ship
        self.shields[i] = self._make_shield(ship)

    def _sfx(self, name):
        """Fire a presentation sound event.

        ALWAYS records (name,) in self._sfx_events (render_model drains it
        into the model's "sfx" list — the separate-process sim is silent,
        so the render process plays it, plan_12 D.1), and ALSO plays it
        immediately when a SoundBank is attached (the in-process path —
        single player + the client, where the sim and the SoundBank are
        the same object). Pure side effect: the sim never reads sound
        state, so determinism is untouched."""
        self._sfx_events.append((name,))
        if self.sound:
            self.sound.play(name)

    def _sfx_throttled(self, name, min_interval):
        """Throttled variant for rapid-fire weapons (see SoundBank).

        The throttle lives in the SoundBank (wall clock), so the event is
        recorded ONLY when the bank actually plays — the piped "sfx" list
        carries what the ear hears, not what the sim tried to fire (~100
        gun shots/s would flood the pipe). With no SoundBank (the
        separate-process sim) there is no throttle, so every event is
        recorded and the RENDER process's SoundBank applies the same
        throttle when it plays the list — same behavior, throttled once."""
        if self.sound:
            if self.sound.play_throttled(name, min_interval):
                self._sfx_events.append((name, min_interval))
        else:
            self._sfx_events.append((name, min_interval))

    def _sfx_thruster(self, on):
        """Engine loop on/off (see SoundBank.set_thruster)."""
        self._sfx_events.append(("thruster", on))
        if self.sound:
            self.sound.set_thruster(on)

    def reset(self):
        # Reset ALL player ships: each to center, idle. (Both players spawn at
        # center in v1 — no ship-vs-ship collision, so they pass through
        # each other.)
        for p in self.players:
            p.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
            p.vel = pygame.Vector2(0, 0)
            p.angle = -math.pi / 2
            p.prev_pos = p.pos.copy()
            p.prev_angle = p.angle
            p.dead = False   # a full reset revives dead ships
            p.reset_shield()
            p.targeting_on = False
            p.tracked = 0
            p.reset_missiles()
            p.reset_lasers()
            p.reset_sensors()
        self.bullets.clear()
        self.enemy_bullets.clear()
        self.beams.clear()
        self._beam_events.clear()   # stale beam events are gone
        self._remote_input_echo.clear()   # stale echoes are gone
        self.missiles.clear()
        self.particles.clear()
        self.asteroids.clear()
        self.enemies.clear()
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.cam.pos = self.ship.pos.copy()
        # A full reset clears the world's missiles — any pending handoff
        # offsets + ghost positions are stale (their buffer copies are
        # gone).
        self._missile_handoff_offsets.clear()
        self._missile_handoff_ghost_pos.clear()
        self._missile_handoff_buf_pos.clear()
        self._bullet_handoff_buf_pos.clear()   # stale bullet handoff
        self._sfx_thruster(False)   # never carry the engine loop across a reset
        if self.test_mode:
            self._setup_test_scene()
            return
        update_field(self.asteroids, [p.pos for p in self.players], 0,
                     rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)

    def respawn_player(self, i):
        """Respawn ONE player's ship (per-player death).

        The dead ship comes back at center, idle, with a fresh shield and
        weapons — the same per-ship reset `reset()` applies to every ship —
        plus the SPAWN_PROTECT timer (the world is cleared of projectiles
        so the respawn is not instant-killed, mirroring `reset`). The
        WORLD (asteroids, enemies, the other player's ship) is UNTOUCHED:
        this is a per-player respawn, not a game reset. No-op if the ship
        is not dead (a live player pressing R does nothing).

        The caller (the sim thread, via the 'respawn' command) must not
        call this while `game_over` in single player — single player uses
        the full `reset()` instead (the R key routes there)."""
        p = self.players[i]
        if not p.dead:
            return
        p.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        p.vel = pygame.Vector2(0, 0)
        p.angle = -math.pi / 2
        p.prev_pos = p.pos.copy()
        p.prev_angle = p.angle
        p.dead = False
        p.reset_shield()
        p.targeting_on = False
        p.tracked = 0
        p.reset_missiles()
        p.reset_lasers()
        p.reset_sensors()
        # Clear the world's projectiles so the respawn is not
        # instant-killed (mirrors reset(); presentation + hazard state).
        self.bullets.clear()
        self.enemy_bullets.clear()
        self.beams.clear()
        self._beam_events.clear()   # stale beam events are gone
        self.missiles.clear()
        self.protect_timer = SPAWN_PROTECT

    # --- networking: whole-sim snapshot (see Ship.snapshot for the
    # ship-level classification) ---
    #
    # SYNCED (serialized, must be identical on both peers):
    #   players: one Ship.snapshot() per player (25-field tuple, ship.py)
    #   enemies: variant tag + e.snapshot(); tag 'mote'/'test'/'ai' —
    #            MoteEnemy and TestTarget are AIEnemy subclasses, so they
    #            must be checked first (_enemy_tag)
    #   bullets, enemy_bullets, missiles: pos/vel/owner/life (+ boost);
    #            a missile's target is the enemy's ship id (re-wired on
    #            apply)
    #   asteroids: id/pos/vel/size/angle/spin/verts
    #   rng state: the sim's single rng — without it the peers diverge on
    #              the next roll
    #   game_over, protect_timer
    #   AIEnemy._next_id, Asteroid._next_id: CLASS-level id counters (not
    #              instance state — a naive snapshot misses them)
    #
    # NOT SYNCED (presentation / config / local timing — never serialized):
    #   stars (regenerated locally), shield (derived Surface), cam
    #   (re-derived on apply), beams, particles, acc (local timing),
    #   render resources, test_mode, sound

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

        # Player ships: apply each Ship.snapshot() to the matching slot.
        for p, p_s in zip(self.players, players_s):
            p.apply_snapshot(p_s)

        # Enemies: build with a THROWAWAY rng so construction (which rolls
        # ship.angle and consumes _next_id) never burns the real rng
        # stream — apply_snapshot overwrites the synced fields anyway.
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
            px, py, vx, vy, owner, _life, _bid = b_s   # + id
            b = Bullet(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                       owner=owner)
            b.apply_snapshot(b_s)   # also restores b.id
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
            px, py, vx, vy, owner, _life, _boost, _tid, _mid = m_s
            m = Missile(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                        owner=owner)
            m.apply_snapshot(m_s)   # also restores m.id
            self.missiles.append(m)
        id_map = {e.ship.id: e for e in self.enemies}
        for m in self.missiles:
            m.target = id_map.get(m._target_id)

        # Asteroids: construct minimally (the constructor's throwaway vel comes
        # from the GLOBAL random module, not the sim's rng); apply_snapshot
        # rebuilds radius from ROCK_SIZES[size] and restores the id.
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

    # --- the render-model -----------------------------------------------------
    #
    # A PLAIN-DATA structure the sim publishes and the render thread
    # consumes: the 11-tuple snapshot() (SYNCED state) PLUS the
    # presentation-only fields draw() reads. Plain data (no pygame objects,
    # no live references) is what makes the atomic reference swap race-free.
    #
    # render_model() is a PURE BUILDER — it must not mutate sim state (the
    # parity test asserts that). The wire snapshot is UNCHANGED: the rich
    # model is local-only (D1), the wire stays pruned.

    def render_model(self):
        """Capture the whole sim as ONE plain-data render-model: no pygame
        objects, no live references — safe to publish by reference swap.

        Carries every field draw() reads (the parity test in
        test_render_model.py asserts this exhaustively). Layout:

          snapshot      the 11-tuple snapshot() (SYNCED state)
          sim_time      the sim clock (seconds) at this step
          step_alpha    self.acc / STEP (the render thread's interpolation
                        alpha)
          camera        (pos, vel, dampening) — the cam.update() target
          stars         [(x, y, r), ...]
          asteroids     [(pos, angle, verts), ...]
          enemies       [(tag, ship_id, pos, angle, vel, acc_smooth,
                         collision_radius, local_poly, shield_dump,
                         shield_clock, flame_mags), ...]
                         ship_id lets draw() resolve a beam's target_id by
                         plain lookup; shield_dump/shield_clock drive the
                         shield-impact flash; flame_mags the exhaust.
          bullets       [(pos, vel), ...]
          enemy_bullets [(pos, vel), ...]
          missiles      [(pos, vel, boost, life), ...]
          beams         [(local_start, target_id, d, vis_end, age, ttl,
                         owner), ...]
                         target_id = the enemy's ship id (None for a rock
                         beam); owner = the FIRING player's index — draw()
                         anchors each beam to its owner's interpolated pose
                         (the host's sim runs BOTH players, so self.beams
                         holds the remote player's beams too).
          particles     [(pos, vel, color, life, max_life), ...]
          players       [per-ship dict, ...] — see _ship_render_pack
          game_over     bool
          protect_timer float
          sfx           [(name, *params), ...] — the presentation sound
                        events fired since the LAST publish (drained from
                        self._sfx_events here): ("laser", 0.12) /
                        ("beam",) / ("missile_launch",) / ("explosion",) /
                        ("shield_hit", 0.15) / ("thruster", on) / ...
                        The separate-process sim is silent (sound=None),
                        so the render process plays this list via its own
                        SoundBank (plan_12 D.1). In-process (SP/client)
                        the events already played on the local bank; the
                        list is then inert (nothing plays it twice).
        """
        ship = self.ship
        # Drain the SFX events fired since the last publish (plan_12 D.1).
        # render_model() is called once per sim step (the publish), so this
        # is the "per publish" drain: the queue holds at most one step's
        # worth of events. The list is plain data (tuples of str/float/
        # bool) — safe to pickle over the model pipe.
        sfx = self._sfx_events
        self._sfx_events = []
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
                          tuple(e.ship.collision.local_poly),
                          e.ship.shield_dump,
                          e.ship.shield_clock,
                          dict(e.ship.flame_mags)))
                        for e in self.enemies],
            "bullets": [((b.pos.x, b.pos.y), (b.vel.x, b.vel.y))
                        for b in self.bullets],
            "enemy_bullets": [((b.pos.x, b.pos.y), (b.vel.x, b.vel.y))
                              for b in self.enemy_bullets],
            "missiles": [((m.pos.x, m.pos.y), (m.vel.x, m.vel.y),
                          m.boost, m.life) for m in self.missiles],
            # --- beams: target_id replaces the live enemy reference; owner is the
            # firing player's index (draw() anchors each beam to its
            # owner's pose, not the local ship's) ---
            "beams": [((beam[0],
                        beam[1].ship.id if beam[1] is not None else None,
                        (beam[2].x, beam[2].y),
                        (beam[3].x, beam[3].y),
                        beam[4], beam[5], beam[6]))
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
            # --- presentation sound events since the last publish (D.1) ---
            "sfx": sfx,
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
            "weapons": [(w.cooldown, w.charge, w.lock_progress, w.burst_count)
                        for w in p.weapons],
            # Per-player death — draw() shows the dead player the
            # respawn screen (and skips the corpse's shield ring).
            "dead": p.dead,
        }

    def handle_events(self, command_sink=None):
        """Returns False when the window should close.

        `command_sink`: a callable receiving one of 'targeting' / 'sensor' /
        'scan' / 'reset' for the T/V/G/R/F keys. When given (the host,
        whose sim runs on the SimThread), the keys are ROUTED to the sim
        thread instead of mutating live state — the sim thread applies
        them at the top of its next iteration (the event -> sim-command
        queue). When None (single-player + the client, whose sim runs on
        this thread), the keys mutate the live state exactly as before.
        QUIT/ESC (exit) and F3 (the render diagnostic) are always
        handled here.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_t and not self.game_over:
                    if command_sink is not None:
                        command_sink("targeting")
                    else:
                        self.ship.targeting_on = not self.ship.targeting_on
                elif event.key == pygame.K_r:
                    if self.game_over:
                        # Single player (2P never sets the global flag):
                        # R is a full game reset.
                        if command_sink is not None:
                            command_sink("reset")
                        else:
                            self.reset()
                    elif len(self.players) == 2:
                        # Per-player death: R respawns the LOCAL player
                        # only. Host routes to the sim thread; the client
                        # sets a flag run_client drains into a T_RESPAWN.
                        # respawn_player is a no-op unless the ship is dead.
                        if command_sink is not None:
                            command_sink("respawn", self.local_index)
                        else:
                            self._respawn_requested = True
                elif event.key == pygame.K_v and not self.game_over:
                    if command_sink is not None:
                        command_sink("sensor")
                    else:
                        self.ship.sensor_on = not self.ship.sensor_on
                elif event.key == pygame.K_g and not self.game_over:
                    if command_sink is not None:
                        command_sink("scan")
                    else:
                        self.ship.fire_scan()
                elif event.key == pygame.K_f and (self.test_mode or self.game_over):
                    if command_sink is not None:
                        command_sink("reset")
                    else:
                        self.reset()
                elif event.key == pygame.K_F3:
                    self.debug_net = not self.debug_net   # 7.8 net debug overlay + log
        return True

    # The fixed-step loop's hiccup policy. `dt` is the UNCLAMPED frame time,
    # so the sim clock tracks real time (a hiccup frame advances the sim
    # its full duration, spread over this or the next frame or two). Two
    # guards keep a pathological stall (GC pause, window drag, OS suspend)
    # from spiraling:
    #   MAX_STEPS_PER_FRAME — at most this many fixed steps per frame
    #     (5 = 83 ms of sim time).
    #   ACC_BACKLOG_CAP — the accumulator is capped, so a long stall drops
    #     its excess backlog; the bounded catch-up then drains it at <=
    #     MAX_STEPS_PER_FRAME steps/frame.
    MAX_STEPS_PER_FRAME = 5
    ACC_BACKLOG_CAP = 0.25

    def update(self, dt, keys):
        # Sample input once per frame; apply it to each fixed step.
        inp = ShipInput.from_keys(keys)
        # dt is the UNCLAMPED frame time — the sim clock tracks real time.
        self.acc = min(self.acc + dt, self.ACC_BACKLOG_CAP)
        for _ in range(self.MAX_STEPS_PER_FRAME):
            if self.acc < STEP:
                break
            self._step(STEP, inp)
            self.acc -= STEP
    def _step(self, dt, inp):
        self.sim_time += dt   # sim clock: every fixed step advances it
        if not self.game_over:
            # Movement + fire for EVERY player. Player 0 uses the local keys'
            # input; the remote player (index 1) uses the host's latest
            # received input (self.remote_input). Single-player is a
            # 1-ship list, so this loop runs once with the local input.
            for i, p in enumerate(self.players):
                # A dead ship is FROZEN — no movement, firing, or
                # targeting; it stays in the world but is no longer a
                # collision or enemy target. Death is per-player, not
                # global.
                if p.dead:
                    continue
                p_inp = inp if i == 0 else self.remote_input
                # Per-ship targeting: each player's tracked count /
                # laser target / missile target / sensor contacts are
                # set BEFORE p.update() so this tick's
                # _allocate()/_update_lasers()/_update_missiles() see
                # them.
                p.tracked = sum(
                    1 for e in self.enemies
                    if e.pos.distance_to(p.pos) <= TARGETING_RANGE)
                p.laser_target = self._pick_laser_target(p)
                p.missile_target = self._pick_missile_target(p)
                self._update_contacts(p)
                # LOCKSTEP INVARIANT (plan_14): the fire decision must be a
                # PURE FUNCTION OF INPUT-LOCAL STATE — this player's input +
                # its own weapon state (cooldown / charge / allocation, all
                # synced via the ship snapshot). The old world-count gates
                # (len(self.bullets) >= MAX_BULLETS) are GONE: a world count
                # is not reproducible by the ghost, so gating fire on it
                # desynced bullet_seq / cooldown (the double-bullet). The
                # bullet population is policed by the cull-oldest safety net
                # below (performance only — it never changes whether a shot
                # fires). Any future term added to the fire path must be
                # input-local or the lockstep (and the id-dedup) breaks.
                if i == 1:
                    # Echo the input actually applied to player 1 this tick
                    # (the client's rewind replay uses it). With the world
                    # cap gone, p_inp is the RAW input — exactly what the
                    # host applies, so the echo is still correct. Tagged
                    # with the tick's START (sim_time was already advanced
                    # to the tick's END at the top of _step); an end-tag
                    # would be off by one tick.
                    self._remote_input_echo.append(
                        (self.sim_time - dt, p_inp))
                shots, beams, missiles = p.update(dt, p_inp)

                for shot in shots:
                    # Unique player-prefixed id (player_index, per-ship
                    # bullet_seq); the counter is synced via the ship
                    # snapshot, so the client's ghost derives the same ids
                    # (the no-double-draw dedup key).
                    bid = (i, p.bullet_seq)
                    p.bullet_seq += 1
                    self.bullets.append(Bullet(shot.pos, shot.vel,
                                               owner=shot.owner, bid=bid))
                # Cull-oldest performance safety net (plan_14): the lists
                # are in fire order (newest at the back), so pop(0) is the
                # oldest. Sized above the natural count, so this never
                # binds in normal play — it only guards against a bug that
                # spawns unbounded projectiles. It NEVER changes whether a
                # shot fires (that decision is input-local, above).
                while len(self.bullets) > BULLET_POOL_MAX:
                    self.bullets.pop(0)
                # S3: fire sounds. The gun fires ~100 shots/s (FIRE_COOLDOWN
                # 0.01), so the laser blip is throttled to a human rate.
                if shots:
                    self._sfx_throttled("laser", SFX_LASER_MIN_INTERVAL)

                for m in missiles:
                    # Unique player-prefixed id (player_index, per-ship
                    # seq); the counter is synced via the ship snapshot, so
                    # the client's ghost derives the same ids (the dedup
                    # key).
                    mid = (i, p.missile_seq)
                    p.missile_seq += 1
                    self.missiles.append(Missile(m.pos, m.vel, owner=m.owner,
                                                 target=m.target, mid=mid))
                # Cull-oldest performance safety net (plan_14) — see the
                # bullet cull above.
                while len(self.missiles) > MISSILE_POOL_MAX:
                    self.missiles.pop(0)
                if missiles:
                    self._sfx("missile_launch")

                for beam in beams:
                    # Tag the beam with the FIRING player's index: the
                    # client skips its OWN ship's beams (the ghost draws
                    # those) and only draws the remote ship's — no
                    # double-draw.
                    self._resolve_beam(beam, i)
                if beams:
                    # Hitscan laser: one discharge per charge cycle (~1/s),
                    # so no throttle needed — unlike the rapid-fire gun blip.
                    self._sfx("beam")
            
            self.protect_timer -= dt
            if not self.test_mode:
                # Dead ships are frozen — pass only the ALIVE ships'
                # positions to the field so it doesn't reflow around a
                # corpse.
                update_field(self.asteroids,
                             [p.pos for p in self.players if not p.dead],
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
            # Each enemy targets the NEAREST player. With one player this is
            # that player — bit-identical to the old path.
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
        """Store the remote player's latest ShipInput. The host calls this
        when an 'input' message arrives; _step applies it to player 1
        (index 1) each tick. Single-player never calls it."""
        self.remote_input = inp

    def flush_remote_input_echo(self):
        """Return + clear the input-echo queue (the sim thread calls this
        after each stepped iteration and sends the entries as one
        T_ECHO). Each entry is (sim_time, ShipInput) — the input the host
        applied to player 1 on the tick that STARTS at sim_time (see the
        capture in _step; with the world fire cap gone, plan_14, this is
        the raw input). Empty when the sim did not step player 1
        (single-player, game_over, or a dead player-1 ship)."""
        out = self._remote_input_echo
        self._remote_input_echo = []
        return out

    def _nearest_player(self, pos):
        """The player ship nearest to `pos`. Enemies target the nearest
        player; with one player this is that player. Dead ships are not
        targeted — they are frozen, so an enemy would waste its shots on
        a corpse. All players dead (2P, both dead) falls back to the
        nearest ship so the enemies keep moving instead of erroring."""
        alive = [p for p in self.players if not p.dead]
        if not alive:
            alive = self.players
        return min(alive,
                   key=lambda p: p.pos.distance_squared_to(pos))

    
    def _build_lights(self, ship):
        """Whitelist of things that shine through the fog of war.
        Returns a list of LightSource in world coords, built fresh each frame.
        (per-ship — called with the local ship.)"""
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
        (per-ship, called once per player.)
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
        (per-ship, called once per player.)"""
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
        missile fitted. (per-ship, called once per player.)"""
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


    def _emit_beam_event(self, start, end, local_start, owner):
        """Queue a laser beam for the client (world-space muzzle ->
        endpoint + the sim clock at fire time + the FIRING player's index).
        The sim thread drains `self._beam_events` after each step and sends
        it as a T_BEAM message. `start`/`end` are pygame.Vector2
        (beam.start / vis_end); `owner` is the player index (0 = host,
        1 = client) so the client can skip its OWN ship's beams (the ghost
        already draws those) and only draw the remote ship's.

        `local_start` is the HULL-LOCAL muzzle offset
        (beam.local_start, a (lx, ly) tuple). The client re-anchors the
        beam's origin to the firing ship's INTERPOLATED pose each frame
        (mirrors the host's _draw_world_beam / the original 1p fix) — the
        world-space `start` is frozen at fire time, so drawing it directly
        left the origin behind in empty space as the ship moved during the
        0.15 s flash. The world-space `start` is still sent (sx, sy) for
        diagnostics; the client anchors on the hull-local muzzle and skips
        a beam whose owner has no interpolated pose this frame."""
        self._beam_events.append((start.x, start.y, end.x, end.y,
                                  self.sim_time, owner,
                                  local_start[0], local_start[1]))

    def _resolve_beam(self, beam, owner):
        """Hitscan: hit the first enemy near the beam's end point.

        The beam *line* is drawn to the point on the target's shield/hull
        facing the muzzle, so beams from different muzzles land at different
        spots. Each damage point additionally jitters its shield flash.
        """
        rock, hit_pt = self._beam_blocked(beam)
        if rock is not None:
            self._beam_hit_asteroid(rock, hit_pt, beam, owner)
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
                # The beam's OWNER (firing player's index) is part of
                # the entry — draw() anchors each beam to its owner's pose.
                self.beams.append([beam.local_start, e, d,  vis_end, 0.0,
                                   0.15, owner])
                self._emit_beam_event(beam.start, vis_end,
                                      beam.local_start, owner)
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

    def _beam_hit_asteroid(self, a, hit_pt, beam, owner):
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
        # Owner index on the entry (see _resolve_beam).
        self.beams.append([beam.local_start, None, d, vis_end, 0.0, 0.15,
                           owner])
        self._emit_beam_event(beam.start, vis_end,
                              beam.local_start, owner)

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
        # prediction ghost's ship on a client (the reticle
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
        ghost's ship)."""
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
        """Handle a hit on a player ship (takes the ship).
        Returns True if the ship survives."""
        if ship.register_hit():
            impact = ship.shield_impact_point(source_pos)
            shield_burst(self.particles, impact, rng=self.rng)
            # Throttled: 3 enemies can hit ~20/s; the ping is 0.3 s long.
            self._sfx_throttled("shield_hit", SFX_SHIELD_HIT_MIN_INTERVAL)
            return True
        # Per-player death. The ship is marked dead (frozen — see
        # _step); the game_over flag is set ONLY in single player (where
        # one dead ship ends the game, as before). In 2P the OTHER player
        # keeps playing — the dead player respawns with R (see
        # respawn_player).
        ship.dead = True
        if len(self.players) == 1:
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

        # enemy bullet vs EACH player ship (shield surface if
        # up, else swept hull). A bullet is consumed by the first ship it
        # hits; one dead ship ends the game (game_over).
        if self.protect_timer <= 0:
            for b in self.enemy_bullets:
                if b.life <= 0:
                    continue
                for p in self.players:
                    if p.dead:
                        continue   # a frozen corpse takes no hits
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
                if p.dead:
                    continue   # a frozen corpse takes no hits
                for e in self.enemies:
                    if p.collision_overlaps_ship(e.ship):
                        if not self._handle_ship_hit(p, e.pos):
                            break
                if self.game_over:
                    break

        # EACH player ship vs asteroid — swept hull polygon vs circle
        if self.protect_timer <= 0 and not self.game_over:
            for p in self.players:
                if p.dead:
                    continue   # a frozen corpse takes no hits
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
        # The ENTIRE render path reads the plain-data RenderModel, not live sim
        # state (the grep gate in test_m2b_local asserts no self.* live
        # reads — what makes the atomic model swap safe). The model is
        # built once per frame here when the caller doesn't supply one.
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
        for (tag, _id, pos, angle, _vel, _acc, _cr, _poly,
             s_dump, s_clock, flame_mags) in model["enemies"]:
            _draw_world_enemy(screen, self.cam, tag, pos, angle, standins,
                              s_dump, s_clock, flame_mags)
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
        # Laser beams from the model — drawn BEFORE the ships. Each beam is
        # anchored to its OWNER's interpolated pose (the host's sim runs
        # BOTH players, so model["beams"] holds the remote player's beams
        # too; anchoring them to the local ship made them look like the
        # local player fired). The end resolves the target by ship_id.
        local_rpos, local_rangle = _ship_pose(local_pack, model["step_alpha"])
        for beam in model["beams"]:
            owner = beam[6]
            if 0 <= owner < len(model["players"]):
                bpos, bang = _ship_pose(model["players"][owner],
                                        model["step_alpha"])
            else:
                # Owner not in the model this frame (shouldn't happen —
                # the sim always has every player): fall back to the
                # local pose rather than dropping the 0.15 s flash.
                bpos, bang = local_rpos, local_rangle
            _draw_world_beam(screen, self.cam, bpos, bang,
                             beam, enemies_by_id, standins)
        for pos, vel in model["enemy_bullets"]:
            _draw_world_bullet(screen, self.cam, pos, ENEMY_BULLET_COLOR)
        for pos, vel, color, life, max_life in model["particles"]:
            _draw_world_particle(screen, self.cam, pos, vel, color, life,
                                 max_life)
        # EVERY player ship, each with its own shield ring. The stand-in is
        # built lazily from that player's fixed hull + loadout; its
        # presentation is synced from the model pack.
        if not model["game_over"]:
            for i, pack in enumerate(model["players"]):
                # A dead ship DISAPPEARS until it respawns (not drawn).
                if pack["dead"]:
                    continue
                standin = self._get_standin(i)
                rpos, rangle = _draw_local_ship(screen, self.cam, standin,
                                                pack, model["step_alpha"])
                # A dead ship is a frozen wreck — no spawn-protection ring.
                if model["protect_timer"] > 0 and not pack["dead"]:
                    sh = self.shields[i]
                    ssx, ssy = self.cam.to_screen(rpos)
                    screen.blit(sh, (ssx - sh.get_width() // 2,
                                     ssy - sh.get_height() // 2))
        # Fog: the local stand-in must carry the local ship's current pose +
        # synced presentation. The ships loop sets these when it draws the
        # local ship, but on game_over the loop is skipped — so set them
        # explicitly here (idempotent when the loop already ran).
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
            # Single player (2P never sets the global flag): full game over.
            draw_game_over(screen, self.big_font, self.font)
        elif model["players"][self.local_index]["dead"]:
            # Per-player death — only the DEAD player sees the
            # respawn screen; the other player keeps playing (no screen).
            draw_respawn(screen, self.big_font, self.font)
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

    # --- networking: remote-interpolation render ---
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

        # Asteroids: fill + edge polygon at the interpolated center (the shape
        # and spin come from the latest snapshot; only the center is
        # interpolated).
        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        # Remote enemies as their REAL hulls. Buffer entry: (tag, x, y,
        # angle, vx, vy, id, shield_dump, shield_clock, power_used,
        # flame_mags).
        for (tag, x, y, ang, vx, vy, _eid, s_dump, s_clock, _p_used,
             flame_mags) in pos['enemies']:
            self._draw_remote_enemy_hull(screen, tag, x, y, ang,
                                         s_dump, s_clock, flame_mags)
        # Remote projectiles. The entry carries the missile id (mid) too —
        # unused here (remote_view has no ghost to dedup against).
        for (x, y, vx, vy, kind, owner, boost, _mid) in pos['bullets']:
            self._draw_remote_bullet(screen, x, y, vx, vy, kind, boost)
        # All player ships, by index, as their REAL hulls at the
        # interpolated (pos, angle). Entry: (x, y, angle, dead,
        # shield_dump, shield_clock, flame_mags, brownout, power_factor).
        # A dead ship is not drawn. The presentation fields are fed to the
        # ship (flash / exhaust / brownout crackle); _update_arcs(dt) is
        # driven here because the client never runs Ship.update for the
        # remote stand-in.
        for i, (x, y, ang, dead, s_dump, s_clock, flame_mags,
                brownout, pf) \
                in enumerate(pos['ships']):
            if dead:
                self.players[i].brownout = False
                self.players[i]._update_arcs(dt)
                continue
            self.players[i].shield_dump = s_dump
            self.players[i].shield_clock = s_clock
            self.players[i].flame_mags = dict(flame_mags)
            self.players[i].brownout = brownout
            self.players[i].power_factor = pf
            self.players[i]._update_arcs(dt)
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        draw_hud(screen, self.font, self.enemies, self.ship)
        return pos

    def push_snapshot(self, sim_time, snap, now=None, real_time=None, rate=1.0):
        """The single seam where a received authoritative snapshot enters
        the remote peer: record it in the interpolation buffer, then feed
        the local ship's prediction ghost — seed on the first snapshot,
        reconcile on every one after. The ghost is the LOCAL player's
        ship, so it takes snap[0][self.local_index].

        When `now` (the ghost's OWN current sim clock — the caller passes
        `ghost.sim_time`, advanced to this frame before the poll loop)
        is given, the reconcile is a REWIND — the ghost is set to the
        authoritative snapshot and the inputs the host applied since the
        snapshot are replayed (the prediction is rebuilt, not snapped).
        Reconciling at the ghost's own clock (not snap_time + rtt_ow)
        keeps the reconcile from jumping the clock forward. When `now`
        is None (tests / v1 fallback) it defaults to `sim_time`, so the
        replay span is 0 and this is a pure full snap."""
        self.snap_buf.push(sim_time, snap)
        local_s = snap[0][self.local_index]
        if not self.ghost.seeded:
            # Pass the snapshot's sim time so the ghost's physics clock
            # starts at the right value (the first reconcile_rewind uses
            # ghost.sim_time as `now`).
            self.ghost.seed(local_s, snap_time=sim_time, real_time=real_time,
                         rate=rate)
        else:
            # The rewind replay steps the ghost with the inputs the host
            # applied since the snapshot — and the host's replayed ticks
            # saw the live enemies, so the replay must see the buffer's
            # enemies too (the laser target is picked from them each
            # replayed step). Proxies come from the buffer at the CURRENT
            # render point; None (no window yet / bare Game) replays with
            # no targets.
            enemies = None
            rp = getattr(self, "render_point", None)
            if rp is not None:
                rp_now = rp.now()
                if rp_now is not None:
                    pos = self.snap_buf.positions_at(rp_now)
                    if pos is not None:
                        enemies = self._ghost_enemy_proxies(pos)
            # Measure the ghost's displacement ACROSS the reconcile (the
            # "snap size") for the net debug overlay + log: ~0 with
            # dead-reckoning rewind + identical input; large = the
            # prediction diverged from authority.
            _bx, _by = self.ghost.ship.pos.x, self.ghost.ship.pos.y
            # Ghost clocks captured BEFORE the reconcile (phase diagnosis;
            # CSV log). _last_step_time is the tick boundary the ghost's
            # sim pose sits on; (snap_time - _last_step_time) is the phase
            # offset the reconcile corrects, so it bounds snap_px.
            self.last_ghost_sim_time = self.ghost._sim_time
            self.last_ghost_last_step_time = self.ghost._last_step_time
            self.last_ghost_acc = self.ghost._acc
            self.last_ghost_phase_ticks = (
                sim_time - self.ghost._last_step_time) / STEP
            # RENDER pose before the reconcile — the player sees
            # render_pose, not ship.pos.
            _br, _ = self.ghost.render_pose()
            # Ghost's missile_seq BEFORE the rewind replay (the replay
            # re-runs the fire tick and can bump it again) and the
            # snapshot's authoritative seq — the two must agree for the
            # ids to line up.
            _seq_before = self.ghost.ship.missile_seq
            _local_before = [tuple(m.id) for m in self.ghost.local_missiles
                             if m.id is not None]
            # Ghost's bullet_seq BEFORE the rewind replay (the replay
            # re-runs the fire tick and can bump it again) — the Phase 0
            # drift measurement (plan_14). The host's authoritative
            # bullet_seq is local_s[24]; the gap (ghost - host) is the
            # seq drift the cap causes. The ghost's PREDICTED gun cooldown
            # (pre-reconcile) is captured too — the secondary drift signal
            # (the host's at-snapshot cooldown is local_s[18][gun][0]).
            _bseq_before = self.ghost.ship.bullet_seq
            _gun_idx = next((i for i, w in enumerate(self.ghost.ship.weapons)
                             if w.comp.bullet_speed > 0), None)
            _g_cd_pred = (self.ghost.ship.weapons[_gun_idx].cooldown
                          if _gun_idx is not None else 0.0)
            self.ghost.reconcile_rewind(
                local_s, sim_time, sim_time if now is None else now,
                enemies=enemies, real_time=real_time, rate=rate)
            self.last_snap_px = math.hypot(
                self.ghost.ship.pos.x - _bx, self.ghost.ship.pos.y - _by)
            # The render pose's displacement across the reconcile.
            _ar, _ = self.ghost.render_pose()
            self.last_render_snap_px = math.hypot(
                _ar.x - _br.x, _ar.y - _br.y)
            # Capture the replay length for the net-debug CSV (the
            # sawtooth diagnosis) — the whole-tick count the rewind replayed
            # + the raw (now - snap_time) span it was truncated from.
            self.last_replay_ticks = self.ghost._last_replay_ticks
            self.last_replay_span = self.ghost._last_replay_span
            # The steps the ghost took between the previous
            # reconcile and this one (the sawtooth diagnosis) — read +
            # reset the accumulator predicted_view feeds.
            self.last_advance_steps = self._dbg_steps_since_reconcile
            self._dbg_steps_since_reconcile = 0
            self.snap_count += 1
            _mtel.log("RECONCILE", sim_t=sim_time, r_snap_t=sim_time,
                      r_snap_seq=local_s[21], r_seq_before=_seq_before,
                      r_seq_after=self.ghost.ship.missile_seq,
                      r_buf_ids=_mtel.fmt_ids(
                          tuple(m[8]) for m in snap[4]
                          if m[8] is not None),
                      r_local_before=_mtel.fmt_ids(_local_before),
                      r_local_after=_mtel.fmt_ids(
                          tuple(m.id) for m in self.ghost.local_missiles
                          if m.id is not None))
            # TELEMETRY (Phase 0, plan_14): the BULLET fire-drift row. The
            # host's authoritative bullet_seq is local_s[24]; the ghost's is
            # _bseq_before (pre-replay) / .ship.bullet_seq (post-replay). The
            # PRIMARY drift signal is the seq gap (ghost - host): a monotonic
            # counter, so a growing gap = the ghost fired shots the host
            # didn't (its cap didn't bind while the host's did). The SECONDARY
            # signal is the gun cooldown at the snapshot's time: the host's
            # is local_s[18][gun][0] (at snap_time); the ghost's PREDICTED
            # value (_g_cd_pred, captured pre-reconcile) is the ghost's
            # estimate of the same instant. The gun is the weapon with
            # bullet_speed>0 (same loadout on both peers, so the index
            # matches).
            _host_cd = (local_s[18][_gun_idx][0]
                        if _gun_idx is not None else 0.0)
            _btel.log("RECONCILE", sim_t=sim_time, r_snap_t=sim_time,
                      r_snap_bseq=local_s[24],
                      r_gseq_before=_bseq_before,
                      r_gseq_after=self.ghost.ship.bullet_seq,
                      r_glocal_ids=_btel.fmt_ids(
                          tuple(b.id) for b in self.ghost.local_bullets
                          if b.id is not None),
                      r_buf_bids=_btel.fmt_ids(
                          tuple(b[6]) for b in snap[2]
                          if b[6] is not None),
                      r_gcd="%.4f" % _g_cd_pred,
                      r_host_cd="%.4f" % _host_cd)
            # Hand the ghost's OWN missiles back to the BUFFER. Two conditions:
            # (a) seq < the snapshot's authoritative missile_seq (ship
            # field 21), and (b) the buffer's NEWEST snapshot actually
            # carries a missile with that id. The seq alone is not
            # enough: the first snapshot to carry the missile arrives up
            # to a snapshot interval after the launch, and the render
            # point reaches it only ~INTERP_DELAY after THAT — culling on
            # seq alone deleted the ghost copy while the buffer's copy
            # was still in the future (the missile vanished, then
            # reappeared). The buffer's copy is the one that actually
            # hits (only the host's missile has collision; the ghost's
            # copy is presentation-only).
            _latest = self.snap_buf.latest_snapshot()
            # handback_missiles MARKS the ghost's missiles pending but
            # does NOT cull them — the ghost keeps drawing them until the
            # buffer's copy is VISIBLE (the render point reaches the
            # carrying snapshot, ~INTERP_DELAY later). predicted_view
            # performs the actual handoff on that frame: it removes the
            # pending missile from ghost.local_missiles (so the buffer
            # copy is no longer dedup-skipped) and seeds the decoupled
            # render offset (ghost_pos - interp_pos), which the decay
            # then smooths to the buffer's position. Drawn continuously —
            # no blink.
            self.ghost.handback_missiles(
                local_s[21],
                latest_missiles=_latest[4] if _latest is not None else None)
            # The ghost's OWN bullets, the same way (the lightweight
            # mirror). `local_s[24]` is the snapshot's authoritative
            # bullet_seq; `_latest[2]` is the newest snapshot's player-
            # bullet list (each entry's id at field 6). The ghost keeps
            # drawing a pending bullet until predicted_view sees the
            # buffer's copy ADVANCE, then hands it off (the buffer draws
            # it pulled forward by V*delay — the closed-form handoff).
            self.ghost.handback_bullets(
                local_s[24],
                latest_bullets=_latest[2] if _latest is not None else None)

    def predicted_view(self, dt, keys, host_time=None, real_time=None):
        """Draw the frame with the LOCAL ship taken from the prediction
        ghost instead of the sim (client-side prediction).

        Mirrors remote_view, except the local ship is the ghost: each frame
        the ghost is advanced with the local player's input at the sim's
        FIXED rate (`ghost.advance(dt, inp)` — a fixed-step accumulator,
        not one step per display frame), and the ghost's ship is drawn at
        its own (predicted) position. Remote entities (enemies, asteroids)
        come from the interpolation buffer at the RENDER POINT
        (`self.render_point.now()` — the newest ARRIVED snapshot's stamp
        minus the adaptive delay, chased at a bounded per-frame rate). The
        buffer's 'ships' entry is drawn for every player EXCEPT the local
        one (as its real hull at the interpolated (pos, angle)); the local
        ship's buffer entry is NOT drawn — it IS the local ship, now
        predicted.

        `host_time`: the caller's estimate of the host's sim clock. When
        given, it becomes self.sim_time — the client's sim_time is the
        HOST's clock as carried by the wire, not the display's wall clock —
        but the REMOTE render no longer reads it: the render point is
        anchored on the buffer's newest snapshot (netcode.RenderPoint),
        because a model of the host clock and the data disagree under
        jitter/loss. When None (tests), the clock is left as-is.

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

        # The render point + buffer window are computed BEFORE the ghost
        # advances, so the ghost can see the buffer's enemies while it
        # steps (its laser target is picked from them — the ghost must
        # fire a beam on the same tick the host does). The render point
        # is anchored on the buffer's newest ARRIVED snapshot (newest
        # stamp - adaptive delay, chased at a bounded per-frame rate —
        # netcode.RenderPoint). None until the first snapshot arrives.
        rp = self.render_point.now()
        if rp is None:
            # No snapshot has arrived yet: nothing to render from.
            return None
        pos = self.snap_buf.positions_at(rp)
        if pos is None:
            # The render point exists but the buffer holds no window yet
            # (fewer than two snapshots, or the point is before the first
            # snapshot): not enough data to interpolate the remote
            # entities.
            return None

        # Lightweight enemy proxies from the buffer's enemy entries (pos +
        # vel + angle + id + shield state). Fed to the ghost's advance
        # (laser target) and reused below for the targeting reticle — one
        # build per frame.
        ghost_enemies = self._ghost_enemy_proxies(pos)

        inp = ShipInput.from_keys(keys)
        # Pass the client's real time + the HostTimeEstimator + the
        # RTT-measured one-way latency so the ghost's sim clock is the
        # HOST's clock (host_time.now(real_time) + rtt_ow), not the
        # client's real time — the ghost's step grid is locked to the
        # host's tick grid, so the reconcile's phase offset is 0. See
        # PredictedShip.advance.
        # On the LIVE client the client loop already advanced the ghost
        # to this frame BEFORE the poll loop (so the reconcile sees a
        # current-frame clock — a 1-frame-stale clock caused the ~8 px
        # snap), so this advance is a no-op on the step grid (the
        # branch-2 clock is a pure function of real_time, steps are
        # boundary-based). It still runs for TESTS, which call
        # predicted_view directly and rely on it to step the ghost.
        # Accumulate the step count across frames (the reconcile happens
        # earlier in the frame, before this advance); push_snapshot reads
        # + resets the accumulator, so last_advance_steps = the steps the
        # ghost took between the previous reconcile and this one.
        self._dbg_steps_since_reconcile += self.ghost.advance(
            dt, inp, enemies=ghost_enemies,
            real_time=real_time,
            host_time=getattr(self, "host_time", None),
            rtt_ow=getattr(self, "_rtt_ow", 0.0))
        # Frame time of the frame that also ran the reconcile (the log
        # line is written in this same frame) — the snap's timing context.
        self.last_frame_dt = dt
        # The ghost's SMOOTHED render pose (the decoupled visual layer —
        # render_pose_smoothed chases the unsmoothed render pose at a
        # bounded rate, so a burst frame's 2-tick jump is spread over 2
        # frames instead of lurching the ship). The local ship + its
        # beams + the camera are drawn at this pose so they move
        # together and glide instead of stepping in 60 Hz jumps. The
        # unsmoothed render_pose() is still used for the reconcile
        # diagnostics (last_render_snap_px) — the phase-lock is
        # measured on the unsmoothed pose, and the smoothed pose is a
        # pure render helper (it never touches the physics). The fog
        # still reads self.ghost.ship.pos (the current pose) — exactly
        # like the host's _draw_local_ship.
        gpos, gangle = self.ghost.render_pose_smoothed(dt)
        # Advance + cull the client-side death bursts (presentation).
        self._step_death_bursts(dt)
        # Age + cull the remote player's laser beams (T_BEAM events).
        self._step_remote_beams(dt)

        screen = self.screen
        self.cam.update(dt, gpos, self.ghost.ship.vel,
                        self.ghost.ship.dampening)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)

        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        # Remote player's laser beams — T_BEAM events (one per discharge),
        # drawn before the ships. Entry: [local_start, end, age, ttl, owner];
        # local_start is the hull-local muzzle offset, end the world endpoint.
        # The origin is re-anchored to the firing ship's INTERPOLATED pose
        # each frame (the muzzle is frozen at fire time, so a dead ship's
        # beam anchors to its death pose). Presentation only.
        for (local_start, end, age, ttl, owner) in self.remote_beams:
            if not (0 <= owner < len(pos['ships'])):
                # Firing ship not in the buffer this frame (index out of
                # range): no pose to anchor to — skip the beam (it is a
                # 0.15 s flash; a missing frame is imperceptible).
                continue
            _sx, _sy, _sang = pos['ships'][owner][:3]
            rpos = pygame.Vector2(_sx, _sy)
            rangle = _sang
            fwd = pygame.Vector2(math.cos(rangle), math.sin(rangle))
            right = pygame.Vector2(-fwd.y, fwd.x)
            start = rpos + fwd * local_start[0] + right * local_start[1]
            fade = 1.0 - age / ttl
            c = tuple(int(ch * fade) for ch in LASER_COLOR)
            pygame.draw.line(screen, c, self.cam.to_screen(start),
                             self.cam.to_screen(end), 2)
        # Remote enemies as their REAL hulls: the buffer carries
        # (tag, x, y, angle, vx, vy, id, shield_dump, shield_clock,
        # power_used, flame_mags); the angle is lerp'd with lerp_angle and
        # the hull is fixed on both peers, so the per-tag stand-in draws it
        # faithfully. Shield state + flame mags feed the stand-in so the
        # impact flash and exhaust render (mirrors the host's model path).
        for (tag, x, y, ang, vx, vy, _eid, s_dump, s_clock, _p_used,
             flame_mags) in pos['enemies']:
            self._draw_remote_enemy_hull(screen, tag, x, y, ang,
                                         s_dump, s_clock, flame_mags)
        # Remote projectiles: every bullet and missile in the buffer, at its
        # interpolated position. DEDUP: the buffer also carries the LOCAL
        # player's own missiles (the host's snapshot includes them ~100 ms
        # after the ghost fired them); a buffer copy whose id matches a
        # ghost missile is SKIPPED — the ghost draws it (predicted,
        # immediate). The id is a (player_index, seq) tuple; the buffer's
        # copy arrives via JSON as a LIST, so tuple() normalizes both sides.
        # Hand off PENDING ghost missiles to the buffer once the buffer's
        # copy is visible AND advancing (a new buffer missile is FROZEN for
        # its pop-in window — handing off on the first visible frame would
        # stall it). On the handoff frame: remove the missile from
        # local_missiles (so the buffer copy is no longer dedup-skipped)
        # and record the ghost's position — the render offset is computed
        # against the interpolated position on the first buffer draw. Must
        # run BEFORE local_missile_ids is built.
        _pending = [m for m in self.ghost.local_missiles if m.pending]
        if _pending:
            # Hand off only when the buffer's copy is MOVING, not merely
            # visible: a new buffer missile pops in at its curr position
            # and is FROZEN for the rest of that window (no earlier
            # snapshot to lerp from), so handing off on the first visible
            # frame would stall it for up to a snapshot interval. _hbuf
            # tracks the buffer's position per id across frames; the first
            # ADVANCING frame is the handoff frame. _hbuf is seeded during
            # the frozen frames so the decay cap below sees a real
            # displacement from the handoff frame on.
            _hbuf = self._missile_handoff_buf_pos
            _hgpos = self._missile_handoff_ghost_pos
            _buf_now = {tuple(b[7]): (b[0], b[1]) for b in pos['bullets']
                        if b[4] == 'missile' and b[7] is not None}
            _kept = []
            for m in self.ghost.local_missiles:
                _mid = tuple(m.id)
                _bp = _buf_now.get(_mid)
                if m.pending and _bp is not None:
                    _prev = _hbuf.get(_mid)
                    _hbuf[_mid] = _bp
                    if _prev is not None and _prev != _bp:
                        # The buffer's copy has ADVANCED since it first
                        # appeared — the pop-in freeze is over. Hand off.
                        _hgpos[_mid] = (m.pos.x, m.pos.y)
                        _mtel.log("HANDBACK", h_host_seq="",
                                  h_buf_ids="", h_before="", h_after="",
                                  h_culled=_mtel.fmt_ids([_mid]),
                                  h_jump="")
                        continue
                _kept.append(m)
            self.ghost.local_missiles = _kept
        local_missile_ids = {tuple(m.id) for m in self.ghost.local_missiles}
        # BULLET HANDOFF — the lightweight mirror of the missile handoff above,
        # minus the offset/decay/cap (a bullet is a straight line, so the
        # handoff is closed-form). Hand off a PENDING ghost bullet on the
        # frame the buffer's copy has ADVANCED (not merely visible — a new
        # buffer bullet is FROZEN for its pop-in window). Remove the bullet
        # from local_bullets (so the buffer copy is no longer dedup-skipped)
        # — the buffer then draws it PULLED FORWARD by V*delay
        # (buffer_pos + V*delay == ghost_pos, a no-op visually). Must run
        # BEFORE local_bullet_ids is built.
        _bbuf = self._bullet_handoff_buf_pos
        _bnow = {tuple(b[7]): (b[0], b[1]) for b in pos['bullets']
                 if b[4] == 'player' and b[7] is not None}
        _bkept = []
        for b in self.ghost.local_bullets:
            _bid = tuple(b.id) if b.id is not None else None
            _bp = _bnow.get(_bid)
            if b.pending and _bp is not None:
                _bprev = _bbuf.get(_bid)
                _bbuf[_bid] = _bp
                if _bprev is not None and _bprev != _bp:
                    # The buffer's copy has ADVANCED since it first
                    # appeared — the pop-in freeze is over. Hand off.
                    _mtel.log("HANDBACK", h_host_seq="",
                              h_buf_ids="", h_before="", h_after="",
                              h_culled=_mtel.fmt_ids([_bid]),
                              h_jump="")
                    continue
            _bkept.append(b)
        self.ghost.local_bullets = _bkept
        local_bullet_ids = {tuple(b.id) for b in self.ghost.local_bullets
                            if b.id is not None}
        # Pull-forward magnitude: the buffer's copy is drawn V*delay ahead of
        # its interpolated position so it lands where the ghost was
        # drawing it (buffer_pos + V*delay == ghost_pos). `delay` is the
        # client's current adaptive delay (render point = newest - delay).
        _bdelay = self.latency.delay
        # Missile render offset (the handback fix): the buffer's copy of a
        # handed-off missile is INTERP_DELAY seconds BEHIND the ghost's
        # predicted position, so drawing it at its interpolated position
        # would make it JUMP BACKWARD. On the FIRST frame the buffer's
        # copy is drawn, compute the offset as (ghost_pos - interp_pos)
        # and add it to the rendering; each subsequent frame decay it by
        # MISSILE_HANDOFF_DECAY (frame-rate independent: decay^(dt*60))
        # until it drops below MISSILE_HANDOFF_OFFSET_EPS.
        # _missile_handoff_ghost_pos: {mid: (gx, gy)} — the ghost's
        # position at the handoff frame (set above), consumed on the
        # first buffer draw. _missile_handoff_offsets: {mid: (dx, dy)} —
        # the computed offset, decayed each frame.
        _hgo = self._missile_handoff_offsets
        _hgpos = self._missile_handoff_ghost_pos
        _hbuf = self._missile_handoff_buf_pos
        # The buffer's per-missile position THIS frame (drives the
        # displacement-based shrinkage cap below).
        _buf_pos = {}
        for _b in pos['bullets']:
            if _b[4] == 'missile' and _b[7] is not None:
                _buf_pos[tuple(_b[7])] = (_b[0], _b[1])
        # Decay existing offsets (frame-rate independent), CAPPED by the
        # buffer's ACTUAL displacement this frame so the rendered position
        # (buffer_pos + offset) NEVER moves backward (see
        # MISSILE_HANDOFF_MAX_SHRINK in config): the plain decay shrinks
        # the offset faster than the missile advances once INTERP_DELAY is
        # large, so the missile would visibly reverse. Capping the
        # shrinkage at a fraction of the buffer's OWN advance guarantees
        # the net rendered motion is always forward. It also handles the
        # POP-IN frames: a buffer missile that just appeared is FROZEN for
        # the first interval (B_disp = 0), so the cap freezes the offset
        # too (no backward drift while the buffer is stationary).
        if _hgo:
            _decay = MISSILE_HANDOFF_DECAY ** (dt * 60.0)
            for _mid in list(_hgo):
                _dx, _dy = _hgo[_mid]
                _mag = math.hypot(_dx, _dy)
                if _mag < MISSILE_HANDOFF_OFFSET_EPS:
                    del _hgo[_mid]
                    _hbuf.pop(_mid, None)
                    continue
                _bp = _buf_pos.get(_mid)
                if _bp is not None:
                    _prev = _hbuf.get(_mid)
                    _hbuf[_mid] = _bp
                    if _prev is not None:
                        _b_disp = math.hypot(_bp[0] - _prev[0],
                                             _bp[1] - _prev[1])
                        _shrink = min(_mag * (1.0 - _decay),
                                      MISSILE_HANDOFF_MAX_SHRINK * _b_disp)
                    else:
                        # First frame the buffer's copy is tracked (the
                        # pop-in frame): it is frozen (no earlier
                        # position), so freeze the offset too (no
                        # backward drift while the buffer is stationary).
                        _shrink = 0.0
                else:
                    # The buffer's copy left the window this frame: no
                    # displacement to cap against — use the plain decay
                    # (the offset is about to be culled anyway).
                    _shrink = _mag * (1.0 - _decay)
                _s = 1.0 - _shrink / _mag
                _dx *= _s
                _dy *= _s
                if math.hypot(_dx, _dy) < MISSILE_HANDOFF_OFFSET_EPS:
                    del _hgo[_mid]
                    _hbuf.pop(_mid, None)
                else:
                    _hgo[_mid] = (_dx, _dy)
        # TELEMETRY: the DEDUP ground truth — which buffer projectiles were
        # DRAWN vs SKIPPED this frame, against the ghost's in-flight ids.
        # log_dedup writes a row only when the triple CHANGES, so a
        # sustained flight logs a handful of transition rows, not 60 Hz.
        _buf_drawn = []
        _buf_skipped = []
        # Bullet-only DEDUP trackers (Phase 0, plan_14): a double-draw is a
        # ghost bullet whose id has NO counterpart in the buffer (the ghost
        # fired a shot the host's cap suppressed, so its id drifted from the
        # host's — the buffer draws the host's version under a different id,
        # and the ghost draws its own). The analyzer flags ghost bullets not
        # in (buf_bdrawn ∪ buf_bskipped).
        _buf_bdrawn = []
        _buf_bskipped = []
        for (x, y, vx, vy, kind, owner, boost, mid) in pos['bullets']:
            if kind == 'missile' and mid is not None \
                    and tuple(mid) in local_missile_ids:
                _buf_skipped.append(tuple(mid))
                continue   # drawn from the ghost (predicted) instead
            # The ghost's OWN gun shots — same dedup as missiles. The id's
            # first element is the FIRING player's index, so:
            #   * a bullet the ghost is still drawing (id in
            #     local_bullet_ids) is SKIPPED (the ghost draws it);
            #   * a LOCAL bullet the ghost has handed off (id[0] ==
            #     local_index, id NOT in local_bullet_ids) is drawn PULLED
            #     FORWARD by V*delay (buffer_pos + V*delay == ghost_pos);
            #   * the REMOTE player's bullets (id[0] != local_index) are
            #     drawn at their raw interpolated position (no pull).
            # Enemy bullets (mid is None) are drawn as-is.
            if kind == 'player' and mid is not None:
                _bid_t = tuple(mid)
                if _bid_t in local_bullet_ids:
                    _buf_skipped.append(_bid_t)
                    _buf_bskipped.append(_bid_t)
                    continue   # drawn from the ghost (predicted) instead
                _buf_drawn.append(_bid_t)
                _buf_bdrawn.append(_bid_t)
                # (plan_15: the local-bullet pull-forward is GONE — the
                # ghost now draws its own bullets at the render point, so
                # the buffer's handed-off copy is already seamless here.)
            if kind == 'missile' and mid is not None:
                _mid_t = tuple(mid)
                _buf_drawn.append(_mid_t)
                # Apply the decoupled render offset. On the FIRST
                # frame the buffer's copy is drawn, compute the offset
                # from the ghost's handoff position minus the
                # interpolated position. Subsequent frames use the
                # stored (decaying) offset.
                if _mid_t in _hgpos:
                    _gx, _gy = _hgpos.pop(_mid_t)
                    _hgo[_mid_t] = (_gx - x, _gy - y)
                    x += _hgo[_mid_t][0]
                    y += _hgo[_mid_t][1]
                elif _mid_t in _hgo:
                    x += _hgo[_mid_t][0]
                    y += _hgo[_mid_t][1]
            self._draw_remote_bullet(screen, x, y, vx, vy, kind, boost)
        if (local_missile_ids or local_bullet_ids or _buf_drawn
                or _buf_skipped):
            _mtel.log_dedup(self.sim_time,
                            local_missile_ids | local_bullet_ids,
                            _buf_drawn, _buf_skipped)
            # TELEMETRY (Phase 0, plan_14): the BULLET-only DEDUP row
            # (ghost in-flight bullet ids vs the buffer's drawn/skipped
            # bullet ids). A ghost id absent from both is a double-draw
            # candidate (the ghost fired a shot the host's cap suppressed).
            _btel.log_dedup(self.sim_time, local_bullet_ids,
                            _buf_bdrawn, _buf_bskipped)
        # Remote ships: every player EXCEPT the local one, from the buffer by
        # index (ships matched by index; the local ship is the ghost,
        # drawn below). Each is drawn as its REAL hull at the
        # interpolated (pos, angle) — the remote hull is known from the
        # join/welcome handshake, so self.players[i] IS the remote
        # ship's hull/loadout.
        # Buffer entry: (x, y, angle, dead, shield_dump, shield_clock,
        # flame_mags, brownout, power_factor). A DEAD remote ship is NOT
        # drawn (it disappears until it respawns) — and on the
        # alive->dead transition the client spawns its OWN explosion
        # burst (the client never runs the sim, so it never sees the
        # host's authoritative `burst()`). Shield state + flame mags are
        # fed to the stand-in so the impact flash and exhaust render
        # (mirrors the host's model path).
        # Brownout arcs: the client never runs Ship.update for the remote
        # stand-in, so we drive _update_arcs(dt) ourselves each frame
        # (the host's sim does this every tick). The arcs are random
        # presentation (never synced — the snapshot has no arcs field),
        # so the client generates its OWN bolts, scaled by
        # (1 - power_factor). _update_arcs is called in BOTH branches:
        # live -> spawns + ages arcs; dead -> only ages them out (a dead
        # ship stops spawning but its last bolts fade). Presentation
        # only: no collision, no sim feedback.
        for i, (x, y, ang, dead, s_dump, s_clock, flame_mags,
                brownout, pf) \
                in enumerate(pos['ships']):
            if i == self.local_index:
                continue
            if dead:
                if not self.players[i].dead:
                    self._death_burst(i, pygame.Vector2(x, y))
                self.players[i].dead = True
                self.players[i].brownout = False
                self.players[i]._update_arcs(dt)
                continue
            self.players[i].dead = False
            self.players[i].shield_dump = s_dump
            self.players[i].shield_clock = s_clock
            self.players[i].flame_mags = dict(flame_mags)
            self.players[i].brownout = brownout
            self.players[i].power_factor = pf
            self.players[i]._update_arcs(dt)
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        # The player's OWN laser beams — the ghost's predicted beams, drawn
        # BEFORE the local ship (mirrors the host's draw() order). Each
        # entry is [local_start, end, age, ttl]: local_start is the
        # HULL-LOCAL muzzle offset, end the world-space endpoint.
        # The origin is RE-ANCHORED to the ghost's CURRENT pose each
        # frame (the same math the host's _draw_world_beam uses): the
        # ghost's Beam carries the world-space muzzle at fire time, so
        # drawing it directly would leave the origin behind in empty
        # space as the ghost moved during the 0.15 s flash. The ghost's
        # pose (gpos/gangle — the INTERPOLATED render pose) is the same
        # pose the local ship is drawn at below, so the beam origin stays
        # glued to the drawn ship. Presentation only.
        gfwd = pygame.Vector2(math.cos(gangle), math.sin(gangle))
        gright = pygame.Vector2(-gfwd.y, gfwd.x)
        for (local_start, end, age, ttl) in self.ghost.local_beams:
            start = gpos + gfwd * local_start[0] + gright * local_start[1]
            fade = 1.0 - age / ttl
            c = tuple(int(ch * fade) for ch in LASER_COLOR)
            pygame.draw.line(screen, c, self.cam.to_screen(start),
                             self.cam.to_screen(end), 2)

        # The LOCAL ship, from the ghost (its own flame_mags render).
        # The ghost is FROZEN while dead (advance() takes 0 steps), so it
        # does not snap. It is NOT drawn while dead (it disappears until
        # it respawns) — and on the alive->dead transition the client
        # spawns its OWN explosion burst at the death pose.
        if self.ghost.ship.dead:
            if not self._local_was_dead:
                self._death_burst(self.local_index, pygame.Vector2(gpos))
            self._local_was_dead = True
        else:
            self._local_was_dead = False
            # Draw at the INTERPOLATED render pose (gpos/gangle) — the same
            # sub-step interpolation the host's local ship gets, so the
            # client's ship glides instead of stepping in 60 Hz jumps.
            # (The raw self.ghost.ship.pos is the post-step pose.)
            self.ghost.ship.draw(screen, self.cam, gpos, gangle)

        # The player's OWN gun shots: the ghost's predicted bullets, drawn
        # at the RENDER POINT (rp) — NOT at the ghost's _last_step_time —
        # so they are phase-locked to the enemy's time domain and strike
        # the enemy where it is DRAWN (the pass-through / early-disappear
        # fix, plan_15). The bullet's stored pos is at _last_step_time;
        # D is the exact gap to the render point. t_off clamps so a bullet
        # younger than D renders at the muzzle (it had not left yet at the
        # render point). Drawn the same way the host draws the local
        # player's bullets (a BULLET_COLOR dot). Presentation only: no
        # collision, no sim feedback.
        _D = self.ghost._last_step_time - rp
        for b in self.ghost.local_bullets:
            _a = BULLET_LIFE - b.life
            _t_off = min(_a, max(_D, 0.0))
            _rx = b.pos.x - b.vel.x * _t_off
            _ry = b.pos.y - b.vel.y * _t_off
            s = self.cam.to_screen(pygame.Vector2(_rx, _ry))
            pygame.draw.circle(screen, BULLET_COLOR, (int(s.x), int(s.y)), 3)
        # The player's OWN missiles — the ghost's predicted missiles, drawn
        # immediately at their predicted (homing) positions so the player
        # sees their fire the instant they launch — not ~100 ms later when
        # the host's next snapshot arrives. Drawn with the host's missile
        # visual (body line + nose + boost exhaust flicker). The buffer's
        # copy of each of these is SKIPPED above (dedup by id), so the
        # same missile never appears twice. Presentation only.
        for m in self.ghost.local_missiles:
            _draw_world_missile(screen, self.cam,
                                (m.pos.x, m.pos.y),
                                (m.vel.x, m.vel.y), m.boost, m.life)

        # Targeting reticle: the client has no enemy list
        # to target (it never runs the sim), so run the SAME lead math
        # the host uses against the buffer's enemy proxies. The reticle
        # leads from the GHOST (where the player is), not from the stale
        # players[0]. The proxies are the SAME ones built above
        # for the ghost's laser target (one build per frame — they are
        # also the fog's reticle-light sources below).
        proxies = (ghost_enemies
                   if TARGETING_ASSIST and self.ghost.ship.targeting_on
                   else [])
        for e in proxies:
            self._draw_lead(screen, e, self.ghost.ship)

        # Fog of war (predicted_view previously skipped it
        # entirely, so fire did not glow through the dark like on the
        # host). The light bubble follows the ghost (the local ship);
        # the lights are the interpolated bullets/missiles plus a reticle
        # light at each lead point (mirrors _build_lights' guard so the
        # light and the reticle appear/disappear together).
        lights = self._remote_fog_lights(pos, self.ghost.ship, local_missile_ids,
                                 handoff_offsets=self._missile_handoff_offsets,
                                 local_bullet_ids=local_bullet_ids,
                                 delay=_bdelay,
                                 local_index=self.local_index)
        # The player's OWN gun shots glow too — mirrors the
        # host's _build_lights, which adds a LightSource per player bullet.
        # Offset by the SAME render-point shift as the sprite (plan_15) so
        # the glow follows the bullet.
        for b in self.ghost.local_bullets:
            _a = BULLET_LIFE - b.life
            _t_off = min(_a, max(_D, 0.0))
            lights.append(LightSource(
                pygame.Vector2(b.pos.x - b.vel.x * _t_off,
                               b.pos.y - b.vel.y * _t_off), 30, 0.5))
        # The player's OWN missiles glow too (the buffer's copy of
        # each was skipped above, so light them from the predicted pos).
        for m in self.ghost.local_missiles:
            lights.append(LightSource(m.pos, 30, 0.5))
        for e in proxies:
            p = e.lead_point(self.ghost.ship.pos, BULLET_SPEED,
                             TARGETING_USE_ACCEL)
            if p is not None and (p - self.ghost.ship.pos).length() \
                    <= TARGETING_RANGE:
                lights.append(LightSource(p, 50, 0.6))
        draw_fog(screen, self.ghost.ship, self.cam, self.light_tex,
                 self.fog_surf, self.light_surf, lights)

        # The client-side death explosions, drawn AFTER the fog so
        # they are visible (the host draws its authoritative particles
        # before the fog, but the client's fog would otherwise hide a
        # burst behind the local light bubble).
        self._draw_death_bursts(screen, self.cam)

        # The ghost's scan pulse ring + sensor contacts, drawn
        # ABOVE the fog — the SAME two calls the host's model path makes
        # (fog -> scan pulse -> contacts -> HUD). The ghost's contacts
        # are built every step from the buffer's enemy proxies (the
        # proxy's .ship carries power_used from the buffer +
        # power_idle_total derived from the tag's loadout — the same
        # signature the host's _update_contacts reads off the live
        # enemy), so given the same sensor state the client sees the
        # same blips/arrows the host does. The pulse ring is the
        # expanding circle on a G press (Ship._draw_scan_pulse, driven
        # by the ghost's scan_pulse).
        if not self.ghost.ship.dead:
            # The pulse ring + contacts anchor on the INTERPOLATED
            # render pose (gpos) — the same pose the ship is drawn at.
            self.ghost.ship._draw_scan_pulse(screen, self.cam, gpos)
            _draw_sensor_contacts_model(screen, self.cam, self.font,
                                        gpos, self.ghost.ship.contacts)

        # HUD: the LOCAL (ghost) ship's power/shield/velocity, and the real
        # (interpolated) enemy count from the buffer. self.ship is players[0]
        # (the HOST's ship on a client) and self.enemies is the client's stale
        # initial spawn (the client never runs the sim), so both would be
        # wrong here.
        draw_hud(screen, self.font, pos['enemies'], self.ghost.ship)
        # Per-player death — the dead player sees the respawn screen
        # (the ghost is frozen while dead, so the world keeps rendering
        # behind it and the other player's ship is still visible + moving).
        # The OTHER player does not see it (their ghost is alive).
        if self.ghost.ship.dead:
            draw_respawn(screen, self.big_font, self.font)
        return pos

    # --- full remote rendering helpers -----------------------------------------
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
        """The presentation stand-in for player `i` ),
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

    def _draw_remote_enemy_hull(self, screen, tag, x, y, ang,
                            shield_dump=0.0, shield_clock=0.0,
                            flame_mags=None):
        """Draw a remote enemy as its REAL hull at the interpolated
        (pos, angle) (replaces the 10 px dot, the D4 defect). The
        stand-in's Ship.draw takes explicit pos/angle, so the stand-in's
        own (never-stepped) state is irrelevant.

        `shield_dump` / `shield_clock` (from the buffer's enemy entry,
        lerp'd across the window) are fed to the stand-in before the
        draw, so Ship._draw_shield renders the enemy's shield-impact
        flash — the same blue->white glow the host shows.

        `flame_mags` (from the buffer's enemy entry, the CURRENT
        snapshot — not lerp'd) is fed the same way, so Ship.draw renders
        the enemy's thruster exhaust — the same flames the host's model
        path shows. None (or an empty dict) draws no exhaust."""
        e = self._get_remote_enemies().get(tag)
        if e is None:
            # Unknown tag (e.g. 'test'): fall back to the coarse dot so
            # an unexpected enemy type still renders something.
            self._draw_remote_enemy(screen, x, y)
            return
        e.ship.shield_dump = shield_dump
        e.ship.shield_clock = shield_clock
        e.ship.flame_mags = dict(flame_mags) if flame_mags else {}
        e.ship.draw(screen, self.cam, pygame.Vector2(x, y), ang,
                    fill=e.hull.fill or ENEMY_FILL,
                    edge=e.hull.edge or ENEMY_EDGE,
                    flame_out=ENEMY_FLAME, flame_in=ENEMY_FLAME)

    def _draw_remote_bullet(self, screen, x, y, vx, vy, kind, boost):
        """Draw one remote projectile at its interpolated position
        (the D3 defect: the client rendered no bullets at
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

    def _remote_fog_lights(self, pos, local_ship, local_missile_ids=(),
                       handoff_offsets=None, local_bullet_ids=(),
                       delay=0.0, local_index=0):
        """LightSources for the client's fog of war (
        predicted_view previously skipped draw_fog entirely, so fire did
        not glow through the dark like on the host). Mirrors
        `_build_lights` for the data the client actually has: the
        interpolated bullets/missiles (the buffer's 'bullets' entries)
        and the local (ghost) ship's position. The targeting-reticle
        lights are added by the caller (they need the proxy's
        lead_point, which the caller already computes for the reticle).
        `local_ship` is the ghost's ship (the client's light bubble
        follows the player, not the stale players[0]).

        `local_missile_ids` (a set of the ghost's own missile ids)
        is used to SKIP the buffer's copy of the local player's own
        missiles — those glow from the ghost's predicted position instead
        (added by the caller), so the same missile does not light twice.

        `handoff_offsets` ({mid: (dx, dy)}) is the decoupled render
        offset for missiles just handed back to the buffer. The fog light
        is placed at the same offset position as the missile sprite, so
        the glow follows the missile smoothly during the handback
        transition.

        `local_bullet_ids` (a set of the ghost's own bullet ids) is
        used to SKIP the buffer's copy of the local player's own gun
        shots — those glow from the ghost's predicted position instead
        (added by the caller), so the same bullet does not light twice.
        A LOCAL bullet the ghost has handed off (its id NOT in
        local_bullet_ids, id[0] == local_index) is lit at its RAW
        interpolated position (plan_15 removed the pull-forward — the
        ghost draws its own bullets at the render point, so the
        buffer's handed-off copy is already seamless). The REMOTE
        player's bullets (id[0] != local_index) are lit at their raw
        position.
        `local_index` is the local player's index (the id's first element
        is the firing player's index, so it distinguishes local from
        remote)."""
        lights = []
        for (x, y, vx, vy, kind, owner, boost, mid) in pos['bullets']:
            if kind == 'missile' and mid is not None \
                    and tuple(mid) in local_missile_ids:
                continue   # glows from the ghost's predicted pos instead
            # The ghost's OWN gun shots — same skip as missiles. A bullet the
            # ghost is still drawing is skipped (it glows from the ghost's
            # predicted pos); a LOCAL bullet the ghost has handed off is
            # lit PULLED FORWARD by V*delay (the closed-form handoff — the
            # same pull the sprite gets). The REMOTE player's bullets
            # (id[0] != local_index) are lit at their raw position (no
            # pull — they are not predicted by the ghost).
            if kind == 'player' and mid is not None:
                _bid_t = tuple(mid)
                if _bid_t in local_bullet_ids:
                    continue   # glows from the ghost's predicted pos
                # (plan_15: the local-bullet pull-forward is GONE — the
                # ghost now draws its own bullets at the render point, so
                # the buffer's handed-off copy is already seamless here.)
            # Apply the handoff offset to the fog light (same
            # position as the missile sprite).
            if kind == 'missile' and mid is not None and handoff_offsets:
                _off = handoff_offsets.get(tuple(mid))
                if _off is not None:
                    x += _off[0]
                    y += _off[1]
            if kind == 'player':
                lights.append(LightSource(pygame.Vector2(x, y), 30, 0.5))
            elif kind == 'enemy':
                lights.append(LightSource(pygame.Vector2(x, y), 24, 0.4))
            else:   # missile
                lights.append(LightSource(pygame.Vector2(x, y), 30, 0.5))
        return lights

    # --- client-side death explosions --------------------------------------------
    #
    # The client never runs the sim, so it never sees the host's `burst()`
    # (the authoritative explosion particles). When a ship flips alive->dead
    # in the buffer (the local ghost, or a remote ship), the client spawns
    # its OWN presentation burst at the death pose so the ship "explodes"
    # the way it does in single player, then disappears (the dead ship is
    # not drawn until it respawns). These are pure presentation: they are
    # stepped/drawn only in predicted_view and never fed back into the sim.

    def _death_burst(self, index, pos):
        """Spawn a client-side explosion burst for ship `index` at world
        `pos` (a Vector2). Mirrors the host's death `burst(..., big=True)`
        (particles.py) so the client's explosion looks like the host's.
        Only the LOCAL peer (the client) calls this — the host already has
        the authoritative particles."""
        from .particles import Particle
        rng = random
        n = 30
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(40, 160) * 1.5
            vel = pygame.Vector2(math.cos(a) * speed, math.sin(a) * speed)
            self._death_bursts.setdefault(index, []).append(
                Particle(pos, vel, rng.choice(PARTICLE_COLORS),
                         rng.uniform(0.4, 0.9)))

    def _step_remote_beams(self, dt):
        """Age + cull the remote player's laser beams (presentation
        only). Called once per frame from predicted_view. Mirrors
        PredictedShip.step_local_beams (the ghost's OWN beams) and the
        host's `for b in self.beams: b[4] += dt` + ttl cull: each entry is
        [start, end, age, ttl]; age advances by dt and an entry is dropped
        once age >= ttl (the 0.15 s beam flash)."""
        if not self.remote_beams:
            return
        for b in self.remote_beams:
            b[2] += dt
        self.remote_beams = [b for b in self.remote_beams if b[2] < b[3]]

    def _step_death_bursts(self, dt):
        """Advance + cull the client-side death bursts (presentation only).
        Called once per frame from predicted_view."""
        if not self._death_bursts:
            return
        for index in list(self._death_bursts):
            parts = self._death_bursts[index]
            for p in parts:
                p.update(dt)
            parts = [p for p in parts if p.life > 0]
            if parts:
                self._death_bursts[index] = parts
            else:
                del self._death_bursts[index]

    def _draw_death_bursts(self, screen, cam):
        """Draw the client-side death bursts (mirrors _draw_world_particle).
        Called from predicted_view AFTER the fog so the explosion is visible
        (the host draws its particles before the fog, but the client's fog
        would otherwise hide a burst behind the local light bubble)."""
        for parts in self._death_bursts.values():
            for p in parts:
                _draw_world_particle(screen, cam,
                                     (p.pos.x, p.pos.y),
                                     (p.vel.x, p.vel.y),
                                     p.color, p.life, p.max_life)

    def _ghost_enemy_proxies(self, pos):
        """Lightweight enemy proxies from the buffer's enemy entries for
        the prediction ghost's weapon targeting.

        Each buffer enemy entry is (tag, x, y, angle, vx, vy, id,
        shield_dump, shield_clock). One _GhostEnemyProxy per
        entry, built fresh each frame — presentation only, never stepped,
        never fed to the sim. The proxy exposes `.pos` (the ghost's laser
        target math reads only that — `Ship._in_wedge` /
        `_pick_laser_target`), `.id` (track a specific enemy
        across frames), and a `.ship` stand-in with the power fields the
        sensor-contact signature needs.

        The power fields ARE in the buffer now. `power_used` is
        carried on the enemy entry (ship_s[22], lerp'd across the window
        — see interp_positions); `power_idle_total` is a LOADOUT CONSTANT
        the client derives from the tag's stand-in ship (the enemy
        hull+loadout is fixed on both peers — the same stand-in
        `_build_remote_enemy_standins` builds for the remote render). The
        host's `_update_contacts` reads `e.ship.power_used -
        e.ship.power_idle_total`; the ghost's `_update_contacts` reads the
        same off these proxies, so the signature matches.
        """
        standins = self._get_remote_enemies()
        proxies = []
        for (tag, x, y, ang, vx, vy, eid, _s_dump, _s_clock, power_used,
             _flame_mags) in pos['enemies']:
            e = standins.get(tag)
            # power_idle_total is a loadout constant (sum of the fitted
            # components' power_idle). The stand-in's ship is built with
            # the EXACT same loadout the host uses, so its
            # power_idle_total is the right value. Fall back to 0.0 for an
            # unknown tag (e.g. 'test') — its signature then reads
            # power_used, which is fine (test targets are never in a 2P
            # snapshot).
            idle = e.ship.power_idle_total if e is not None else 0.0
            proxies.append(_GhostEnemyProxy(
                pygame.Vector2(x, y), pygame.Vector2(vx, vy),
                ang, eid, power_used, idle))
        return proxies

    def _targeting_proxies(self, pos):
        """Lightweight targeting proxies from the buffer's enemy entries
        enough of an AIEnemy for `lead_point` /
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
        # The buffer's enemy entry also carries (shield_dump,
        # shield_clock, power_used, flame_mags) — unused here (the reticle
        # only needs the pose), but the unpack must match the 11-tuple
        # shape.
        for (tag, x, y, ang, vx, vy, _eid, _s_dump, _s_clock, _p_used,
             _flame_mags) in pos['enemies']:
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
