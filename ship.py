"""Player ship: hull-driven, per-thruster movement and rendering.

The hull (polygon + thruster/weapon slots) lives in hulls.py as a HullType.
This module keeps the runtime Ship: state, the fixed-step update, and
rendering that reads geometry from self.hull.

Per-thruster physics pipeline (one fixed step):
  1. _apply_rotation  — Q/E turns the hull
  2. _set_demands     — map input (+ auto-stop guidance) to per-thruster demand (0..1)
  3. _allocate        — resolve each thruster's allocation (0..1). Phase 2:
                        power/compute budgets. For now: always 1.0.
  4. _resolve_forces  — per-thruster force = demand * allocation * comp.thrust,
                        summed into a world-space accel; also sets flame mags
  5. _integrate       — vel += accel*dt, clamp, deadband, pos += vel*dt

A thruster's Slot.orientation is the direction of the FORCE on the ship;
the exhaust/flame is rendered in the opposite direction.

Networked-ready (unchanged):
- update() takes a ShipInput (flat, serializable intent).
- update() expects a fixed timestep for deterministic simulation.
- snapshot()/apply_snapshot() expose the only synced state: (pos, vel, angle).
- flame_mags is presentation-only.
"""
import math
import random
from dataclasses import dataclass

import pygame

from .bullets import Shot, Beam

from .config import (WIDTH, HEIGHT, ROT_SPEED, MAX_SPEED,
                     STOP_GAIN, MAX_STOP_ACCEL, STOP_DEADBAND,
                     SHIP_COLOR, SHIP_EDGE, FLAME_OUT, FLAME_IN,
                     POWER_HYSTERESIS, AUTO_STOP_COMPUTE, SHIELD_DUMP_DECAY,
                     SHIELD_OVAL_A, SHIELD_OVAL_B, SHIELD_COLOR_DIM,
                     SHIELD_COLOR_BRIGHT, SHIELD_OFFLINE_FACTOR, SHIELD_OFFLINE_DRAIN,
                     ARC_GLOW, ARC_CORE, TARGETING_POWER, TARGETING_COMPUTE_BASE,
                     TARGETING_COMPUTE_PER_TARGET, LASER_DUMP_DECAY, LASER_COLOR)

from .hulls import DEFAULT_HULL, default_loadout, Slot, ComponentType

from .intent import ShipInput

def wrapped_delta(a, b, size):
    """Shortest signed distance from a to b on a wrapped axis."""
    d = (b - a) % size
    return d - size if d > size / 2 else d


@dataclass
class Thruster:
    """Runtime state for one fitted thruster (one per thruster slot).

    demand:     0..1, how hard the player/guidance wants it to fire this tick
    allocation: 0..1, what it's actually allowed (power/compute, Phase 2)
    force:      resolved force magnitude this tick (drives the flame)
    """
    slot: Slot
    comp: ComponentType
    demand: float = 0.0
    allocation: float = 1.0
    force: float = 0.0



@dataclass
class Weapon:
    """Runtime state for one fitted weapon (one per weapon slot).

    cooldown:   seconds until it can fire again
    allocation: 0..1 power allocation this tick (0 = can't fire)
    """
    slot: Slot
    comp: ComponentType
    cooldown: float = 0.0
    allocation: float = 1.0
    charge: float = 0.0


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _dim_color(c, f):
    return tuple(int(ch * f) for ch in c)

class Ship:
    def __init__(self, ship_id=0, hull=None, loadout=None):
        self.id = ship_id
        self.hull = hull or DEFAULT_HULL
        # slot_name -> ComponentType (the fitted parts, all types)
        self.components = dict(loadout or default_loadout(self.hull))
        # runtime thruster instances (thruster slots only)
        self.thrusters = []
        self.thrusters_by_name = {}
        for slot in self.hull.slots:
            if slot.slot_type == 'thruster' and slot.name in self.components:
                t = Thruster(slot, self.components[slot.name])
                self.thrusters.append(t)
                self.thrusters_by_name[slot.name] = t
        # runtime weapon instances (weapon slots only)
        self.weapons = []
        self.weapons_by_name = {}
        for slot in self.hull.slots:
            if slot.slot_type == 'weapon' and slot.name in self.components:
                w = Weapon(slot, self.components[slot.name])
                self.weapons.append(w)
                self.weapons_by_name[slot.name] = w

        # shield runtime state
        self.shield_slot = None
        self.shield_comp = None
        for slot in self.hull.slots:
            if slot.slot_type == 'shield' and slot.name in self.components:
                self.shield_slot = slot
                self.shield_comp = self.components[slot.name]
                break
        self.shield_charge = (self.shield_comp.shield_max_charge
                              if self.shield_comp else 0.0)
        self.shield_dump = 0.0
        self.shield_clock = 0.0
        # targeting assist: player-toggled paid system (T key)
        self.targeting_on = False
        self.tracked = 0    # enemies within TARGETING_RANGE; set by Game each tick
        self.laser_target = None   # enemy candidate; set by Game each tick
        self.laser_dump = 0.0      # decaying power spike from discharges


        # Phase 2: power/compute budgets, derived from the fitted parts
        self.power_supply = sum(c.power_supply for c in self.components.values())
        self.compute_supply = sum(c.compute_supply for c in self.components.values())
        self.power_used = 0.0
        self.compute_used = 0.0
        self.brownout = False
        self.power_factor = 1.0
        self.compute_alloc = 1.0
        self.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.vel = pygame.Vector2(0, 0)
        self.accel = pygame.Vector2(0, 0)
        self.angle = -math.pi / 2
        self.flame_mags = {}   # presentation only — never serialized
        self.arcs = [] # presentation only: brownout lightning (pts, age, ttl)
        self.arc_clock = 0.0
        self.dampening = False
        self.prev_pos = self.pos.copy()
        self.prev_angle = self.angle

    # --- geometry helpers (read from the hull) ---

    def slot(self, name):
        for s in self.hull.slots:
            if s.name == name:
                return s
        return None

    @property
    def muzzle(self):
        """Local muzzle point: the first weapon slot, else the hull nose."""
        for s in self.hull.slots:
            if s.slot_type == 'weapon':
                return s.position
        return self.hull.nose

    @property
    def collision_radius(self):
        """Collision radius from the hull (per-hull, not a global)."""
        return self.hull.collision_radius

    @property
    def shield_on(self):
        return self.shield_comp is not None and self.shield_charge > 0

    def shield_impact_point(self, world_pos):
        """Point on the shield oval in the direction of world_pos."""
        if self.shield_comp is None:
            return world_pos
        d = world_pos - self.pos
        if d.length_squared() < 1e-6:
            return self.pos
        d = d.normalize()
        fwd, right = self.axes()
        dx, dy = d.dot(fwd), d.dot(right)
        t = 1.0 / math.sqrt((dx / SHIELD_OVAL_A) ** 2 + (dy / SHIELD_OVAL_B) ** 2)
        return self.pos + fwd * (t * dx) + right * (t * dy)

    def shield_contains(self, world_pos):
        """True if world_pos is inside the shield oval."""
        if self.shield_comp is None:
            return False
        d = world_pos - self.pos
        fwd, right = self.axes()
        lx, ly = d.dot(fwd), d.dot(right)
        return (lx / SHIELD_OVAL_A) ** 2 + (ly / SHIELD_OVAL_B) ** 2 <= 1.0

    def reset_lasers(self):
        self.laser_target = None
        self.laser_dump = 0.0
        for w in self.weapons:
            w.charge = 0.0


    def axes(self):
        fwd = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
        right = pygame.Vector2(-fwd.y, fwd.x)
        return fwd, right

    def to_world(self, lx, ly):
        fwd, right = self.axes()
        return self.pos + fwd * lx + right * ly

    # --- networking (unchanged) ---

    def snapshot(self):
        a = math.atan2(math.sin(self.angle), math.cos(self.angle))
        return (self.pos.x, self.pos.y, self.vel.x, self.vel.y, a)

    def sync_render(self, alpha):
        self.rpos = self.prev_pos.lerp(self.pos, alpha)
        da = (self.angle - self.prev_angle + math.pi) % (2 * math.pi) - math.pi
        self.rangle = self.prev_angle + da * alpha

    def apply_snapshot(self, s):
        self.pos = pygame.Vector2(s[0], s[1])
        self.vel = pygame.Vector2(s[2], s[3])
        self.angle = s[4]

    # --- simulation: per-thruster pipeline ---


    def update(self, dt, inp):
        """... Returns (shots, beams) fired this tick; the caller routes
        them into its bullet/beam lists."""
        self.prev_pos = self.pos.copy()
        self.prev_angle = self.angle
        self.dampening = inp.stop
        self._update_shield(dt)
        self._apply_rotation(dt, inp)
        self._set_demands(inp)
        self._allocate(inp)
        accel = self._resolve_forces()
        self.accel = accel.copy()
        shots = self._fire(dt, inp)
        beams = self._update_lasers(dt, inp)   # after _allocate: sees this tick's brownout
        self._integrate(dt, accel)
        self._update_arcs(dt)
        return shots, beams

    def _apply_rotation(self, dt, inp):
        self.angle += inp.turn * ROT_SPEED * self.hull.turn_rate_factor * dt

    def _set_demands(self, inp):
        """Map player input (and auto-stop guidance) to per-thruster demand.

        Keys are not netted: W+S commands both the main engines and the
        reverse thruster. The forces cancel in _resolve_forces, but the
        power draw doesn't — inefficient firing is the player's cost.
        """
        for t in self.thrusters:
            t.demand = 0.0
        if inp.thrust_fwd:
            self._demand('forward_s', inp.thrust_fwd)
            self._demand('forward_p', inp.thrust_fwd)
        #if inp.thrust_rev:
        #    self._demand('reverse', inp.thrust_rev)
        if inp.thrust_rev:
            for t in self.thrusters:
                if t.slot.orientation == (-1, 0):
                    t.demand = max(t.demand, inp.thrust_rev)
        if inp.thrust_right:
            self._demand('to_right', inp.thrust_right)
        if inp.thrust_left:
            self._demand('to_left', inp.thrust_left)
        if inp.stop:
            self._stop_demands()


    def _demand(self, name, amount):
        t = self.thrusters_by_name.get(name)
        if t is not None:
            t.demand = max(t.demand, min(1.0, amount))

    def _stop_demands(self):
        """Auto-stop (B): a guidance system that commands thrusters against
        the current velocity. Limited by what the fitted thrusters can push."""
        fwd, right = self.axes()
        retro = -(self.vel * (STOP_GAIN * self.compute_alloc))
        cap = MAX_STOP_ACCEL * self.hull.max_speed_factor
        if retro.length() > cap:
            retro.scale_to_length(cap)
        r_fwd = retro.dot(fwd)
        r_right = retro.dot(right)
        # Push along each local axis using the thrusters aligned with it.
        self._axis_demand((1, 0), r_fwd)
        self._axis_demand((-1, 0), -r_fwd)
        self._axis_demand((0, 1), r_right)
        self._axis_demand((0, -1), -r_right)

    def _axis_demand(self, orient, amount):
        """Set demand on thrusters whose force direction == orient so their
        combined thrust produces `amount` of force (clamped to capacity)."""
        if amount <= 0:
            return
        aligned = [t for t in self.thrusters if t.slot.orientation == orient]
        capacity = sum(t.comp.thrust for t in aligned)
        if capacity <= 0:
            return
        d = min(1.0, amount / capacity)
        for t in aligned:
            t.demand = max(t.demand, d)

    def _allocate(self,inp):
        """Resolve each thruster's allocation (0..1) from power/compute.

        Power: fixed priority with a latched brownout. When total demand
        exceeds supply by the hysteresis margin the brownout latches on
        and lower-priority thrusters shed first; it latches off only when
        demand drops back below supply minus the margin (no flapping).
        The shield's draw (steady + hit-dump) is subtracted from the
        budget before thrusters are allocated.
        Compute: continuous scale applied to auto-stop guidance.
        """
        # --- power ---
        idle = sum(c.power_idle for c in self.components.values())
        active = [(t, t.comp.power_active * t.demand) for t in self.thrusters]
        active += [(w, w.comp.power_active) for w in self.weapons if inp.fire]
        # lasers draw power while charging (charge is last tick's value)
        active += [(w, w.comp.power_active) for w in self.weapons
            if w.comp.laser_range > 0 and 0.0 < w.charge < 1.0]
        
        shield_active = 0.0
        if self.shield_on:
            shield_active = self.shield_comp.power_active + self.shield_dump
        # Targeting: fixed draw while on, like the shield — a priority
        # system that can TRIGGER a brownout but isn't shed itself.
        targeting_active = TARGETING_POWER if self.targeting_on else 0.0
        laser_active = self.laser_dump
        total = (idle + sum(need for _, need in active) + shield_active
                 + targeting_active)
        self.power_used = total

        if not self.brownout and total > self.power_supply * (1.0 + POWER_HYSTERESIS):
            self.brownout = True
        elif self.brownout and total < self.power_supply * (1.0 - POWER_HYSTERESIS):
            self.brownout = False


        # How much of the non-idle demand can the reactor actually meet?
        # 1.0 = fully powered; <1.0 while browned out (whole-ship sag).
        non_idle = (sum(need for _, need in active) + shield_active
                    + targeting_active)
        if self.brownout and non_idle > 0:
            self.power_factor = min(1.0, (self.power_supply - idle) / non_idle)
        else:
            self.power_factor = 1.0

        if self.brownout:
            remaining = max(0.0, self.power_supply - idle - shield_active)
            for t, need in sorted(active, key=lambda p: p[0].comp.priority):
                give = min(need, remaining)
                t.allocation = (give / need) if need > 0 else 1.0
                remaining -= give
        else:
            for t in self.thrusters:
                t.allocation = 1.0
            for w in self.weapons:
                w.allocation = 1.0

        # --- compute (guidance) ---
        comp_demand = sum(t.comp.compute_demand * t.demand for t in self.thrusters)
        if self.dampening:
            comp_demand += AUTO_STOP_COMPUTE
        if self.targeting_on:
            comp_demand += (TARGETING_COMPUTE_BASE
                            + TARGETING_COMPUTE_PER_TARGET * self.tracked)
        self.compute_used = comp_demand
        self.compute_alloc = (min(1.0, self.compute_supply / comp_demand)
                              if comp_demand > 0 else 1.0)
        # Brownout sags compute too: auto-stop guidance weakens on a spike.
        self.compute_alloc *= self.power_factor


    def _resolve_forces(self):
        """Per-thruster force = demand * allocation * comp.thrust, summed into
        a world-space accel. Also sets flame magnitudes for rendering."""
        fwd, right = self.axes()
        accel = pygame.Vector2(0, 0)
        self.flame_mags = {}
        for t in self.thrusters:
            t.force = t.demand * t.allocation * t.comp.thrust
            ox, oy = t.slot.orientation
            accel += fwd * (ox * t.force) + right * (oy * t.force)
            if t.slot.flame_key and t.comp.thrust > 0:
                mag = t.force / t.comp.thrust   # == demand * allocation, 0..1
                self.flame_mags[t.slot.flame_key] = max(
                    self.flame_mags.get(t.slot.flame_key, 0.0), mag)
        return accel

    def _fire(self, dt, inp):
        """Execution stage: tick weapon cooldowns, fire ready weapons.

        A weapon fires when the input is held, its cooldown has elapsed,
        and it has power allocation. The cooldown advances only for shots
        that actually fire, so a ship that can't fire (capped by the
        caller, or browned out) fires the instant it can.
        """
        shots = []
        for w in self.weapons:
            if w.comp.laser_range > 0:
                continue   # laser: no ballistic shots, _update_lasers() owns it
            w.cooldown -= dt
            if inp.fire and w.cooldown <= 0 and w.allocation > 0:
                fwd, right = self.axes()
                ox, oy = w.slot.orientation
                vel = (fwd * ox + right * oy) * w.comp.bullet_speed
                shots.append(Shot(self.to_world(*w.slot.position), vel, self.id))
                w.cooldown = w.comp.fire_cooldown
        return shots

    def _integrate(self, dt, accel):
        self.vel += accel * dt
        cap = MAX_SPEED * self.hull.max_speed_factor
        if self.vel.length() > cap:
            self.vel.scale_to_length(cap)
        if self.vel.length() < STOP_DEADBAND:
            self.vel = pygame.Vector2(0, 0)
        self.pos += self.vel * dt


    def _update_shield(self, dt):
        """Recharge the shield, decay the hit-dump, sag under brownout."""
        if self.shield_comp is None:
            return
        # Deep brownout: shield goes offline, charge drains.
        if self.brownout and self.power_factor < SHIELD_OFFLINE_FACTOR:
            self.shield_charge = max(0.0,
                                     self.shield_charge - SHIELD_OFFLINE_DRAIN * dt)
            self.shield_dump = max(0.0, self.shield_dump - SHIELD_DUMP_DECAY * dt)
            return
        if self.shield_charge < self.shield_comp.shield_max_charge:
            rate = self.shield_comp.shield_recharge_rate * self.power_factor
            self.shield_charge = min(self.shield_comp.shield_max_charge,
                                     self.shield_charge + rate * dt)
        if self.shield_dump > 0:
            self.shield_clock += dt
            self.shield_dump = max(0.0, self.shield_dump - SHIELD_DUMP_DECAY * dt)
        else:
            self.shield_clock = 0.0

    def _update_arcs(self, dt):
        """Presentation-only: spawn/age lightning arcs during brownout.

        Never serialized. Spawn rate scales with brownout severity
        (1 - power_factor): a deep sag crackles harder.
        """
        for arc in self.arcs:
            arc[1] += dt
        self.arcs = [a for a in self.arcs if a[1] < a[2]]
        if not self.brownout:
            return
        severity = 1.0 - self.power_factor
        self.arc_clock += dt
        interval = 0.45 - 0.15 * severity   # 0.25s -> 0.10s as it deepens
        if self.arc_clock >= interval:
            self.arc_clock = 0.0
            self.arcs.append(self._make_arc())

    def _make_arc(self):
        """One jagged arc in LOCAL ship space, from a random hull vertex
        pointing outward. Local space so it tracks the rendered ship even
        with network render interpolation."""
        verts = self.hull.polygon
        lx, ly = verts[random.randrange(len(verts))]
        d = pygame.Vector2(lx, ly)
        if d.length_squared() < 1e-6:
            d = pygame.Vector2(1, 0)
        else:
            d = d.normalize()
        d.rotate(random.uniform(-50, 50))
        pts = [(lx, ly)]
        x, y = lx, ly
        for _ in range(random.randint(3, 5)):
            d.rotate(random.uniform(-55, 55))
            x += d.x * random.uniform(6, 14)
            y += d.y * random.uniform(6, 14)
            pts.append((x, y))
        ttl = random.uniform(0.05, 0.12)
        return [pts, 0.0, ttl]


    def register_hit(self):
        """Register a hit on the shield. Returns True if it absorbed the hit."""
        if not self.shield_on:
            return False
        self.shield_charge -= 1.0
        self.shield_dump += self.shield_comp.power_hit
        return True

    def reset_shield(self):
        if self.shield_comp is not None:
            self.shield_charge = self.shield_comp.shield_max_charge
            self.shield_dump = 0.0
            self.shield_clock = 0.0


    def _laser_in_wedge(self, w, target_pos):
        """True if target_pos is inside laser w's firing wedge and range.

        The wedge is measured from the nose: 0 deg = straight ahead,
        positive = starboard. The component stores the arc in degrees
        (the human-facing stat); the test works in radians via
        wrapped_delta, same as the rest of the sim.
        """
        d = target_pos - self.pos
        dist = d.length()
        if dist < 1 or dist > w.comp.laser_range:
            return False
        ang_deg = math.degrees(wrapped_delta(self.angle,
                                         math.atan2(d.y, d.x), 2 * math.pi))
        return w.comp.laser_arc_start_deg <= ang_deg <= w.comp.laser_arc_end_deg


    def _update_lasers(self, dt, inp):
        """Laser charge/discharge state machine. Returns Beam events.

        IDLE -> CHARGING:   target in wedge, charge < 1
        CHARGING -> READY:  charge reaches 1
        -> IDLE (reset):    target leaves wedge, target gone, or brownout
        READY -> DISCHARGE: inp.laser_fire (R)
        """
        # decay the discharge spike (mirrors the shield's hit-dump)
        if self.laser_dump > 0:
            self.laser_dump = max(0.0, self.laser_dump - LASER_DUMP_DECAY * dt)

        beams = []
        for w in self.weapons:
            if w.comp.laser_range <= 0:
                continue                      # not a laser
            t = self.laser_target
            if t is None or self.brownout or not self._laser_in_wedge(w, t.pos):
                w.charge = 0.0                # broken: full reset
                continue
            if w.charge < 1.0:
                w.charge = min(1.0, w.charge + dt / w.comp.laser_charge_time)
                continue
            if inp.laser_fire:
                beams.append(Beam(self.to_world(*w.slot.position), t.pos,
                              self.id, w.comp.laser_damage))
                self.laser_dump += w.comp.laser_discharge_dump
                w.charge = 0.0
        return beams

    # --- rendering (reads geometry from self.hull) ---
    def draw(self, screen, cam, pos=None, angle=None, fill=None, edge=None,
             flame_out=None, flame_in=None):
        if fill is None:
            fill = SHIP_COLOR
        if edge is None:
            edge = SHIP_EDGE
        if flame_out is None:
            flame_out = FLAME_OUT
        if flame_in is None:
            flame_in = FLAME_IN
        if pos is None:
            pos = self.pos
        if angle is None:
            angle = self.angle
        fwd = pygame.Vector2(math.cos(angle), math.sin(angle))
        right = pygame.Vector2(-fwd.y, fwd.x)

        def w(lx, ly):
            return pos + fwd * lx + right * ly

        # Flames: group thruster slots by flame bucket, render each. The
        # flame points along the EXHAUST direction = -orientation.
        by_key = {}
        for slot in self.hull.slots:
            if slot.slot_type == 'thruster' and slot.flame_key:
                by_key.setdefault(slot.flame_key, []).append(slot)
        
        for key, slots in by_key.items():
            mag = self.flame_mags.get(key, 0.0)
            if mag <= 0:
                continue
            for slot in slots:
                lx, ly = slot.position
                dx, dy = slot.flame_dir
                self._flame(screen, cam, fwd, right, w(lx, ly), dx, dy, mag,
                        scale=slot.flame_scale, width=slot.flame_width,
                        flame_out=flame_out, flame_in=flame_in)

        # Hull.
        pts = [cam.to_screen(w(*p)) for p in self.hull.polygon]
        pygame.draw.polygon(screen, fill, pts)
        pygame.draw.polygon(screen, edge, pts, 2)

        # Cockpit + engine nozzles + gun muzzles.
        cx, cy = self.hull.cockpit
        pygame.draw.circle(screen, edge, cam.to_screen(w(cx, cy)), 2)
        for slot in self.hull.slots:
            if slot.slot_type == 'thruster' and slot.flame_key == 'forward':
                nx, ny = slot.position
                pygame.draw.circle(screen, edge, cam.to_screen(w(nx, ny)), 3)
            elif slot.slot_type == 'weapon':
                gx, gy = slot.position
                pygame.draw.circle(screen, edge, cam.to_screen(w(gx, gy)), 2)

        self._draw_shield(screen, cam, pos, angle)
        self._draw_laser_charge(screen, cam, w)
        self._draw_arcs(screen, cam, w)


    def _flame(self, screen, cam, fwd, right, base_world, dx, dy, mag,
           scale=1.0, width=3, flame_out=FLAME_OUT, flame_in=FLAME_IN):
        base = cam.to_screen(base_world)
        length = (6 + 14 * mag) * scale + random.random() * 5 * scale
        tip = base + fwd * (dx * length) + right * (dy * length)
        pygame.draw.line(screen, flame_out, base, tip, width)
        pygame.draw.line(screen, flame_in, base, base + (tip - base) * 0.5, 1)

    def _draw_shield(self, screen, cam, pos, angle):
        """Draw the shield flash: invisible until hit, then flashes + flickers."""
        if self.shield_comp is None or self.shield_dump <= 0:
            return
        fwd = pygame.Vector2(math.cos(angle), math.sin(angle))
        right = pygame.Vector2(-fwd.y, fwd.x)
        # envelope: 1 at the hit, decays to 0 over the dump
        env = min(1.0, self.shield_dump / self.shield_comp.power_hit)
        # flicker: ~6Hz oscillation with a little per-frame jitter
        flicker = 0.6 + 0.4 * math.sin(self.shield_clock * 40.0)
        flicker *= (0.8 + 0.2 * random.random())
        alpha = env * flicker
        if alpha <= 0.02:
            return
        color = _lerp_color(SHIELD_COLOR_DIM, SHIELD_COLOR_BRIGHT, alpha)
        pts = []
        for i in range(32):
            th = 2 * math.pi * i / 32
            world = pos + fwd * (SHIELD_OVAL_A * math.cos(th)) \
                         + right * (SHIELD_OVAL_B * math.sin(th))
            pts.append(cam.to_screen(world))
        pygame.draw.polygon(screen, _dim_color(color, 0.5), pts, 5)  # glow
        pygame.draw.polygon(screen, color, pts, 2)                   # core

    def _draw_arcs(self, screen, cam, w):
        """Draw brownout lightning: blue glow under a white core, fading out."""
        if not self.arcs:
            return
        for pts, age, ttl in self.arcs:
            fade = 1.0 - age / ttl
            sp = [cam.to_screen(w(*p)) for p in pts]
            pygame.draw.lines(screen, _dim_color(ARC_GLOW, fade), False, sp, 3)
            pygame.draw.lines(screen, _dim_color(ARC_CORE, fade), False, sp, 1)

    def _draw_laser_charge(self, screen, cam, to_world):
        """Draw a charge effect at each laser's muzzle: rays pulled
        inward toward the weapon point as charge builds.

        Rays light up in sequence; each one's inner end advances from
        the outer radius to the muzzle over its slice of charge, so at
        full charge they've all converged on the point. Only visible
        while a target is in the wedge (charge resets to 0 otherwise),
        so it doubles as a lock indicator. Presentation only.
        """
        n = 8
        outer = 26.0
        rot = pygame.time.get_ticks() * 0.0008   # slow swirl, ~8s per rev
        for wpn in self.weapons:
            if wpn.comp.laser_range <= 0:
                continue                      # not a laser
            if wpn.charge <= 0:
                continue
            m = cam.to_screen(to_world(*wpn.slot.position))
            for i in range(n):
                prog = wpn.charge * n - i     # this ray's 0..1 progress
                if prog <= 0:
                    continue
                prog = min(1.0, prog)
                ang = 2 * math.pi * i / n + rot
                r_in = outer - prog * (outer - 2.0)
                x1 = m.x + math.cos(ang) * r_in
                y1 = m.y + math.sin(ang) * r_in
                x2 = m.x + math.cos(ang) * outer
                y2 = m.y + math.sin(ang) * outer
                bright = 0.35 + 0.65 * prog
                pygame.draw.line(screen, _dim_color(LASER_COLOR, bright),
                                 (x1, y1), (x2, y2), 2)
            if wpn.charge >= 1.0:
                # converged: pulse a bright point at the muzzle
                phase = pygame.time.get_ticks() * 0.012
                r = 3 + int(2 * (0.5 + 0.5 * math.sin(phase)))
                pygame.draw.circle(screen, LASER_COLOR,
                                   (int(m.x), int(m.y)), r)
