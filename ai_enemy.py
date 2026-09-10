"""AI enemy: a Ship driven by steering, not a keyboard.

Composition, not subclassing: an AIEnemy *owns* a Ship (same hull +
loadout as the player for now) and each tick runs steering that emits a
ShipInput. The Ship does all the physics, power, and firing; the AI only
decides turn/thrust/fire. This is the reusability test — the same Ship
the keyboard drives, now driven by a brain.
"""
import math
import random

import pygame

from .config import (ENEMY_HP, ENEMY_ENGAGE_RANGE, ENEMY_ORBIT_OFFSET,
                     ENEMY_AVOID_RADIUS, ENEMY_AVOID_WEIGHT,
                     ENEMY_COURSE_MARGIN, MAX_SPEED,
                     ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME, TARGETING_MAX_LEAD,
                     ENEMY_AVOID_BUFFER, ENEMY_BULLET_SPEED)


from .hulls import ENEMY_HULL, enemy_loadout

from .ship import Ship

from .intent import ShipInput


class AIEnemy:
    """A Ship with an AI brain. Owns hp (composition); the Ship owns the
    shield, physics, and weapons."""

    _next_id = 1   # player is ship_id 0; enemies get 1, 2, 3, ...

    def __init__(self, pos):
        self.ship = Ship(ship_id=self._next_id, hull=ENEMY_HULL,
                         loadout=enemy_loadout())
        AIEnemy._next_id += 1
        self.ship.pos = pygame.Vector2(pos)
        self.ship.angle = random.uniform(0, 2 * math.pi)
        self.hp = ENEMY_HP
        self._acc_smooth = pygame.Vector2(0,0)

    # --- collision surface (mirrors how Game hits the player) ---

    @property
    def pos(self):
        return self.ship.pos

    @property
    def collision_radius(self):
        return self.ship.collision_radius

    def register_hit(self, source_pos):
        """Route a hit through the shield first, then hp. True if alive."""
        if self.ship.register_hit(source_pos):
            return True          # shield absorbed it
        self.hp -= 1
        return self.hp > 0

    # --- the brain: steering -> ShipInput (no physics here) ---
    def _steer(self, player, asteroids):
        to_player = player.pos - self.ship.pos
        dist = to_player.length()

        # lead aim: where the player will be when our bullet arrives
        lead_t = min(dist / ENEMY_BULLET_SPEED, 0.5)
        aim_point = player.pos + player.vel * lead_t
        to_aim = aim_point - self.ship.pos
        if to_aim.length() > 1:
            to_aim.normalize_ip()
        else:
            to_aim = pygame.Vector2(1, 0)

        # rock avoidance: a deflection on top of the aim, not a takeover
        avoid, danger = self._avoid(asteroids)

        # nose: on the player, bent away by rocks
        desired = to_aim + avoid * ENEMY_AVOID_WEIGHT
        if desired.length() < 0.01:
            desired = to_aim
        desired.normalize_ip()
        desired_angle = math.atan2(desired.y, desired.x)

        diff = (desired_angle - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = math.copysign(1.0, diff) if abs(diff) > 0.08 else 0.0

        # emergency: on a collision course? Hard RCS sidestep, retro damper on.
        threat = self._on_course(asteroids)
        if threat is not None:
            to_rock = threat.pos - self.ship.pos
            if to_rock.length() < 1:
                to_rock = pygame.Vector2(1, 0)
            to_rock.normalize_ip()
            # sidestep away from where the rock WILL be (stable, geometric)
            rel = self.ship.vel - threat.vel
            t_ca = to_rock.length_squared() / max(rel.dot(to_rock), 1e-6)
            rock_ca = threat.pos + threat.vel * t_ca
            away = self.ship.pos - rock_ca
            if away.length() < 1:
                away = pygame.Vector2(1, 0)
            away.normalize_ip()
            perp = pygame.Vector2(-to_rock.y, to_rock.x)
            if perp.dot(away) < 0:
                perp = -perp
            fwd, right = self.ship.axes()
            lat = perp.dot(right)
            thrust_left = min(1.0, -lat) if lat < 0 else 0.0
            thrust_right = min(1.0, lat) if lat > 0 else 0.0
            thrust_fwd = 0.0
            stop = True
        else:
            fwd, right = self.ship.axes()
            # base orbit thrust: approach when far, strafe tangentially at range
            if dist > ENEMY_ORBIT_OFFSET:
                thrust_fwd = 1.0 if abs(diff) < 0.5 else 0.0
                thrust_rev = 0.0
                lat = 0.0
            else:
                thrust_fwd = 0.0
                thrust_rev = 0.0
                if to_player.length() > 1:
                    tp = to_player.copy(); tp.normalize_ip()
                else:
                    tp = pygame.Vector2(1, 0)
                tangent = pygame.Vector2(-tp.y, tp.x)
                lat = tangent.dot(right)

            # escape thrust: a rock threat moves the ship NOW. Decompose the
            # avoidance direction into the local frame and push out of the
            # rock's path — this is what actually escapes a rock behind us.
            if danger > 0.05 and avoid.length() > 0.01:
                avoid_dir = avoid.copy().normalize()
                a_fwd = avoid_dir.dot(fwd)
                a_right = avoid_dir.dot(right)
                if danger > 0.5:
                    # strong threat: escape overrides the orbit
                    thrust_fwd = max(0.0, a_fwd) * danger
                    thrust_rev = max(0.0, -a_fwd) * danger
                    lat = a_right * danger
                else:
                    # mild threat: nudge on top of the orbit
                    thrust_fwd = max(thrust_fwd, a_fwd * danger)
                    thrust_rev = max(thrust_rev, -a_fwd) * danger
                    lat = lat + a_right * danger

            thrust_left = min(1.0, -lat) if lat < -0.2 else 0.0
            thrust_right = min(1.0, lat) if lat > 0.2 else 0.0
            stop = False

        # fire: aligned with the lead aim, in range
        aim = math.atan2(aim_point.y - self.ship.pos.y,
                         aim_point.x - self.ship.pos.x)
        aim_diff = (aim - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        fire = abs(aim_diff) < 0.25 and dist < ENEMY_ENGAGE_RANGE

        return ShipInput(turn=turn, thrust_fwd=thrust_fwd,
                         thrust_left=thrust_left, thrust_right=thrust_right,
                         stop=stop, fire=fire)


    def _avoid(self, asteroids):
        """Repel from nearby rocks, urgency-weighted, with lookahead.
        Returns (avoid_vector, max_urgency)."""
        avoid = pygame.Vector2(0, 0)
        danger = 0.0
        for a in asteroids:
            d = self.ship.pos - a.pos
            dist_a = d.length()
            if dist_a < ENEMY_AVOID_RADIUS:
                hit_dist = (a.collision_radius + self.collision_radius
                            + ENEMY_AVOID_BUFFER)
                t = min(dist_a / max(MAX_SPEED, 1.0), 1.0)
                rock = a.pos + a.vel * t
                d = self.ship.pos - rock
                dist_a = d.length()
                if dist_a < 1:
                    d = pygame.Vector2(random.uniform(-1, 1),
                                       random.uniform(-1, 1))
                    dist_a = d.length()
                d.normalize_ip()
                urgency = 1.0 - (dist_a - hit_dist) / max(ENEMY_AVOID_RADIUS - hit_dist, 1.0)
                urgency = max(0.0, min(1.0, urgency))
                avoid += d * (urgency * urgency)
                danger = max(danger, urgency)
        return avoid, danger


    def _on_course(self, asteroids):
        """Return the rock we're on a collision course with (worst miss), or None."""
        vel = self.ship.vel
        if vel.length() < 1.0:
            return None
        r2 = ENEMY_AVOID_RADIUS ** 2
        worst, worst_miss = None, float('inf')
        for a in asteroids:
            to_rock = a.pos - self.ship.pos
            d2 = to_rock.length_squared()
            if d2 > r2:
                continue
            rel = vel - a.vel
            closing = rel.dot(to_rock)
            if closing <= 0:
                continue
            t_ca = d2 / closing
            miss = (to_rock - rel * t_ca).length()
            limit = a.collision_radius + self.collision_radius + ENEMY_COURSE_MARGIN
            if miss < limit and miss < worst_miss:
                worst, worst_miss = a, miss
        return worst


    def update(self, dt, player, asteroids):
        inp = self._steer(player, asteroids)
        shots = self.ship.update(dt, inp)
        k = 1.0 - math.exp(-dt / 0.15)
        self._acc_smooth += (self.ship.accel - self._acc_smooth) * k
        return shots

    def draw(self, screen, cam):
        self.ship.draw(screen, cam, fill=ENEMY_FILL, edge=ENEMY_EDGE,
                       flame_out=ENEMY_FLAME, flame_in=ENEMY_FLAME)

    def predict_path(self, horizon, steps):
        """Presentation-only predicted future positions for the targeting
        assist. Constant-velocity base + a decaying acceleration term: the
        AI re-plans every tick, so its instantaneous accel is only
        trustworthy near-term — the fade keeps the far end from over-curving
        when the enemy turns."""
        pos = self.ship.pos.copy()
        vel = self.ship.vel.copy()
        acc = self.ship.accel.copy()
        pts = [pos.copy()]
        dt = horizon / steps
        for i in range(steps):
            w = 1.0 - i / steps          # accel influence fades over the horizon
            pos += vel * dt + 0.5 * acc * (w * dt * dt)
            vel += acc * (w * dt)
            pts.append(pos.copy())
        return pts

    def lead_point(self, shooter_pos, bullet_speed, use_accel=True):
        """Intercept solution: the world point the shooter should aim at so a
        bullet of `bullet_speed` fired from `shooter_pos` meets this enemy.

        Solves |E(T) - shooter| = bullet_speed * T, where E(T) is the enemy's
        predicted position (constant velocity + optional constant accel).
        Iterates to convergence — the mapping is a contraction whenever the
        bullet can actually catch the enemy, so it converges exactly when a
        hit is possible. Returns None when there's no valid solution (enemy
        moving away faster than the bullet, or lead time out of range).
        """
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
