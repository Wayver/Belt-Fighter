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
                     ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME, TARGETING_MAX_LEAD)


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

    # --- collision surface (mirrors how Game hits the player) ---

    @property
    def pos(self):
        return self.ship.pos

    @property
    def collision_radius(self):
        return self.ship.collision_radius

    def register_hit(self, source_pos):
        """Route a hit through the shield first, then hp. True if alive."""
        if self.ship.register_hit():
            return True          # shield absorbed it
        self.hp -= 1
        return self.hp > 0

    # --- the brain: steering -> ShipInput (no physics here) ---

    def _steer(self, player, asteroids):
        to_player = player.pos - self.ship.pos
        dist = to_player.length()

        # seek an orbit point offset to the side, so we circle not ram
        if dist > 1:
            perp = pygame.Vector2(-to_player.y, to_player.x).normalize()
            target = player.pos + perp * ENEMY_ORBIT_OFFSET
        else:
            target = player.pos
        seek = target - self.ship.pos
        if seek.length() > 1:
            seek.normalize_ip()
        else:
            seek = pygame.Vector2(0, 0)

        # avoid: repel from nearby rocks, urgency-weighted, with lookahead
        avoid = pygame.Vector2(0, 0)
        danger = 0.0
        for a in asteroids:
            d = self.ship.pos - a.pos
            dist_a = d.length()
            if dist_a < ENEMY_AVOID_RADIUS:
                hit_dist = a.collision_radius + self.collision_radius
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

        desired = seek * (1.0 - 0.8 * danger) + avoid * ENEMY_AVOID_WEIGHT
        if desired.length() < 0.01:
            desired = seek
        desired.normalize_ip()
        desired_angle = math.atan2(desired.y, desired.x)

        diff = (desired_angle - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = math.copysign(1.0, diff) if abs(diff) > 0.05 else 0.0

        # emergency: on a collision course? Hand the sticks to the
        # auto-damper (same compute-paid guidance as the player's B)
        stop = self._on_course(asteroids)
        if stop:
            thrust_fwd = thrust_left = thrust_right = 0.0
        else:
            # lateral: RCS moves the ship sideways while the nose is
            # still swinging, so the dodge starts immediately
            fwd, right = self.ship.axes()
            lat = desired.dot(right)
            thrust_left = min(1.0, -lat) if lat < -0.2 else 0.0
            thrust_right = min(1.0, lat) if lat > 0.2 else 0.0
            thrust_fwd = 1.0 if (danger < 0.5 or abs(diff) < 0.6) else 0.0

        aim = math.atan2(player.pos.y - self.ship.pos.y,
                         player.pos.x - self.ship.pos.x)
        aim_diff = (aim - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        fire = abs(aim_diff) < 0.25 and dist < ENEMY_ENGAGE_RANGE

        return ShipInput(turn=turn, thrust_fwd=thrust_fwd,
                         thrust_left=thrust_left, thrust_right=thrust_right,
                         stop=stop, fire=fire)

    def _on_course(self, asteroids):
        """True if our current velocity takes us into a rock: closing on
        it, and the miss distance at closest approach under the combined
        radii + margin."""
        vel = self.ship.vel
        if vel.length() < 1.0:
            return False
        r2 = ENEMY_AVOID_RADIUS ** 2
        for a in asteroids:
            to_rock = a.pos - self.ship.pos
            d2 = to_rock.length_squared()
            if d2 > r2:
                continue
            rel = vel - a.vel
            closing = rel.dot(to_rock)
            if closing <= 0:
                continue
            t_ca = d2 / closing                    # time to closest approach
            miss = (to_rock - rel * t_ca).length()
            if miss < a.collision_radius + self.collision_radius + ENEMY_COURSE_MARGIN:
                return True
        return False


    def update(self, dt, player, asteroids):
        """Run one fixed step. Returns the Ship's Shot events (route them
        into the enemy bullet list)."""
        inp = self._steer(player, asteroids)
        return self.ship.update(dt, inp)

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
        a = self.ship.accel if use_accel else pygame.Vector2(0, 0)
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
