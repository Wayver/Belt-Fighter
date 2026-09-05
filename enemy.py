"""Enemy ships: chase, circle, avoid rocks, and shoot back."""
import math
import random

import pygame

from .config import (WIDTH, HEIGHT, ENEMY_ACCEL, ENEMY_MAX_SPEED,
                     ENEMY_ROT_SPEED, ENEMY_RADIUS, ENEMY_HP,
                     ENEMY_BULLET_SPEED, ENEMY_FIRE_COOLDOWN,
                     ENEMY_FIRE_SPREAD, ENEMY_ENGAGE_RANGE,
                     ENEMY_ORBIT_OFFSET, ENEMY_AVOID_RADIUS,
                     ENEMY_AVOID_WEIGHT, ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME)
from .bullets import EnemyBullet


class EnemyShip:
    NOSE = (14, 0)
    WING_P = (-10, -9)
    TAIL = (-6, 0)
    WING_S = (-10, 9)

    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.angle = random.uniform(0, 2 * math.pi)
        self.hp = ENEMY_HP
        self.collision_radius = ENEMY_RADIUS
        self.fire_timer = random.uniform(0.5, ENEMY_FIRE_COOLDOWN)
        self.thrusting = False

    def axes(self):
        fwd = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
        right = pygame.Vector2(-fwd.y, fwd.x)
        return fwd, right

    def to_world(self, lx, ly):
        fwd, right = self.axes()
        return self.pos + fwd * lx + right * ly

    def update(self, dt, ship, enemy_bullets, asteroids):
        # steer toward a point offset to the side of the player, so the
        # enemy circles instead of ramming
        to_player = ship.pos - self.pos
        dist = to_player.length()
        if dist > 1:
            perp = pygame.Vector2(-to_player.y, to_player.x).normalize()
            target = ship.pos + perp * ENEMY_ORBIT_OFFSET
        else:
            target = ship.pos

        # seek: unit vector toward the orbit point
        seek = target - self.pos
        if seek.length() > 1:
            seek.normalize_ip()
        else:
            seek = pygame.Vector2(0, 0)

        # avoid: repel from nearby rocks, weighted by urgency (squared,
        # so close rocks dominate), using a short lookahead of the
        # rock's own motion
        avoid = pygame.Vector2(0, 0)
        danger = 0.0
        for a in asteroids:
            d = self.pos - a.pos
            dist_a = d.length()
            if dist_a < ENEMY_AVOID_RADIUS:
                hit_dist = a.collision_radius + ENEMY_RADIUS
                # where the rock will be if we reach it at top speed
                t = min(dist_a / max(ENEMY_MAX_SPEED, 1.0), 1.0)
                rock = a.pos + a.vel * t
                d = self.pos - rock
                dist_a = d.length()
                if dist_a < 1:
                    d = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                    dist_a = d.length()
                d.normalize_ip()
                urgency = 1.0 - (dist_a - hit_dist) / max(ENEMY_AVOID_RADIUS - hit_dist, 1.0)
                urgency = max(0.0, min(1.0, urgency))
                avoid += d * (urgency * urgency)
                danger = max(danger, urgency)

        # blend: avoidance takes over as a rock gets close
        desired = seek * (1.0 - 0.8 * danger) + avoid * ENEMY_AVOID_WEIGHT
        if desired.length() < 0.01:
            desired = seek

        desired_angle = math.atan2(desired.y, desired.x)
        diff = (desired_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.thrusting = False
        if abs(diff) > 0.05:
            self.angle += math.copysign(ENEMY_ROT_SPEED * dt, diff)
            # don't burn while swinging wide of a rock we're about to hit
            if danger < 0.5 or abs(diff) < 0.6:
                self.thrusting = True

        fwd, _ = self.axes()
        if self.thrusting:
            # ease off the burn while dodging so we don't accelerate in
            self.vel += fwd * ENEMY_ACCEL * (1.0 - 0.5 * danger) * dt
        if self.vel.length() > ENEMY_MAX_SPEED:
            self.vel.scale_to_length(ENEMY_MAX_SPEED)
        self.pos += self.vel * dt

        # fire when roughly aimed at the player and in range
        self.fire_timer -= dt
        if self.fire_timer <= 0 and dist < ENEMY_ENGAGE_RANGE:
            aim = math.atan2(ship.pos.y - self.pos.y, ship.pos.x - self.pos.x)
            aim_diff = (aim - self.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(aim_diff) < 0.25:
                spread = random.uniform(-ENEMY_FIRE_SPREAD, ENEMY_FIRE_SPREAD)
                bvel = pygame.Vector2(math.cos(self.angle + spread),
                                      math.sin(self.angle + spread)) * ENEMY_BULLET_SPEED
                enemy_bullets.append(EnemyBullet(self.to_world(*self.NOSE), bvel))
                self.fire_timer = ENEMY_FIRE_COOLDOWN

    def draw(self, screen, cam):
        if self.thrusting:
            base = cam.to_screen(self.to_world(-10, 0))
            fwd, _ = self.axes()
            tip = base - fwd * (8 + random.random() * 6)
            pygame.draw.line(screen, ENEMY_FLAME, base, tip, 3)
        pts = [cam.to_screen(self.to_world(*p))
               for p in (self.NOSE, self.WING_P, self.TAIL, self.WING_S)]
        pygame.draw.polygon(screen, ENEMY_FILL, pts)
        pygame.draw.polygon(screen, ENEMY_EDGE, pts, 2)
