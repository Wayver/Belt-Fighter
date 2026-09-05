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
                     ENEMY_AVOID_RADIUS, ENEMY_AVOID_WEIGHT, MAX_SPEED,
                     ENEMY_FILL, ENEMY_EDGE, ENEMY_FLAME)

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

        # blend: avoidance takes over as a rock gets close
        desired = seek * (1.0 - 0.8 * danger) + avoid * ENEMY_AVOID_WEIGHT
        if desired.length() < 0.01:
            desired = seek
        desired_angle = math.atan2(desired.y, desired.x)

        # turn: sign of the wrapped angle error
        diff = (desired_angle - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = math.copysign(1.0, diff) if abs(diff) > 0.05 else 0.0

        # thrust: burn while chasing, ease off while dodging
        thrust_fwd = 1.0 if (danger < 0.5 or abs(diff) < 0.6) else 0.0

        # fire: aimed at the player and in range (Ship handles the cooldown)
        aim = math.atan2(player.pos.y - self.ship.pos.y,
                         player.pos.x - self.ship.pos.x)
        aim_diff = (aim - self.ship.angle + math.pi) % (2 * math.pi) - math.pi
        fire = abs(aim_diff) < 0.25 and dist < ENEMY_ENGAGE_RANGE

        return ShipInput(turn=turn, thrust_fwd=thrust_fwd, fire=fire)

    def update(self, dt, player, asteroids):
        """Run one fixed step. Returns the Ship's Shot events (route them
        into the enemy bullet list)."""
        inp = self._steer(player, asteroids)
        return self.ship.update(dt, inp)

    def draw(self, screen, cam):
        self.ship.draw(screen, cam, fill=ENEMY_FILL, edge=ENEMY_EDGE,
                       flame_out=ENEMY_FLAME, flame_in=ENEMY_FLAME)
