"""Asteroids: drift, spin, and split when shot."""
import math
import random

import pygame

from .config import WIDTH, HEIGHT, ROCK_SIZES, ROCK_FILL, ROCK_EDGE


class Asteroid:
    def __init__(self, pos, size, vel=None, rng=None):
        rng = rng or random
        cfg = ROCK_SIZES[size]
        self.size = size
        self.radius = cfg['radius']
        self.collision_radius = cfg['radius'] * 1.0
        self.pos = pygame.Vector2(pos)
        if vel is None:
            speed = rng.uniform(*cfg['speed'])
            a = rng.uniform(0, 2 * math.pi)
            self.vel = pygame.Vector2(math.cos(a) * speed, math.sin(a) * speed)
        else:
            self.vel = pygame.Vector2(vel)
        self.spin = rng.uniform(*cfg['spin']) * rng.choice((-1, 1))
        self.angle = rng.uniform(0, 2 * math.pi)
        self.verts = self._make_rock(self.radius, rng=rng)

    @staticmethod
    def _make_rock(radius, n=10, rng=None):
        rng = rng or random
        verts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            r = radius * rng.uniform(0.72, 1.25)
            verts.append(pygame.Vector2(math.cos(a) * r, math.sin(a) * r))
        return verts

    # --- networking ---
    #
    # SYNCED: pos, vel, size, angle, spin, verts (the rock shape — both
    # peers must draw the same rock). radius / collision_radius are
    # derived from size via ROCK_SIZES, so they are rebuilt, not stored.

    def snapshot(self):
        return (
            self.pos.x, self.pos.y,
            self.vel.x, self.vel.y,
            self.size,
            self.angle, self.spin,
            tuple((v.x, v.y) for v in self.verts),
        )

    def apply_snapshot(self, s):
        (px, py, vx, vy, size, angle, spin, verts) = s
        self.size = size
        cfg = ROCK_SIZES[size]
        self.radius = cfg['radius']
        self.collision_radius = cfg['radius'] * 1.0
        self.pos = pygame.Vector2(px, py)
        self.vel = pygame.Vector2(vx, vy)
        self.angle = angle
        self.spin = spin
        self.verts = [pygame.Vector2(x, y) for (x, y) in verts]

    def update(self, dt):
        self.pos += self.vel * dt
        self.angle += self.spin * dt

    def draw(self, screen, cam):
        sx, sy = cam.to_screen(self.pos)
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        pts = []
        for v in self.verts:
            pts.append((sx + v.x * ca - v.y * sa, sy + v.x * sa + v.y * ca))
        pygame.draw.polygon(screen, ROCK_FILL, pts)
        pygame.draw.polygon(screen, ROCK_EDGE, pts, 2)