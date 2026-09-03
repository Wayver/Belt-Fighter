"""Asteroids: drift, spin, and split when shot."""
import math
import random

import pygame

from .config import WIDTH, HEIGHT, ROCK_SIZES, ROCK_FILL, ROCK_EDGE


class Asteroid:
    def __init__(self, pos, size, vel=None):
        cfg = ROCK_SIZES[size]
        self.size = size
        self.radius = cfg['radius']
        self.collision_radius = cfg['radius'] * 0.8
        self.score = cfg['score']
        self.pos = pygame.Vector2(pos)
        if vel is None:
            speed = random.uniform(*cfg['speed'])
            a = random.uniform(0, 2 * math.pi)
            self.vel = pygame.Vector2(math.cos(a) * speed, math.sin(a) * speed)
        else:
            self.vel = pygame.Vector2(vel)
        self.spin = random.uniform(*cfg['spin']) * random.choice((-1, 1))
        self.angle = random.uniform(0, 2 * math.pi)
        self.verts = self._make_rock(self.radius)

    @staticmethod
    def _make_rock(radius, n=10):
        verts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            r = radius * random.uniform(0.72, 1.25)
            verts.append(pygame.Vector2(math.cos(a) * r, math.sin(a) * r))
        return verts

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
