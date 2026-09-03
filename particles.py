"""Explosion particles."""
import math
import random

import pygame

from .config import PARTICLE_COLORS


class Particle:
    def __init__(self, pos, vel, color, life):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt

    def draw(self, screen, cam):
        a = max(0.0, self.life / self.max_life)
        tail = self.pos - self.vel * 0.03
        s1 = cam.to_screen(self.pos)
        s2 = cam.to_screen(tail)
        pygame.draw.line(screen, self.color, (s1.x, s1.y), (s2.x, s2.y), max(1, int(2 * a)))

def burst(particles, pos, radius, big=False):
    n = 30 if big else 14
    for _ in range(n):
        a = random.uniform(0, 2 * math.pi)
        speed = random.uniform(40, 160) * (1.5 if big else 1.0)
        vel = pygame.Vector2(math.cos(a) * speed, math.sin(a) * speed)
        particles.append(Particle(pos, vel, random.choice(PARTICLE_COLORS),
                                  random.uniform(0.4, 0.9)))
