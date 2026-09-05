"""Player and enemy bullets."""
import pygame
from .config import BULLET_LIFE, ENEMY_BULLET_LIFE

from dataclasses import dataclass

@dataclass(frozen=True)
class Shot:
    """A fired projectile in world space.

    Produced by Ship._fire(); the Game routes it into the right bullet
    list (player vs enemy). The ship stays agnostic of bullet classes.
    """
    pos: pygame.Vector2
    vel: pygame.Vector2
    owner: int


class Bullet:
    def __init__(self, pos, vel, owner=0):
        self.pos = pos
        self.vel = vel
        self.life = BULLET_LIFE
        self.owner = owner  # index into Game.ships; this is the future sync field

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt


class EnemyBullet:
    def __init__(self, pos, vel, owner=0):
        self.pos = pos
        self.vel = vel
        self.life = ENEMY_BULLET_LIFE
        self.owner = owner

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt
