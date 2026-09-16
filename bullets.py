"""Player and enemy bullets."""
import pygame
import math
from .config import (BULLET_LIFE, ENEMY_BULLET_LIFE,
                     MISSILE_SPEED, MISSILE_ACCEL, MISSILE_BOOST_TIME,
                     MISSILE_TURN_RATE, MISSILE_LIFE, MISSILE_DAMAGE)

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

@dataclass(frozen=True)
class MissileShot:
    """A launched homing missile in world space.

    Produced by Ship._update_missiles(); the Game routes it into the
    missile list and wraps it in a Missile (which owns the guidance).
    """
    pos: pygame.Vector2
    vel: pygame.Vector2
    owner: int
    target: object

@dataclass(frozen=True)
class Beam:
    """A hitscan laser discharge in world space (muzzle -> target)."""
    start: pygame.Vector2
    end: pygame.Vector2
    owner: int
    damage: int = 1
    local_start: tuple = None   # hull-local muzzle coords, for re-anchoring


class Bullet:
    def __init__(self, pos, vel, owner=0):
        self.pos = pos
        self.prev_pos = pos.copy()      # NEW
        self.vel = vel
        self.life = BULLET_LIFE
        self.owner = owner

    def update(self, dt):
        self.prev_pos = self.pos.copy()  # NEW
        self.pos += self.vel * dt
        self.life -= dt

class EnemyBullet:
    def __init__(self, pos, vel, owner=0):
        self.pos = pos
        self.prev_pos = pos.copy()      # NEW
        self.vel = vel
        self.life = ENEMY_BULLET_LIFE
        self.owner = owner

    def update(self, dt):
        self.prev_pos = self.pos.copy()  # NEW
        self.pos += self.vel * dt
        self.life -= dt

class Missile:
    def __init__(self, pos, vel, owner=0, target=None):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.vel = vel.copy()
        self.owner = owner
        self.target = target              # AIEnemy ref (or None)
        self.life = MISSILE_LIFE
        self.boost = MISSILE_BOOST_TIME
        self.dmg = MISSILE_DAMAGE

    def update(self, dt):
        self.prev_pos = self.pos.copy()
        if self.boost > 0:
            # straight-line boost, ramp to cruise speed
            self.boost -= dt
            self.vel += self.vel.normalize() * MISSILE_ACCEL * dt
            if self.vel.length() > MISSILE_SPEED:
                self.vel.scale_to_length(MISSILE_SPEED)
        elif self.target is not None and self.target.hp > 0:
            # seek: steer toward the live intercept point, turn-rate limited
            aim = self.target.lead_point(self.pos, self.vel.length())
            if aim is None:
                aim = self.target.pos          # falling back: chase current pos
            desired = aim - self.pos
            if desired.length() > 1:
                desired.normalize_ip()
            cur = self.vel.copy()
            speed = cur.length()
            cur.normalize_ip()
            ang = math.atan2(desired.y, desired.x)
            cur_ang = math.atan2(cur.y, cur.x)
            diff = (ang - cur_ang + math.pi) % (2 * math.pi) - math.pi
            max_turn = MISSILE_TURN_RATE * dt
            turn = max(-max_turn, min(max_turn, diff))
            new_ang = cur_ang + turn
            self.vel = pygame.Vector2(math.cos(new_ang), math.sin(new_ang)) * speed
        self.pos += self.vel * dt
        self.life -= dt
