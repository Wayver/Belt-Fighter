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
    def __init__(self, pos, vel, owner=0, bid=None):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.vel = vel
        self.life = BULLET_LIFE
        self.owner = owner
        # Unique id (player_index, bullet_seq), assigned by the host at
        # fire (Game._step). The client's ghost derives the same id and
        # dedups its predicted bullets against the buffer's copy (no
        # double-draw). None in single-player; presentation only — never
        # feeds the sim.
        self.id = bid
        # Handback state (client-only): True once the buffer's newest
        # snapshot carries this bullet (PredictedShip.handback_bullets);
        # predicted_view then hands it to the buffer on the frame the
        # buffer's copy ADVANCES. Always False on the host.
        self.pending = False

    def snapshot(self):
        # The id (player_index, bullet_seq) is carried too — the client's
        # dedup key; a 2-tuple (None-safe).
        return (self.pos.x, self.pos.y, self.vel.x, self.vel.y,
                self.owner, self.life, self.id)

    def apply_snapshot(self, s):
        (px, py, vx, vy, owner, life, bid) = s
        self.pos = pygame.Vector2(px, py)
        self.prev_pos = self.pos.copy()   # no interpolation artifact on restore
        self.vel = pygame.Vector2(vx, vy)
        self.owner = owner
        self.life = life
        self.id = bid                     # restore the unique id

    def update(self, dt):
        self.prev_pos = self.pos.copy()
        self.pos += self.vel * dt
        self.life -= dt

class EnemyBullet:
    def __init__(self, pos, vel, owner=0):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.vel = vel
        self.life = ENEMY_BULLET_LIFE
        self.owner = owner

    def snapshot(self):
        return (self.pos.x, self.pos.y, self.vel.x, self.vel.y,
                self.owner, self.life)

    def apply_snapshot(self, s):
        (px, py, vx, vy, owner, life) = s
        self.pos = pygame.Vector2(px, py)
        self.prev_pos = self.pos.copy()   # no interpolation artifact on restore
        self.vel = pygame.Vector2(vx, vy)
        self.owner = owner
        self.life = life

    def update(self, dt):
        self.prev_pos = self.pos.copy()
        self.pos += self.vel * dt
        self.life -= dt

class Missile:
    def __init__(self, pos, vel, owner=0, target=None, mid=None):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.vel = vel.copy()
        self.owner = owner
        self.target = target              # AIEnemy ref (or None)
        self.life = MISSILE_LIFE
        self.boost = MISSILE_BOOST_TIME
        self.dmg = MISSILE_DAMAGE
        # Unique id (player_index, seq), assigned by the host at launch
        # (Game._step). The client dedups its ghost missiles against the
        # buffer's copy (no double-draw). None in single-player.
        self.id = mid

    def snapshot(self):
        # target is stored as the enemy's ship id (or None); Game
        # apply_snapshot() re-wires it to the live AIEnemy object.
        # The id (player_index, seq) is carried too — the client's dedup
        # key; a 2-tuple (None-safe).
        return (self.pos.x, self.pos.y, self.vel.x, self.vel.y,
                self.owner, self.life, self.boost,
                self.target.ship.id if self.target is not None else None,
                self.id)

    def apply_snapshot(self, s):
        (px, py, vx, vy, owner, life, boost, target_id, mid) = s
        self.pos = pygame.Vector2(px, py)
        self.prev_pos = self.pos.copy()   # no interpolation artifact on restore
        self.vel = pygame.Vector2(vx, vy)
        self.owner = owner
        self.life = life
        self.boost = boost
        self.target = None                # Game wires the real ref
        self._target_id = target_id
        self.id = mid                     # restore the unique id

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
