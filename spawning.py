"""Spawning: persistent sector-based rock field, enemy respawns, starfield.

The world is infinite, so rocks aren't spawned in waves anymore. Instead
the world is divided into SECTOR_SIZE cells; every sector within
ACTIVE_SECTORS of the player group is kept topped up to ROCKS_PER_SECTOR
rocks, and rocks that drift past DESPAWN_RADIUS are culled. Rocks persist
where you leave them, so the field feels continuous rather than a
treadmill.

Multiplayer-ready: update_field() takes a list of player positions
(one today, one per ship later) and keys everything off their centroid.
Sector ids are plain (int, int) pairs, so they're trivial to sync.

Seedable: every random-using function takes an optional `rng` (a
random.Random instance). Default None falls back to the global random
module, so existing call sites are unchanged. Pass the same rng through
to make a run reproducible.
"""
import math
import random

import pygame

from .config import (WIDTH, HEIGHT, SECTOR_SIZE, ACTIVE_SECTORS,
                     ROCKS_PER_SECTOR, DESPAWN_RADIUS, SPAWN_CLEAR_RADIUS,
                     MAX_SPAWNS_PER_TICK)
from .asteroid import Asteroid
from .enemy import EnemyShip
from .intent import ShipInput
from .ai_enemy import AIEnemy, MoteEnemy

MOTE_CHANCE = 0.3

class TestTarget(AIEnemy):
    """Stationary, non-firing target for the test range. Same collision
    surface and shield/hp routing as a real enemy; the brain is
    replaced with 'sit still'."""
    def __init__(self, pos, rng=None):
        super().__init__(pos, rng=rng)
        self.hp = 50            # ENEMY_HP is 1; survive a test session
        self.ship.angle = math.pi / 2   # nose toward the player
        self.ship.vel = pygame.Vector2(0, 0)

    def update(self, dt, player, asteroids):
        self.ship.vel = pygame.Vector2(0, 0)   # kill any drift
        shots, _, _ = self.ship.update(dt, ShipInput(turn=0, thrust_fwd=0.0,
                                                 thrust_left=0.0,
                                                 thrust_right=0.0,
                                                 stop=False, fire=False))
        return shots


def _far_pos(ship, min_dist, tries=20, rng=None):
    """Pick a random position at least min_dist from the ship."""
    rng = rng or random
    for _ in range(tries):
        p = pygame.Vector2(rng.uniform(-WIDTH, 2 * WIDTH),
                           rng.uniform(-HEIGHT, 2 * HEIGHT))
        if p.distance_to(ship.pos) > min_dist:
            return p
    return pygame.Vector2(rng.uniform(-WIDTH, 2 * WIDTH),
                          rng.uniform(-HEIGHT, 2 * HEIGHT))

# old enemy
#def spawn_enemy(enemies, ship, min_dist=400):
#    enemies.append(EnemyShip(_far_pos(ship, min_dist)))
def spawn_enemy(enemies, ship, min_dist=400, rng=None):
    rng = rng or random
    p = _far_pos(ship, min_dist, rng=rng)
    if rng.random() < MOTE_CHANCE:
        enemies.append(MoteEnemy(p, rng=rng))
    else:
        enemies.append(AIEnemy(p, rng=rng))


def make_stars(n=90, rng=None):
    rng = rng or random
    return [(rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1),
             rng.choice((1, 1, 2))) for _ in range(n)]


# --- sector field ---

def sector_of(pos):
    """Integer sector id for a world position (floors correctly for
    negative coords)."""
    return (int(pos.x // SECTOR_SIZE), int(pos.y // SECTOR_SIZE))


def _spawn_pos(sector, players, tries=8, rng=None):
    """Random point inside the sector, clear of every player."""
    rng = rng or random
    sx, sy = sector
    for _ in range(tries):
        p = pygame.Vector2(rng.uniform(sx * SECTOR_SIZE, (sx + 1) * SECTOR_SIZE),
                           rng.uniform(sy * SECTOR_SIZE, (sy + 1) * SECTOR_SIZE))
        if all(p.distance_to(pl) > SPAWN_CLEAR_RADIUS for pl in players):
            return p
    return None


def _spawn_size(wave, rng=None):
    """The field gets meaner as waves rise: more small/medium rocks."""
    rng = rng or random
    if wave <= 1:
        return 'large'
    return rng.choices(('large', 'medium', 'small'),
                       weights=(0.5, 0.3, 0.2))[0]


def update_field(asteroids, players, wave, dt, rng=None):
    """Top up under-populated sectors and cull rocks that drifted away.

    Call once per fixed step.
    asteroids: the game's rock list (mutated in place)
    players:   list of Vector2 positions — one entry today, one per ship later
    wave:      current difficulty wave (shifts the spawn size mix)
    dt:        fixed step size (kept for API symmetry / future use)
    rng:       optional random.Random for reproducible fields (default global)
    """
    rng = rng or random
    centroid = pygame.Vector2(sum(p.x for p in players) / len(players),
                              sum(p.y for p in players) / len(players))

    # tag each rock with its sector, then cull anything beyond the fog
    for a in asteroids:
        a.sector = sector_of(a.pos)
    asteroids[:] = [a for a in asteroids
                    if a.pos.distance_to(centroid) < DESPAWN_RADIUS]

    counts = {}
    for a in asteroids:
        counts[a.sector] = counts.get(a.sector, 0) + 1

    cs, ct = sector_of(centroid)
    spawned = 0
    for sx in range(cs - ACTIVE_SECTORS, cs + ACTIVE_SECTORS + 1):
        for sy in range(ct - ACTIVE_SECTORS, ct + ACTIVE_SECTORS + 1):
            if spawned >= MAX_SPAWNS_PER_TICK:
                break
            need = ROCKS_PER_SECTOR - counts.get((sx, sy), 0)
            for _ in range(need):
                pos = _spawn_pos((sx, sy), players, rng=rng)
                if pos is not None:
                    asteroids.append(Asteroid(pos, _spawn_size(wave, rng=rng),
                                              rng=rng))
                    counts[(sx, sy)] = counts.get((sx, sy), 0) + 1
                    spawned += 1
                if spawned >= MAX_SPAWNS_PER_TICK:
                    break