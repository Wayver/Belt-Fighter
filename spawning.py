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
"""
import math
import random

import pygame

from .config import (WIDTH, HEIGHT, SECTOR_SIZE, ACTIVE_SECTORS,
                     ROCKS_PER_SECTOR, DESPAWN_RADIUS, SPAWN_CLEAR_RADIUS,
                     MAX_SPAWNS_PER_TICK)
from .asteroid import Asteroid
from .enemy import EnemyShip


def _far_pos(ship, min_dist, tries=20):
    """Pick a random position at least min_dist from the ship."""
    for _ in range(tries):
        p = pygame.Vector2(random.uniform(-WIDTH, 2 * WIDTH),
                           random.uniform(-HEIGHT, 2 * HEIGHT))
        if p.distance_to(ship.pos) > min_dist:
            return p
    return pygame.Vector2(random.uniform(-WIDTH, 2 * WIDTH),
                          random.uniform(-HEIGHT, 2 * HEIGHT))


def spawn_enemy(enemies, ship, min_dist=400):
    enemies.append(EnemyShip(_far_pos(ship, min_dist)))


def make_stars(n=90):
    return [(random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1),
             random.choice((1, 1, 2))) for _ in range(n)]


# --- sector field ---

def sector_of(pos):
    """Integer sector id for a world position (floors correctly for
    negative coords)."""
    return (int(pos.x // SECTOR_SIZE), int(pos.y // SECTOR_SIZE))


def _spawn_pos(sector, players, tries=8):
    """Random point inside the sector, clear of every player."""
    sx, sy = sector
    for _ in range(tries):
        p = pygame.Vector2(random.uniform(sx * SECTOR_SIZE, (sx + 1) * SECTOR_SIZE),
                           random.uniform(sy * SECTOR_SIZE, (sy + 1) * SECTOR_SIZE))
        if all(p.distance_to(pl) > SPAWN_CLEAR_RADIUS for pl in players):
            return p
    return None


def _spawn_size(wave):
    """The field gets meaner as waves rise: more small/medium rocks."""
    if wave <= 1:
        return 'large'
    return random.choices(('large', 'medium', 'small'),
                          weights=(0.5, 0.3, 0.2))[0]


def update_field(asteroids, players, wave, dt):
    """Top up under-populated sectors and cull rocks that drifted away.

    Call once per fixed step.
    asteroids: the game's rock list (mutated in place)
    players:   list of Vector2 positions — one entry today, one per ship later
    wave:      current difficulty wave (shifts the spawn size mix)
    dt:        fixed step size (kept for API symmetry / future use)
    """
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
                pos = _spawn_pos((sx, sy), players)
                if pos is not None:
                    asteroids.append(Asteroid(pos, _spawn_size(wave)))
                    counts[(sx, sy)] = counts.get((sx, sy), 0) + 1
                    spawned += 1
                if spawned >= MAX_SPAWNS_PER_TICK:
                    break
