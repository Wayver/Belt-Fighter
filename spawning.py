"""Spawning: persistent sector-based rock field, enemy respawns, starfield.

The world is infinite, so rocks aren't spawned in waves anymore. Instead
the world is divided into SECTOR_SIZE cells; every sector within
ACTIVE_SECTORS of ANY alive player is kept topped up to ROCKS_PER_SECTOR
rocks, and rocks that drift past FIELD_KEEP_RADIUS of every player are
culled. Rocks persist where you leave them, so the field feels continuous
rather than a treadmill.

Multiplayer-ready: update_field() takes a list of player positions (one
per ship) and keys the field off the UNION of each player's active region
(plan_21 S1: per-player, not the centroid — the centroid sat between the
two players in 2P and left their local sectors empty). Sector ids are
plain (int, int) pairs, so they're trivial to sync.

Seedable: every random-using function takes an optional `rng` (a
random.Random instance). Default None falls back to the global random
module, so existing call sites are unchanged. Pass the same rng through
to make a run reproducible.
"""
import math
import random

import pygame

from .config import (WIDTH, HEIGHT, SECTOR_SIZE, ACTIVE_SECTORS,
                     ROCKS_PER_SECTOR, FIELD_KEEP_RADIUS, SPAWN_CLEAR_RADIUS,
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


def _spawn_size(rng=None):
    """Random rock size: mostly large, some medium/small."""
    rng = rng or random
    return rng.choices(('large', 'medium', 'small'),
                       weights=(0.5, 0.3, 0.2))[0]


def update_field(asteroids, players, dt, rng=None):
    """Top up under-populated sectors and cull rocks that drifted away.

    Call once per fixed step.

    plan_21 S1: the field is the UNION of a per-player active region (a
    3x3 block of sectors around EACH alive player), not a single
    centroid-anchored blob. A rock is kept if it is within
    FIELD_KEEP_RADIUS of AT LEAST ONE alive player (per-player
    min-distance), not the centroid. This gives each 2P player a full
    field centered on themselves (the old centroid blob sat between the
    players and left their local sectors empty) and stops rocks near a
    player's screen edge from being culled while visible.

    asteroids: the game's rock list (mutated in place)
    players:   list of Vector2 positions — one per alive ship
    dt:        fixed step size (kept for API symmetry / future use)
    rng:       optional random.Random for reproducible fields (default global)
    """
    rng = rng or random
    if not players:
        # No one alive to anchor the field (2P, both players dead waiting to
        # respawn). Leave the field as-is — no cull, no top-up — until a
        # respawn gives it players again.
        return

    # Cull: keep a rock if it is within FIELD_KEEP_RADIUS of at least one
    # alive player (per-player min-distance, not the centroid).
    def _kept(a):
        for p in players:
            if a.pos.distance_to(p) < FIELD_KEEP_RADIUS:
                return True
        return False
    for a in asteroids:
        a.sector = sector_of(a.pos)
    asteroids[:] = [a for a in asteroids if _kept(a)]

    # Count surviving rocks per sector.
    counts = {}
    for a in asteroids:
        counts[a.sector] = counts.get(a.sector, 0) + 1

    # Top up the UNION of each player's 3x3 active region. `region` tracks
    # sectors already scheduled this tick so overlapping regions (players
    # close together) are not double-filled. The GLOBAL spawn cap bounds the
    # total per tick (deterministic fill order; see plan_21 §3.3).
    region = set()
    spawned = 0
    for p in players:
        ps, pt = sector_of(p)
        for sx in range(ps - ACTIVE_SECTORS, ps + ACTIVE_SECTORS + 1):
            for sy in range(pt - ACTIVE_SECTORS, pt + ACTIVE_SECTORS + 1):
                if spawned >= MAX_SPAWNS_PER_TICK:
                    break
                key = (sx, sy)
                if key in region:
                    continue
                region.add(key)
                need = ROCKS_PER_SECTOR - counts.get(key, 0)
                for _ in range(need):
                    pos = _spawn_pos(key, players, rng=rng)
                    if pos is not None:
                        asteroids.append(Asteroid(pos, _spawn_size(rng=rng),
                                                  rng=rng))
                        counts[key] = counts.get(key, 0) + 1
                        spawned += 1
                    if spawned >= MAX_SPAWNS_PER_TICK:
                        break
            if spawned >= MAX_SPAWNS_PER_TICK:
                break
        if spawned >= MAX_SPAWNS_PER_TICK:
            break