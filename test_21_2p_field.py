"""plan_21 S1: the 2P asteroid field is the UNION of per-player regions,
not a single centroid blob.

Asserts:
  1. A rock within FIELD_KEEP_RADIUS of a player (but beyond the old
     centroid cull radius) SURVIVES the cull — the old centroid cull
     removed it (wink-out).
  2. Each player's local sector gets topped up when the players are far
     apart — the old centroid top-up left the players' local sectors
     empty (sparse / off-center field).
  3. 1P: one player -> the field is anchored on that player (own sector
     populated, nothing beyond FIELD_KEEP_RADIUS).
  4. update_field is deterministic (same seed + same players -> same
     field).

Headless: no display, no full Game, no netcode — just the field logic.
Run from /mnt:  python -m ship5.test_21_2p_field
"""
import random

import pygame

from . import config as C
from .spawning import update_field, sector_of
from .asteroid import Asteroid


def _run(players, ticks=600, seed=42):
    """Run the field for `ticks` fixed steps with stationary players.

    Resets the class-level rock id counter first (like test_determinism.py)
    so two runs with the same seed get the same ids — the counter is global
    and would otherwise keep incrementing across runs.
    """
    Asteroid._next_id = 1
    rng = random.Random(seed)
    asteroids = []
    for _ in range(ticks):
        update_field(asteroids, players, C.TICK, rng=rng)
        for a in asteroids:
            a.update(C.TICK)
    return asteroids


def test_rock_near_player_survives_centroid_cull():
    # P0 at origin, P1 at (1500, 0) -> centroid (750, 0).
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    # A rock 2100 px from P0 but 2850 px from the centroid (750, 0). The
    # old centroid cull (DESPAWN_RADIUS 2600) removed it (2850 > 2600);
    # the new per-player keep radius (FIELD_KEEP_RADIUS 2200) keeps it
    # (2100 < 2200).
    rock = Asteroid(pygame.Vector2(-2100, 0), 'medium',
                    vel=pygame.Vector2(0, 0), rng=random.Random(1))
    asteroids = [rock]
    update_field(asteroids, [p0, p1], C.TICK, rng=random.Random(2))
    assert any(a.id == rock.id for a in asteroids), \
        "rock near P0 (far from centroid) was culled — wink-out not fixed"


def test_each_players_local_sector_topped_up():
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    asteroids = _run([p0, p1])
    # Each player's own sector must be populated (the old centroid top-up
    # left both empty when the players were 1500 px apart).
    s0 = sector_of(p0)
    s1 = sector_of(p1)
    n0 = sum(1 for a in asteroids if sector_of(a.pos) == s0)
    n1 = sum(1 for a in asteroids if sector_of(a.pos) == s1)
    assert n0 >= 1, f"P0's own sector empty ({n0} rocks) — sparse field"
    assert n1 >= 1, f"P1's own sector empty ({n1} rocks) — sparse field"


def test_1p_field_anchored_on_player():
    p0 = pygame.Vector2(0, 0)
    asteroids = _run([p0])
    # With one player the field is anchored on them: their own sector is
    # populated and no rock is farther than FIELD_KEEP_RADIUS.
    s0 = sector_of(p0)
    assert sum(1 for a in asteroids if sector_of(a.pos) == s0) >= 1, \
        "1P: player's own sector empty"
    assert all(a.pos.distance_to(p0) < C.FIELD_KEEP_RADIUS
               for a in asteroids), \
        "1P: a rock survived beyond FIELD_KEEP_RADIUS"


def test_field_deterministic():
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    a = _run([p0, p1], seed=7)
    b = _run([p0, p1], seed=7)
    sa = sorted((x.id, round(x.pos.x, 3), round(x.pos.y, 3)) for x in a)
    sb = sorted((x.id, round(x.pos.x, 3), round(x.pos.y, 3)) for x in b)
    assert sa == sb, "field not deterministic for the same seed + players"


def main():
    checks = [
        ("rock near player survives centroid cull",
         test_rock_near_player_survives_centroid_cull),
        ("each player's local sector topped up",
         test_each_players_local_sector_topped_up),
        ("1P field anchored on player", test_1p_field_anchored_on_player),
        ("field deterministic", test_field_deterministic),
    ]
    ok = True
    for name, fn in checks:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            ok = False
            print(f"FAIL: {name}: {e}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())