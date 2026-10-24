"""plan_21 S2: the host culls rocks outside the remote player's viewport
before serializing the snapshot (serialize_snapshot's `cull_center`).

Asserts:
  1. A rock within the cull rect is SENT (present in the wire).
  2. A rock outside the cull rect is DROPPED (absent from the wire).
  3. cull_center=None (default) sends ALL rocks (backward-compatible).
  4. _remote_player_pos returns the client's (player 1) pos for a 2P
     snapshot and None for a 1P snapshot.

Headless: no display, no sockets — just the serialization. Run from /mnt:
  python -m ship5.test_21_wire_cull
"""
import math

from . import config as C
from .net import serialize_snapshot, _remote_player_pos, RockShapeTracker


def _ship_snap(px, py, ship_id):
    """A minimal 25-field Ship.snapshot() (only pos.x/pos.y/id matter here)."""
    return (px, py, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, False, 0.0, 0.0,
            False, 0.0, 0.0, 0.0, 0.0, False, 1.0, (), ship_id, False, 0,
            0.0, (), 0)


def _rock(rid, x, y):
    """A minimal 9-tuple rock (id, x, y, vx, vy, size, angle, spin, verts)."""
    verts = tuple((50.0 * math.cos(2 * math.pi * j / 10),
                   50.0 * math.sin(2 * math.pi * j / 10)) for j in range(10))
    return (rid, x, y, 0.0, 0.0, 'large', 0.0, 0.0, verts)


def _snap(players, rocks):
    """A minimal 11-tuple Game.snapshot() with the given players + rocks."""
    return (tuple(players), (), (), (), (), tuple(rocks),
            None, False, 0.0, 1, len(rocks) + 1)


def test_rock_in_cull_rect_is_sent():
    # Client (player 1) at (1500, 0). A rock at the cull center is sent.
    snap = _snap([_ship_snap(0, 0, 0), _ship_snap(1500, 0, 1)],
                 [_rock(1, 1500.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker(),
                              cull_center=(1500.0, 0.0))
    assert len(wire[5]) == 1, "rock at the cull center was dropped"


def test_rock_outside_cull_rect_is_dropped():
    # Client at (1500, 0). A rock 3500 px away is outside the cull rect
    # (half-width WIDTH/2 + WIRE_CULL_MARGIN = 960 + 640 = 1600). Dropped.
    snap = _snap([_ship_snap(0, 0, 0), _ship_snap(1500, 0, 1)],
                 [_rock(1, 5000.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker(),
                              cull_center=(1500.0, 0.0))
    assert len(wire[5]) == 0, "rock outside the cull rect was sent"


def test_cull_rect_boundary():
    # A rock exactly at the cull-rect edge (WIDTH/2 + WIRE_CULL_MARGIN from
    # the center) is KEPT (the test is `>`, so the boundary is inclusive);
    # one just beyond it is dropped.
    edge = C.WIDTH / 2 + C.WIRE_CULL_MARGIN
    snap = _snap([_ship_snap(0, 0, 0), _ship_snap(1500, 0, 1)],
                 [_rock(1, 1500.0 + edge, 0.0),
                  _rock(2, 1500.0 + edge + 1.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker(),
                              cull_center=(1500.0, 0.0))
    ids = [r[0] for r in wire[5]]
    assert 1 in ids, "rock at the cull-rect edge was dropped"
    assert 2 not in ids, "rock just beyond the cull-rect edge was sent"


def test_no_cull_sends_all_rocks():
    # cull_center=None (default) sends ALL rocks (backward-compatible).
    snap = _snap([_ship_snap(0, 0, 0), _ship_snap(1500, 0, 1)],
                 [_rock(1, 0.0, 0.0), _rock(2, 5000.0, 0.0),
                  _rock(3, -5000.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker())
    assert len(wire[5]) == 3, "cull_center=None should send all rocks"


def test_remote_player_pos():
    # 2P snapshot -> the client's (player 1) pos.
    snap2 = _snap([_ship_snap(0, 0, 0), _ship_snap(1500, 0, 1)], [])
    assert _remote_player_pos(snap2) == (1500.0, 0.0), \
        "_remote_player_pos should return player 1's pos"
    # 1P snapshot -> None (no remote peer to cull for).
    snap1 = _snap([_ship_snap(0, 0, 0)], [])
    assert _remote_player_pos(snap1) is None, \
        "_remote_player_pos should return None for a 1P snapshot"


def main():
    checks = [
        ("rock in cull rect is sent", test_rock_in_cull_rect_is_sent),
        ("rock outside cull rect is dropped",
         test_rock_outside_cull_rect_is_dropped),
        ("cull rect boundary", test_cull_rect_boundary),
        ("no cull sends all rocks", test_no_cull_sends_all_rocks),
        ("_remote_player_pos", test_remote_player_pos),
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