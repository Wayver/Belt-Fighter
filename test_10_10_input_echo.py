"""Session 10.10: the input-ECHO wire change — the high-speed snapback fix.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_10_input_echo

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

The high-speed snapback (deferred from 7.6, carried through 10.9) is the
ghost's reconcile yank. The dominant visible component is IN-FLIGHT INPUT:
the client's reconcile_rewind replays the ticks since the snapshot with the
client's OWN sent input, but the host actually applied the latest input it
had RECEIVED (up to one-way latency stale). On an input CHANGE (turn/thrust)
the ghost predicts with the new input while the host is still on the old one
-> divergence ~MAX_SPEED x latency -> the reconcile snaps the ghost back.

The fix (10.10): the host ECHOES the input it actually applied to player 1
(the client's ship), per tick, over a new T_ECHO message (the T_BEAM pattern
— per-tick event data that must not bloat the pruned snapshot; the snapshot
stays pruned, D1). The client's rewind replay uses the ECHO buffer instead of
its own sent input. A dropped echo degrades gracefully (the replay holds the
last echoed input — exactly what the host itself did).

This test proves it (value-level):
  1. ECHO-CAPTURE  — Game._step records the POST-CAP player-1 input, tagged
                     at the tick's START (sim_time - dt), and ONLY player 1's
                     (player 0's input is never echoed). flush returns +
                     clears.
  2. WORLD-CAP     — when the world is full (len(bullets) >= MAX_BULLETS) the
                     echo carries fire=False (the POST-CAP input — the ghost's
                     step() does NOT apply the world cap, so echoing the raw
                     input would re-introduce a fire divergence). When the
                     world has room the echo carries fire=True.
  3. SHAPE/ROUND-TRIP — the T_ECHO message the sim thread sends (a list of
                     [sim_time, inp_dict] entries) survives the wire and
                     inverts back to the exact (sim_time, ShipInput) pairs.
  4. ECHO-REPLAY   — the rewind's replay is driven by the ECHO buffer: with
                     the own buffer on no-thrust and the echo on thrust, the
                     ghost MOVES after the rewind (it followed the echo, not
                     its own input). This is the snapback collapsing.
  5. ECHO-PRIORITY — with BOTH buffers populated but DIFFERENT, the echo wins
                     (the ghost follows the echo's input, not the own's).
  6. FALLBACK      — with an EMPTY echo buffer the replay falls back to the
                     own buffer (the 7.6 behavior: initial transient / loss).
  7. BOUND/ORDER   — record_echo bounds to INPUT_BUFFER_MAX (2 s) of sim time
                     (oldest dropped) and preserves arrival order.
  8. RESET-CLEAR   — Game.reset clears the echo queue (no stale echoes).

PROVEN-REAL: the ECHO-REPLAY + ECHO-PRIORITY checks FAIL if the replay is
reverted to read the own buffer (the ghost would not move / would follow the
own input instead of the echo's).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT, TICK, MAX_BULLETS
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .bullets import Bullet
from .intent import ShipInput
from .netcode import PredictedShip
from .game import Game, STEP
from .net import (T_ECHO, serialize_input, deserialize_input,
                  encode_frame, extract_frames)
from .fog import make_light_texture

# A gun-fitted test hull (the 10.7 pattern): the scout polygon + the stock
# slots + a 'gun' weapon slot. default_loadout(hull) maps 'gun' -> GUN_TYPE,
# so the stock loadout gives this hull a pulse gun (needed for the WORLD-CAP
# gate — the world cap strips fire/missile_fire when the world is full).
GUN_HULL = HullType(
    id='gun_test',
    polygon=((18, 0), (14, 3.5), (8, 6.5), (0, 8), (-8, 8), (-12, 11),
             (-12, 5), (-9, 3), (-9, -3), (-12, -5), (-12, -11), (-8, -8),
             (0, -8), (8, -6.5), (14, -3.5)),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN,
           REACTOR, COMPUTER, SHIELD, SENSOR),
    base_mass=1.0,
    collision_radius=12.0,
    nose=(18, 0),
    cockpit=(8, 0),
    max_speed_factor=1.0,
    turn_rate_factor=1.0,
    fill=(200, 200, 200),
    edge=(255, 255, 255),
)
GUN_LOADOUT = default_loadout(GUN_HULL)

REPLAY_TICKS = 6   # the rewind window (ticks) for the replay gates
EPS = 1e-6


class Keys:
    """A minimal stand-in for pygame.key.get_pressed() — idle (all keys
    released). Game.update(dt, keys) calls ShipInput.from_keys(keys), so
    player 0's input is driven by this (idle), while player 1's input is
    set directly on g.remote_input."""
    def __getitem__(self, k):
        return 0


IDLE_KEYS = Keys()


def make_res():
    """The pygame res tuple a Game needs (headless)."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return (screen, font, big_font, light_tex, fog_surf, light_surf)


def make_host(res):
    """A 2P host Game (player 0 = the gun hull, player 1 = the client's
    placeholder ship), ISOLATED (no enemies/asteroids, update_field no-op'd)
    so the only thing moving a ship is the input. Mirrors the
    test_interpolation (l) isolation."""
    import ship5.game as _gm
    g = Game(*res, hull=GUN_HULL, loadout=GUN_LOADOUT, seed=1234,
             players=2, local_index=0)
    g.enemies.clear()
    g.asteroids.clear()
    return g


def make_replay_ghost():
    """A bare prediction ghost (the client's local ship = player 1) seeded
    from a plain Ship at the origin, vel 0, angle 0. A plain Ship (no gun)
    keeps the replay self-contained — only turn/thrust/fire move it, so the
    echo-vs-own input selection is the only thing that matters."""
    s = Ship()
    s.pos = pygame.Vector2(0.0, 0.0)
    s.vel = pygame.Vector2(0.0, 0.0)
    s.angle = 0.0
    g = PredictedShip(local_index=1)
    g.seed(s.snapshot())
    return g, s.snapshot()


def main():
    pygame.init()
    ok = True
    res = make_res()

    # --- (1) ECHO-CAPTURE: Game._step records the POST-CAP player-1 input,
    # tagged at the tick's START (sim_time - dt), and ONLY player 1's.
    g = make_host(res)
    _gm = __import__("ship5.game", fromlist=["update_field"])
    orig_uf = _gm.update_field
    _gm.update_field = lambda *a, **k: None
    try:
        inps = [ShipInput(thrust_fwd=1.0),
                ShipInput(turn=1.0),
                ShipInput(thrust_fwd=1.0, fire=True)]
        for i, inp in enumerate(inps):
            g.remote_input = inp
            g.update(STEP, IDLE_KEYS)   # player 0 = idle (never echoed)
        echo = g.flush_remote_input_echo()
        # One entry per tick, tagged at the tick's START (i*STEP), carrying
        # the player-1 input (NOT player 0's idle input).
        cap_ok = (len(echo) == 3
                  and all(abs(st - i * STEP) < EPS
                          for (st, _), i in zip(echo, range(3)))
                  and all(e == inp for (_, e), inp in zip(echo, inps)))
        # flush CLEARS the queue (the next flush is empty).
        cleared = g.flush_remote_input_echo() == []
        if cap_ok and cleared:
            print(f"PASS: echo-capture — {len(echo)} entries, tagged at the "
                  f"tick START (i*STEP), carrying the player-1 input (not "
                  f"player 0's); flush returns + clears")
        else:
            ok = False
            print(f"FAIL: echo-capture — cap_ok={cap_ok} cleared={cleared} "
                  f"echo={[(round(st, 4), e) for st, e in echo]}")
    finally:
        _gm.update_field = orig_uf

    # --- (2) WORLD-CAP: the echo carries the POST-CAP input. When the world
    # is full (len(bullets) >= MAX_BULLETS) fire is stripped -> the echo
    # carries fire=False (the ghost's step() does NOT apply the world cap, so
    # echoing the raw input would re-introduce a fire divergence). When the
    # world has room the echo carries fire=True.
    g = make_host(res)
    _gm = __import__("ship5.game", fromlist=["update_field"])
    orig_uf = _gm.update_field
    _gm.update_field = lambda *a, **k: None
    try:
        # Fill the world to the cap with real Bullets (the cap checks
        # len(self.bullets) >= MAX_BULLETS).
        g.bullets = [Bullet(pygame.Vector2(100.0, 100.0),
                            pygame.Vector2(1.0, 0.0), owner=1)
                     for _ in range(MAX_BULLETS)]
        g.remote_input = ShipInput(fire=True)
        g.update(STEP, IDLE_KEYS)
        echo_full = g.flush_remote_input_echo()
        # The world is full -> fire is stripped -> the echo carries
        # fire=False (the POST-CAP input).
        wc_full = (len(echo_full) == 1
                   and echo_full[0][1].fire is False
                   and len(g.bullets) == MAX_BULLETS)   # no new bullet fired
        # Control: the world has room -> fire is NOT stripped -> the echo
        # carries fire=True.
        g2 = make_host(res)
        g2.remote_input = ShipInput(fire=True)
        g2.update(STEP, IDLE_KEYS)
        echo_room = g2.flush_remote_input_echo()
        wc_room = (len(echo_room) == 1
                   and echo_room[0][1].fire is True)
        if wc_full and wc_room:
            print(f"PASS: world-cap — full world -> echo fire=False "
                  f"(POST-CAP, no new bullet: {len(g.bullets)} "
                  f"== MAX_BULLETS); room -> echo fire=True")
        else:
            ok = False
            print(f"FAIL: world-cap — full={wc_full} room={wc_room} "
                  f"(full echo={echo_full}, room echo={echo_room})")
    finally:
        _gm.update_field = orig_uf

    # --- (3) SHAPE/ROUND-TRIP: the T_ECHO message the sim thread sends (a
    # list of [sim_time, inp_dict] entries) survives the wire and inverts
    # back to the exact (sim_time, ShipInput) pairs.
    g = make_host(res)
    _gm = __import__("ship5.game", fromlist=["update_field"])
    orig_uf = _gm.update_field
    _gm.update_field = lambda *a, **k: None
    try:
        inps = [ShipInput(turn=1.0, thrust_fwd=1.0),
                ShipInput(turn=-1.0, fire=True),
                ShipInput()]
        for inp in inps:
            g.remote_input = inp
            g.update(STEP, IDLE_KEYS)
        echo = g.flush_remote_input_echo()
        # Build the T_ECHO message the sim thread sends (the same shape).
        msg = {"type": T_ECHO,
               "entries": [[t, serialize_input(inp)] for (t, inp) in echo]}
        msgs, rest = extract_frames(encode_frame(msg))
        rt_ok = (rest == b"" and len(msgs) == 1
                 and msgs[0].get("type") == T_ECHO
                 and len(msgs[0]["entries"]) == len(inps))
        if rt_ok:
            for (st, d), (ost, oinp) in zip(msgs[0]["entries"], echo):
                if abs(st - ost) > EPS or deserialize_input(d) != oinp:
                    rt_ok = False
                    break
        if rt_ok:
            print(f"PASS: shape/round-trip — T_ECHO with {len(inps)} entries "
                  f"survives the wire; sim_time + input invert exactly")
        else:
            ok = False
            print(f"FAIL: shape/round-trip — rt_ok={rt_ok} "
                  f"(msgs={msgs}, rest={rest!r})")
    finally:
        _gm.update_field = orig_uf

    # --- (4) ECHO-REPLAY: the rewind's replay is driven by the ECHO buffer.
    # The own buffer is on no-thrust, the echo on thrust -> the ghost MOVES
    # after the rewind (it followed the echo, not its own input). This is
    # the snapback collapsing. PROVEN-REAL: reverting the replay to read the
    # own buffer makes this FAIL (the ghost would not move).
    g, snap = make_replay_ghost()
    for i in range(REPLAY_TICKS):
        g.record_input(i * TICK, ShipInput())          # own: no-thrust
        g.record_echo(i * TICK, ShipInput(thrust_fwd=1.0))   # echo: thrust
    g.reconcile_rewind(snap, 0.0, REPLAY_TICKS * TICK)
    disp = g.ship.pos.length()
    er_ok = disp > 1.0   # the ghost moved (followed the echo's thrust)
    if er_ok:
        print(f"PASS: echo-replay — own=no-thrust, echo=thrust -> the ghost "
              f"moved {disp:.3f}px after the rewind (followed the ECHO, not "
              f"its own input — the snapback collapsing)")
    else:
        ok = False
        print(f"FAIL: echo-replay — the ghost did not move "
              f"(disp={disp:.3f}px): the replay is not driven by the echo "
              f"buffer (it read the own buffer, which is no-thrust)")

    # --- (5) ECHO-PRIORITY: with BOTH buffers populated but DIFFERENT, the
    # echo wins (the ghost follows the echo's input, not the own's).
    g, snap = make_replay_ghost()
    for i in range(REPLAY_TICKS):
        g.record_input(i * TICK, ShipInput(thrust_fwd=1.0))   # own: thrust
        g.record_echo(i * TICK, ShipInput())                  # echo: no-thrust
    g.reconcile_rewind(snap, 0.0, REPLAY_TICKS * TICK)
    disp = g.ship.pos.length()
    ep_ok = disp < EPS   # the ghost did NOT move (followed the echo's no-thrust)
    if ep_ok:
        print(f"PASS: echo-priority — own=thrust, echo=no-thrust -> the ghost "
              f"did NOT move (disp={disp:.3f}px): the echo wins over the own "
              f"buffer")
    else:
        ok = False
        print(f"FAIL: echo-priority — the ghost moved {disp:.3f}px: the "
              f"replay followed the OWN buffer, not the echo (the echo must "
              f"win when both are populated)")

    # --- (6) FALLBACK: with an EMPTY echo buffer the replay falls back to
    # the own buffer (the 7.6 behavior: initial transient / loss).
    g, snap = make_replay_ghost()
    for i in range(REPLAY_TICKS):
        g.record_input(i * TICK, ShipInput(thrust_fwd=1.0))   # own: thrust
        # no record_echo -> the echo buffer is empty
    g.reconcile_rewind(snap, 0.0, REPLAY_TICKS * TICK)
    disp = g.ship.pos.length()
    fb_ok = disp > 1.0   # the ghost moved (fell back to the own buffer's thrust)
    if fb_ok:
        print(f"PASS: fallback — empty echo, own=thrust -> the ghost moved "
              f"{disp:.3f}px (fell back to the own buffer — the 7.6 "
              f"behavior)")
    else:
        ok = False
        print(f"FAIL: fallback — the ghost did not move (disp={disp:.3f}px): "
              f"with an empty echo buffer the replay must fall back to the "
              f"own buffer")

    # --- (7) BOUND/ORDER: record_echo bounds to INPUT_BUFFER_MAX (2 s) of
    # sim time (oldest dropped) and preserves arrival order.
    g, _ = make_replay_ghost()
    # Record entries spanning 3 s (more than the 2 s bound).
    for i in range(180):   # 180 ticks = 3 s
        g.record_echo(i * TICK, ShipInput())
    # The oldest entries (before t = 3s - 2s = 1s) are dropped.
    bound_ok = (abs(g._echo_buffer[0][0] - 1.0 * TICK * 60) < 1e-3
                and len(g._echo_buffer) == 120)   # 2 s = 120 ticks remain
    # Order is preserved (arrival order == sim_time order).
    order_ok = all(g._echo_buffer[i][0] <= g._echo_buffer[i + 1][0]
                   for i in range(len(g._echo_buffer) - 1))
    if bound_ok and order_ok:
        print(f"PASS: bound/order — record_echo bounds to INPUT_BUFFER_MAX "
              f"(2 s): {len(g._echo_buffer)} entries remain (oldest "
              f"dropped), arrival order preserved")
    else:
        ok = False
        print(f"FAIL: bound/order — bound_ok={bound_ok} order_ok={order_ok} "
              f"(len={len(g._echo_buffer)}, first="
              f"{g._echo_buffer[0][0] if g._echo_buffer else None})")

    # --- (8) RESET-CLEAR: Game.reset clears the echo queue (no stale
    # echoes carry across a reset/respawn).
    g = make_host(res)
    _gm = __import__("ship5.game", fromlist=["update_field"])
    orig_uf = _gm.update_field
    _gm.update_field = lambda *a, **k: None
    try:
        g.remote_input = ShipInput(thrust_fwd=1.0)
        g.update(STEP, IDLE_KEYS)
        pre = len(g._remote_input_echo)
        g.reset()
        post = len(g._remote_input_echo)
        rc_ok = pre >= 1 and post == 0
        if rc_ok:
            print(f"PASS: reset-clear — reset clears the echo queue "
                  f"({pre} -> {post})")
        else:
            ok = False
            print(f"FAIL: reset-clear — pre={pre} post={post} (reset did "
                  f"not clear the echo queue)")
    finally:
        _gm.update_field = orig_uf

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()