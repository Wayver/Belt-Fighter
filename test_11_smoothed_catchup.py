"""11.x SMOOTHED-POSE CATCH-UP regression gate.

The bug: `render_pose_smoothed` capped its per-frame catch-up at the ship's
ACTUAL speed * dt. At rest (vel == 0, no turn input) that cap is 0, so once a
stall/burst left the smoothed pose BEHIND the render target, the gap could
NEVER close — the ship sat still while its smoothed pose (and therefore its
muzzle) stayed stuck at the pre-stall position. Bullets fired from that stale
pose appeared to come from empty space and never got closer.

The fix is a VELOCITY-BASED CONTROLLER (feedforward + proportional): the
smoothed pose's velocity = the ship's velocity (feedforward: track the ship's
motion exactly, zero steady-state lag) + GAIN * the gap (proportional: close
any stall gap, damped). The gap closes with a time constant of 1/GAIN,
independent of the ship's motion (the feedforward cancels it), so the gap
closes at rest AND while the ship is moving (no rubber-band).

This gate is a pure function of the ghost's internal render state (no Game,
no network), so it is deterministic and fast:

  (1) POSITION CATCH-UP AT REST — the smoothed pose starts 100 px behind the
      (static) render target, the ship is at rest (vel 0, no input). Over
      successive frames the gap must MONOTONICALLY shrink and reach ~0. With
      the old actual-speed cap the gap would stay 100 px forever (FAIL).
  (2) ANGLE CATCH-UP AT REST — same, for a stale heading while not turning.
  (3) STEADY-STATE ZERO LAG — a ship moving at max speed with the smoothed
      pose seeded ON the target must stay ON the target (the feedforward must
      track the ship exactly, no lag).
  (4) NO RUBBER-BAND ON START MOVING — a stall leaves a gap, then the ship
      starts moving at max speed. The gap must keep CLOSING (not grow) — the
      feedforward cancels the ship's motion, so the gap dynamics are
      independent of the ship's velocity. A constant-velocity slide (the
      floor fix) has no feedforward and rubber-bands here.

Run:  python -m ship5.test_11_smoothed_catchup
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import TICK, MAX_SPEED, ROT_SPEED
from .hulls import SILAS_HULL, default_loadout
from .ship import Ship
from .intent import ShipInput
from .netcode import PredictedShip

EPS = 1e-3
DT = TICK  # one 60 Hz frame


def make_ghost():
    s = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g.seed(s.snapshot())
    return g


def _set_rest(g, pos, angle):
    """Put the ghost at rest at `pos`/`angle` with a static render target.

    vel 0 + no input + _acc 0 makes render_pose() return exactly (pos,
    angle) every frame (the pre-step integrates zero velocity / zero turn).
    """
    g.ship.pos = pygame.Vector2(pos)
    g.ship.angle = angle
    g.ship.vel = pygame.Vector2(0.0, 0.0)
    g._acc = 0.0
    g._last_inp = ShipInput()  # turn 0, no thrust


def test_position_catchup_at_rest():
    g = make_ghost()
    _set_rest(g, (100.0, 0.0), 0.0)
    # Start the smoothed pose 100 px BEHIND the target (the post-stall gap).
    g._smooth_pos = pygame.Vector2(0.0, 0.0)
    g._smooth_angle = 0.0
    g._smooth_dead = False

    target = pygame.Vector2(100.0, 0.0)
    prev_gap = (target - g._smooth_pos).length()
    closed = False
    for _ in range(60):  # up to 1 s of frames
        sp, _ = g.render_pose_smoothed(DT)
        gap = (target - sp).length()
        # The gap must never GROW (monotonic approach).
        assert gap <= prev_gap + EPS, (
            f"gap grew {prev_gap:.3f} -> {gap:.3f} (should only shrink)")
        prev_gap = gap
        if gap < EPS:
            closed = True
            break
    assert closed, (
        f"smoothed pose never caught up at rest: final gap {prev_gap:.3f} px "
        f"(old actual-speed cap = 0 at rest -> gap is permanent)")
    return True


def test_angle_catchup_at_rest():
    g = make_ghost()
    _set_rest(g, (0.0, 0.0), 0.0)
    # Start the smoothed heading 0.5 rad STALE (the post-stall angle gap).
    g._smooth_pos = pygame.Vector2(0.0, 0.0)
    g._smooth_angle = 0.5
    g._smooth_dead = False

    target_angle = 0.0
    prev_gap = abs((target_angle - g._smooth_angle + math.pi)
                   % (2 * math.pi) - math.pi)
    closed = False
    for _ in range(60):
        _, sa = g.render_pose_smoothed(DT)
        gap = abs((target_angle - sa + math.pi) % (2 * math.pi) - math.pi)
        assert gap <= prev_gap + EPS, (
            f"angle gap grew {prev_gap:.3f} -> {gap:.3f}")
        prev_gap = gap
        if gap < EPS:
            closed = True
            break
    assert closed, (
        f"smoothed angle never caught up at rest: final gap {prev_gap:.3f} rad "
        f"(old actual-turn-rate cap = 0 when not turning -> stale heading)")
    return True


def test_steady_state_zero_lag():
    """A ship moving at max speed, smoothed pose seeded ON the target, must
    stay EXACTLY ON the target every frame (the hybrid snaps to the target
    in steady state — zero per-frame error, no vibration)."""
    g = make_ghost()
    v = MAX_SPEED * g.ship.hull.max_speed_factor
    g.ship.pos = pygame.Vector2(0.0, 0.0)
    g.ship.angle = 0.0
    g.ship.vel = pygame.Vector2(v, 0.0)
    g._acc = 0.0
    g._last_inp = ShipInput()
    # Seed the smoothed pose exactly on the current render target.
    g._smooth_pos = pygame.Vector2(0.0, 0.0)
    g._smooth_angle = 0.0
    g._smooth_dead = False

    max_lag = 0.0
    for _ in range(30):
        # Advance the physics pose by one tick (what advance() does per step)
        # so the render target moves at the ship's actual (max) speed.
        g.ship.pos += g.ship.vel * TICK
        sp, _ = g.render_pose_smoothed(DT)
        # The target this frame is the new physics pose (vel*TICK pre-step,
        # _acc 0 -> base = s.pos). The hybrid SNAPS to the target in steady
        # state, so the error must be EXACTLY 0 (no vibration).
        lag = (g.ship.pos - sp).length()
        max_lag = max(max_lag, lag)
    assert max_lag < EPS, (
        f"steady-state error {max_lag:.4f} px — the smoothed pose must be "
        f"EXACTLY on the target in steady state (the hybrid snaps; a "
        f"feedforward-only tracker leaves a per-frame vibration)")
    return True


def test_steady_state_no_vibration():
    """The vibration regression: at a CRUISING speed (below max), the
    smoothed pose must be EXACTLY the render target every frame — including
    on a BURST frame where the target advances 2 ticks in one frame (the
    worst-case steady-state gap). A feedforward-only tracker (no snap)
    leaves a per-frame error of up to ~1 tick (the feedforward integrates
    over the smooth frame time dt, but the target advances in ragged tick
    chunks) — a constant vibration visible from game start. The hybrid
    snaps whenever the gap is <= 2 ticks of motion at max speed, which
    covers the worst-case steady-state gap, so the error is exactly 0.
    """
    g = make_ghost()
    v = 300.0  # a realistic cruising speed (below max)
    g.ship.pos = pygame.Vector2(0.0, 0.0)
    g.ship.angle = 0.0
    g.ship.vel = pygame.Vector2(v, 0.0)
    g._acc = 0.0
    g._last_inp = ShipInput()
    g._smooth_pos = pygame.Vector2(0.0, 0.0)
    g._smooth_angle = 0.0
    g._smooth_dead = False

    max_err = 0.0
    for i in range(120):
        # Alternate 1-tick and 2-tick (burst) frames — the sim clock
        # advances 0, 1, or 2 ticks per frame in real play.
        ticks = 2 if i % 3 == 0 else 1
        g.ship.pos += g.ship.vel * TICK * ticks
        sp, _ = g.render_pose_smoothed(DT)
        err = (g.ship.pos - sp).length()
        max_err = max(max_err, err)
    assert max_err < EPS, (
        f"steady-state vibration: max per-frame error {max_err:.4f} px "
        f"(the smoothed pose must be EXACTLY the target in steady state — "
        f"the hybrid snaps; a feedforward-only tracker vibrates)")
    return True


def test_no_rubberband_on_start_moving():
    """The key scenario: a stall leaves a gap, then the player starts
    moving. The gap must keep CLOSING (not grow / rubber-band) when the
    ship starts moving. This is the property the feedforward gives: the
    gap dynamics are d(gap)/dt = -GAIN*gap, INDEPENDENT of the ship's
    velocity (the feedforward cancels the ship's motion). A
    constant-velocity slide (the floor fix) has no feedforward, so when
    the ship starts moving at max speed the slide (also at max speed)
    can't close the gap — it stays stuck or grows (the rubber-band).
    """
    g = make_ghost()
    # Start at rest with a 100 px gap (a stall just happened).
    _set_rest(g, (100.0, 0.0), 0.0)
    g._smooth_pos = pygame.Vector2(0.0, 0.0)
    g._smooth_angle = 0.0
    g._smooth_dead = False

    # Let the gap close a bit at rest (a few frames).
    for _ in range(5):
        g.render_pose_smoothed(DT)
    prev_gap = (pygame.Vector2(100.0, 0.0) - g._smooth_pos).length()

    # Now the player starts moving at max speed. The gap must keep
    # closing (not grow) — the feedforward cancels the ship's motion, so
    # the gap dynamics are independent of the ship's velocity.
    v = MAX_SPEED * g.ship.hull.max_speed_factor
    g.ship.vel = pygame.Vector2(v, 0.0)
    closed = False
    for _ in range(60):
        # Advance the physics pose (the target moves at the ship's speed).
        g.ship.pos += g.ship.vel * TICK
        sp, _ = g.render_pose_smoothed(DT)
        gap = (g.ship.pos - sp).length()
        # The gap must not grow when the ship starts moving (no
        # rubber-band). Allow a 1 px tolerance for the feedforward to
        # catch up to the new velocity.
        assert gap <= prev_gap + 1.0, (
            f"gap grew when the ship started moving: {prev_gap:.2f} -> "
            f"{gap:.2f} px (rubber-band — no feedforward)")
        prev_gap = gap
        if gap < EPS:
            closed = True
            break
    assert closed, (
        f"gap did not close while the ship was moving: final gap "
        f"{prev_gap:.2f} px (the slide can't close the gap at max speed "
        f"— no feedforward)")
    return True


def main():
    pygame.init()
    ok = True
    for name, fn in [
        ("position catch-up at rest", test_position_catchup_at_rest),
        ("angle catch-up at rest", test_angle_catchup_at_rest),
        ("steady-state zero lag", test_steady_state_zero_lag),
        ("steady-state no vibration", test_steady_state_no_vibration),
        ("no rubber-band on start moving", test_no_rubberband_on_start_moving),
    ]:
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