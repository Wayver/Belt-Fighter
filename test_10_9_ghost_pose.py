"""Session 10.9: the prediction ghost's INTERPOLATED render pose.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_9_ghost_pose

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.9 makes the CLIENT's local ship (the prediction ghost) GLIDE like the
host's local ship instead of stepping in 60 Hz jumps. Before 10.9 the
ghost was drawn at its RAW post-step pose (self.ghost.ship.pos in
predicted_view). PredictedShip.advance() steps the ship in fixed 60 Hz
TICK chunks, but the display runs at its own rate (~62 Hz), so the raw
pose moves in discrete MAX_SPEED*TICK (~8.67 px) jumps whose phase drifts
against the display (freeze-jump-freeze-jump) = the "jagged / jittery"
local ship the user reported in live 2P.

The host's local ship does NOT have this problem: it is drawn at
_ship_pose(pack, step_alpha) — INTERPOLATED between the model's prev +
curr poses at the render clock's alpha. 10.9 gives the ghost the SAME
sub-step interpolation:

  * netcode.PredictedShip.advance() captures the ship's pose BEFORE each
    step into _prev_pos / _prev_angle.
  * render_pose() returns lerp(_prev, curr, acc/TICK) with the SAME
    shortest-arc angle math as the host's _ship_pose (game.py:335).
  * alpha is clamped to [0, 1] (the 70460de rule: on a hiccup the acc can
    exceed one step — render the latest simulated state, never
    extrapolate past the current pose).
  * seed() / reconcile() / reconcile_rewind() reset _prev = curr so a
    reconcile never adds a phantom lerp from the pre-reconcile pose.

render_pose() is a RENDER helper only — it does NOT touch self._ship.pos
(the ghost's actual simulated pose, which the no-teleport / rewind gates
and the missile/bullet handoff math read). predicted_view draws the local
ship, its beams (the 10.4c re-anchor), the scan pulse + sensor contacts,
and the camera target at render_pose() instead of the raw pose.

This test proves it (value-level — the lerp math is deterministic, so an
exact gate is possible):
  1. LERP-MATH     — render_pose() lerps between _prev + curr at the
                     right alpha: midpoint exact, endpoints exact at
                     alpha 0 (== _prev) and alpha 1 (== curr); the angle
                     uses the shortest-arc delta (including a wrap).
  2. NO-EXTRAPOLATE— alpha is clamped to [0, 1]: a hiccup (acc > TICK)
                     renders the CURRENT pose exactly (never past it); a
                     negative acc renders the PREV pose exactly.
  3. RECONCILE-ANCHOR — after a reconcile the render pose is the
                     reconciled pose EXACTLY (no phantom lerp from the
                     pre-reconcile _prev), even with a fractional acc.
  4. ADVANCE-CAPTURE — the regression lock: right after a whole number of
                     steps (acc == 0) render_pose returns the pose BEFORE
                     the last step (captured in advance), NOT the seeded
                     pose. This FAILS if the _prev capture in advance()
                     is removed (the render pose would lag a full step).
  5. NO-MUTATE     — render_pose() does not touch the ghost's actual
                     simulated pose (self._ship.pos / .angle).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import TICK
from .hulls import SILAS_HULL, default_loadout
from .ship import Ship
from .intent import ShipInput
from .netcode import PredictedShip

EPS = 1e-6   # pose-comparison tolerance (pygame Vector2 lerp is ~exact)


def make_ghost():
    """A bare prediction ghost seeded from a real silas ship snapshot.

    A bare Ship snapshot is a valid seed (the ghost is self-contained —
    no Game needed), which keeps this gate minimal + deterministic. The
    ship is given a non-zero velocity (0, 300) so each whole step moves it
    ~5 px — the reconcile-anchor + advance-capture checks need a
    measurable per-step displacement (a resting ship under thrust-only
    acceleration moves sub-pixel over a couple of ticks)."""
    s = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    s.vel = pygame.Vector2(0.0, 300.0)
    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g.seed(s.snapshot())
    return g


def _short_arc(a0, a1):
    """The host's _ship_pose shortest-arc delta (game.py:335)."""
    return (a1 - a0 + math.pi) % (2 * math.pi) - math.pi


def main():
    pygame.init()
    ok = True

    # --- (1) LERP-MATH: render_pose lerps _prev -> curr at alpha = acc/TICK.
    # Directly set the internal state (render_pose is a pure function of
    # _prev_pos/_prev_angle/_acc + the ship's pos/angle) so the lerp math
    # is checked exactly, independent of the physics.
    g = make_ghost()
    g._prev_pos = pygame.Vector2(0.0, 0.0)
    g._prev_angle = 0.0
    g.ship.pos = pygame.Vector2(100.0, 0.0)
    g.ship.angle = 0.0

    # Midpoint (alpha 0.5) -> (50, 0).
    g._acc = 0.5 * TICK
    rp, ra = g.render_pose()
    mid_ok = abs(rp.x - 50.0) < EPS and abs(rp.y - 0.0) < EPS and abs(ra) < EPS
    # Endpoint alpha 0 -> _prev exactly (0, 0).
    g._acc = 0.0
    rp0, ra0 = g.render_pose()
    a0_ok = abs(rp0.x) < EPS and abs(rp0.y) < EPS and abs(ra0) < EPS
    # Endpoint alpha 1 -> curr exactly (100, 0).
    g._acc = TICK
    rp1, ra1 = g.render_pose()
    a1_ok = abs(rp1.x - 100.0) < EPS and abs(rp1.y) < EPS and abs(ra1) < EPS
    if mid_ok and a0_ok and a1_ok:
        print(f"PASS: lerp-math — midpoint (50,0) + endpoints exact "
              f"(alpha 0 == _prev, alpha 1 == curr)")
    else:
        ok = False
        print(f"FAIL: lerp-math — mid={mid_ok} a0={a0_ok} a1={a1_ok} "
              f"(midpoint / alpha-0 / alpha-1 not exact)")

    # Angle: shortest-arc lerp. Straight case (0 -> pi/2, alpha 0.5 -> pi/4).
    g._prev_angle = 0.0
    g.ship.angle = math.pi / 2
    g._acc = 0.5 * TICK
    _, ra = g.render_pose()
    ang_straight = abs(ra - math.pi / 4) < EPS
    # Wrap case: _prev = +3.0, curr = -3.0 (the SAME direction, 2 pi apart
    # the long way). The shortest arc is +0.283 rad (3.0 -> 3.283 == -3.0
    # + 2pi), so the midpoint is 3.0 + 0.1415 == pi (not the long way
    # around through 0).
    g._prev_angle = 3.0
    g.ship.angle = -3.0
    da = _short_arc(3.0, -3.0)
    want_mid = 3.0 + da * 0.5
    g._acc = 0.5 * TICK
    _, ra_wrap = g.render_pose()
    ang_wrap = abs(ra_wrap - want_mid) < EPS
    if ang_straight and ang_wrap:
        print(f"PASS: lerp-math angle — straight midpoint pi/4 + wrap "
              f"midpoint {want_mid:.4f} (shortest arc {da:.4f} rad)")
    else:
        ok = False
        print(f"FAIL: lerp-math angle — straight={ang_straight} "
              f"wrap={ang_wrap} (got {ra_wrap:.4f}, want {want_mid:.4f})")

    # --- (2) NO-EXTRAPOLATE: alpha clamped to [0, 1] (the 70460de rule).
    g = make_ghost()
    g._prev_pos = pygame.Vector2(0.0, 0.0)
    g._prev_angle = 0.0
    g.ship.pos = pygame.Vector2(100.0, 0.0)
    g.ship.angle = 0.0
    # Hiccup: acc = 2*TICK (two steps behind) -> alpha clamps to 1 -> the
    # CURRENT pose exactly (never extrapolated past it).
    g._acc = 2.0 * TICK
    rp, _ = g.render_pose()
    no_extrap = abs(rp.x - 100.0) < EPS and abs(rp.y) < EPS
    # Negative acc (defensive) -> alpha clamps to 0 -> the PREV pose.
    g._acc = -0.5 * TICK
    rp_neg, _ = g.render_pose()
    no_neg = abs(rp_neg.x) < EPS and abs(rp_neg.y) < EPS
    if no_extrap and no_neg:
        print("PASS: no-extrapolate — hiccup (acc=2*TICK) renders curr "
              "exactly; negative acc renders _prev exactly (alpha clamped "
              "to [0,1])")
    else:
        ok = False
        print(f"FAIL: no-extrapolate — hiccup={no_extrap} neg={no_neg} "
              f"(alpha not clamped to [0,1])")

    # --- (3) RECONCILE-ANCHOR: a reconcile resets _prev = curr, so the
    # render pose is the reconciled pose EXACTLY (no phantom lerp from the
    # pre-reconcile _prev), even with a fractional acc.
    g = make_ghost()
    thrust = ShipInput(thrust_fwd=1.0)
    # Move the ghost a step + a half step: _prev != curr, acc fractional.
    g.advance(1.5 * TICK, thrust)
    pre_recon = g.render_pose()
    pre_frac = 0.0 < g._acc < TICK   # confirm we have a fractional acc
    # Reconcile to a far pose C.
    c = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    c.pos = pygame.Vector2(500.0, 500.0)
    c.angle = 0.5
    g.reconcile(c.snapshot())
    rp, ra = g.render_pose()
    recon_ok = (abs(rp.x - 500.0) < EPS and abs(rp.y - 500.0) < EPS
                and abs(ra - 0.5) < EPS)
    # The pre-reconcile render pose must have been a genuine lerp (not
    # already at C) — otherwise this check is vacuous.
    moved = math.hypot(pre_recon[0].x - 500.0, pre_recon[0].y - 500.0) > 1.0
    if recon_ok and pre_frac and moved:
        print(f"PASS: reconcile-anchor — after reconcile the render pose "
              f"is C exactly (no phantom lerp; pre-reconcile was "
              f"{moved:.0f}px away with a fractional acc)")
    else:
        ok = False
        print(f"FAIL: reconcile-anchor — recon={recon_ok} pre_frac="
              f"{pre_frac} moved={moved} (a reconcile added a phantom "
              f"lerp, or the setup was vacuous)")

    # --- (4) ADVANCE-CAPTURE: the regression lock. Right after a whole
    # number of steps (acc == 0) render_pose returns the pose BEFORE the
    # last step (captured in advance), NOT the seeded pose. If the _prev
    # capture in advance() is removed, _prev stays at the seeded pose and
    # this FAILS (the render pose lags a full step).
    g = make_ghost()
    thrust = ShipInput(thrust_fwd=1.0)
    g.advance(TICK, thrust)          # 1 step: ship at A1, _prev = A (seed)
    a1 = pygame.Vector2(g.ship.pos)  # capture the pose after step 1
    g.advance(TICK, thrust)          # 2 steps: ship at B, _prev = A1
    acc_zero = abs(g._acc) < EPS
    rp, _ = g.render_pose()          # alpha 0 -> _prev (should be A1)
    cap_ok = (acc_zero
              and math.hypot(rp.x - a1.x, rp.y - a1.y) < EPS
              and math.hypot(a1.x - g.ship.pos.x, a1.y - g.ship.pos.y) > 1.0)
    if cap_ok:
        print(f"PASS: advance-capture — after 2 whole steps (acc==0) the "
              f"render pose is the pre-last-step pose "
              f"({a1.x:.2f},{a1.y:.2f}), not the seeded pose "
              f"(captured in advance)")
    else:
        ok = False
        print(f"FAIL: advance-capture — acc_zero={acc_zero} "
              f"render=({rp.x:.2f},{rp.y:.2f}) want pre-last-step "
              f"({a1.x:.2f},{a1.y:.2f}): the _prev capture in advance() "
              f"is missing (render pose lags a full step)")

    # --- (5) NO-MUTATE: render_pose() is a render helper — it must not
    # touch the ghost's actual simulated pose (the no-teleport / rewind
    # gates + the missile/bullet handoff math read self._ship.pos).
    g = make_ghost()
    g.advance(1.5 * TICK, thrust)
    before = (pygame.Vector2(g.ship.pos), g.ship.angle)
    for _ in range(3):
        g.render_pose()
    after = (pygame.Vector2(g.ship.pos), g.ship.angle)
    no_mut = (before[0] == after[0] and before[1] == after[1])
    if no_mut:
        print("PASS: no-mutate — render_pose() leaves the ghost's "
              "simulated pose untouched (repeated calls)")
    else:
        ok = False
        print(f"FAIL: no-mutate — render_pose() mutated the simulated "
              f"pose ({before} -> {after})")

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()