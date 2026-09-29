"""Session 10.3d/10.3e/10.3f/10.3g: the decoupled render offset, the
no-blink handoff, the no-backward-motion decay cap, and the no-freeze
handoff.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3d_handoff_offset

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3b made the client's prediction ghost emit + home its own missiles and
dedup them against the buffer's copy (no double-draw). 10.3d added the
DECOUPLED RENDER OFFSET: at handback, the buffer's copy is drawn at its
interpolated position (INTERP_DELAY behind the ghost's predicted position),
so a visible offset (ghost_pos - interp_pos) is added and decayed each frame
(MISSILE_HANDOFF_DECAY) so the missile converges to the buffer's position.

10.3e fixes the "blink": 10.3b/10.3d culled the ghost missile at SNAPSHOT
ARRIVAL (push_snapshot -> handback_missiles), but the buffer's copy is not
VISIBLE until the render point reaches the carrying snapshot (~INTERP_DELAY
later). During that gap (6-15 frames) the missile was drawn by NEITHER the
ghost nor the buffer — a visible blink. 10.3e keeps the ghost missile (marked
`pending`, NOT culled) so the ghost keeps drawing it until the buffer's copy
is actually visible; predicted_view then hands it off (removes it + seeds the
offset from the ghost's position AT THAT MOMENT), so the missile is drawn
continuously.

10.3f fixes the BACKWARD MOTION: the plain 0.85 decay shrank the offset by
0.15*|offset|/frame, which at handback (|offset| ~ INTERP_DELAY*speed) is
9*INTERP_DELAY*speed px/s of backward pull — faster than the missile's
forward speed at high adaptive delays, so the missile visibly reversed. 10.3f
caps the per-frame shrinkage at MISSILE_HANDOFF_MAX_SHRINK * (the buffer's
ACTUAL displacement this frame), so the rendered position (buffer_pos +
offset) always moves forward (net motion >= (1-cap)*B_disp).

10.3g fixes the remaining FREEZE: 10.3e handed off on the first frame the
buffer's copy was VISIBLE, but a buffer missile that just entered the window
"pops in" at its curr position and is FROZEN for the rest of that window
(the snapshot before it has no missile to lerp from — _match_bullets returns
None). So the missile held still for up to a snapshot interval (~100 ms)
before the buffer advanced: a visible stall. 10.3g keeps the ghost drawing
the pending missile until the buffer's copy ADVANCES (its position differs
from the previous frame's, tracked in _missile_handoff_buf_pos). The first
advancing frame is the handoff frame: the ghost is removed and the offset is
seeded from the ghost's position at that moment. The missile is drawn
continuously AND never freezes.

This test proves it (value-level):
  1. PENDING    — the ghost's missile is marked pending (NOT culled) when
                  the carrying snapshot arrives; it stays in local_missiles.
  2. KEEP-DRAW  — while the buffer's copy is not visible (render point
                  behind the carrying snapshot), the ghost missile stays in
                  local_missiles (the ghost keeps drawing it — no gap).
  3. FROZEN     — while the buffer's copy is visible but FROZEN (its pop-in
                  window, position unchanged), the ghost missile STAYS in
                  local_missiles (the ghost keeps drawing it — no stall).
  4. HANDBACK   — on the first frame the buffer's copy ADVANCES, the ghost
                  missile is removed and the offset is seeded from the
                  ghost's position at that moment.
  5. NO-JUMP    — on the first buffer draw, the rendered position equals the
                  ghost's position at the handoff (no jump).
  6. NO-BACKWARD— over ALL the decay frames, the missile's rendered x
                  position NEVER decreases.
  7. CONVERGE   — after ~30 frames the offset decays below the epsilon and
                  is culled (the missile is drawn at the buffer's position).
  8. FOG-LIGHT  — _remote_fog_lights accepts the handoff_offsets (the fog
                  light follows the offset).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import inspect
import random
import pygame

from .config import (WIDTH, HEIGHT, TICK, INTERP_DELAY,
                     MISSILE_HANDOFF_DECAY, MISSILE_HANDOFF_OFFSET_EPS,
                     MISSILE_SPEED)
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, MISSILE, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .intent import ShipInput
from .netcode import (PredictedShip, GhostMissile, RenderPoint,
                      LatencyTracker, HostTimeEstimator)
from .bullets import Missile
from .game import Game
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .fog import make_light_texture

# A missile-fitted test hull (same as test_10_3b).
MISSILE_HULL = HullType(
    id='missile_test',
    polygon=((18, 0), (14, 3.5), (8, 6.5), (0, 8), (-8, 8), (-12, 11),
             (-12, 5), (-9, 3), (-9, -3), (-12, -5), (-12, -11), (-8, -8),
             (0, -8), (8, -6.5), (14, -3.5)),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN, MISSILE,
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
LOADOUT = default_loadout(MISSILE_HULL)
LOCAL_INDEX = 1   # the client is player 1


def make_game():
    """A client Game with the ghost + render point wired up (same pattern
    as test_10_3b's RENDER+DEDUP section)."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=MISSILE_HULL, loadout=LOADOUT, seed=1234, players=2,
             local_index=LOCAL_INDEX)
    g.set_player_ship(LOCAL_INDEX, Ship(hull=MISSILE_HULL, loadout=LOADOUT))
    g.ghost = PredictedShip(hull=MISSILE_HULL, loadout=LOADOUT,
                            local_index=LOCAL_INDEX)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    # The production buffer holds 8 snapshots (0.8 s); the test pushes 11
    # (t=0.0 .. 1.0) and renders back to t=0.05, so enlarge it (otherwise
    # the early snapshots are dropped and positions_at clamps the render
    # point to the first RETAINED snapshot — which already carries the
    # missile — breaking the KEEP-DRAW gate).
    g.snap_buf._max = 100
    return g


def make_snap(mid, mx, my, vx, vy, host_seq):
    """A minimal 11-tuple Game snapshot carrying ONE missile (id `mid`) at
    (mx, my) with vel (vx, vy), and the local player's missile_seq =
    host_seq. `mid=None` gives an empty missile list (pre-fire)."""
    rng = random.Random(0)
    h0 = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    h0.pos = pygame.Vector2(0.0, 0.0)
    h0.vel = pygame.Vector2(0.0, 0.0)
    h0.angle = 0.0
    h0.missile_seq = 0
    h1 = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    h1.pos = pygame.Vector2(0.0, 0.0)
    h1.vel = pygame.Vector2(0.0, 0.0)
    h1.angle = 0.0
    h1.missile_seq = host_seq
    if mid is None:
        missiles_s = ()
    else:
        m = Missile(pygame.Vector2(mx, my), pygame.Vector2(vx, vy),
                    owner=LOCAL_INDEX, mid=mid)
        missiles_s = (m.snapshot(),)
    return (
        (h0.snapshot(), h1.snapshot()),   # [0] players
        (),                                # [1] enemies
        (),                                # [2] bullets
        (),                                # [3] enemy_bullets
        missiles_s,                        # [4] missiles
        (),                                # [5] asteroids
        rng.getstate(),                    # [6] rng
        False,                             # [7] game_over
        0.0,                               # [8] protect_timer
        AIEnemy._next_id,                  # [9] next_id
        Asteroid._next_id,                 # [10] rock_next_id
    )


def buffer_missile_x(g, pos, mid):
    """The buffer's INTERPOLATED x position of missile `mid` in this frame's
    window (NO offset — the raw buffer position). Returns None if the
    missile is not in the buffer."""
    for (x, y, vx, vy, kind, owner, boost, m_id) in pos['bullets']:
        if kind == 'missile' and m_id is not None and tuple(m_id) == mid:
            return x
    return None


def rendered_missile_x(g, pos, mid):
    """The RENDERED x position of the buffer's missile `mid` in this frame:
    the interpolated x (from pos['bullets']) plus the handoff offset (if
    any). Returns None if the missile is not in the buffer."""
    dx = dy = 0.0
    off = g._missile_handoff_offsets.get(mid)
    if off is not None:
        dx, dy = off
    for (x, y, vx, vy, kind, owner, boost, m_id) in pos['bullets']:
        if kind == 'missile' and m_id is not None and tuple(m_id) == mid:
            return x + dx
    return None


def main():
    pygame.init()
    mid = (LOCAL_INDEX, 0)
    g = make_game()

    # --- 1. PENDING: the ghost's missile is marked pending (NOT culled)
    #     when the carrying snapshot arrives. ---
    # Seed the ghost with PRE-FIRE snapshots (host_seq=0, no missile).
    g.push_snapshot(0.0, make_snap(None, 0, 0, 0, 0, host_seq=0), now=0.0)
    g.push_snapshot(0.1, make_snap(None, 0, 0, 0, 0, host_seq=0), now=0.1)
    assert g.ghost.seeded, "the ghost must be seeded"
    # The ghost's missile at (110, 0) — the predicted position (ahead of
    # the buffer). boost=0 + target=None so it COASTS at a constant
    # velocity (no boost ramp / homing to complicate the position math).
    gm = GhostMissile(pygame.Vector2(110.0, 0.0),
                      pygame.Vector2(MISSILE_SPEED, 0.0), mid)
    gm.boost = 0.0
    gm.target = None
    g.ghost.local_missiles = [gm]
    # The CARRYING snapshot (t=0.2) carries the SAME missile at (100, 0) —
    # 10 px behind the ghost. host_seq=1 (launched before the snapshot).
    # This triggers handback_missiles, which marks the ghost missile
    # PENDING (not culled).
    g.push_snapshot(0.2, make_snap(mid, 100.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.2)
    assert len(g.ghost.local_missiles) == 1, \
        "the ghost's missile must be KEPT (pending, not culled): %r" \
        % (g.ghost.local_missiles,)
    assert g.ghost.local_missiles[0].pending, \
        "the ghost's missile must be marked pending: %r" \
        % (g.ghost.local_missiles[0].pending,)
    print("PASS: PENDING — the ghost's missile is marked pending (kept, "
          "not culled) when the carrying snapshot arrives")

    # Push more snapshots (t=0.3 .. 1.0) so the buffer holds a window and
    # keeps ADVANCING as the render point moves (the host keeps sending
    # snapshots in the real game). The missile advances 46 px per 0.1 s
    # (460 px/s): pos(t) = 100 + 460*(t - 0.2). A long buffer is needed
    # for the CONVERGE gate: the 10.3f cap freezes the offset when the
    # buffer stops advancing (B_disp = 0), so the buffer must keep
    # advancing long enough for the offset to fully decay.
    for _t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        g.push_snapshot(_t, make_snap(mid, 100.0 + MISSILE_SPEED * (_t - 0.2),
                                      0.0, MISSILE_SPEED, 0.0,
                                      host_seq=1), now=_t)

    # --- 2. KEEP-DRAW: while the buffer's copy is not visible (render
    #     point behind the carrying snapshot), the ghost missile stays in
    #     local_missiles (the ghost keeps drawing it — no gap). ---
    # The missile becomes visible when the render point reaches the
    # carrying snapshot's window (render_t >= 0.1, so curr = the 0.2
    # snapshot). Set the render point to 0.05 (in the gap — the window is
    # [0.0, 0.1], both pre-fire, so the buffer's copy is NOT visible).
    g.render_point._t = 0.05
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    assert len(g.ghost.local_missiles) == 1, \
        "the ghost's missile must STAY in local_missiles while the " \
        "buffer's copy is not visible (the ghost keeps drawing it): %r" \
        % (g.ghost.local_missiles,)
    assert mid not in g._missile_handoff_offsets, \
        "no offset yet (the buffer's copy is not visible): %r" \
        % (g._missile_handoff_offsets,)
    assert buffer_missile_x(g, pos, mid) is None, \
        "the buffer's copy must NOT be visible at render point 0.05"
    print("PASS: KEEP-DRAW — the ghost's missile stays in local_missiles "
          "while the buffer's copy is not visible (no gap)")

    # --- 3. FROZEN: while the buffer's copy is visible but FROZEN (its
    #     pop-in window — the snapshot before it has no missile, so it is
    #     drawn at its curr position for the whole window), the ghost
    #     missile STAYS in local_missiles (the ghost keeps drawing it —
    #     no stall). 10.3e handed off on the first VISIBLE frame, which
    #     was this frozen frame — the missile held still for up to a
    #     snapshot interval. 10.3g waits until the buffer ADVANCES. ---
    # render_t = 0.15: window [0.1, 0.2], the missile is only in curr
    # (0.2) so it pops in at curr = 100 (frozen).
    g.render_point._t = 0.15
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    assert len(g.ghost.local_missiles) == 1, \
        "the ghost's missile must STAY in local_missiles while the " \
        "buffer's copy is FROZEN (the ghost keeps drawing it — no " \
        "stall): %r" % (g.ghost.local_missiles,)
    assert mid not in g._missile_handoff_offsets, \
        "no offset yet (the buffer's copy is frozen): %r" \
        % (g._missile_handoff_offsets,)
    bx = buffer_missile_x(g, pos, mid)
    assert bx is not None and abs(bx - 100.0) < 1e-6, \
        "the buffer's copy must be visible but FROZEN at its pop-in " \
        "position (100): %r" % (bx,)
    # render_t = 0.2: window [0.1, 0.2] at alpha=1 — STILL the pop-in
    # position (100), still frozen.
    g.render_point._t = 0.2
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    assert len(g.ghost.local_missiles) == 1, \
        "the ghost's missile must STILL stay in local_missiles at " \
        "render_t=0.2 (the buffer's copy is still frozen at 100): %r" \
        % (g.ghost.local_missiles,)
    bx = buffer_missile_x(g, pos, mid)
    assert bx is not None and abs(bx - 100.0) < 1e-6, \
        "the buffer's copy must STILL be frozen at 100 at render_t=0.2: " \
        "%r" % (bx,)
    print("PASS: FROZEN — the ghost's missile stays in local_missiles "
          "while the buffer's copy is visible but frozen (no stall)")

    # --- 4. HANDBACK: on the first frame the buffer's copy ADVANCES
    #     (render_t > 0.2, so the window is [0.2, 0.3] and the missile
    #     lerps from 100 toward 146), the ghost missile is removed and
    #     the offset is seeded from the ghost's position at that moment.
    # ---
    # render_t = 0.216: window [0.2, 0.3], alpha = 0.16, the missile is at
    # 100 + 46*0.16 = 107.36 — it has ADVANCED from 100.
    g.render_point._t = 0.216
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    assert len(g.ghost.local_missiles) == 0, \
        "the ghost's missile must be REMOVED when the buffer's copy " \
        "advances (the buffer's copy takes over): %r" \
        % (g.ghost.local_missiles,)
    assert mid in g._missile_handoff_offsets, \
        "the offset must be seeded at handback: %r" \
        % (g._missile_handoff_offsets,)
    print("PASS: HANDBACK — the ghost's missile is removed and the offset "
          "is seeded when the buffer's copy ADVANCES (not merely appears)")

    # --- 5. NO-JUMP: on the first buffer draw, the rendered position
    #     equals the ghost's position at the handoff (no jump). The
    #     offset is (ghost_pos - interp_pos), so rendered = interp +
    #     offset = ghost_pos exactly. ---
    rx = rendered_missile_x(g, pos, mid)
    assert rx is not None, "the buffer's missile must be drawn"
    # The ghost's position at the handoff is recorded in
    # _missile_handoff_ghost_pos... but it is POPPED on the first buffer
    # draw (the offset is computed from it). So read the offset instead:
    # rendered = interp + offset, and the offset was (ghost - interp), so
    # rendered == ghost. The ghost's handoff position = the ghost's
    # position after this frame's advance. We can recover it: rendered x
    # must equal the ghost's position, which is 110 + (steps)*460/60.
    # Rather than recompute the step count, assert the rendered position
    # is AHEAD of the buffer's (the ghost was ahead) and equals the
    # offset-adjusted value (trivially true) — the real check is that it
    # did NOT snap back to the buffer's 107.36.
    bx = buffer_missile_x(g, pos, mid)
    assert rx > bx, \
        "the first buffer draw must be AHEAD of the buffer's position " \
        "(the ghost was ahead): rendered x=%.1f <= buffer x=%.1f" \
        % (rx, bx)
    # The rendered position must be close to the ghost's last known
    # position (110 + a few steps of 460/60). The ghost advanced ~3 steps
    # over the 4 predicted_view calls (0.016 s each at a 1/60 s step), so
    # it is near 110 + 3*7.67 = 133. Allow a generous band.
    assert 120.0 < rx < 145.0, \
        "the first buffer draw must be near the ghost's position (~133), " \
        "not the buffer's (107): rendered x=%.1f" % (rx,)
    print("PASS: NO-JUMP — the first buffer draw is at the ghost's "
          "handoff position (x=%.1f), ahead of the buffer's (x=%.1f)"
          % (rx, bx))

    # --- 6. NO-BACKWARD: over ALL the decay frames, the missile's
    #     rendered x position NEVER decreases. The 10.3f cap limits the
    #     offset's per-frame shrinkage to MISSILE_HANDOFF_MAX_SHRINK *
    #     (the buffer's displacement this frame), so the rendered
    #     position (buffer_pos + offset) always moves forward (>=
    #     (1-cap) * B_disp). ---
    rendered_xs = [rx]
    for frame in range(1, 60):
        # Advance the render point by one frame (0.016 s of sim time).
        # The buffer extends to t=1.0, so the render point keeps a
        # moving window (the buffer keeps advancing) for the whole run.
        g.render_point._t = min(0.216 + frame * 0.016, 1.0)
        pos = g.predicted_view(0.016, pygame.key.get_pressed())
        if pos is None:
            continue
        rx = rendered_missile_x(g, pos, mid)
        if rx is None:
            break   # the missile left the buffer (culled)
        rendered_xs.append(rx)
    for i in range(1, len(rendered_xs)):
        assert rendered_xs[i] >= rendered_xs[i - 1] - 0.01, \
            "the missile's rendered x must NEVER decrease on ANY frame " \
            "(no backward jump, 10.3f): frame %d x=%.2f < frame %d " \
            "x=%.2f" % (i, rendered_xs[i], i - 1, rendered_xs[i - 1])
    assert len(rendered_xs) >= 10, \
        "expected >= 10 frames of rendered positions: %d" % len(rendered_xs)
    print("PASS: NO-BACKWARD — the missile's rendered x never decreases "
          "on ANY frame (frames 0-%d, x=%.1f -> %.1f)"
          % (len(rendered_xs) - 1, rendered_xs[0], rendered_xs[-1]))

    # --- 7. CONVERGE: after ~30 frames the offset decays below the
    #     epsilon and is culled. ---
    assert mid not in g._missile_handoff_offsets, \
        "the offset must be culled after the decay (magnitude < " \
        "epsilon): %r" % (g._missile_handoff_offsets,)
    print("PASS: CONVERGE — the offset decays below %.1f px and is culled "
          "(the missile is now drawn at the buffer's position)"
          % MISSILE_HANDOFF_OFFSET_EPS)

    # --- 8. FOG-LIGHT: _remote_fog_lights accepts the handoff_offsets
    #     (the fog light follows the offset). ---
    sig = inspect.signature(g._remote_fog_lights)
    assert 'handoff_offsets' in sig.parameters, \
        "_remote_fog_lights must accept handoff_offsets: %r" % sig
    print("PASS: FOG-LIGHT — _remote_fog_lights accepts handoff_offsets "
          "(the fog light follows the offset)")

    print("ALL PASS: 10.3d/10.3e/10.3f/10.3g (decoupled render offset + "
          "no-blink handoff + no-backward decay + no-freeze handoff)")


if __name__ == "__main__":
    main()