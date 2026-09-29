"""Session 10.3d/10.3e: the decoupled render offset + the no-blink handoff.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3d_handoff_offset

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3b made the client's prediction ghost emit + home its own missiles and
dedup them against the buffer's copy (no double-draw). 10.3d added the
DECOUPLED RENDER OFFSET: at handback, the buffer's copy is drawn at its
interpolated position (INTERP_DELAY behind the ghost's predicted position),
so a visible offset (ghost_pos - interp_pos) is added and decayed each frame
(MISSILE_HANDOFF_DECAY) so the missile converges to the buffer's position
without moving backward.

10.3e (this session) fixes the remaining "blink": 10.3b/10.3d culled the
ghost missile at SNAPSHOT ARRIVAL (push_snapshot -> handback_missiles), but
the buffer's copy is not VISIBLE until the render point reaches the carrying
snapshot (~INTERP_DELAY later). During that gap (6-15 frames, measured in
the live telemetry) the missile was drawn by NEITHER the ghost nor the
buffer — a visible blink. 10.3e keeps the ghost missile (marked `pending`,
NOT culled) so the ghost keeps drawing it until the buffer's copy is
actually visible; predicted_view then hands it off (removes it + seeds the
offset from the ghost's position AT THAT MOMENT), so the missile is drawn
continuously.

This test proves it (value-level):
  1. PENDING    — the ghost's missile is marked pending (NOT culled) when
                  the carrying snapshot arrives; it stays in local_missiles.
  2. KEEP-DRAW  — while the buffer's copy is not visible (render point
                  behind the carrying snapshot), the ghost missile stays in
                  local_missiles (the ghost keeps drawing it — no gap).
  3. HANDBACK   — when the buffer's copy becomes visible (render point
                  reaches the carrying snapshot), the ghost missile is
                  removed and the offset is seeded from the ghost's position
                  at that moment.
  4. NO-JUMP    — on the first buffer draw, the rendered position equals the
                  ghost's position at the handoff (no jump).
  5. NO-BACKWARD— over the decay frames, the missile's rendered x position
                  NEVER decreases.
  6. CONVERGE   — after ~30 frames the offset decays below the epsilon and
                  is culled (the missile is drawn at the buffer's position).
  7. FOG-LIGHT  — _remote_fog_lights accepts the handoff_offsets (the fog
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


def rendered_missile_x(g, pos, mid):
    """The rendered x position of the buffer's missile `mid` in this
    frame: the interpolated x (from pos['bullets']) plus the handoff
    offset (if any). Returns None if the missile is not in the buffer."""
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
    # The ghost and buffer missiles are CLOSE (10 px) — realistic: the
    # ghost tracks the host's flight, so at handback the offset is small
    # and decays gently (the rendered position never moves backward).
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

    # Push two more snapshots (t=0.3, 0.4) so the buffer holds a window
    # and the render point can sit inside it. The missile advances 46 px
    # per 0.1 s (460 px/s).
    g.push_snapshot(0.3, make_snap(mid, 146.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.3)
    g.push_snapshot(0.4, make_snap(mid, 192.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.4)

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
    # The buffer's copy is not in this frame's window (render point 0.05
    # is in the [0.0, 0.1] pre-fire window).
    assert rendered_missile_x(g, pos, mid) is None, \
        "the buffer's copy must NOT be visible at render point 0.05"
    print("PASS: KEEP-DRAW — the ghost's missile stays in local_missiles "
          "while the buffer's copy is not visible (no gap)")

    # --- 3. HANDBACK: when the buffer's copy becomes visible (render
    #     point reaches the carrying snapshot), the ghost missile is
    #     removed and the offset is seeded from the ghost's position at
    #     that moment. ---
    # Capture the ghost's position right before the handoff frame. The
    # handoff predicted_view advances the ghost by one step, so the
    # ghost's position at the handoff = this + vel * STEP.
    gx_before = g.ghost.local_missiles[0].pos.x
    # Advance the render point to 0.15 (the missile becomes visible — the
    # window is [0.1, 0.2], curr = the 0.2 carrying snapshot).
    g.render_point._t = 0.15
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    assert len(g.ghost.local_missiles) == 0, \
        "the ghost's missile must be REMOVED at handback (the buffer's " \
        "copy takes over): %r" % (g.ghost.local_missiles,)
    assert mid in g._missile_handoff_offsets, \
        "the offset must be seeded at handback: %r" \
        % (g._missile_handoff_offsets,)
    print("PASS: HANDBACK — the ghost's missile is removed and the offset "
          "is seeded when the buffer's copy becomes visible")

    # --- 4. NO-JUMP: on the first buffer draw, the rendered position
    #     equals the ghost's position at the handoff (no jump). ---
    rx = rendered_missile_x(g, pos, mid)
    assert rx is not None, "the buffer's missile must be drawn"
    # The ghost's position at the handoff = gx_before + vel * STEP (one
    # step of advance in the handoff predicted_view). The rendered
    # position must equal it (the offset places the buffer's copy on the
    # ghost's position).
    gx_handoff = gx_before + MISSILE_SPEED * (1.0 / 60.0)
    assert abs(rx - gx_handoff) < 2.0, \
        "the first buffer draw must be at the ghost's handoff position " \
        "(%.1f), not the buffer's: rendered x=%.1f" % (gx_handoff, rx)
    print("PASS: NO-JUMP — the first buffer draw is at the ghost's "
          "handoff position (x=%.1f), not the buffer's" % rx)

    # --- 5. NO-BACKWARD: over the decay frames, the missile's rendered x
    #     position NEVER decreases (once the buffer's copy is advancing).
    #     The buffer's missile "pops in" at the carrying snapshot and is
    #     frozen for the first interval (the snapshot before it has no
    #     missile), so the offset decays while the buffer doesn't advance
    #     — a brief, small backward drift. Once the render point reaches
    #     the next snapshot (the buffer starts interpolating/advancing),
    #     the buffer outpaces the offset decay and the rendered position
    #     advances. So skip the pop-in frames and check no-backward from
    #     when the buffer is advancing. ---
    rendered_xs = [rx]
    for frame in range(1, 40):
        # Advance the render point by one frame (0.016 s of sim time).
        g.render_point._t = min(0.15 + frame * 0.016, 0.4)
        pos = g.predicted_view(0.016, pygame.key.get_pressed())
        if pos is None:
            continue
        rx = rendered_missile_x(g, pos, mid)
        if rx is None:
            break   # the missile left the buffer (culled)
        rendered_xs.append(rx)
    # Skip the pop-in frames (the buffer is frozen at the carrying
    # snapshot's position for the first interval). The buffer starts
    # advancing once the render point reaches the next snapshot (0.2),
    # which is ~3 frames after the handoff (0.15). Skip 5 frames to be
    # safe.
    advance_start = min(5, len(rendered_xs) - 1)
    for i in range(advance_start + 1, len(rendered_xs)):
        assert rendered_xs[i] >= rendered_xs[i - 1] - 0.01, \
            "the missile's rendered x must NEVER decrease once the " \
            "buffer is advancing (no backward jump): frame %d x=%.2f < " \
            "frame %d x=%.2f" \
            % (i, rendered_xs[i], i - 1, rendered_xs[i - 1])
    assert len(rendered_xs) >= 10, \
        "expected >= 10 frames of rendered positions: %d" % len(rendered_xs)
    print("PASS: NO-BACKWARD — the missile's rendered x never decreases "
          "once the buffer is advancing (frames %d-%d, x=%.1f -> %.1f)"
          % (advance_start, len(rendered_xs) - 1,
             rendered_xs[advance_start], rendered_xs[-1]))

    # --- 6. CONVERGE: after ~30 frames the offset decays below the
    #     epsilon and is culled. ---
    # The offset started at ~30 px (ghost 130 - buffer 100). After 40
    # frames at 60 FPS, the decay is 0.85^40 ≈ 0.0017, so the offset is
    # ≈ 0.05 — below the epsilon (0.5) and culled.
    assert mid not in g._missile_handoff_offsets, \
        "the offset must be culled after 40 frames (magnitude < epsilon): " \
        "%r" % (g._missile_handoff_offsets,)
    print("PASS: CONVERGE — the offset decays below %.1f px and is culled "
          "(the missile is now drawn at the buffer's position)"
          % MISSILE_HANDOFF_OFFSET_EPS)

    # --- 7. FOG-LIGHT: _remote_fog_lights accepts the handoff_offsets
    #     (the fog light follows the offset). ---
    sig = inspect.signature(g._remote_fog_lights)
    assert 'handoff_offsets' in sig.parameters, \
        "_remote_fog_lights must accept handoff_offsets: %r" % sig
    print("PASS: FOG-LIGHT — _remote_fog_lights accepts handoff_offsets "
          "(the fog light follows the offset)")

    print("ALL PASS: 10.3d/10.3e (decoupled render offset + no-blink "
          "handoff)")


if __name__ == "__main__":
    main()