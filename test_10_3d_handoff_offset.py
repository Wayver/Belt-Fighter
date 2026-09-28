"""Session 10.3d: the decoupled render offset (the handback fix).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3d_handoff_offset

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3b made the client's prediction ghost emit + home its own missiles and
dedup them against the buffer's copy (no double-draw). But the handback
moment — when the ghost's missile is culled and the buffer's copy takes
over — still showed a visible BACKWARD JUMP: the buffer's copy is drawn at
its INTERPOLATED position (INTERP_DELAY seconds behind the ghost's
predicted position), so the missile snapped backward the instant the ghost
copy stopped drawing.

10.3d fixes this with a DECOUPLED RENDER OFFSET:
  * At handback, the ghost's position at cull time is recorded
    (ghost.handback_missiles returns {mid: (gx, gy)}).
  * On the FIRST frame the buffer's copy is drawn, predicted_view computes
    the offset as (ghost_pos - interp_pos) — the vector that, added to the
    buffer's interpolated position, places the sprite where the ghost was
    (no jump).
  * Each subsequent frame, the offset is decayed by MISSILE_HANDOFF_DECAY
    (0.85, frame-rate independent: decay^(dt*60)), so the missile
    converges to the buffer's position smoothly, moving FORWARD the whole
    time (the offset shrinks slower than the missile advances).
  * The offset is removed when its magnitude drops below
    MISSILE_HANDOFF_OFFSET_EPS (0.5 px — invisible).

This test proves it (value-level):
  1. HANDBACK   — the ghost's missile is culled and its cull-time position
                  is recorded.
  2. NO-JUMP    — on the first buffer draw, the rendered position equals
                  the ghost's cull-time position (no jump).
  3. NO-BACKWARD— over the decay frames, the missile's rendered x position
                  NEVER decreases.
  4. CONVERGE   — after ~30 frames the offset decays below the epsilon and
                  is culled (the missile is drawn at the buffer's position).
  5. FOG-LIGHT  — _remote_fog_lights accepts the handoff_offsets (the fog
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


def make_seed_ship_s():
    """A ship snapshot with the missile weapon at FULL lock, the ship at
    the origin facing +x, vel 0."""
    s = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    s.pos = pygame.Vector2(0.0, 0.0)
    s.vel = pygame.Vector2(0.0, 0.0)
    s.angle = 0.0
    for w in s.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 1.0
    return s.snapshot()


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

    # --- 1. HANDBACK: the ghost's missile is culled and its cull-time
    #     position is recorded. ---
    g = make_game()
    # Seed the ghost with a PRE-FIRE snapshot (host_seq=0, no missile).
    g.push_snapshot(0.0, make_snap(None, 0, 0, 0, 0, host_seq=0), now=0.0)
    assert g.ghost.seeded, "the ghost must be seeded"
    # The ghost's missile at (130, 0) — the predicted position (ahead of
    # the buffer).
    g.ghost.local_missiles = [
        GhostMissile(pygame.Vector2(130.0, 0.0),
                     pygame.Vector2(MISSILE_SPEED, 0.0), mid)]
    # The buffer carries the SAME missile at (100, 0) — 30 px behind the
    # ghost. The snapshot's missile_seq = 1 (launched before the snapshot).
    g.push_snapshot(0.1, make_snap(mid, 100.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.1)
    assert len(g.ghost.local_missiles) == 0, \
        "the ghost's missile must be culled at handback: %r" \
        % (g.ghost.local_missiles,)
    assert mid in g._missile_handoff_ghost_pos, \
        "the ghost's cull-time position must be recorded: %r" \
        % (g._missile_handoff_ghost_pos,)
    gx, gy = g._missile_handoff_ghost_pos[mid]
    assert abs(gx - 130.0) < 1e-6 and abs(gy) < 1e-6, \
        "the recorded ghost position must be (130, 0): (%.1f, %.1f)" \
        % (gx, gy)
    print("PASS: HANDBACK — the ghost's missile is culled and its "
          "cull-time position (130, 0) is recorded")

    # Push two more snapshots (t=0.2, 0.3) so the buffer holds a window
    # and the render point can sit inside it. The missile advances 46 px
    # per 0.1 s (460 px/s).
    g.push_snapshot(0.2, make_snap(mid, 146.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.2)
    g.push_snapshot(0.3, make_snap(mid, 192.0, 0.0, MISSILE_SPEED, 0.0,
                                   host_seq=1), now=0.3)
    # Advance the render point to its first anchor (newest - delay =
    # 0.3 - 0.1 = 0.2).
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    assert abs(g.render_point.now() - 0.2) < 1e-6, \
        "the render point must be at 0.2: %r" % g.render_point.now()

    # --- 2. NO-JUMP: on the first buffer draw, the rendered position
    #     equals the ghost's cull-time position (130, 0). ---
    pos = g.predicted_view(0.016, pygame.key.get_pressed())
    assert pos is not None, "predicted_view must return a window"
    rx = rendered_missile_x(g, pos, mid)
    assert rx is not None, "the buffer's missile must be drawn"
    # The render point is at 0.2, so the interpolated position is the
    # curr snapshot's position (146, 0) (alpha=1). The offset is
    # (130 - 146, 0) = (-16, 0). The rendered position is 146 + (-16) =
    # 130 — the ghost's cull-time position (no jump).
    assert abs(rx - 130.0) < 1.0, \
        "the first buffer draw must be at the ghost's position (130), " \
        "not the buffer's (146): rendered x=%.1f" % rx
    assert mid in g._missile_handoff_offsets, \
        "the offset must be computed on the first buffer draw: %r" \
        % (g._missile_handoff_offsets,)
    dx, dy = g._missile_handoff_offsets[mid]
    assert abs(dx - (-16.0)) < 1.0 and abs(dy) < 1.0, \
        "the offset must be (ghost - interp) = (-16, 0): (%.1f, %.1f)" \
        % (dx, dy)
    assert mid not in g._missile_handoff_ghost_pos, \
        "the ghost position must be consumed (popped) after the first " \
        "draw: %r" % (g._missile_handoff_ghost_pos,)
    print("PASS: NO-JUMP — the first buffer draw is at the ghost's "
          "position (x=%.1f), not the buffer's (x=146); offset=(%.1f, %.1f)"
          % (rx, dx, dy))

    # --- 3. NO-BACKWARD: over the decay frames, the missile's rendered x
    #     position NEVER decreases. ---
    rendered_xs = [rx]
    for frame in range(1, 40):
        g.latency.tick(0.016)
        g.render_point.advance(0.016, g.snap_buf.newest_time())
        pos = g.predicted_view(0.016, pygame.key.get_pressed())
        if pos is None:
            continue
        rx = rendered_missile_x(g, pos, mid)
        if rx is None:
            break   # the missile left the buffer (culled)
        rendered_xs.append(rx)
    for i in range(1, len(rendered_xs)):
        assert rendered_xs[i] >= rendered_xs[i - 1] - 0.01, \
            "the missile's rendered x must NEVER decrease (no backward " \
            "jump): frame %d x=%.2f < frame %d x=%.2f" \
            % (i, rendered_xs[i], i - 1, rendered_xs[i - 1])
    assert len(rendered_xs) >= 10, \
        "expected >= 10 frames of rendered positions: %d" % len(rendered_xs)
    print("PASS: NO-BACKWARD — the missile's rendered x never decreases "
          "over %d frames (first=%.1f, last=%.1f)"
          % (len(rendered_xs), rendered_xs[0], rendered_xs[-1]))

    # --- 4. CONVERGE: after ~30 frames the offset decays below the
    #     epsilon and is culled. ---
    # The offset started at -16. After 30 frames at 60 FPS, the decay is
    # 0.85^30 ≈ 0.0076, so the offset is ≈ -0.12 — below the epsilon
    # (0.5) and culled.
    assert mid not in g._missile_handoff_offsets, \
        "the offset must be culled after 40 frames (magnitude < epsilon): " \
        "%r" % (g._missile_handoff_offsets,)
    print("PASS: CONVERGE — the offset decays below %.1f px and is culled "
          "(the missile is now drawn at the buffer's position)"
          % MISSILE_HANDOFF_OFFSET_EPS)

    # --- 5. FOG-LIGHT: _remote_fog_lights accepts the handoff_offsets
    #     (the fog light follows the offset). ---
    sig = inspect.signature(g._remote_fog_lights)
    assert 'handoff_offsets' in sig.parameters, \
        "_remote_fog_lights must accept handoff_offsets: %r" % sig
    print("PASS: FOG-LIGHT — _remote_fog_lights accepts handoff_offsets "
          "(the fog light follows the offset)")

    print("ALL PASS: 10.3d (decoupled render offset — no backward jump at "
          "handback)")


if __name__ == "__main__":
    main()