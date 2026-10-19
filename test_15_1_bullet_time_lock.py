"""plan_15 Stage 1 — the bullet time-domain lock.

Verifies that predicted_view draws the ghost's OWN gun shots at the RENDER
POINT (rp), not at the ghost's _last_step_time:

    render_pos = b.pos - b.vel * min(age, D)
    D          = ghost._last_step_time - rp      (the exact gap)

This is the pass-through / early-disappear fix: the bullet is phase-locked
to the enemy's time domain so it strikes the enemy where it is DRAWN.

Three checks (each on a fresh client so the ghost clock / buffer are
independent):

  1. FROZEN AT MUZZLE — a bullet younger than D (age < D) renders at the
     muzzle (b.pos - b.vel*age), NOT at its raw position. The bullet had
     not left the muzzle yet at the render point.
  2. FLYING AT FULL SPEED — a bullet older than D (age > D) renders at
     b.pos - b.vel*D (the exact render-point position), NOT at its raw
     position and NOT at the muzzle. The offset is exactly D, not age.
  3. NO PULL-FORWARD AT HANDOFF — a handed-off bullet (pending, buffer
     copy advancing) is drawn by the buffer at its RAW interpolated
     position (no V*delay pull-forward — the ghost already drew it at the
     render point, so the handoff is seamless).

The local ship is at the origin (the fog light follows it), so the test
places each bullet's muzzle near the origin and asserts the bullet is
drawn at the OFFSET position (lit) and NOT at its raw position.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import BULLET_SPEED, BULLET_LIFE, TICK, WIDTH, HEIGHT
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .bullets import Bullet
from .netcode import (PredictedShip, RenderPoint, LatencyTracker,
                      HostTimeEstimator)
from .game import Game
from .fog import make_light_texture

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
LOADOUT = default_loadout(GUN_HULL)
LOCAL_INDEX = 1   # the client is player 1


def bullet_px(screen, cam, wx, wy, r=10):
    """Count BULLET_COLOR-ish pixels in an r x r box around world (wx, wy).
    A player bullet is BULLET_COLOR (255, 230, 120) — a warm yellow."""
    s = cam.to_screen(pygame.Vector2(wx, wy))
    cx, cy = int(s.x), int(s.y)
    n = 0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = cx + dx, cy + dy
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                continue
            px = screen.get_at((x, y))
            if px[0] > 200 and px[1] > 180 and px[2] < 180:
                n += 1
    return n


def build_client(bullets=None):
    """A fresh client Game with a seeded ghost, a 3-snapshot buffer (the
    local ship at the origin, gun NOT firing, optional `bullets` in the
    scratch world), and the render point advanced one frame. Returns
    (g, screen). `bullets` (a list of Bullet) is carried in every
    snapshot — used by the handoff check, which needs the buffer to hold
    the local bullet."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=GUN_HULL, loadout=LOADOUT, seed=1234, players=2,
             local_index=LOCAL_INDEX)
    g.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    g.ghost = PredictedShip(hull=GUN_HULL, loadout=LOADOUT,
                            local_index=LOCAL_INDEX)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    # A scratch game carrying the local ship at the origin, gun NOT firing.
    tmp = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, light_tex,
               fog_surf, light_surf, hull=GUN_HULL, loadout=LOADOUT,
               seed=1234, players=2)
    tmp.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    cs = tmp.players[LOCAL_INDEX]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 999.0   # the ghost does NOT fire during the frame
    if bullets is not None:
        tmp.bullets = bullets
    for k in range(3):
        g.push_snapshot(k * 0.1, tmp.snapshot(), now=k * 0.1)
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    return g, screen


def main():
    pygame.init()

    # OBSOLETE (plan_15 REVERT): this test asserts the plan_15 render-point
    # behavior (frozen-at-muzzle, flying-at-full-speed, no pull-forward at
    # handoff). The plan_15 Stage 1 change was REVERTED because the
    # render-point shift pinned each bullet at the muzzle for the first D
    # seconds (the "firing from empty space" + "direction change" — the
    # ship-vs-bullet time-domain gap). The ghost's own bullets are now
    # drawn at the PRESENT (b.pos), and the buffer's handed-off copy is
    # pulled forward by V*delay (see test_10_7 gate 5). Re-enable this
    # test only if the render-point draw is reintroduced.
    print("SKIP: 15.1 is OBSOLETE — the plan_15 render-point draw was "
          "reverted (bullets are drawn at the present, b.pos). The "
          "handoff pull-forward is covered by test_10_7 gate 5.")
    return

    # --- 1. FROZEN AT MUZZLE: a bullet younger than D (age < D) renders at
    #     the muzzle (b.pos - b.vel*age), NOT at its raw position. The
    #     muzzle is at the origin (b.pos - b.vel*age = (0,0)), so the bullet
    #     is drawn at the origin (lit) and NOT at its raw position. ---
    g1, screen = build_client()
    # Muzzle at the origin: b.pos = (60,0), age 0.05 -> muzzle = 60 - 860*0.05
    # = 17 (NOT the origin). To put the muzzle AT the origin, b.pos =
    # b.vel*age = 860*0.05 = 43.
    g1.ghost.local_bullets = [
        Bullet(pygame.Vector2(BULLET_SPEED * 0.05, 0.0),
               pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g1.ghost.local_bullets[0].life = BULLET_LIFE - 0.05
    g1.ghost.local_bullets[0].pending = False
    b1 = g1.ghost.local_bullets[0]
    rp1 = g1.render_point.now()
    D1 = g1.ghost._last_step_time - rp1
    age1 = BULLET_LIFE - b1.life
    assert age1 < D1, \
        "the test needs age < D (frozen at muzzle): age=%.4f D=%.4f" \
        % (age1, D1)
    t_off1 = min(age1, max(D1, 0.0))
    exp1 = (b1.pos.x - b1.vel.x * t_off1, b1.pos.y - b1.vel.y * t_off1)
    g1.predicted_view(0.016, pygame.key.get_pressed())
    assert len(g1.ghost.local_bullets) == 1, \
        "the ghost must keep its bullet (no handoff this frame): %r" \
        % (g1.ghost.local_bullets,)
    assert bullet_px(screen, g1.cam, exp1[0], exp1[1]) > 0, \
        "the young bullet must be DRAWN at the muzzle (%.1f, %.1f) = " \
        "b.pos - b.vel*age: %d px" % (exp1[0], exp1[1],
                                      bullet_px(screen, g1.cam, exp1[0],
                                                exp1[1]))
    assert bullet_px(screen, g1.cam, b1.pos.x, b1.pos.y) == 0, \
        "the young bullet must NOT be drawn at its raw position (%.1f, " \
        "%.1f) — it is frozen at the muzzle: %d px" \
        % (b1.pos.x, b1.pos.y,
           bullet_px(screen, g1.cam, b1.pos.x, b1.pos.y))
    print("PASS: FROZEN AT MUZZLE — a bullet younger than D (age=%.4f < "
          "D=%.4f) renders at the muzzle (%.1f, %.1f), not its raw "
          "(%.1f, %.1f)" % (age1, D1, exp1[0], exp1[1], b1.pos.x,
                            b1.pos.y))

    # --- 2. FLYING AT FULL SPEED: a bullet older than D (age > D) renders
    #     at b.pos - b.vel*D (the exact render-point position), NOT at its
    #     raw position and NOT at the muzzle. The offset is exactly D. ---
    g2, screen = build_client()
    # age 0.2 (> D ~0.1). Muzzle at the origin: b.pos = b.vel*age = 860*0.2
    # = 172. The render-point position = b.pos - b.vel*D = 172 - 860*D.
    g2.ghost.local_bullets = [
        Bullet(pygame.Vector2(BULLET_SPEED * 0.2, 0.0),
               pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g2.ghost.local_bullets[0].life = BULLET_LIFE - 0.2
    g2.ghost.local_bullets[0].pending = False
    b2 = g2.ghost.local_bullets[0]
    rp2 = g2.render_point.now()
    D2 = g2.ghost._last_step_time - rp2
    age2 = BULLET_LIFE - b2.life
    assert age2 > D2, \
        "the test needs age > D (flying at full speed): age=%.4f D=%.4f" \
        % (age2, D2)
    t_off2 = min(age2, max(D2, 0.0))
    assert abs(t_off2 - D2) < 1e-9, \
        "the offset must be exactly D (not age) when age > D: t_off=%.4f " \
        "D=%.4f" % (t_off2, D2)
    exp2 = (b2.pos.x - b2.vel.x * t_off2, b2.pos.y - b2.vel.y * t_off2)
    muzzle2 = (b2.pos.x - b2.vel.x * age2, b2.pos.y - b2.vel.y * age2)
    g2.predicted_view(0.016, pygame.key.get_pressed())
    assert len(g2.ghost.local_bullets) == 1, \
        "the ghost must keep its bullet (no handoff this frame): %r" \
        % (g2.ghost.local_bullets,)
    assert bullet_px(screen, g2.cam, exp2[0], exp2[1]) > 0, \
        "the old bullet must be DRAWN at the render-point position " \
        "(%.1f, %.1f) = b.pos - b.vel*D: %d px" \
        % (exp2[0], exp2[1], bullet_px(screen, g2.cam, exp2[0], exp2[1]))
    assert bullet_px(screen, g2.cam, b2.pos.x, b2.pos.y) == 0, \
        "the old bullet must NOT be drawn at its raw position (%.1f, " \
        "%.1f): %d px" % (b2.pos.x, b2.pos.y,
                          bullet_px(screen, g2.cam, b2.pos.x, b2.pos.y))
    assert bullet_px(screen, g2.cam, muzzle2[0], muzzle2[1]) == 0, \
        "the old bullet must NOT be drawn at the muzzle (%.1f, %.1f) — " \
        "it is flying at full speed: %d px" \
        % (muzzle2[0], muzzle2[1],
           bullet_px(screen, g2.cam, muzzle2[0], muzzle2[1]))
    print("PASS: FLYING AT FULL SPEED — a bullet older than D (age=%.4f > "
          "D=%.4f) renders at b.pos - b.vel*D (%.1f, %.1f), not its raw "
          "(%.1f, %.1f) or muzzle (%.1f, %.1f)"
          % (age2, D2, exp2[0], exp2[1], b2.pos.x, b2.pos.y, muzzle2[0],
             muzzle2[1]))

    # --- 3. NO PULL-FORWARD AT HANDOFF: a handed-off bullet (pending,
    #     buffer copy advancing) is drawn by the buffer at its RAW
    #     interpolated position (no V*delay pull-forward). Mirrors
    #     test_10_7 gate 5's setup (buffer bullet at (300,0), pre-seeded
    #     advance) but asserts the RAW position. ---
    # The buffer carries the local bullet (id (1,0)) at (300,0) in all
    # snapshots. The ghost's copy is PENDING (the buffer's newest snapshot
    # carries it). Pre-seed the handoff buffer position to a DIFFERENT spot
    # so the "advance" is detected on the first frame.
    g3, screen = build_client(bullets=[
        Bullet(pygame.Vector2(300.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0)),
    ])
    g3.ghost.local_bullets = [
        Bullet(pygame.Vector2(100.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g3.ghost.local_bullets[0].pending = True
    g3._bullet_handoff_buf_pos[(LOCAL_INDEX, 0)] = (200.0, 0.0)
    g3.latency.tick(0.016)
    g3.render_point.advance(0.016, g3.snap_buf.newest_time())
    # The raw buffer position of the (1,0) bullet in this frame's window.
    _rp3 = g3.render_point.now()
    _pos3 = g3.snap_buf.positions_at(_rp3)
    _raw3 = None
    for (x, y, vx, vy, kind, owner, boost, mid) in _pos3['bullets']:
        if kind == 'player' and mid is not None \
                and tuple(mid) == (LOCAL_INDEX, 0):
            _raw3 = (x, y, vx, vy)
    assert _raw3 is not None, \
        "the buffer must carry the (1,0) bullet in the window: %r" \
        % _pos3['bullets']
    _delay3 = g3.latency.delay
    _pulled3 = (_raw3[0] + _raw3[2] * _delay3,
                _raw3[1] + _raw3[3] * _delay3)
    g3.predicted_view(0.016, pygame.key.get_pressed())
    assert len(g3.ghost.local_bullets) == 0, \
        "the handed-off bullet must be REMOVED from the ghost (the buffer " \
        "draws it now): %r" % (g3.ghost.local_bullets,)
    assert bullet_px(screen, g3.cam, _raw3[0], _raw3[1]) > 0, \
        "the handed-off bullet must be DRAWN at its RAW interpolated " \
        "position (%.1f, %.1f), no pull-forward: %d px" \
        % (_raw3[0], _raw3[1],
           bullet_px(screen, g3.cam, _raw3[0], _raw3[1]))
    assert bullet_px(screen, g3.cam, _pulled3[0], _pulled3[1]) == 0, \
        "the handed-off bullet must NOT be drawn PULLED FORWARD at " \
        "(%.1f, %.1f) = raw + V*delay (plan_15 removed the pull-forward): " \
        "%d px" % (_pulled3[0], _pulled3[1],
                   bullet_px(screen, g3.cam, _pulled3[0], _pulled3[1]))
    print("PASS: NO PULL-FORWARD AT HANDOFF — the handed-off bullet is "
          "removed from the ghost and drawn by the buffer at its RAW "
          "interpolated position (%.1f, %.1f), NOT pulled forward to "
          "(%.1f, %.1f)" % (_raw3[0], _raw3[1], _pulled3[0], _pulled3[1]))

    print("ALL PASS: 15.1 (ghost's own gun shots drawn at the render point "
          "— frozen at the muzzle when young, flying at full speed when "
          "old, seamless handoff with no pull-forward)")


if __name__ == "__main__":
    main()