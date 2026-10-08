"""Session 10.7: the prediction ghost's own gun shots are deduped against
the buffer's copy (no double-draw) — the lightweight id-based mirror of the
10.3b ghost-missile dedup.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_7_bullet_dedup

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

Before 10.7 the client drew its OWN gun shots immediately (Session 7.3,
ghost.local_bullets) AND the host's snapshot carried those same shots
(snapshot index 2), so the interpolation buffer drew them a second time —
the "criss-cross" double-draw.

The fix (lightweight, id-based — see the 10.7 plan):
  * Each player bullet carries a unique id (player_index, bullet_seq). The
    host assigns it at fire and syncs bullet_seq via the ship snapshot
    (field 24, like missile_seq); the ghost derives the SAME id
    deterministically. No ID-assignment message.
  * predicted_view SKIPS the buffer's copy of a bullet whose id is in
    ghost.local_bullets (the ghost draws it, predicted, immediate).
  * The ghost hands a bullet off to the buffer once the buffer's copy is
    visible AND has ADVANCED (the 10.3g wait-until-advancing rule, reused);
    the buffer then draws it PULLED FORWARD by V*delay (the closed-form
    handoff — no offset state, no decay, no backward cap, because a bullet
    is a straight line).

This test proves it (value-level):
  1. ID-SEQ        — the ghost's bullets get id (local_index, seq) and
                     bullet_seq increments per fire (the dedup key the
                     host's ids match).
  2. RECONCILE-DEDUP — the reconcile REWIND re-runs the fire tick (apply_
                     snapshot resets bullet_seq to the host's pre-fire
                     value, then the replay re-fires). The ghost must NOT
                     hold two bullets of one id — the guard skips the
                     re-fire.
  3. HANDBACK      — handback_bullets marks a ghost bullet pending (kept,
                     not culled) once seq < the snapshot's authoritative
                     bullet_seq AND the newest snapshot carries its id;
                     seq >= host_seq is not pending.
  4. RENDER+DEDUP  — predicted_view draws the ghost's bullet, SKIPS the
                     buffer's copy of the SAME bullet (id match), and still
                     draws the OTHER player's buffer bullet (id mismatch).
  5. PULL-FORWARD  — a handed-off bullet (pending, buffer copy advancing)
                     is removed from the ghost and drawn by the buffer
                     PULLED FORWARD by V*delay (ahead of the raw buffer
                     position, in the velocity direction).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import BULLET_SPEED, TICK, WIDTH, HEIGHT
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .bullets import Bullet
from .intent import ShipInput
from .netcode import (PredictedShip, RenderPoint, LatencyTracker,
                      HostTimeEstimator)
from .game import Game
from .fog import make_light_texture

# A gun-fitted test hull: the scout polygon + the stock slots + a 'gun'
# weapon slot. default_loadout(hull) maps 'gun' -> GUN_TYPE (hulls.py), so
# the stock loadout gives this hull a pulse gun.
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


def make_seed_ship_s():
    """A ship snapshot with the ship at the origin facing +x, vel 0, gun
    cooldown 0 (ready to fire). The ghost is seeded from this, so its
    muzzle + fire match the host's at the fire tick."""
    s = Ship(hull=GUN_HULL, loadout=LOADOUT)
    s.pos = pygame.Vector2(0.0, 0.0)
    s.vel = pygame.Vector2(0.0, 0.0)
    s.angle = 0.0
    for w in s.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 0.0
    return s.snapshot()


def make_ghost():
    return PredictedShip(hull=GUN_HULL, loadout=LOADOUT,
                         local_index=LOCAL_INDEX)


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


def main():
    pygame.init()

    # --- 1. ID-SEQ: the ghost's bullets get id (local_index, seq) and
    #     bullet_seq increments per fire (the dedup key the host's ids
    #     match). ---
    ghost1 = make_ghost()
    ghost1.seed(make_seed_ship_s())
    ghost1.step(TICK, ShipInput(fire=True))
    assert len(ghost1.local_bullets) == 1, \
        "the ghost must emit a bullet on the fire tick: %r" \
        % (ghost1.local_bullets,)
    b0 = ghost1.local_bullets[0]
    assert b0.id == (LOCAL_INDEX, 0), \
        "the first bullet must have id (%d, 0): %r" % (LOCAL_INDEX, b0.id)
    assert ghost1.ship.bullet_seq == 1, \
        "bullet_seq must bump to 1 after the first fire: %r" \
        % ghost1.ship.bullet_seq
    # Fire again (the gun's cooldown has elapsed after a few ticks).
    for _ in range(8):
        ghost1.step(TICK, ShipInput())
        ghost1.step_local_bullets(TICK)
    ghost1.step(TICK, ShipInput(fire=True))
    ids = [b.id for b in ghost1.local_bullets]
    assert (LOCAL_INDEX, 1) in ids, \
        "a second bullet must have id (%d, 1): %r" % (LOCAL_INDEX, ids)
    assert ghost1.ship.bullet_seq == 2, \
        "bullet_seq must bump to 2 after the second fire: %r" \
        % ghost1.ship.bullet_seq
    print("PASS: ID-SEQ — the ghost's bullets get id (local_index, seq) and "
          "bullet_seq increments per fire (ids %r, seq %d)"
          % (ids, ghost1.ship.bullet_seq))

    # --- 2. RECONCILE-DEDUP: the reconcile REWIND re-runs the fire tick
    #     (apply_snapshot resets bullet_seq to the host's pre-fire value,
    #     then the replay re-fires). The ghost must NOT hold two bullets of
    #     one id — the guard skips the re-fire (same id). Without it the
    #     rewind duplicates the in-flight bullet, and because the buffer's
    #     copy is suppressed by the id-dedup, BOTH visible ones are
    #     collision-less ghost bullets. ---
    ghost2 = make_ghost()
    ghost2.seed(make_seed_ship_s())
    # The prediction fires at host time TICK (advance's step) — one ghost
    # bullet, id (1,0). Stamp the inputs for the replay too.
    ghost2.record_input(0.0, ShipInput())
    ghost2.advance(TICK, ShipInput())
    ghost2.record_input(TICK, ShipInput(fire=True))
    ghost2.advance(TICK, ShipInput(fire=True))
    assert len(ghost2.local_bullets) == 1, \
        "the prediction must fire exactly one bullet: %r" \
        % (ghost2.local_bullets,)
    assert ghost2.local_bullets[0].id == (LOCAL_INDEX, 0)
    # The host's snapshot: taken at S=0 (PRE-fire — bullet_seq=0, cooldown
    # 0, because the snapshot reflects the state after the tick that ended
    # at 0, which is before the fire tick at TICK). The rewind replays
    # [0, 2*TICK], which INCLUDES the fire tick at TICK — so the replay
    # re-fires the same bullet the prediction already fired.
    host2 = Ship(hull=GUN_HULL, loadout=LOADOUT)
    host2.pos = pygame.Vector2(0.0, 0.0)
    host2.vel = pygame.Vector2(0.0, 0.0)
    host2.angle = 0.0
    host2.bullet_seq = 0
    for w in host2.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 0.0
    snap2 = host2.snapshot()
    assert snap2[24] == 0, \
        "the snapshot must be pre-fire (bullet_seq 0): %r" % snap2[24]
    ghost2.reconcile_rewind(snap2, 0.0, 2 * TICK)
    ids2 = [tuple(b.id) for b in ghost2.local_bullets]
    assert ids2.count((LOCAL_INDEX, 0)) == 1, \
        "the rewind re-fire must NOT duplicate the in-flight bullet " \
        "(one id, one bullet): %r" % ids2
    assert ghost2.ship.bullet_seq == 1, \
        "the resynced seq must be the post-fire value: %r" \
        % ghost2.ship.bullet_seq
    print("PASS: RECONCILE-DEDUP — the rewind re-fire does not duplicate "
          "the in-flight ghost bullet (ids %r, seq %d)"
          % (ids2, ghost2.ship.bullet_seq))

    # --- 3. HANDBACK: handback_bullets marks a ghost bullet PENDING (kept,
    #     not culled) once BOTH hold: seq < the snapshot's authoritative
    #     bullet_seq (ship field 24) AND the newest snapshot carries its id.
    #     seq >= host_seq was fired after the snapshot — not pending. The
    #     ghost KEEPS DRAWING a pending bullet until predicted_view sees the
    #     buffer's copy ADVANCE, then hands it off. ---
    ghost3 = make_ghost()
    b0 = Bullet(pygame.Vector2(100.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
                owner=1, bid=(LOCAL_INDEX, 0))
    b1 = Bullet(pygame.Vector2(110.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
                owner=1, bid=(LOCAL_INDEX, 1))
    ghost3.local_bullets = [b0, b1]
    # host_seq=1, newest snapshot carries bullet 0 only -> b0 pending, b1
    # not. BOTH stay in local_bullets (no cull).
    ghost3.handback_bullets(1, latest_bullets=[
        (100.0, 0.0, BULLET_SPEED, 0.0, 1, 1.0, (LOCAL_INDEX, 0))])
    assert [b.id for b in ghost3.local_bullets] == \
        [(LOCAL_INDEX, 0), (LOCAL_INDEX, 1)], \
        "pending bullets are KEPT (not culled) until the buffer's copy " \
        "advances: %r" % [b.id for b in ghost3.local_bullets]
    assert b0.pending and not b1.pending, \
        "seq < host_seq (and in the buffer) must be pending; seq >= " \
        "host_seq not: b0.pending=%r b1.pending=%r" \
        % (b0.pending, b1.pending)
    # host_seq=2, newest snapshot carries both -> b1 also pending. Both stay.
    ghost3.handback_bullets(2, latest_bullets=[
        (100.0, 0.0, BULLET_SPEED, 0.0, 1, 1.0, (LOCAL_INDEX, 0)),
        (110.0, 0.0, BULLET_SPEED, 0.0, 1, 1.0, (LOCAL_INDEX, 1))])
    assert b0.pending and b1.pending, \
        "both bullets are in the buffer now: b0.pending=%r b1.pending=%r" \
        % (b0.pending, b1.pending)
    assert len(ghost3.local_bullets) == 2, \
        "pending bullets stay in local_bullets: %r" % ghost3.local_bullets
    # host_seq=0 (pre-fire) -> nothing pending.
    ghost3.local_bullets = [b0, b1]
    b0.pending = b1.pending = False
    ghost3.handback_bullets(0, latest_bullets=[])
    assert not b0.pending and not b1.pending, \
        "a pre-fire snapshot (host_seq 0) must mark nothing pending: " \
        "b0.pending=%r b1.pending=%r" % (b0.pending, b1.pending)
    print("PASS: HANDBACK — ghost bullets with seq < the snapshot's "
          "authoritative bullet_seq (and in the newest snapshot) are marked "
          "pending (kept, not culled); seq >= host_seq are not pending")

    # --- 4. RENDER + DEDUP: predicted_view draws the ghost's bullet,
    #     SKIPS the buffer's copy of the SAME bullet (id match), and still
    #     draws the OTHER player's buffer bullet (id mismatch). The buffer's
    #     (1,0) copy is placed at a DIFFERENT position than the ghost's so
    #     the skip is observable (in reality they coincide — the buffer is
    #     ~delay behind — but separating them proves the skip specifically).
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

    # A scratch game carrying TWO player bullets: the CLIENT's own (id
    # (1,0)) at (300,0) and the HOST's (id (0,0)) at (150,50) — both within
    # the on-screen area (the camera is centered on the ghost at the origin).
    tmp = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, light_tex,
               fog_surf, light_surf, hull=GUN_HULL, loadout=LOADOUT,
               seed=1234, players=2)
    tmp.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    # The ghost is SEEDED from tmp's snapshot (push_snapshot -> ghost.seed),
    # so the ghost's ship pose — and thus the camera — comes from
    # tmp.players[LOCAL_INDEX], NOT g.players[LOCAL_INDEX]. Set it at the
    # origin so the camera centers there and the bullets land in the open
    # (not under the top-left HUD, which is drawn after the fog).
    cs = tmp.players[LOCAL_INDEX]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 999.0   # the ghost does NOT fire during the frame
    tmp.bullets = [
        Bullet(pygame.Vector2(300.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0)),
        Bullet(pygame.Vector2(150.0, 50.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=0, bid=(0, 0)),
    ]
    # Push a 3-snapshot stream (t = 0, 0.1, 0.2) so the buffer holds a
    # window to interpolate. All carry the same two bullets.
    for k in range(3):
        g.push_snapshot(k * 0.1, tmp.snapshot(), now=k * 0.1)
    # The ghost's OWN bullet (id (1,0)) at (100,0) — NOT pending, so it is
    # drawn by the ghost (not handed off). predicted_view must draw THIS and
    # skip the buffer's copy (id (1,0) at (300,0)), while still drawing the
    # host's (id (0,0) at (150,50)).
    g.ghost.local_bullets = [
        Bullet(pygame.Vector2(100.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g.ghost.local_bullets[0].pending = False
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())

    assert len(g.ghost.local_bullets) == 1, \
        "the ghost must keep exactly its own bullet (no fire, no handoff " \
        "this frame): %r" % (g.ghost.local_bullets,)
    assert bullet_px(screen, g.cam, 100.0, 0.0) > 0, \
        "the ghost's bullet (id (1,0)) must be DRAWN at (100,0)"
    assert bullet_px(screen, g.cam, 300.0, 0.0) == 0, \
        "the buffer's copy of the client's bullet (id (1,0)) at (300,0) " \
        "must be SKIPPED (the ghost draws it): %d px" \
        % bullet_px(screen, g.cam, 300.0, 0.0)
    assert bullet_px(screen, g.cam, 150.0, 50.0) > 0, \
        "the HOST's buffer bullet (id (0,0)) must still be DRAWN at (150,50)"
    print("PASS: RENDER+DEDUP — the ghost's bullet draws at (100,0); the "
          "buffer's copy of the SAME bullet (id (1,0)) at (300,0) is "
          "SKIPPED; the host's bullet (id (0,0)) draws at (150,50)")

    # --- 5. PULL-FORWARD: a handed-off bullet (pending, buffer copy
    #     ADVANCING) is removed from the ghost and drawn by the buffer
    #     PULLED FORWARD by V*delay — ahead of the raw buffer position, in
    #     the velocity direction. The closed-form handoff: buffer_pos +
    #     V*delay == ghost_pos (a no-op visually). ---
    g2 = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
              hull=GUN_HULL, loadout=LOADOUT, seed=1234, players=2,
              local_index=LOCAL_INDEX)
    g2.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    g2.ghost = PredictedShip(hull=GUN_HULL, loadout=LOADOUT,
                             local_index=LOCAL_INDEX)
    g2.host_time = HostTimeEstimator()
    g2.latency = LatencyTracker()
    g2.render_point = RenderPoint(g2.latency)
    # A scratch game carrying ONE client bullet (id (1,0)) at (300,0) moving
    # +x at BULLET_SPEED.
    tmp2 = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, light_tex,
                fog_surf, light_surf, hull=GUN_HULL, loadout=LOADOUT,
                seed=1234, players=2)
    tmp2.set_player_ship(LOCAL_INDEX, Ship(hull=GUN_HULL, loadout=LOADOUT))
    # Same as gate 4: the ghost is seeded from tmp2's snapshot, so the
    # ship pose (and camera) comes from tmp2.players[LOCAL_INDEX].
    cs2 = tmp2.players[LOCAL_INDEX]
    cs2.pos = pygame.Vector2(0.0, 0.0)
    cs2.vel = pygame.Vector2(0.0, 0.0)
    cs2.angle = 0.0
    for w in cs2.weapons:
        if w.comp.bullet_speed > 0:
            w.cooldown = 999.0
    tmp2.bullets = [
        Bullet(pygame.Vector2(300.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0)),
    ]
    for k in range(3):
        g2.push_snapshot(k * 0.1, tmp2.snapshot(), now=k * 0.1)
    # The ghost's OWN bullet (id (1,0)), PENDING (the buffer's newest
    # snapshot carries it). Pre-seed the handoff buffer position to a
    # DIFFERENT spot so the "advance" is detected on the first frame (the
    # buffer's copy has moved since it first appeared).
    g2.ghost.local_bullets = [
        Bullet(pygame.Vector2(100.0, 0.0), pygame.Vector2(BULLET_SPEED, 0.0),
               owner=1, bid=(LOCAL_INDEX, 0))]
    g2.ghost.local_bullets[0].pending = True
    g2._bullet_handoff_buf_pos[(LOCAL_INDEX, 0)] = (200.0, 0.0)
    g2.latency.tick(0.016)
    g2.render_point.advance(0.016, g2.snap_buf.newest_time())
    # The raw buffer position of the (1,0) bullet in this frame's window.
    _rp = g2.render_point.now()
    _pos = g2.snap_buf.positions_at(_rp)
    _raw = None
    for (x, y, vx, vy, kind, owner, boost, mid) in _pos['bullets']:
        if kind == 'player' and mid is not None and tuple(mid) == (LOCAL_INDEX, 0):
            _raw = (x, y, vx, vy)
    assert _raw is not None, \
        "the buffer must carry the (1,0) bullet in the window: %r" \
        % _pos['bullets']
    _delay = g2.latency.delay
    _ex, _ey = _raw[0] + _raw[2] * _delay, _raw[1] + _raw[3] * _delay
    g2.predicted_view(0.016, pygame.key.get_pressed())
    assert len(g2.ghost.local_bullets) == 0, \
        "the handed-off bullet must be REMOVED from the ghost (the buffer " \
        "draws it now): %r" % (g2.ghost.local_bullets,)
    assert bullet_px(screen, g2.cam, _ex, _ey) > 0, \
        "the handed-off bullet must be DRAWN PULLED FORWARD at " \
        "(%.1f, %.1f) = raw + V*delay: %d px" \
        % (_ex, _ey, bullet_px(screen, g2.cam, _ex, _ey))
    # The pulled-forward position is AHEAD of the raw buffer position in the
    # velocity direction (the handoff never moves the bullet backward).
    assert _ex > _raw[0], \
        "the pulled-forward x (%.1f) must be ahead of the raw buffer x " \
        "(%.1f) for a +x bullet: the handoff is forward-only" \
        % (_ex, _raw[0])
    print("PASS: PULL-FORWARD — the handed-off bullet is removed from the "
          "ghost and drawn by the buffer PULLED FORWARD at (%.1f, %.1f) "
          "(raw (%.1f, %.1f) + V*delay, delay=%.3f s)"
          % (_ex, _ey, _raw[0], _raw[1], _delay))

    print("ALL PASS: 10.7 (ghost's own gun shots deduped against the "
          "buffer; no double-draw; closed-form pull-forward handoff)")


if __name__ == "__main__":
    main()