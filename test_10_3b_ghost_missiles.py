"""Session 10.3b (Step 2): the prediction ghost emits + homes its own
missiles.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_3b_ghost_missiles

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.3b makes the CLIENT's local ship (the prediction ghost) emit homing
missiles, the same launch the host's sim produces. Before 10.3b the ghost
discarded the missiles `s.update()` returned, so the player's own missile
appeared only when the host's next snapshot arrived (~100 ms later) — and
the buffer's copy of the SAME missile would then be drawn a second time
(the double-draw).

The fix (mirrors 10.3a beams / 7.3 bullets):
  * step() picks the ghost's missile target from the buffer's enemy
    proxies (the SAME nearest-in-range math the host's
    _pick_missile_target uses) and KEEPS the missiles s.update() returns
    as presentation GhostMissiles in ghost.local_missiles.
  * Each GhostMissile carries the SAME unique id the host assigns at
    launch — (player_index, per-ship seq) — and the ghost's missile_seq is
    bumped to match (resynced to the host's via the ship snapshot on
    reconcile), so the client can dedup it against the buffer's copy.
  * Homing is CLIENT-SIDE: the missile steers toward the buffer's enemy
    proxy for its target id (re-resolved each step; coasts if the target
    left the buffer), using the same turn-rate-limited steering math as the
    host's Missile.update.

This test proves it (value-level):
  1. SAME-TICK  — for a scripted input that fires a missile at a buffer
                  enemy, the ghost emits a missile on the same tick the
                  host does, with the same muzzle pos + launch vel + id.
  2. HOMING     — after the boost phase, the ghost missile steers toward
                  an off-axis target (its velocity turns toward it).
  3. COAST      — when the target leaves the buffer, the missile coasts
                  (straight line, velocity unchanged).
  4. CULL       — a missile is culled after MISSILE_LIFE.
  5. ID-SEQ     — the id is (local_index, seq) and seq increments per
                  launch (the dedup key the host's ids match).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import MISSILE_LIFE, TICK, WIDTH, HEIGHT
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, MISSILE, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .ship import Ship
from .intent import ShipInput
from .netcode import (PredictedShip, GhostMissile, RenderPoint,
                      LatencyTracker, HostTimeEstimator)
from .game import Game, _GhostEnemyProxy
from .fog import make_light_texture

# A missile-fitted test hull: the scout polygon + the stock slots + a
# 'missile' weapon slot. default_loadout(hull) maps 'missile' -> MISSILE_TYPE
# (hulls.py), so the stock loadout gives this hull a homing missile.
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
    """A ship snapshot with the missile weapon at FULL lock (lock_progress
    1.0), the ship at the origin facing +x, vel 0. The ghost is seeded from
    this, so its muzzle + launch match the host's at the fire tick."""
    s = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    s.pos = pygame.Vector2(0.0, 0.0)
    s.vel = pygame.Vector2(0.0, 0.0)
    s.angle = 0.0
    for w in s.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 1.0
    return s.snapshot()


def make_ghost():
    return PredictedShip(hull=MISSILE_HULL, loadout=LOADOUT,
                         local_index=LOCAL_INDEX)


def main():
    pygame.init()

    # --- 1. SAME-TICK: the ghost emits a missile on the same tick the host
    #     does, with the same muzzle pos + launch vel + id. ---
    proxy = _GhostEnemyProxy(pygame.Vector2(100.0, 0.0), pygame.Vector2(0, 0),
                             0.0, 1, 0.0, 0.0)
    # The host: a bare Ship in the same state, missile_target set (as
    # Game._step does via _pick_missile_target), lock ready, missile_fire.
    host = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    host.pos = pygame.Vector2(0.0, 0.0)
    host.vel = pygame.Vector2(0.0, 0.0)
    host.angle = 0.0
    host.missile_target = proxy
    for w in host.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 1.0
    _shots, _beams, host_ms = host.update(TICK, ShipInput(missile_fire=True))
    assert len(host_ms) == 1, "the host must launch a missile: %r" % host_ms
    hm = host_ms[0]

    ghost = make_ghost()
    ghost.seed(make_seed_ship_s())
    ghost.step(TICK, ShipInput(missile_fire=True), enemies=[proxy])
    assert len(ghost.local_missiles) == 1, \
        "the ghost must emit a missile on the same tick the host does: " \
        "local_missiles=%r" % (ghost.local_missiles,)
    gm = ghost.local_missiles[0]
    assert gm.pos.distance_to(hm.pos) < 1e-6, \
        "the ghost's missile muzzle must match the host's: %r vs %r" \
        % (gm.pos, hm.pos)
    assert gm.vel.distance_to(hm.vel) < 1e-6, \
        "the ghost's missile launch vel must match the host's: %r vs %r" \
        % (gm.vel, hm.vel)
    assert gm.id == (LOCAL_INDEX, 0), \
        "the ghost's missile id must be (local_index, seq)=(%d, 0): %r" \
        % (LOCAL_INDEX, gm.id)
    print("PASS: SAME-TICK — the ghost fires a missile on the same tick the "
          "host does (muzzle %r, vel %r, id %r)"
          % (tuple(round(v, 1) for v in gm.pos),
             tuple(round(v, 1) for v in gm.vel), gm.id))

    # --- 2. HOMING: after the boost phase, the ghost missile steers toward
    #     an off-axis target (its velocity turns toward it). ---
    proxy2 = _GhostEnemyProxy(pygame.Vector2(100.0, 100.0),
                              pygame.Vector2(0, 0), 0.0, 1, 0.0, 0.0)
    ghost2 = make_ghost()
    ghost2.seed(make_seed_ship_s())
    ghost2.step(TICK, ShipInput(missile_fire=True), enemies=[proxy2])
    gm2 = ghost2.local_missiles[0]
    # Launch is +x (the missile slot orientation); the target is off-axis
    # at (100, 100), so the missile must turn toward +y to home in.
    assert gm2.vel.x > 0 and abs(gm2.vel.y) < 1e-6, \
        "the launch vel must be +x (the slot orientation): %r" % gm2.vel
    # Step through the boost phase (0.35 s = 21 ticks) and into seeking.
    for _ in range(40):
        ghost2.step_local_missiles(TICK, [proxy2])
    assert gm2.vel.y > 0, \
        "after the boost the missile must have turned toward the off-axis " \
        "target (vel.y > 0): vel=%r" % gm2.vel
    print("PASS: HOMING — after the boost the ghost missile steers toward "
          "the off-axis target (vel %r)"
          % (tuple(round(v, 1) for v in gm2.vel),))

    # --- 3. COAST: when the target leaves the buffer, the missile coasts
    #     (straight line, velocity unchanged). ---
    proxy3 = _GhostEnemyProxy(pygame.Vector2(100.0, 100.0),
                              pygame.Vector2(0, 0), 0.0, 1, 0.0, 0.0)
    ghost3 = make_ghost()
    ghost3.seed(make_seed_ship_s())
    ghost3.step(TICK, ShipInput(missile_fire=True), enemies=[proxy3])
    gm3 = ghost3.local_missiles[0]
    # Get through the boost phase (while the target is present).
    for _ in range(25):
        ghost3.step_local_missiles(TICK, [proxy3])
    vel_before = gm3.vel.copy()
    # Now the target is GONE (empty enemies list) -> the missile coasts.
    for _ in range(10):
        ghost3.step_local_missiles(TICK, [])
    assert gm3.vel.distance_to(vel_before) < 1e-6, \
        "a coasting missile (target gone) must keep its velocity: %r vs %r" \
        % (gm3.vel, vel_before)
    print("PASS: COAST — when the target leaves the buffer the missile "
          "coasts (velocity unchanged %r)"
          % (tuple(round(v, 1) for v in gm3.vel),))

    # --- 4. CULL: a missile is culled after MISSILE_LIFE. ---
    proxy4 = _GhostEnemyProxy(pygame.Vector2(100.0, 0.0), pygame.Vector2(0, 0),
                              0.0, 1, 0.0, 0.0)
    ghost4 = make_ghost()
    ghost4.seed(make_seed_ship_s())
    ghost4.step(TICK, ShipInput(missile_fire=True), enemies=[proxy4])
    assert len(ghost4.local_missiles) == 1
    n_ticks = int(MISSILE_LIFE / TICK) + 5
    for _ in range(n_ticks):
        ghost4.step_local_missiles(TICK, [proxy4])
    assert len(ghost4.local_missiles) == 0, \
        "the missile must be culled after MISSILE_LIFE (%.1f s): %r" \
        % (MISSILE_LIFE, ghost4.local_missiles)
    print("PASS: CULL — a missile is culled after MISSILE_LIFE (%.1f s)"
          % MISSILE_LIFE)

    # --- 5. ID-SEQ: the id is (local_index, seq) and seq increments per
    #     launch (the dedup key the host's ids match). ---
    proxy5 = _GhostEnemyProxy(pygame.Vector2(100.0, 0.0), pygame.Vector2(0, 0),
                              0.0, 1, 0.0, 0.0)
    ghost5 = make_ghost()
    ghost5.seed(make_seed_ship_s())
    ghost5.step(TICK, ShipInput(missile_fire=True), enemies=[proxy5])
    assert ghost5.local_missiles[0].id == (LOCAL_INDEX, 0)
    assert ghost5.ship.missile_seq == 1, \
        "the ghost's missile_seq must bump to 1 after the first launch: %r" \
        % ghost5.ship.missile_seq
    # Re-lock: hold the target for the lock time (0.8 s = 48 ticks) to
    # rebuild lock_progress, then fire again.
    for _ in range(48):
        ghost5.step(TICK, ShipInput(), enemies=[proxy5])
        ghost5.step_local_missiles(TICK, [proxy5])
    ghost5.step(TICK, ShipInput(missile_fire=True), enemies=[proxy5])
    ids = [m.id for m in ghost5.local_missiles]
    assert (LOCAL_INDEX, 1) in ids, \
        "the second missile must have id (%d, 1): %r" % (LOCAL_INDEX, ids)
    assert ghost5.ship.missile_seq == 2, \
        "the ghost's missile_seq must bump to 2 after the second launch: %r" \
        % ghost5.ship.missile_seq
    print("PASS: ID-SEQ — ids are (local_index, seq) and seq increments per "
          "launch (ids %r, seq %d)" % (ids, ghost5.ship.missile_seq))

    # --- 6. RENDER + DEDUP: predicted_view draws the ghost's missile,
    #     suppresses the buffer's copy of the SAME missile (id match), and
    #     still draws the OTHER player's buffer missile (id mismatch). ---
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

    # --- 7. RECONCILE-DEDUP: the reconcile REWIND re-runs the fire tick
    #     (apply_snapshot resets missile_seq to the host's pre-fire value,
    #     then the replay re-fires). The ghost must NOT hold two missiles
    #     of one id — the prediction's in-flight copy stays, the replay's
    #     re-fire is skipped (same id). This is the live-test "two
    #     missiles" bug: without the dedup guard the rewind duplicates the
    #     in-flight missile, and because the buffer's copy is suppressed by
    #     the id-dedup, BOTH visible ones are collision-less ghost missiles
    #     that pass through the target and orbit it. ---
    ghost7 = make_ghost()
    ghost7.seed(make_seed_ship_s())
    proxy7 = _GhostEnemyProxy(pygame.Vector2(100.0, 0.0), pygame.Vector2(0, 0),
                              0.0, 1, 0.0, 0.0)
    # The prediction fires at host time TICK (advance's step) — one ghost
    # missile, id (1,0). Stamp the inputs for the replay too (the rewind
    # replays from the input buffer, not from advance's args).
    ghost7.record_input(0.0, ShipInput())
    ghost7.advance(TICK, ShipInput(), enemies=[proxy7])
    ghost7.record_input(TICK, ShipInput(missile_fire=True))
    ghost7.advance(TICK, ShipInput(missile_fire=True), enemies=[proxy7])
    assert len(ghost7.local_missiles) == 1, \
        "the prediction must fire exactly one missile: %r" \
        % (ghost7.local_missiles,)
    assert ghost7.local_missiles[0].id == (LOCAL_INDEX, 0)
    # The host's snapshot: taken at S=0 (PRE-fire — missile_seq=0, lock
    # 1.0, because the snapshot reflects the state after the tick that
    # ended at 0, which is before the fire tick at TICK). The rewind
    # replays [0, 2*TICK], which INCLUDES the fire tick at TICK — so the
    # replay re-fires the same missile the prediction already fired.
    host7 = Ship(hull=MISSILE_HULL, loadout=LOADOUT)
    host7.pos = pygame.Vector2(0.0, 0.0)
    host7.vel = pygame.Vector2(0.0, 0.0)
    host7.angle = 0.0
    host7.missile_seq = 0
    for w in host7.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 1.0
    snap7 = host7.snapshot()
    assert snap7[21] == 0, \
        "the snapshot must be pre-fire (missile_seq 0): %r" % snap7[21]
    ghost7.reconcile_rewind(snap7, 0.0, 2 * TICK, enemies=[proxy7])
    ids7 = [tuple(m.id) for m in ghost7.local_missiles]
    assert ids7.count((LOCAL_INDEX, 0)) == 1, \
        "the rewind re-fire must NOT duplicate the in-flight missile " \
        "(one id, one missile): %r" % ids7
    assert ghost7.ship.missile_seq == 1, \
        "the resynced seq must be the post-fire value: %r" \
        % ghost7.ship.missile_seq
    print("PASS: RECONCILE-DEDUP — the rewind re-fire does not duplicate "
          "the in-flight ghost missile (ids %r, seq %d)"
          % (ids7, ghost7.ship.missile_seq))

    # --- 8. HANDBACK: once the snapshot that carries a ghost missile has
    #     arrived, the missile is marked PENDING (handed back to the
    #     buffer) — but NOT culled (10.3e). The snapshot's authoritative
    #     missile_seq (ship field 21) is the exact count of missiles the
    #     host had launched as of the snapshot, so a ghost missile with
    #     seq < host_seq is provably in the buffer — mark it pending.
    #     seq >= host_seq was launched after the snapshot — not pending
    #     (handed back by the next snapshot). The seq comparison is exact
    #     and phase-free (no dependence on the replay span).
    #
    #     10.3e: the ghost KEEPS DRAWING a pending missile until the
    #     buffer's copy is VISIBLE (the render point reaches the carrying
    #     snapshot, ~INTERP_DELAY later) — predicted_view then removes it
    #     and seeds the decoupled render offset. Culling at snapshot
    #     arrival (the pre-10.3e behavior) left a 6-15 frame gap where
    #     the missile was drawn by neither the ghost nor the buffer (the
    #     "blink"). So handback_missiles now marks, not culls: the
    #     pending missile stays in local_missiles (and is dedup-skipped
    #     from the buffer) until predicted_view hands it off. ---
    ghost8 = make_ghost()
    m0 = GhostMissile(pygame.Vector2(100.0, 0.0),
                      pygame.Vector2(460.0, 0.0), (LOCAL_INDEX, 0))
    m1 = GhostMissile(pygame.Vector2(110.0, 0.0),
                      pygame.Vector2(460.0, 0.0), (LOCAL_INDEX, 1))
    ghost8.local_missiles = [m0, m1]
    # host_seq=1: the snapshot carries missile 0 only -> m0 pending, m1
    # not. BOTH stay in local_missiles (no cull).
    ghost8.handback_missiles(1)
    assert [m.id for m in ghost8.local_missiles] == \
        [(LOCAL_INDEX, 0), (LOCAL_INDEX, 1)], \
        "pending missiles are KEPT (not culled) until the buffer's copy " \
        "is visible: %r" % [m.id for m in ghost8.local_missiles]
    assert m0.pending and not m1.pending, \
        "seq < host_seq must be marked pending; seq >= host_seq not: " \
        "m0.pending=%r m1.pending=%r" % (m0.pending, m1.pending)
    # host_seq=2: the snapshot carries both -> m1 also pending. Both stay.
    ghost8.handback_missiles(2)
    assert m0.pending and m1.pending, \
        "both missiles are in the buffer now: m0.pending=%r m1.pending=%r" \
        % (m0.pending, m1.pending)
    assert len(ghost8.local_missiles) == 2, \
        "pending missiles stay in local_missiles: %r" \
        % ghost8.local_missiles
    # host_seq=0: the snapshot carries none (pre-fire) -> nothing pending.
    ghost8.local_missiles = [m0, m1]
    m0.pending = m1.pending = False
    ghost8.handback_missiles(0)
    assert not m0.pending and not m1.pending, \
        "a pre-fire snapshot (host_seq 0) must mark nothing pending: " \
        "m0.pending=%r m1.pending=%r" % (m0.pending, m1.pending)
    assert len(ghost8.local_missiles) == 2, \
        "a pre-fire snapshot must cull nothing: %r" \
        % ghost8.local_missiles
    print("PASS: HANDBACK — ghost missiles with seq < the snapshot's "
          "authoritative missile_seq are marked pending (kept, not "
          "culled); seq >= host_seq are not pending")

    # A snapshot whose buffer carries TWO missiles: the CLIENT's own
    # (id (1,0)) at (100,0) and the HOST's (id (0,0)) at (150,50) — both
    # within the on-screen area (the camera is centered on the ghost at the
    # origin, so world x/y within ~±350 px are visible). The client ship
    # (index 1) is missile-fitted with lock 0 (so the ghost does not fire
    # during the frame).
    tmp = Game(pygame.Surface((WIDTH, HEIGHT)), font, big_font, light_tex,
               fog_surf, light_surf, hull=MISSILE_HULL, loadout=LOADOUT,
               seed=1234, players=2)
    tmp.set_player_ship(LOCAL_INDEX, Ship(hull=MISSILE_HULL, loadout=LOADOUT))
    cs = tmp.players[LOCAL_INDEX]
    cs.pos = pygame.Vector2(0.0, 0.0)
    cs.vel = pygame.Vector2(0.0, 0.0)
    cs.angle = 0.0
    for w in cs.weapons:
        if w.comp.missile_speed > 0:
            w.lock_progress = 0.0
    from .bullets import Missile
    tmp.missiles = [
        Missile(pygame.Vector2(100.0, 0.0), pygame.Vector2(460.0, 0.0),
                owner=1, mid=(LOCAL_INDEX, 0)),
        Missile(pygame.Vector2(150.0, 50.0), pygame.Vector2(460.0, 0.0),
                owner=0, mid=(0, 0)),
    ]
    # Push a 3-snapshot stream (t = 0, 0.1, 0.2) so the buffer holds a window
    # to interpolate (positions_at needs >= 2 snapshots). All carry the
    # same two missiles.
    for k in range(3):
        g.push_snapshot(k * 0.1, tmp.snapshot(), now=k * 0.1)
    # The ghost's OWN missile (id (1,0)) at (100,0) — the same missile the
    # buffer carries. predicted_view must draw THIS and skip the buffer's
    # copy (id match), while still drawing the host's (id (0,0)).
    g.ghost.local_missiles = [
        GhostMissile(pygame.Vector2(100.0, 0.0), pygame.Vector2(460.0, 0.0),
                     (LOCAL_INDEX, 0))]
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())

    def orange_px(wx, wy, r=8):
        s = g.cam.to_screen(pygame.Vector2(wx, wy))
        cx, cy = int(s.x), int(s.y)
        n = 0
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                    continue   # clamp to the screen
                px = screen.get_at((x, y))
                if px[0] > 180 and px[1] > 60 and px[1] < 200 and px[2] < 120:
                    n += 1
        return n

    assert len(g.ghost.local_missiles) == 1, \
        "the ghost must keep exactly its own missile (no fire this frame): " \
        "%r" % (g.ghost.local_missiles,)
    assert orange_px(100.0, 0.0) > 0, \
        "the ghost's missile (id (1,0)) must be DRAWN at (100,0)"
    assert orange_px(100.0, 0.0) < 40, \
        "the buffer's copy of the client's missile (id (1,0)) must be " \
        "SUPPRESSED (only the ghost's single missile draws, not two): %d px" \
        % orange_px(100.0, 0.0)
    assert orange_px(150.0, 50.0) > 0, \
        "the HOST's buffer missile (id (0,0)) must still be DRAWN at (150,50)"
    print("PASS: RENDER+DEDUP — the ghost's missile draws at (100,0); the "
          "buffer's copy of the SAME missile (id (1,0)) is suppressed "
          "(%d px, single missile); the host's missile (id (0,0)) draws at "
          "(150,50) (%d px)" % (orange_px(100.0, 0.0), orange_px(150.0, 50.0)))

    print("ALL PASS: 10.3b (ghost emits + homes its own missiles; no "
          "double-draw)")


if __name__ == "__main__":
    main()