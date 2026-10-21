"""Plan 14 Phase 3: the ghost's MISSILE fire path is in lockstep with the
host's.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_14_3_missile_lockstep

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

Phase 3 removed the ghost's LOCAL-DRAW missile fire gate
(`len(self.local_missiles) < MAX_MISSILES` in PredictedShip.step) and added
the matching cull-oldest safety net (MISSILE_POOL_MAX), mirroring the Phase 1
host edit and the Phase 2 bullet edit. The fire decision is now input-local
on BOTH peers for missiles too (the lock-rebuild is the real rate limiter —
lock_progress resets to 0 on launch and takes 0.8 s to rebuild), so the
ghost must reproduce the host's launches exactly: the same missiles, the
same ids, the same missile_seq.

This test drives a 2P host Game and a client-side ghost with the SAME input
stream (player 1 holds the missile key) and ONE stationary target in the
wedge (a TestTarget, 300 px ahead of the nose — the missile locks + launches
on it on both peers, so the fire path is actually exercised; the natural
missile count is small (~3.75/player at the 0.8 s lock-rebuild cadence),
so a sustained-fire run needs several launches to be meaningful). A fixed one-way wire latency L is used (the
snapshot arriving now was taken L ticks ago). The ghost's PREDICTION (the
explicit steps between reconciles, run from the last resync with the local
input) must match the host's ACTUAL state at the same instant — measured
PRE-reconcile (a post-reconcile comparison would be masked by the rewind
replay, which rebuilds the prediction from authority). The host's input-echo
is forwarded to the ghost each tick so the rewind replay has the host's
actual input; the reconcile replay sees the target through the buffer's
enemy proxies (the render point's window), exactly as in live play.

  1. SEQ LOCKSTEP — at every reconcile the ghost's PREDICTED missile_seq
     (pre-reconcile, at the host's current time) equals the host's. A
     reintroduced fire gate (world- or local-count) would make the ghost
     launch a different set of missiles than the host and this gap would go
     nonzero.
  2. LOCK LOCKSTEP — the ghost's predicted missile lock_progress equals the
     host's at the same instant, to float tolerance. The lock is the
     missile's rate limiter; if it drifted, the next launch would desync
     even if the current seqs happen to agree.
  3. NO GHOST-ONLY ID — every id in the ghost's local_missiles was launched
     by the host (it is in the host's fired set). A ghost-only id is the
     double-missile (the ghost draws it, predicted, and the buffer's copy —
     which has no matching id — draws too).
  4. GHOST LAUNCHES THE SAME MISSILES — the ghost's total launch count
     equals the host's over the run.

Missiles are SLOW (the rate limiter is the lock rebuild: after a launch
lock_progress resets to 0 and takes 0.8 s to rebuild, so the cadence is
~0.8 s per launch; 3.0 s life), so 300 ticks (~5 s) yields ~6 launches per
player — enough to exercise the fire path several times. The ghost's
local_missiles list stays small (natural ~3.75) — that is fine; the
lockstep + no-ghost-only-id checks are the real gates, not the count.

The world is ISOLATED (no asteroids, update_field no-op'd; the single
TestTarget is stationary and non-firing) so the only thing that removes a
missile is its own life — collisions can't skew the fire or the pool. Both
ships sit at spawn and never move (idle input except the missile key), so
the pose is stable and the fire is the only dynamics.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import (WIDTH, HEIGHT, TICK, MISSILE_LIFE, MISSILE_POOL_MAX)
from .fog import make_light_texture
from .game import Game, STEP, _GhostEnemyProxy, _GhostPlayerProxy
from .hulls import (HullType,
                    FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT,
                    GUN, MISSILE, REACTOR, COMPUTER, SHIELD, SENSOR,
                    default_loadout)
from .intent import ShipInput
from .netcode import PredictedShip, RenderPoint, LatencyTracker
from .ship import Ship
from .spawning import TestTarget

N_TICKS = 300          # ~5 s: ~6 missile launches per player (the 0.8 s
                       # lock-rebuild cadence — lock_progress resets to 0
                       # on launch and takes 0.8 s to rebuild)
RECONCILE_EVERY = 10   # a snapshot every 10 ticks (60 Hz sim, 6 Hz snaps)
L = 3                  # one-way wire latency, in ticks (the snapshot is
                       # L ticks stale when it arrives)
EPS = 1e-6
LOCAL_INDEX = 1        # the client is player 1

# A missile-fitted test hull (the same one test_10_3b uses): the scout
# polygon + the stock slots + a 'missile' weapon slot. default_loadout(hull)
# maps 'missile' -> MISSILE_TYPE, so the stock loadout gives this hull a
# homing missile.
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

IDLE = ShipInput()                 # player 0 (the host's own ship): idle
FIRE = ShipInput(missile_fire=True)   # player 1 (the client): holds the
                                      # missile key


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
    """A 2P host Game, ISOLATED (no asteroids, update_field no-op'd) with
    ONE stationary TestTarget 300 px ahead of the players' nose — the
    missile lock target. Both players spawn at center facing -y (up), so
    the target at (WIDTH/2, HEIGHT/2 - 300) is straight ahead: inside the
    ±45° lock arc and well inside MISSILE_LOCK_RANGE (900)."""
    import ship5.game as _gm
    g = Game(*res, hull=MISSILE_HULL, loadout=LOADOUT, seed=1234,
             players=2, local_index=0)
    # Player 1 is a PLACEHOLDER (the client's hull/loadout is unknown until
    # the join handshake) — swap in the missile-fitted ship the handshake
    # would install, so the host's sim steps player 1 with the SAME
    # weapons the ghost has.
    g.set_player_ship(1, Ship(hull=MISSILE_HULL, loadout=LOADOUT))
    g.asteroids.clear()
    _gm.update_field = lambda *a, **k: None
    # Game.__init__ spawns 3 AI enemies — clear them so the TestTarget is
    # the ONLY enemy (both peers' _pick_missile_target then pick the same
    # one, and the ghost's proxy tracks it).
    g.enemies.clear()
    g.enemies.append(TestTarget(
        pygame.Vector2(WIDTH / 2, HEIGHT / 2 - 300.0), rng=g.rng))
    # plan_19: in 2P the other player's ship is a targeting candidate
    # (team 0 != team 1), and both ships spawn at center (d=0) — p1's
    # nearest candidate would be p0, not the TestTarget. Move p0 2000 px
    # out of the 900 px lock range so the TestTarget is the nearest
    # candidate on BOTH peers (p0 is idle — no input — so it stays).
    g.players[0].pos = pygame.Vector2(WIDTH / 2 - 2000.0, HEIGHT / 2)
    return g


def make_client(res, host):
    """A client Game (local_index=1) whose ghost is built from the host's
    player-1 hull/loadout — the same loadout the host applies, so the
    weapon state (cooldown/charge/lock_progress) is bit-identical. The
    render point + latency tracker are wired so the reconcile REWIND replay
    sees the buffer's enemy proxies (the target) — exactly as in live
    play; without them the replay would run with no target and the
    re-fired launch would be skipped (a false lockstep break)."""
    g = Game(*res, hull=MISSILE_HULL, loadout=LOADOUT, seed=1234,
             players=2, local_index=LOCAL_INDEX)
    p1 = host.players[LOCAL_INDEX]
    g.set_player_ship(LOCAL_INDEX,
                      Ship(hull=p1.hull, loadout=p1.components))
    g.ghost = PredictedShip(hull=p1.hull, loadout=p1.components,
                            local_index=LOCAL_INDEX)
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def missile_index(ship):
    """The index of the homing missile weapon (missile_speed > 0) in
    ship.weapons — the same weapon both peers fire; the loadout is
    identical, so the index matches on host and ghost."""
    return next(i for i, w in enumerate(ship.weapons)
                if w.comp.missile_speed > 0)


def main():
    pygame.init()
    ok = True
    res = make_res()

    host = make_host(res)
    client = make_client(res, host)
    ghost = client.ghost
    p1 = host.players[LOCAL_INDEX]
    mi = missile_index(p1)
    # The prediction's target: the buffer proxy the ghost locks (the SAME
    # target the host's _pick_missile_target picks — the only enemy).
    target = host.enemies[0]
    proxy = _GhostEnemyProxy(target.pos, target.ship.vel, target.ship.angle,
                             target.ship.id, 0.0, 0.0)
    # plan_19: the remote-player proxy (p0 — the host's own ship, 2000 px
    # out of lock range). The ghost's candidate set must match the
    # host's (enemies + the other player's ship), so the direct step
    # below feeds it alongside the enemy proxy — exactly as the buffer
    # would in live play. p0 is idle + stationary, so its pose is
    # constant.
    p0 = host.players[0]
    p_proxy = _GhostPlayerProxy(p0.pos, p0.vel, p0.angle, 0, p0.dead,
                                p0.shield_oval)

    # Seed the ghost at the host's INITIAL state (t=0), so the prediction
    # runs from the same origin as the host. snaps[t] is the host's
    # snapshot at sim time t*STEP (snaps[0] is the pre-loop state).
    snaps = [host.snapshot()]
    client.push_snapshot(0.0, snaps[0], now=0.0)

    # The host's fired ids, in fire order (the authoritative set the
    # ghost's ids must be a subset of — check 3).
    host_fired = []
    # Per-reconcile drift samples (checks 1 + 2), measured PRE-reconcile
    # (the prediction vs the host's current state at the same instant).
    seq_gaps = []
    lock_deltas = []
    # Ghost launch samples (check 4).
    ghost_launched = 0
    ghost_max_list = 0
    last_echo_idx = 0

    for t in range(1, N_TICKS + 1):
        # --- HOST tick: player 0 idle, player 1 holding the missile key.
        # The host applies self.remote_input to player 1 (the client's
        # input, as received) — set to FIRE each tick (the loopback ideal).
        host.remote_input = FIRE
        h_seq_before = p1.missile_seq
        host._step(STEP, IDLE)
        h_fired = p1.missile_seq - h_seq_before
        host_fired.extend((LOCAL_INDEX, h_seq_before + k)
                          for k in range(h_fired))
        snaps.append(host.snapshot())   # snaps[t] at t*STEP

        # --- Wire delivery + render clock (the game loop's per-frame work):
        # the client receives the host's snapshot this tick (loopback
        # ideal) and advances the render point + latency. The reconcile's
        # REWIND REPLAY reads render_point.now() to build the target
        # proxies from the buffer's window — without a live render point
        # the replay would run targetless and reset the ghost's missile
        # lock (the gun test 14_2 didn't need a target; missiles do).
        client.snap_buf.push(t * STEP, snaps[t])
        client.latency.tick(STEP)
        client.render_point.advance(STEP, client.snap_buf.newest_time())

        # --- Forward the host's new input-echo entries to the ghost (the
        # T_ECHO wire message, in-order). The rewind replay uses these as
        # the host's actual applied input.
        for (st, inp) in host._remote_input_echo[last_echo_idx:]:
            ghost.record_echo(st, inp)
        last_echo_idx = len(host._remote_input_echo)

        # --- GHOST tick: the prediction, the same input at the same step,
        # the same target (the proxy the buffer would hand it).
        g_seq_before = ghost.ship.missile_seq
        ghost.step(STEP, FIRE, enemies=[proxy], player_proxy=p_proxy)
        ghost.step_local_missiles(STEP, [proxy], player_proxy=p_proxy)
        g_fired = ghost.ship.missile_seq - g_seq_before
        ghost_launched += g_fired
        ghost_max_list = max(ghost_max_list, len(ghost.local_missiles))

        # --- RECONCILE every RECONCILE_EVERY ticks: the snapshot arriving
        # now was taken L ticks ago (stamp (t-L)*STEP). Measure the
        # PREDICTION's drift PRE-reconcile (ghost at t*STEP vs host at
        # t*STEP), then resync + replay.
        if t % RECONCILE_EVERY == 0:
            s_idx = t - L
            snap = snaps[s_idx]
            s_time = s_idx * STEP
            # 1 + 2. PRE-reconcile lockstep: the ghost's prediction (at
            # t*STEP) vs the host's actual state (at t*STEP).
            seq_gaps.append(ghost.ship.missile_seq - p1.missile_seq)
            lock_deltas.append(abs(ghost.ship.weapons[mi].lock_progress
                                   - p1.weapons[mi].lock_progress))
            # Resync + rewind replay (rebuilds the prediction from
            # authority over the last L ticks; the replay sees the target
            # through the buffer's enemy proxies).
            client.push_snapshot(s_time, snap, now=t * STEP)

    # --- 1. SEQ LOCKSTEP: the predicted gap is 0 at EVERY reconcile (a
    # single nonzero gap is a regression — a reintroduced fire gate).
    bad_seq = [(i, gap) for i, gap in enumerate(seq_gaps) if gap != 0]
    if not bad_seq:
        print("PASS: seq lockstep — ghost predicted missile_seq == host's "
              "at all %d reconciles (gap 0 every time)" % len(seq_gaps))
    else:
        ok = False
        print("FAIL: seq lockstep — %d/%d reconciles with a nonzero "
              "predicted gap (first: reconcile %d gap %d): %r"
              % (len(bad_seq), len(seq_gaps), bad_seq[0][0],
                 bad_seq[0][1], bad_seq[:5]))

    # --- 2. LOCK LOCKSTEP: the predicted vs actual missile lock_progress
    # agree to float tolerance at every reconcile. The lock is the
    # missile's rate limiter — a drift here desyncs the NEXT launch even
    # if the current seqs agree.
    worst_lock = max(lock_deltas)
    if worst_lock < EPS:
        print("PASS: lock lockstep — max |ghost - host| predicted "
              "lock_progress = %.2e over %d reconciles"
              % (worst_lock, len(lock_deltas)))
    else:
        ok = False
        print("FAIL: lock lockstep — max |ghost - host| = %.6f (>= EPS "
              "%.0e) over %d reconciles"
              % (worst_lock, EPS, len(lock_deltas)))

    # --- 3. NO GHOST-ONLY ID: every id the ghost drew was launched by the
    # host. A ghost-only id is the double-missile (the ghost draws it,
    # predicted, and the buffer's copy — no matching id — draws too).
    ghost_ids = [tuple(m.id) for m in ghost.local_missiles]
    ghost_only = [mid for mid in ghost_ids if mid not in host_fired]
    if not ghost_only:
        print("PASS: no ghost-only id — all %d in-flight ghost missile ids "
              "were launched by the host (host launched %d total)"
              % (len(ghost_ids), len(host_fired)))
    else:
        ok = False
        print("FAIL: no ghost-only id — %d ghost ids the host never "
              "launched: %r" % (len(ghost_only), ghost_only[:5]))

    # --- 4. GHOST LAUNCHES THE SAME MISSILES: the ghost's total launch
    # count equals the host's over the run (a reintroduced fire gate would
    # make it fire fewer), and the presentation list stayed under
    # MISSILE_POOL_MAX (the cull never binds at the natural ~2.5).
    launched_ok = (ghost_launched == len(host_fired)
                   and len(host_fired) >= 2)
    bound_ok = ghost_max_list <= MISSILE_POOL_MAX
    if launched_ok and bound_ok:
        lock_time = p1.weapons[mi].comp.missile_lock_time
        # The missile's rate limiter is the lock rebuild: after a launch
        # lock_progress resets to 0 and takes missile_lock_time to rebuild,
        # so the cadence is ~lock_time per launch (the natural live count
        # is MISSILE_LIFE / lock_time).
        print("PASS: ghost launches the same missiles — %d launches == "
              "the host's %d over %d ticks (lock-rebuild %.1f s cadence), "
              "list max %d (<= MISSILE_POOL_MAX %d; natural ~%.1f)"
              % (ghost_launched, len(host_fired), N_TICKS,
                 lock_time, ghost_max_list,
                 MISSILE_POOL_MAX, MISSILE_LIFE / lock_time))
    else:
        ok = False
        print("FAIL: ghost launches the same missiles — ghost=%d host=%d "
              "(want equal, host >= 2), max_list=%d (<= %d)"
              % (ghost_launched, len(host_fired), ghost_max_list,
                 MISSILE_POOL_MAX))

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()