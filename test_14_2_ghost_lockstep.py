"""Plan 14 Phase 2: the ghost's fire path is in lockstep with the host's.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_14_2_ghost_lockstep

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

Phase 2 removed the ghost's LOCAL-DRAW fire gate
(`len(self.local_bullets) < MAX_BULLETS` in PredictedShip.step) and added
the matching cull-oldest safety net (BULLET_POOL_MAX). The fire decision
is now input-local on BOTH peers (Phase 1 did the host, Phase 2 the
ghost), so the ghost must reproduce the host's fire exactly: the same
shots, the same ids, the same counters.

This test drives a 2P host Game and a client-side ghost with the SAME
input stream (player 1 holds fire), with a fixed one-way wire latency L
(the snapshot arriving now was taken L ticks ago). The ghost's PREDICTION
(the explicit steps between reconciles, run from the last resync with the
local input) must match the host's ACTUAL state at the same instant —
that is the Phase-0 drift signal, measured pre-reconcile (a post-
reconcile comparison would be masked by the rewind replay, which rebuilds
the prediction from authority). The host's input-echo is forwarded to the
ghost each tick so the rewind replay has the host's actual input.

  1. SEQ LOCKSTEP — at every reconcile the ghost's PREDICTED bullet_seq
     (pre-reconcile, at the host's current time) equals the host's
     bullet_seq. The Phase-0 drift was exactly this gap running +1 every
     reconcile cycle; it must now be 0 at every one.
  2. COOLDOWN LOCKSTEP — the ghost's predicted gun cooldown equals the
     host's at the same instant, to float tolerance. The Phase-0
     secondary signal (47% of reconciles |delta| > 5 ms).
  3. NO GHOST-ONLY ID — every id in the ghost's local_bullets was fired
     by the host (it is in the host's fired set). The Phase-0
     "host-never-fired" bullet (the ghost's 13th, 554 captured) is the
     double-draw; it must never appear.
  4. GHOST FIRES THE SAME SHOTS — the ghost fires the SAME SHOTS the host
     does (Phase 4's burst gate is input-local, so both fire the same
     6-shot bursts with the same gaps): the ghost's total fired == the
     host's total fired, and at most 1 shot/tick (the within-burst rate
     is 1 shot / 3 ticks). The FIRE COUNT is the real guard against a
     reintroduced fire gate: a cap on the fire decision would make the
     ghost fire fewer than the host. The presentation list is non-empty
     (the ghost actually fired + kept bullets) and stays under
     BULLET_POOL_MAX (the cull bounds it). NOTE: at the Phase-4 sustained
     rate (~8.6/s) the natural live count is ~13, so the list no longer
     grows past the old MAX_BULLETS 12 — that Phase-2 "grew past 12"
     check is retired; the total-fired equality is the real fire-gate
     guard now. The list also settles a bit below the host's natural ~13
     because the rewind replay re-steps the presentation bullets (aging
     their life by the replay span) — a harmless presentation artifact
     (the buffer draws the authoritative bullets; the ghost's list is
     short-lived prediction), NOT a fire-path bug.

The world is ISOLATED (no enemies/asteroids, update_field no-op'd) so
the only thing that removes a bullet is its own life — collisions can't
skew the fire or the pool. Both ships sit at spawn and never move
(idle input), so the pose is stable and the fire is the only dynamics.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import (WIDTH, HEIGHT, TICK, BULLET_LIFE,
                     BULLET_POOL_MAX, FIRE_COOLDOWN, GUN_BURST_SIZE,
                     GUN_BURST_REST)
from .fog import make_light_texture
from .game import Game, STEP
from .intent import ShipInput
from .netcode import PredictedShip

N_TICKS = 300          # 5 s of sustained fire; steady state at ~90 ticks
RECONCILE_EVERY = 10   # a snapshot every 10 ticks (60 Hz sim, 6 Hz snaps)
L = 3                  # one-way wire latency, in ticks (the snapshot is
                       # L ticks stale when it arrives)
EPS = 1e-6
LOCAL_INDEX = 1        # the client is player 1


class Keys:
    """A minimal stand-in for pygame.key.get_pressed() — idle (all keys
    released). Player 0's input is driven by this (idle)."""
    def __getitem__(self, k):
        return 0


IDLE = ShipInput()     # player 0 (the host's own ship): idle
FIRE = ShipInput(fire=True)   # player 1 (the client): holds fire


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
    """A 2P host Game, ISOLATED (no enemies/asteroids, update_field
    no-op'd) so only bullet life removes bullets. Mirrors
    test_14_1_sustained_fire's isolation."""
    import ship5.game as _gm
    g = Game(*res, seed=1234, players=2, local_index=0)
    g.enemies.clear()
    g.asteroids.clear()
    _gm.update_field = lambda *a, **k: None
    return g


def make_client(res, host):
    """A client Game (local_index=1) whose ghost is built from the host's
    player-1 hull/loadout — the same loadout the host applies, so the
    weapon state (cooldown/charge/allocation) is bit-identical."""
    g = Game(*res, seed=1234, players=2, local_index=LOCAL_INDEX)
    p1 = host.players[LOCAL_INDEX]
    # Ship stores its loadout as `components` (a dict); PredictedShip ->
    # Ship(hull, loadout) rebuilds the same components from it, so the
    # weapon state (cooldown/charge/allocation) is bit-identical.
    g.ghost = PredictedShip(hull=p1.hull, loadout=p1.components,
                            local_index=LOCAL_INDEX)
    return g


def gun_index(ship):
    """The index of the ballistic gun (bullet_speed > 0) in ship.weapons —
    the same weapon both peers fire; the loadout is identical, so the
    index matches on host and ghost."""
    return next(i for i, w in enumerate(ship.weapons)
                if w.comp.bullet_speed > 0)


def main():
    pygame.init()
    ok = True
    res = make_res()

    host = make_host(res)
    client = make_client(res, host)
    ghost = client.ghost
    p1 = host.players[LOCAL_INDEX]
    gi = gun_index(p1)

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
    cd_deltas = []
    # Ghost fire-rate samples (check 4).
    ghost_fired_ticks = 0
    ghost_max_list = 0
    ghost_min_delta = None
    ghost_max_delta = None
    last_echo_idx = 0

    for t in range(1, N_TICKS + 1):
        # --- HOST tick: player 0 idle, player 1 holding fire. The host
        # applies self.remote_input to player 1 (the client's input, as
        # received) — set to FIRE each tick (the loopback ideal).
        host.remote_input = FIRE
        h_seq_before = p1.bullet_seq
        host._step(STEP, IDLE)
        h_fired = p1.bullet_seq - h_seq_before
        host_fired.extend((LOCAL_INDEX, h_seq_before + k)
                          for k in range(h_fired))
        snaps.append(host.snapshot())   # snaps[t] at t*STEP

        # --- Forward the host's new input-echo entries to the ghost (the
        # T_ECHO wire message, in-order). The rewind replay uses these as
        # the host's actual applied input.
        for (st, inp) in host._remote_input_echo[last_echo_idx:]:
            ghost.record_echo(st, inp)
        last_echo_idx = len(host._remote_input_echo)

        # --- GHOST tick: the prediction, the same input at the same step.
        g_seq_before = ghost.ship.bullet_seq
        ghost.step(STEP, FIRE)
        ghost.step_local_bullets(STEP)
        g_fired = ghost.ship.bullet_seq - g_seq_before
        ghost_fired_ticks += g_fired
        ghost_min_delta = (g_fired if ghost_min_delta is None
                           else min(ghost_min_delta, g_fired))
        ghost_max_delta = (g_fired if ghost_max_delta is None
                           else max(ghost_max_delta, g_fired))
        ghost_max_list = max(ghost_max_list, len(ghost.local_bullets))

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
            seq_gaps.append(ghost.ship.bullet_seq - p1.bullet_seq)
            cd_deltas.append(abs(ghost.ship.weapons[gi].cooldown
                                 - p1.weapons[gi].cooldown))
            # Resync + rewind replay (rebuilds the prediction from
            # authority over the last L ticks).
            client.push_snapshot(s_time, snap, now=t * STEP)

    # --- 1. SEQ LOCKSTEP: the predicted gap is 0 at EVERY reconcile (the
    # Phase-0 drift was +1 at 1785/2021 reconciles; a single nonzero gap
    # is a regression).
    bad_seq = [(i, gap) for i, gap in enumerate(seq_gaps) if gap != 0]
    if not bad_seq:
        print("PASS: seq lockstep — ghost predicted bullet_seq == host's "
              "at all %d reconciles (gap 0 every time; Phase 0 was +1 at "
              "1785/2021)" % len(seq_gaps))
    else:
        ok = False
        print("FAIL: seq lockstep — %d/%d reconciles with a nonzero "
              "predicted gap (first: reconcile %d gap %d): %r"
              % (len(bad_seq), len(seq_gaps), bad_seq[0][0],
                 bad_seq[0][1], bad_seq[:5]))

    # --- 2. COOLDOWN LOCKSTEP: the predicted vs actual gun cooldown agree
    # to float tolerance at every reconcile (Phase 0: 47% of reconciles
    # |delta| > 5 ms).
    worst_cd = max(cd_deltas)
    if worst_cd < EPS:
        print("PASS: cooldown lockstep — max |ghost - host| predicted gun "
              "cooldown = %.2e s over %d reconciles (Phase 0: 47%% with "
              "|delta| > 5 ms)" % (worst_cd, len(cd_deltas)))
    else:
        ok = False
        print("FAIL: cooldown lockstep — max |ghost - host| = %.4f s "
              "(>= EPS %.0e) over %d reconciles"
              % (worst_cd, EPS, len(cd_deltas)))

    # --- 3. NO GHOST-ONLY ID: every id the ghost drew was fired by the
    # host. The Phase-0 "host-never-fired" bullet (the ghost's 13th, 554
    # captured) is the double-draw; it must never appear.
    ghost_ids = [tuple(b.id) for b in ghost.local_bullets]
    ghost_only = [bid for bid in ghost_ids if bid not in host_fired]
    if not ghost_only:
        print("PASS: no ghost-only id — all %d in-flight ghost bullet ids "
              "were fired by the host (host fired %d total; Phase 0 "
              "captured 554 host-never-fired)"
              % (len(ghost_ids), len(host_fired)))
    else:
        ok = False
        print("FAIL: no ghost-only id — %d ghost ids the host never "
              "fired: %r" % (len(ghost_only), ghost_only[:5]))

    # --- 4. GHOST FIRES THE SAME SHOTS: Phase 4's burst gate is
    # input-local, so the ghost fires the SAME 6-shot bursts (with the
    # same gaps) as the host — the real lockstep check is that the ghost's
    # TOTAL fired == the host's total fired (a reintroduced fire gate
    # would make the ghost fire fewer). At most 1 shot/tick (the
    # within-burst rate is 1 shot / 3 ticks; the gap ticks fire 0). The
    # presentation list is non-empty (the ghost actually fired + kept
    # bullets) and stays under BULLET_POOL_MAX (the cull never binds at
    # the natural count). NOTE: at the Phase-4 sustained rate (~8.6/s) the
    # natural live count is ~13, so the list no longer grows past the old
    # MAX_BULLETS 12 — that Phase-2 "grew past 12" check is retired; the
    # total-fired equality is the real fire-gate guard now.
    sustained = (GUN_BURST_SIZE /
                 (GUN_BURST_SIZE * FIRE_COOLDOWN + GUN_BURST_REST))
    natural = round(sustained * BULLET_LIFE)   # ~13
    rate_ok = (ghost_max_delta <= 1
               and ghost_fired_ticks == len(host_fired))
    fired_ok = ghost_max_list > 0
    bound_ok = ghost_max_list <= BULLET_POOL_MAX
    if rate_ok and fired_ok and bound_ok:
        print("PASS: ghost fires the same shots — %d shots over %d ticks "
              "(== the host's %d, at most 1/tick), list peaked at %d "
              "(<= BULLET_POOL_MAX %d; natural ~%d)"
              % (ghost_fired_ticks, N_TICKS, len(host_fired),
                 ghost_max_list, BULLET_POOL_MAX, natural))
    else:
        ok = False
        print("FAIL: ghost fires the same shots — fired=%d/%d "
              "min_delta=%r max_delta=%r max_list=%d (want == host's %d, "
              "max 1/tick, > 0, <= %d)"
              % (ghost_fired_ticks, N_TICKS, ghost_min_delta,
                 ghost_max_delta, ghost_max_list, len(host_fired),
                 BULLET_POOL_MAX))

    pygame.quit()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()