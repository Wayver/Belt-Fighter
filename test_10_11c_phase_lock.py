"""10.11c phase-lock regression test.

Proves the phase-locked free-run clock executes EXACTLY 2 physics ticks per
2-tick snapshot interval (not 1/2/3) when the host runs at 1x real time —
the healthy case the user's real setup (separate machines) looks like. The
10.11b bug was a free-running accumulator whose phase drifted relative to
the host's tick grid, so the ghost executed 1/2/3 ticks per interval and the
reconcile corrected the ±1-tick error with an 8.67px snap (the bimodal snap).

The proof (no network, no threads — fully deterministic):
  - A reference Ship (the "host") is stepped at exactly the sim's rate
    (2 ticks per snapshot interval).
  - The ghost's clock is driven by real_time at 1x real time (the
    phase-locked formula), stepping physics on tick-boundary crossings.
  - Every snapshot interval, the reference Ship's pose is applied as the
    snapshot, and the ghost reconciles (apply + replay L of input).
  - ASSERT: over N intervals, the free-run executes EXACTLY 2 ticks per
    interval (the phase-lock), and the snap (displacement across the
    reconcile) is ~0 (the ghost's pose matches the authority + replay).

Run from the repo root:
    python -m ship5.test_10_11c_phase_lock
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

EPS = 1e-6
L = 0.020              # one-way latency (s) — the RTT/2 the client measures
SNAP_INTERVAL = 2 * TICK   # 2 ticks (33.33 ms) — the host's snapshot cadence
N_INTERVALS = 30         # 30 intervals = 1 s of sim time


def main():
    pygame.init()
    ok = True
    thrust = ShipInput(thrust_fwd=1.0)

    # The reference Ship (the "host") — stepped at exactly the sim's rate.
    host = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    host.vel = pygame.Vector2(0.0, 300.0)

    # The ghost — seeded from the host's initial pose.
    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g.seed(host.snapshot(), snap_time=0.0, real_time=0.0)

    host_sim = 0.0     # the host's sim time (1x real time)
    real_time = 0.0    # the client's real time
    ticks_per_interval = []
    snaps = []

    for i in range(N_INTERVALS):
        # Free-run the ghost for one snapshot interval (real_time advances
        # by SNAP_INTERVAL at 1x real time). The ghost's clock advances by
        # SNAP_INTERVAL (2 ticks) and steps physics on tick-boundary
        # crossings.
        steps = g.advance(SNAP_INTERVAL, thrust,
                          real_time=real_time + SNAP_INTERVAL)
        ticks_per_interval.append(steps)

        # Step the reference Ship (the host) 2 ticks (its sim time advances
        # by SNAP_INTERVAL = 2 ticks at 1x real time).
        for _ in range(2):
            host.step(TICK, thrust)
        host_sim += SNAP_INTERVAL
        real_time += SNAP_INTERVAL

        # A snapshot arrives (the host's pose at host_sim). The ghost
        # reconciles: apply the snapshot + replay L of input.
        snap_time = host_sim
        now = snap_time + L
        bx, by = g.ship.pos.x, g.ship.pos.y
        snap = host.snapshot()
        g.record_input(snap_time, thrust)
        g.reconcile_rewind(snap, snap_time, now,
                           real_time=real_time, rate=1.0)
        snaps.append(math.hypot(g.ship.pos.x - bx, g.ship.pos.y - by))

    # --- ASSERT 1: the free-run executes EXACTLY 2 ticks per interval.
    all_two = all(t == 2 for t in ticks_per_interval)
    dist = {t: ticks_per_interval.count(t) for t in set(ticks_per_interval)}
    a1_ok = all_two
    print(("PASS" if a1_ok else "FAIL") +
          f": phase-lock — free-run executed EXACTLY 2 ticks per interval "
          f"({N_INTERVALS} intervals). Distribution: {dist}")
    ok &= a1_ok

    # --- ASSERT 2: the snap is ~0 (the ghost's pose matches the authority
    # + replay — no bimodal snap). The 10.11b bug was a bimodal snap (0 when
    # the free-run was 2 ticks, ~8.67px when it was 1 or 3).
    max_snap = max(snaps)
    mean_snap = sum(snaps) / len(snaps)
    big = sum(1 for s in snaps if s > 1.0)
    a2_ok = max_snap < 1.0
    print(("PASS" if a2_ok else "FAIL") +
          f": no bimodal snap — max snap {max_snap:.3f}px, mean "
          f"{mean_snap:.3f}px, {big}/{N_INTERVALS} intervals > 1px "
          f"(the 10.11b bug was ~8.67px on 47% of intervals)")
    ok &= a2_ok

    # --- ASSERT 3: the ghost's pose matches the host's pose (phase-locked).
    host_pos = host.pos
    ghost_pos = g.ship.pos
    pos_dist = math.hypot(ghost_pos.x - host_pos.x, ghost_pos.y - host_pos.y)
    a3_ok = pos_dist < 1.0
    print(("PASS" if a3_ok else "FAIL") +
          f": ghost matches host — pos dist {pos_dist:.3f}px "
          f"(ghost at {ghost_pos.x:.1f},{ghost_pos.y:.1f}, "
          f"host at {host_pos.x:.1f},{host_pos.y:.1f})")
    ok &= a3_ok

    pygame.quit()
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()