"""10.11b verification: the ghost clock anchored at snap_time + L.

Proves the mechanism end-to-end (no network):
  1. SEED — the ghost's clock starts at the first snapshot's time.
  2. ADVANCE — between snapshots the clock free-runs (n*TICK per step).
  3. RECONCILE with now = snap_time + L — the replay span is L (positive,
     the prediction offset), the clock is anchored to the present, and the
     render pose is at the present (no snap).
  4. BUFFER — the input/echo buffer (2 s) covers the rewind span (L ~ 20 ms).
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


def make_ghost():
    s = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    s.vel = pygame.Vector2(0.0, 300.0)
    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g.seed(s.snapshot())
    return g


def main():
    pygame.init()
    ok = True
    L = 0.020   # one-way latency (s) — the RTT/2 the client measures

    # --- (1) SEED: the clock starts at the first snapshot's time.
    g = make_ghost()
    seed_t = 10.0
    g.seed(g.ship.snapshot(), snap_time=seed_t)
    seed_ok = abs(g.sim_time - seed_t) < EPS
    print(("PASS" if seed_ok else "FAIL") +
          f": seed — ghost clock starts at the snapshot's time "
          f"({g.sim_time:.4f} == {seed_t:.4f})")
    ok &= seed_ok

    # --- (2) ADVANCE: between snapshots the clock free-runs (n*TICK/step).
    g = make_ghost()
    g.seed(g.ship.snapshot(), snap_time=seed_t)
    thrust = ShipInput(thrust_fwd=1.0)
    # Advance 1.5 s of real time in 16 ms frames (the client's frame rate).
    t = seed_t
    frames = 0
    while t < seed_t + 1.5:
        g.advance(0.016, thrust)
        t += 0.016
        frames += 1
    # The clock should be ~1.5 s ahead of the seed (free-run).
    adv_ok = 1.4 < (g.sim_time - seed_t) < 1.6
    print(("PASS" if adv_ok else "FAIL") +
          f": advance — clock free-ran {g.sim_time - seed_t:.3f}s over "
          f"{frames} frames (expect ~1.5s)")
    ok &= adv_ok

    # --- (3) RECONCILE with now = snap_time + L: the span is L (positive),
    # the clock is anchored to the present, and the render pose is at the
    # present (no snap). This is the 10.11b fix: the ghost sits at the
    # real-time present, not at the snapshot's time L in the past.
    g = make_ghost()
    g.seed(g.ship.snapshot(), snap_time=seed_t)
    g.advance(0.05, thrust)   # a little prediction before the snapshot
    # The snapshot arrives at snap_time = seed_t + 0.05 (the host's clock).
    snap_t = seed_t + 0.05
    # The client anchors the ghost at snap_time + L (the present).
    now = snap_t + L
    # Capture the pre-reconcile render pose (the predicted pose).
    pre_pos, _ = g.render_pose()
    # Reconcile: apply the snapshot + replay the span (now - snap_time = L).
    # Use a far pose so the snap is measurable.
    c = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    c.pos = pygame.Vector2(500.0, 500.0)
    c.angle = 0.5
    g.reconcile_rewind(c.snapshot(), snap_t, now)
    # The replay span should be L (positive — the prediction offset).
    span_ok = abs(g._last_replay_span - L) < EPS
    # The clock should be anchored to the present (now = snap_t + L).
    clock_ok = abs(g.sim_time - now) < EPS
    # The render pose should be at the reconciled pose (no phantom lerp).
    rp, ra = g.render_pose()
    render_ok = (abs(rp.x - 500.0) < EPS and abs(rp.y - 500.0) < EPS
                 and abs(ra - 0.5) < EPS)
    recon_ok = span_ok and clock_ok and render_ok
    print(("PASS" if recon_ok else "FAIL") +
          f": reconcile — span={g._last_replay_span*1000:.1f}ms (expect "
          f"{L*1000:.1f}ms, positive), clock={g.sim_time:.4f} (expect "
          f"{now:.4f}), render at reconciled pose (no snap)")
    ok &= recon_ok

    # --- (4) BUFFER: the input/echo buffer (2 s) covers the rewind span
    # (L ~ 20 ms). The buffer bound is INPUT_BUFFER_MAX = 2.0 s, which is
    # >> L, so the replay never runs out of inputs.
    g = make_ghost()
    g.seed(g.ship.snapshot(), snap_time=seed_t)
    # Record inputs over 2.5 s (the buffer should drop the oldest 0.5 s).
    for i in range(150):   # 150 * 16 ms = 2.4 s
        g.record_input(seed_t + i * 0.016, ShipInput(thrust_fwd=1.0))
    buf_span = g._input_buffer[-1][0] - g._input_buffer[0][0]
    buf_ok = buf_span <= 2.0 + EPS and buf_span > 1.9
    # The rewind span (L) is well within the buffer.
    covers_ok = L < buf_span
    print(("PASS" if buf_ok and covers_ok else "FAIL") +
          f": buffer — span={buf_span:.2f}s (bound 2.0s), covers the "
          f"rewind span L={L*1000:.0f}ms")
    ok &= buf_ok and covers_ok

    pygame.quit()
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()