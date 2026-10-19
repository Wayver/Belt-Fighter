"""Analyze bullet_handoff_debug.csv — the handoff-offset measurement.

At each bullet handoff (ghost -> buffer), the offset between the ghost's
last drawn position and the buffer's drawn position (already pulled forward
by V*delay) is decomposed into:
  * off_par  — ALONG the bullet's velocity (signed px). A nonzero value is
               a TIME-DOMAIN mismatch: the ghost draws at its predicted
               clock (host_time + rtt_ow) while the buffer draws at the
               newest snapshot stamp. off_par ~= V * (ghost_present -
               newest). The pull-forward (V*delay) only cancels this if
               delay == (ghost_present - newest).
  * off_perp — PERPENDICULAR to the velocity (px). A nonzero value is a
               PREDICTION-ERROR / MUZZLE mismatch: the ghost's bullet line
               (muzzle + velocity at the fire tick) differs from the
               host's actual bullet line, so the two are parallel lines
               offset laterally. This is the "shifted magically a little
               bit perpendicular" symptom.

A SEAMLESS handoff has off_mag ~= 0 (both components ~0). The analyzer
reports the distribution of each component and flags the worst handoffs.

Run:  python analyze_bullet_handoff.py
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError):
        return float("nan")


def stats(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return None
    return (min(vals), max(vals), sum(vals) / len(vals), len(vals))


def main():
    rows = load("bullet_handoff_debug.csv")
    if not rows:
        print("bullet_handoff_debug.csv is missing or empty — capture a 2P "
              "run with F3 on (debug_net) and fire a few bursts, then re-run.")
        return

    print("=" * 72)
    print("BULLET HANDOFF-OFFSET ANALYSIS  (%d handoffs)" % len(rows))
    print("=" * 72)

    mags = [f(r, "off_mag") for r in rows]
    pars = [f(r, "off_par") for r in rows]
    perps = [f(r, "off_perp") for r in rows]
    delays = [f(r, "delay") for r in rows]
    speeds = [math.hypot(f(r, "bvx"), f(r, "bvy")) for r in rows]

    def show(label, vals, unit="px"):
        s = stats(vals)
        if s is None:
            print("  %-10s (no data)" % label)
            return
        print("  %-10s min=%8.2f  max=%8.2f  mean=%8.2f  n=%d  %s"
              % (label, s[0], s[1], s[2], s[3], unit))

    print("\n[1] OFFSET MAGNITUDE (want ~0 for a seamless handoff)")
    show("off_mag", mags)
    show("off_par", pars)
    show("off_perp", perps)
    show("delay", delays, "s")
    show("speed", speeds, "px/s")

    # How many handoffs are "seamless" (off_mag < 2 px)?
    ok = sum(1 for m in mags if not math.isnan(m) and m < 2.0)
    print("\n  seamless (off_mag < 2 px): %d / %d (%.0f%%)"
          % (ok, len(rows), 100.0 * ok / len(rows)))

    # [2] Decomposition: which component dominates?
    n_par = sum(1 for r in rows
                if abs(f(r, "off_par")) > abs(f(r, "off_perp")) + 1.0)
    n_perp = sum(1 for r in rows
                 if abs(f(r, "off_perp")) > abs(f(r, "off_par")) + 1.0)
    print("\n[2] DOMINANT COMPONENT (which axis is the jump on?)")
    print("  parallel-dominated  (time-domain / clock gap): %d" % n_par)
    print("  perpendicular-dominated (prediction/muzzle):   %d" % n_perp)
    print("  mixed / small:                                   %d"
          % (len(rows) - n_par - n_perp))

    # [3] off_par vs delay: is the parallel offset consistent with
    #     V * (ghost_present - newest)? If off_par ~= V*delay, the pull-
    #     forward is NOT cancelling it (delay is measured from newest, but
    #     the ghost is AHEAD of newest by rtt_ow + the step phase).
    print("\n[3] PARALLEL OFFSET vs DELAY (is the pull-forward enough?)")
    print("  (off_par should be ~0 if the pull-forward fully cancels the")
    print("   time-domain gap; a consistent positive off_par means the")
    print("   ghost is drawn AHEAD of the buffer by more than V*delay)")
    for r in rows[:10]:
        print("    bid=%-6s off_par=%8.2f  off_perp=%8.2f  delay=%6.3f  "
              "speed=%7.1f  ship=(%7.1f,%7.1f ang=%6.2f)"
              % (r["bid"], f(r, "off_par"), f(r, "off_perp"), f(r, "delay"),
                 math.hypot(f(r, "bvx"), f(r, "bvy")),
                 f(r, "ship_x"), f(r, "ship_y"), f(r, "ship_ang")))

    # [4] Worst handoffs (largest off_mag).
    print("\n[4] WORST 10 HANDOFFS (by off_mag)")
    worst = sorted(rows, key=lambda r: -f(r, "off_mag"))[:10]
    for r in worst:
        print("    t=%-12s bid=%-6s off=(%7.2f,%7.2f) mag=%7.2f  "
              "par=%7.2f perp=%7.2f  delay=%6.3f"
              % (r["t"], r["bid"], f(r, "off_x"), f(r, "off_y"),
                 f(r, "off_mag"), f(r, "off_par"), f(r, "off_perp"),
                 f(r, "delay")))

    # [5] Perpendicular offset vs ship speed (is the perp error from the
    #     ship moving between the ghost's fire tick and the host's actual
    #     fire tick? i.e. prediction error scales with ship speed).
    print("\n[5] PERPENDICULAR OFFSET vs SHIP SPEED (prediction error?)")
    # bucket by ship speed
    buckets = {}
    for r in rows:
        sv = math.hypot(f(r, "ship_vx"), f(r, "ship_vy"))
        b = int(sv // 100) * 100
        buckets.setdefault(b, []).append(f(r, "off_perp"))
    for b in sorted(buckets):
        vals = [v for v in buckets[b] if not math.isnan(v)]
        if vals:
            print("  ship speed %4d-%4d: mean perp=%7.2f  max=%7.2f  n=%d"
                  % (b, b + 100, sum(vals) / len(vals), max(vals), len(vals)))

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("  off_par  ~ 0 and off_perp ~ 0  -> seamless (no fix needed).")
    print("  off_par  >> 0 (consistent)     -> time-domain gap: the ghost is")
    print("     drawn ahead of the buffer by more than V*delay. The pull-")
    print("     forward uses `delay` (newest - render point) but the ghost")
    print("     sits at host_time + rtt_ow, which is AHEAD of newest. Fix:")
    print("     pull forward by V*(ghost_present - newest), not V*delay.")
    print("  off_perp >> 0                  -> prediction/muzzle error: the")
    print("     ghost's ship pose (pos/angle) at the fire tick drifted from")
    print("     the host's actual pose, so the bullet line is offset")
    print("     laterally. Scales with ship speed (see [5]). Fix: reduce the")
    print("     ghost's pose error at fire time (tighter reconcile / phase).")


if __name__ == "__main__":
    main()