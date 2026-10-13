"""M4 two-machine CSV analysis.

Reads the fresh F3 logs (host_debug.csv, host_sim_debug.csv, net_debug.csv)
and prints a compact verdict vs the 8.4 baseline. No raw-line dumping —
just distributions + rates, so it's safe to re-run.

Run:  python analyze_m4.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def stats(vals):
    if not vals:
        return "n=0"
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    med = vals[n // 2]
    p95 = vals[int(n * 0.95)]
    return ("n=%d mean=%.4f med=%.4f p95=%.4f min=%.4f max=%.4f"
            % (n, mean, med, p95, vals[0], vals[-1]))


def rate(rows, tcol, vcol):
    """Advance rate of vcol per unit of tcol (should be ~1.0 for sim clock)."""
    if len(rows) < 2:
        return float("nan")
    t0, v0 = float(rows[0][tcol]), float(rows[0][vcol])
    t1, v1 = float(rows[-1][tcol]), float(rows[-1][vcol])
    dt = t1 - t0
    if dt <= 0:
        return float("nan")
    return (v1 - v0) / dt


def main():
    host = load("host_debug.csv")
    sim = load("host_sim_debug.csv")
    net = load("net_debug.csv")

    print("=" * 70)
    print("M4 TWO-MACHINE VERDICT  (baseline: 8.4 host fps mean 21.6)")
    print("=" * 70)

    # ---- HOST frame rate (host_debug.csv, 10 Hz windows) ----
    fps = [float(r["fps"]) for r in host if float(r["fps"]) > 0]
    dt_max = [float(r["dt_max"]) for r in host]
    hic = [int(r["hic"]) for r in host]
    nframes = [int(r["n"]) for r in host]
    tot_frames = sum(nframes)
    tot_hic = sum(hic)
    print("\n[HOST] host_debug.csv  (%d windows, %d frames)"
          % (len(host), tot_frames))
    print("  fps (per-window mean): %s" % stats(fps))
    if fps:
        # overall fps = total frames / total wall time
        wall = sum(float(r["n"]) / float(r["fps"]) for r in host
                   if float(r["fps"]) > 0)
        print("  overall fps (frames/wall): %.1f   <-- headline M3 number"
              % (tot_frames / wall if wall else 0))
    print("  dt_max (longest raw frame/window): %s" % stats(dt_max))
    print("  hiccup frames (>50ms): total=%d  (per-window: %s)"
          % (tot_hic, stats([float(h) for h in hic])))

    # ---- SIM clock rate + render alpha (host_sim_debug.csv, per-frame) ----
    raw_dt = [float(r["raw_dt"]) for r in sim]
    draw_dt = [float(r["draw_dt"]) for r in sim]
    alpha = [float(r["alpha"]) for r in sim]
    sim_rate = rate(sim, "t", "sim_time")
    wall_span = float(sim[-1]["t"]) - float(sim[0]["t"])
    print("\n[SIM] host_sim_debug.csv  (%d frames, %.1f s wall)"
          % (len(sim), wall_span))
    print("  sim-clock rate (sim_time/wall): %.4f  (want ~1.0)" % sim_rate)
    print("  raw_dt (real frame time): %s" % stats(raw_dt))
    print("  draw_dt (clamped, drives draw): %s" % stats(draw_dt))
    print("  render alpha: %s" % stats(alpha))
    # alpha histogram (buckets of 0.1)
    hist = [0] * 10
    for a in alpha:
        b = min(9, int(a * 10))
        hist[b] += 1
    print("  alpha histogram [0-0.1 .. 0.9-1.0]: %s" % hist)
    # how often is alpha pinned at exactly 0?
    zero = sum(1 for a in alpha if a < 0.005)
    print("  alpha < 0.005: %d/%d (%.0f%%)"
          % (zero, len(alpha), 100.0 * zero / len(alpha)))

    # ---- CLIENT (net_debug.csv, per-reconcile ~10 Hz) ----
    snap = [float(r["snap_px"]) for r in net if r["snap_px"] != ""]
    delay = [float(r["delay"]) for r in net if r["delay"] != ""]
    jit = [float(r["jitter_ema"]) for r in net if r["jitter_ema"] != ""]
    buf = [int(r["buf_depth"]) for r in net if r["buf_depth"] != ""]
    cfps = [float(r["fps"]) for r in net if float(r["fps"] or 0) > 0]
    print("\n[CLIENT] net_debug.csv  (%d reconciles)" % len(net))
    print("  snap_px (ghost displacement/reconcile): %s" % stats(snap))
    print("  delay (adaptive interp delay): %s   (8.4 baseline mean 0.146)"
          % stats(delay))
    print("  jitter_ema: %s   (8.4 baseline mean 0.0232)" % stats(jit))
    print("  buf_depth: %s" % stats([float(b) for b in buf]))
    if cfps:
        print("  client fps: %s" % stats(cfps))

    # ---- cross-check: host stall vs client gap ----
    # (informational: are the big client snap_px spikes aligned with host
    #  hiccup windows? The user already said the end-of-run snap = death.)
    if snap:
        big = [(i, s) for i, s in enumerate(snap) if s > 20]
        n = len(snap)
        print("\n[CLIENT] snap_px > 20px spikes: %d total (of %d)"
              % (len(big), n))
        if big:
            idxs = [i for i, _ in big]
            shown = idxs[:10]
            if len(idxs) > 10:
                shown = shown + ["..."]
            print("  at reconcile indices: %s" % shown)
            tail = sum(1 for i in idxs if i >= n - 20)
            print("  of which in the final 20 reconciles: %d  "
                  "(end-of-run cluster = death/reconcile, per user)" % tail)
        # steady-state snap = exclude the >20px spikes (the death)
        steady = [s for s in snap if s <= 20]
        if steady:
            print("  STEADY-STATE snap_px (spikes excluded): %s"
                  % stats(steady))
            print("    (8.4 baseline steady mean ~1.81)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()