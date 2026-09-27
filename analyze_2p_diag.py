"""Diagnose the 2P 'a little worse' run: correlate the client's snap spikes
against host hiccups / sim-clock drops at the same wall-clock time, and
check whether the host sim clock is steady or fluctuating.

Run:  python analyze_2p_diag.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), newline="") as f:
        return list(csv.DictReader(f))


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError):
        return float("nan")


def main():
    host = load("host_debug.csv")        # 10 Hz windows: t,sim_time,fps,dt_max,hic,n
    sim = load("host_sim_debug.csv")     # per frame: t,raw_dt,draw_dt,sim_time,alpha
    net = load("net_debug.csv")          # per reconcile: t,snap_px,...,delay,jitter_ema,...

    print("=" * 72)
    print("2P DIAGNOSTIC  (host=Mac ~29fps, client=4090 ~62fps)")
    print("=" * 72)

    # ---- 1. host sim-clock rate, per 10 Hz window (steady or fluctuating?) ----
    print("\n[1] HOST SIM-CLOCK RATE per 10 Hz window (want ~1.0, steady)")
    # group sim rows into 0.1 s windows by t
    import math
    wins = {}
    for r in sim:
        t = f(r, "t")
        if math.isnan(t):
            continue
        w = int(t * 10)
        wins.setdefault(w, []).append(r)
    rates = []
    for w in sorted(wins):
        rows = wins[w]
        t0, s0 = f(rows[0], "t"), f(rows[0], "sim_time")
        t1, s1 = f(rows[-1], "t"), f(rows[-1], "sim_time")
        if t1 - t0 > 0.05:
            rates.append((w / 10.0, (s1 - s0) / (t1 - t0)))
    if rates:
        vals = [r for _, r in rates]
        import statistics
        print("  windows: %d  rate mean=%.4f med=%.4f min=%.4f max=%.4f"
              % (len(vals), statistics.mean(vals), statistics.median(vals),
                 min(vals), max(vals)))
        # show the worst 5 windows
        worst = sorted(rates, key=lambda x: x[1])[:5]
        print("  worst 5 windows (t, rate): " +
              ", ".join("(%.1f, %.3f)" % (t, r) for t, r in worst))
        # count windows below 0.98
        low = sum(1 for _, r in rates if r < 0.98)
        print("  windows with rate < 0.98: %d / %d" % (low, len(rates)))

    # ---- 2. host fps per window: dips? ----
    print("\n[2] HOST FPS per 10 Hz window")
    fps = [f(r, "fps") for r in host if f(r, "fps") > 0]
    hic = [int(r["hic"]) for r in host]
    import statistics
    print("  fps mean=%.1f med=%.1f min=%.1f max=%.1f"
          % (statistics.mean(fps), statistics.median(fps), min(fps), max(fps)))
    print("  hiccup frames total=%d  windows with >=1 hiccup=%d / %d"
          % (sum(hic), sum(1 for h in hic if h > 0), len(hic)))
    # worst fps windows
    wf = sorted(((f(r, "t"), f(r, "fps"), int(r["hic"])) for r in host),
                key=lambda x: x[1])[:5]
    print("  worst 5 fps windows (t, fps, hic): " +
          ", ".join("(%.1f, %.0f, %d)" % x for x in wf))

    # ---- 3. correlate client snap spikes with host state at same t ----
    print("\n[3] CLIENT SNAP SPIKES vs HOST STATE (same wall-clock t)")
    # index host sim rows by t (per frame) for lookup
    sim_by_t = {}
    for r in sim:
        sim_by_t[round(f(r, "t"), 1)] = r
    # host hiccup windows
    hic_wins = {}
    for r in host:
        t = f(r, "t")
        hic_wins[round(t, 1)] = (int(r["hic"]), f(r, "fps"))

    spikes = [(f(r, "t"), f(r, "snap_px"), f(r, "delay"), f(r, "jitter_ema"))
              for r in net if f(r, "snap_px") > 20]
    print("  snap>20px spikes: %d" % len(spikes))
    # for each spike, find the host state ~0-0.15s earlier (the snapshot the
    # client just reconciled was sent by the host slightly before)
    matched_hic = 0
    host_low_fps = 0
    for t, sp, d, j in spikes:
        # look at host windows in [t-0.3, t]
        found_hic = 0
        found_fps = []
        for dt in (0.0, 0.1, 0.2, 0.3):
            key = round(t - dt, 1)
            if key in hic_wins:
                h, fp = hic_wins[key]
                found_hic += h
                found_fps.append(fp)
        if found_hic > 0:
            matched_hic += 1
        if found_fps and min(found_fps) < 25:
            host_low_fps += 1
    print("  spikes with a host hiccup in the prior 0.3s: %d / %d"
          % (matched_hic, len(spikes)))
    print("  spikes where host fps < 25 in the prior 0.3s: %d / %d"
          % (host_low_fps, len(spikes)))
    # show first 8 spikes with their host context
    print("  first 8 spikes (t, snap, delay, jitter | host hic/fps prior 0.3s):")
    for t, sp, d, j in spikes[:8]:
        ctx = []
        for dt in (0.0, 0.1, 0.2, 0.3):
            key = round(t - dt, 1)
            if key in hic_wins:
                ctx.append("%d/%.0f" % hic_wins[key])
        print("    t=%.1f snap=%.0f delay=%.3f jit=%.3f | %s"
              % (t, sp, d, j, " ".join(ctx)))

    # ---- 4. client delay + jitter over time (drifting up?) ----
    print("\n[4] CLIENT delay + jitter over time (first vs last quarter)")
    def q(rows, k, frac):
        v = [f(r, k) for r in rows if not math.isnan(f(r, k))]
        n = len(v)
        qn = max(1, n // 4)
        return (sum(v[:qn]) / qn, sum(v[-qn:]) / qn)
    d0, d1 = q(net, "delay", 0.25)
    j0, j1 = q(net, "jitter_ema", 0.25)
    s0, s1 = q(net, "snap_px", 0.25)
    print("  delay    first-qtr=%.4f  last-qtr=%.4f" % (d0, d1))
    print("  jitter   first-qtr=%.4f  last-qtr=%.4f" % (j0, j1))
    print("  snap_px  first-qtr=%.2f  last-qtr=%.2f" % (s0, s1))

    # ---- 5. host render alpha over time (pinned at 1.0?) ----
    print("\n[5] HOST RENDER ALPHA over time")
    a_first = [f(r, "alpha") for r in sim[:len(sim) // 4]]
    a_last = [f(r, "alpha") for r in sim[-len(sim) // 4:]]
    import statistics
    pin_first = sum(1 for a in a_first if a >= 0.999) / max(1, len(a_first))
    pin_last = sum(1 for a in a_last if a >= 0.999) / max(1, len(a_last))
    print("  first-qtr: mean=%.3f  pinned@1.0=%.0f%%"
          % (statistics.mean(a_first), 100 * pin_first))
    print("  last-qtr : mean=%.3f  pinned@1.0=%.0f%%"
          % (statistics.mean(a_last), 100 * pin_last))


if __name__ == "__main__":
    main()