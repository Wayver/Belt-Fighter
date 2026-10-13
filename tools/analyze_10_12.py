"""10.12 post-change analysis (no paired baseline — the old CSVs were
overwritten). Judge the run on its own merits:

  1. RTT: rtt_ow should settle near the LAN one-way latency (ping 5-8 ms
     RTT -> L ~ 2.5-4 ms; the 10.11b EMA starts at 8.3 ms and should decay).
  2. SNAP: distribution + the in-flight-input signature — mean snap on
     INPUT-CHANGE reconciles vs steady-input reconciles. Pre-10.12 the
     input-change snap was ~v*(L+12ms) ~ 17 px; post-10.12 it should be
     ~v*(L+1ms) ~ 3-5 px.
  3. PHASE LOCK: lead = ghost_sim_time - newest_stamp should be narrow
     (the 10.11c diagnostic).
  4. REPLAY: n (replay_ticks) consistent with L (n = int(L/TICK)).
  5. HOST: fps, hiccup rate, sim-clock rate (host_debug + host_sim_debug).
  6. CLIENT: fps, hic, delay, jitter_ema, buf_depth.

Run: python analyze_10_12.py   (from the workspace root)
"""
import csv
import statistics as st


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(label, xs):
    print("  %-14s n=%-5d mean=%8.3f  p50=%8.3f  p95=%8.3f  max=%8.3f"
          % (label, len(xs), st.mean(xs), pct(xs, 50), pct(xs, 95), max(xs)))


def main():
    net = load("net_debug.csv")
    hdbg = load("host_debug.csv")
    hsdbg = load("host_sim_debug.csv")

    # ---- trim the pre-game tail: the client logs from the first snapshot;
    # the host sim clock is frozen after death (advance==0 tail). Use the
    # rows where the sim was actually advancing on the host side.
    t0 = float(net[0]["t"])
    t1 = float(net[-1]["t"])
    print("== SESSION: net_debug %.1f s (%d reconciles), %.3f -> %.3f"
          % (t1 - t0, len(net), t0, t1))

    # ---- 1. RTT (one-way estimate) over time ----
    rtt = [float(r["rtt_ow"]) for r in net]
    third = len(rtt) // 3
    print("\n== RTT (one-way, s) — the 10.11b EMA (starts 8.3 ms, clamped 2-250)")
    for name, seg in (("first 1/3", rtt[:third]),
                      ("mid 1/3", rtt[third:2 * third]),
                      ("last 1/3", rtt[2 * third:])):
        print("  %-10s mean=%7.4f  p50=%7.4f  min=%7.4f  max=%7.4f"
              % (name, st.mean(seg), pct(seg, 50), min(seg), max(seg)))
    print("  (ping between the machines: 5-8 ms RTT -> expect L ~ 2.5-4 ms)")

    # ---- 2. SNAP distribution + input-change signature ----
    snaps = [float(r["snap_px"]) for r in net]
    print("\n== SNAP (px, ghost displacement across reconcile)")
    dist("all", snaps)
    for thr in (5, 10, 20, 50):
        n = sum(1 for s in snaps if s > thr)
        print("  >%3d px: %4d (%.1f%%)" % (thr, n, 100.0 * n / len(snaps)))

    # input-change signature: compare each row's 'input' to the previous
    # row's (the input active at the PREVIOUS reconcile). A change means
    # the client's input moved between two reconciles -> the in-flight
    # window is open -> the snap should carry the v*(L+poll) correction.
    chg, steady = [], []
    prev_inp = None
    for r in net:
        inp = r["input"]
        if prev_inp is not None and inp != prev_inp:
            chg.append(float(r["snap_px"]))
        elif prev_inp is not None:
            steady.append(float(r["snap_px"]))
        prev_inp = inp
    print("\n== IN-FLIGHT-INPUT SIGNATURE (the 10.12 target)")
    dist("input CHANGED", chg)
    dist("input steady", steady)
    if chg and steady:
        print("  ratio (changed/steady) = %.2f" % (st.mean(chg) / st.mean(steady)))
    print("  pre-10.12 expectation: changed ~ v*(L+12ms) ~ 17 px at 520 px/s")
    print("  post-10.12 expectation: changed ~ v*(L+1ms)  ~ 3-5 px")

    # ---- 3. phase lock: lead = ghost_sim_time - newest_stamp ----
    lead = [float(r["ghost_sim_time"]) - float(r["newest_stamp"]) for r in net]
    print("\n== PHASE LOCK (lead = ghost_sim_time - newest_stamp, s)")
    dist("lead", lead)
    print("  (narrow + centered ~0 = the ghost clock is locked to the host)")

    # ---- 4. replay ticks ----
    nre = [int(r["replay_ticks"]) for r in net]
    spans = [float(r["replay_span"]) for r in net]
    from collections import Counter
    print("\n== REPLAY (n = int(L/TICK) whole ticks replayed)")
    print("  n histogram: %s" % dict(sorted(Counter(nre).items())))
    dist("replay_span", spans)

    # ---- 5. host ----
    fps = [float(r["fps"]) for r in hdbg if float(r["fps"]) > 0]
    hic = sum(1 for r in hdbg if int(r["hic"]) > 0)
    print("\n== HOST (host_debug.csv, %d frames)" % len(hdbg))
    if fps:
        dist("fps", fps)
    print("  hiccup frames: %d (%.1f%%)" % (hic, 100.0 * hic / len(hdbg)))
    # sim-clock rate from host_sim_debug (the per-frame diagnostic)
    ts = [float(r["t"]) for r in hsdbg]
    ss = [float(r["sim_time"]) for r in hsdbg]
    # trim the frozen post-death tail (sim_time stops advancing)
    adv = [(ts[i], ss[i]) for i in range(1, len(ts)) if ss[i] > ss[i - 1]]
    if adv:
        (ta, sa), (tb, sb) = adv[0], adv[-1]
        rate = (sb - sa) / (tb - ta)
        print("  sim-clock rate (advancing rows): %.4fx  (%.1f s sim / %.1f s wall)"
              % (rate, sb - sa, tb - ta))
    raw = [float(r["raw_dt"]) for r in hsdbg]
    hics = [d for d in raw if d > 0.05]
    print("  raw_dt hiccup frames (>50 ms): %d, max %.0f ms"
          % (len(hics), max(hics) * 1000 if hics else 0))

    # ---- 6. client ----
    cfps = [float(r["fps"]) for r in net if float(r["fps"]) > 0]
    chic = sum(1 for r in net if int(r["hic"]) > 0)
    delay = [float(r["delay"]) for r in net]
    jit = [float(r["jitter_ema"]) for r in net]
    buf = [int(r["buf_depth"]) for r in net]
    print("\n== CLIENT (net_debug.csv)")
    if cfps:
        dist("fps", cfps)
    print("  hiccup reconciles: %d (%.1f%%)" % (chic, 100.0 * chic / len(net)))
    dist("delay", delay)
    dist("jitter_ema", jit)
    print("  buf_depth: min=%d max=%d (cap 16)" % (min(buf), max(buf)))


if __name__ == "__main__":
    main()