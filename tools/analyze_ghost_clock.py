"""11.x: analyze ghost_clock.csv + net_debug.csv + host_debug.csv.

Resolves the "why does the ghost render in 60 Hz steps" question by
measuring, from the per-frame ghost-clock trace:
  1. Does the ghost step in advance() at all? (steps distribution)
  2. Is the step grid re-anchored every frame by the reconcile?
     (correlate ADV steps with whether a REC landed in the same frame)
  3. Where is the clock relative to the tick boundary? (_acc distribution)
  4. Is the MAX_LEAD cap binding? (ht_raw > ht_cap)
  5. How wrong is the host-time estimate? (rate noise, cap, and the
     estimate's implied host sim time vs the snapshot stamps)

Run:  cd /mnt && python -m ship5.tools.analyze_ghost_clock
"""
import csv, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GC = os.path.join(HERE, "ghost_clock.csv")
ND = os.path.join(HERE, "net_debug.csv")
HD = os.path.join(HERE, "host_debug.csv")
TICK = 1.0 / 60.0


def f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load_gc():
    recs = []
    with open(GC) as fh:
        for row in csv.DictReader(fh):
            recs.append(row)
    return recs


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    recs = load_gc()
    adv = [r for r in recs if r["kind"] == "ADV"]
    rec = [r for r in recs if r["kind"] == "REC"]
    print("=== ghost_clock.csv ===")
    print("total rows %d  ADV %d  REC %d" % (len(recs), len(adv), len(rec)))

    # --- 1. steps distribution -------------------------------------------------
    steps = Counter(int(f(r["steps"]) or 0) for r in adv)
    print("\n[1] ADV steps distribution (steps per advance() call):")
    for k in sorted(steps):
        print("    steps=%d : %d  (%.1f%%)" % (k, steps[k], 100.0 * steps[k] / len(adv)))

    # --- 2. correlate steps with same-frame REC --------------------------------
    # Group records by real_time (the frame). A frame "has a snapshot" if a REC
    # shares its real_time.
    frames = {}
    for r in recs:
        rt = r["real_time"]
        fr = frames.setdefault(rt, {"rec": 0, "adv_steps": 0, "adv": 0})
        if r["kind"] == "REC":
            fr["rec"] += 1
        else:
            fr["adv"] += 1
            fr["adv_steps"] += int(f(r["steps"]) or 0)
    with_snap = [fr for fr in frames.values() if fr["rec"] > 0]
    no_snap = [fr for fr in frames.values() if fr["rec"] == 0]
    print("\n[2] frames: %d total, %d with a snapshot (REC), %d without"
          % (len(frames), len(with_snap), len(no_snap)))
    if with_snap:
        s_ws = sum(fr["adv_steps"] for fr in with_snap)
        print("    with-snapshot frames: total ADV steps = %d  (mean %.3f/frame)"
              % (s_ws, s_ws / len(with_snap)))
    if no_snap:
        s_ns = sum(fr["adv_steps"] for fr in no_snap)
        print("    no-snapshot frames:   total ADV steps = %d  (mean %.3f/frame)"
              % (s_ns, s_ns / len(no_snap)))
    # how many with-snapshot frames had zero steps
    ws_zero = sum(1 for fr in with_snap if fr["adv_steps"] == 0)
    print("    with-snapshot frames with 0 steps: %d / %d (%.1f%%)"
          % (ws_zero, len(with_snap), 100.0 * ws_zero / max(1, len(with_snap))))

    # --- 3. _acc distribution --------------------------------------------------
    accs = [f(r["post_acc"]) for r in adv if f(r["post_acc"]) is not None]
    print("\n[3] ADV post_acc (render phase, s) distribution:")
    print("    min %.4f  p50 %.4f  p95 %.4f  max %.4f  (TICK=%.4f)"
          % (min(accs), pct(accs, 50), pct(accs, 95), max(accs), TICK))
    # histogram in 2ms bins
    hist = Counter(int(a / 0.002) for a in accs)
    print("    histogram (2 ms bins):")
    for b in sorted(hist):
        lo = b * 0.002
        print("      %5.3f-%5.3f s : %d" % (lo, lo + 0.002, hist[b]))

    # --- 4. cap binding --------------------------------------------------------
    capbind = 0
    ncap = 0
    for r in adv:
        hr, hc = f(r["ht_raw"]), f(r["ht_cap"])
        if hr is not None and hc is not None:
            ncap += 1
            if hr > hc + 1e-9:
                capbind += 1
    print("\n[4] MAX_LEAD cap binding (ht_raw > ht_cap): %d / %d ADV (%.1f%%)"
          % (capbind, ncap, 100.0 * capbind / max(1, ncap)))

    # --- 5. host-time estimate -------------------------------------------------
    print("\n[5] host-time estimate (from net_debug.csv):")
    if os.path.exists(ND):
        rates, rtt, delay = [], [], []
        with open(ND) as fh:
            for row in csv.DictReader(fh):
                r = f(row.get("host_rate"))
                if r is not None:
                    rates.append(r)
                o = f(row.get("rtt_ow"))
                if o is not None:
                    rtt.append(o)
                d = f(row.get("delay"))
                if d is not None:
                    delay.append(d)
        if rates:
            print("    host_rate: min %.3f  p50 %.3f  p95 %.3f  max %.3f  (n=%d)"
                  % (min(rates), pct(rates, 50), pct(rates, 95), max(rates), len(rates)))
        if rtt:
            print("    rtt_ow:    min %.4f  p50 %.4f  p95 %.4f  max %.4f"
                  % (min(rtt), pct(rtt, 50), pct(rtt, 95), max(rtt)))
        if delay:
            print("    delay:     min %.4f  p50 %.4f  p95 %.4f  max %.4f"
                  % (min(delay), pct(delay, 50), pct(delay, 95), max(delay)))
    # The estimate's implied lead over the newest stamp, per ADV:
    # clock - (ht_cap) should be ~ rtt_ow. And ht_cap vs the newest stamp:
    # in a with-snapshot frame, ht_cap == newest stamp (anchor just moved).
    leads = []
    for r in adv:
        c, hc = f(r["clock"]), f(r["ht_cap"])
        if c is not None and hc is not None:
            leads.append(c - hc)
    if leads:
        print("    clock - ht_cap (should be ~ rtt_ow): min %.4f p50 %.4f max %.4f"
              % (min(leads), pct(leads, 50), max(leads)))

    # --- 6. reconcile replay ---------------------------------------------------
    ns = Counter(int(f(r["n"]) or 0) for r in rec)
    print("\n[6] REC replay tick count n distribution:")
    for k in sorted(ns):
        print("    n=%d : %d  (%.1f%%)" % (k, ns[k], 100.0 * ns[k] / len(rec)))
    spans = [f(r["now"]) - f(r["snap_time"]) for r in rec
             if f(r["now"]) is not None and f(r["snap_time"]) is not None]
    if spans:
        print("    replay span (now - snap_time): min %.4f p50 %.4f max %.4f"
              % (min(spans), pct(spans, 50), max(spans)))

    # --- 7. host sim rate (host_debug.csv) ------------------------------------
    if os.path.exists(HD):
        rows = []
        with open(HD) as fh:
            for row in csv.DictReader(fh):
                t, st = f(row.get("t")), f(row.get("sim_time"))
                if t is not None and st is not None:
                    rows.append((t, st))
        if len(rows) > 10:
            rows.sort()
            dt = rows[-1][0] - rows[0][0]
            ds = rows[-1][1] - rows[0][1]
            print("\n[7] host sim clock rate (host_debug.csv): %.4f x real time"
                  % (ds / dt if dt > 0 else float("nan")))
            # per-second rate
            print("    (over %.1f s wall, sim advanced %.1f s)" % (dt, ds))


if __name__ == "__main__":
    main()