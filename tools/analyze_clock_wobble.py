"""Measure the ghost CLOCK's per-frame wobble (the render-pose wobble = v x clock wobble).

For consecutive ADV rows, the per-frame render-pose change is exactly
v * (clock delta), so the clock's per-frame delta IS the wobble. A smooth
clock advances at ~1x real time (clock_delta ~= real_delta). Freezes
(clock_delta << real_delta) and jumps (clock_delta >> real_delta) are the
wobble. This gives a before/after metric for the host-time-estimate fix.

Run:  cd /mnt && python -m ship5.tools.analyze_clock_wobble
"""
import csv, os
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GC = os.path.join(HERE, "ghost_clock.csv")
TICK = 1.0 / 60.0


def f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    adv = []
    with open(GC) as fh:
        for row in csv.DictReader(fh):
            if row["kind"] == "ADV":
                adv.append(row)
    # keep only live (non-dead) ADV rows with a clock
    adv = [r for r in adv if f(r["clock"]) is not None and r["dead"] == "0"]
    print("ADV live rows: %d" % len(adv))

    # per-frame clock delta + real delta (consecutive ADV rows = consecutive frames)
    rates = []       # clock_delta / real_delta (the clock's instantaneous rate)
    cdeltas = []     # clock_delta (s)
    rdeltas = []     # real_delta (s)
    freezes = 0      # clock_delta < 0.3 * real_delta (the clock barely moved)
    jumps = 0        # clock_delta > 1.7 * real_delta (the clock jumped)
    neg = 0          # clock went BACKWARD
    for a, b in zip(adv, adv[1:]):
        ca, cb = f(a["clock"]), f(b["clock"])
        ra, rb = f(a["real_time"]), f(b["real_time"])
        cd = cb - ca
        rd = rb - ra
        if rd <= 0:
            continue
        cdeltas.append(cd)
        rdeltas.append(rd)
        rates.append(cd / rd)
        if cd < 0.3 * rd:
            freezes += 1
        if cd > 1.7 * rd:
            jumps += 1
        if cd < -1e-6:
            neg += 1
    n = len(rates)
    print("\nper-frame clock rate (clock_delta / real_delta), n=%d:" % n)
    print("  min %.3f  p5 %.3f  p25 %.3f  p50 %.3f  p75 %.3f  p95 %.3f  max %.3f"
          % (min(rates), pct(rates, 5), pct(rates, 25), pct(rates, 50),
             pct(rates, 75), pct(rates, 95), max(rates)))
    print("  freezes (rate<0.3): %d (%.1f%%)" % (freezes, 100.0 * freezes / n))
    print("  jumps   (rate>1.7): %d (%.1f%%)" % (jumps, 100.0 * jumps / n))
    print("  backward (cd<0):    %d (%.1f%%)" % (neg, 100.0 * neg / n))

    # The render-pose wobble in px: v * clock_delta. Use MAX_SPEED 520 px/s
    # (the coasting speed) as the scale.
    V = 520.0
    wob = [abs(cd) * V for cd in cdeltas]
    print("\nrender-pose wobble per frame (|clock_delta| * 520 px/s):")
    print("  p50 %.2f px  p95 %.2f px  max %.2f px"
          % (pct(wob, 50), pct(wob, 95), max(wob)))
    # A smooth 62.5 FPS frame moves 520 * 0.016 = 8.3 px. Anything well above
    # that in a single frame is a visible jump.
    big = sum(1 for w in wob if w > 1.5 * V * 0.016)
    print("  frames moving >1.5x a smooth frame: %d (%.1f%%)"
          % (big, 100.0 * big / len(wob)))

    # Histogram of the clock rate to see the distribution shape
    hist = Counter(int(r * 10) for r in rates)  # 0.1 bins
    print("\nclock-rate histogram (0.1 bins):")
    for b in sorted(hist):
        lo = b / 10.0
        bar = "#" * min(60, hist[b] // max(1, n // 200))
        print("  %4.1f-%4.1f : %5d %s" % (lo, lo + 0.1, hist[b], bar))


if __name__ == "__main__":
    main()