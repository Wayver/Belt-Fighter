"""10.11c: pin down the root cause of the REMAINING snap after the 10.11b
clock-anchor fix.

With the 10.11c instrumentation (ghost_sim_time column), the key test is
the PHASE-DRIFT check: lead = ghost_sim_time - newest_stamp (the ghost's
clock lead over the newest snapshot at reconcile time). If the free-run
clock is not phase-locked to the host's tick grid, lead oscillates between
~0 and ~1 tick and the snap is bimodal (0 when lead~0, ~1 tick when
lead~1). A locked clock keeps lead in a narrow band.

Hypotheses to test:
  H1 (sub-tick remainder): the replay only goes to whole ticks (n = int(L/TICK));
      the sub-tick remainder (L - n*TICK) sits in the accumulator and is applied
      on the NEXT frame, not at the reconcile. So the reconciled pose is behind
      the present by the remainder -> snap ~ v * remainder.
  H2 (in-flight input): on an input CHANGE the ghost uses the new input
      immediately but the host is L behind -> snap ~ v * L on the change frame.
  H3 (clock drift): the ghost's sim time drifts from the host's between
      reconciles -> snap grows over the snapshot interval.

We can't see v or the direction from the CSV, but we CAN correlate snap_px with:
  - the sub-tick remainder (replay_span - replay_ticks*TICK)
  - whether the input CHANGED since the previous reconcile
  - the snapshot index within the interval (drift)
"""
import csv, math, statistics as st
from collections import Counter

TICK = 1/60

def corr(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs))
    sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov / (sx*sy) if sx and sy else 0.0

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def f(row, k):
    v = row.get(k, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None

net = load("net_debug.csv")
t0 = float(net[0]["t"])
has_ghost = "ghost_sim_time" in net[0]
rows = []
for r in net:
    span = f(r, "replay_span")
    rt = f(r, "replay_ticks")
    rem = (span - rt * TICK) if (span is not None and rt is not None) else None
    rows.append({
        "t": float(r["t"]) - t0,
        "snap": f(r, "snap_px"),
        "input": r.get("input", ""),
        "span": span,
        "rt": rt,
        "rem": rem,
        "ow": f(r, "rtt_ow"),
        "newest": f(r, "newest_stamp"),
        "gts": f(r, "ghost_sim_time"),
    })

# ---- PHASE-DRIFT (10.11c): the ghost's clock lead over the newest
# snapshot at reconcile time. lead = ghost_sim_time - newest_stamp.
# Bimodal (oscillating ~0 / ~1 tick) = the free-run clock is not
# phase-locked to the host's tick grid. A narrow band = locked.
if has_ghost:
    print("\n== PHASE-DRIFT: lead = ghost_sim_time - newest_stamp ==")
    leads = [(r["gts"] - r["newest"]) for r in rows
             if r["gts"] is not None and r["newest"] is not None]
    if leads:
        print(f"  n={len(leads)} mean {st.mean(leads)*1000:.1f}ms "
              f"min {min(leads)*1000:.1f} max {max(leads)*1000:.1f} "
              f"std {st.pstdev(leads)*1000:.1f}ms")
        # bucket lead into tick fractions
        lb = {}
        for x in leads:
            lb.setdefault(round(x / TICK, 1), []).append(x)
        print("  lead histogram (ticks):",
              {k: len(v) for k, v in sorted(lb.items())})
        # correlate lead with snap
        ls = [(r["gts"] - r["newest"], r["snap"]) for r in rows
              if r["gts"] is not None and r["newest"] is not None
              and r["snap"] is not None and r["snap"] < 100]
        if len(ls) > 10:
            print(f"  corr(lead, snap) = {corr([x for x, _ in ls], [y for _, y in ls]):.3f}")
    else:
        print("  (no ghost_sim_time data in this CSV)")
else:
    print("\n== PHASE-DRIFT: (ghost_sim_time column absent — pre-10.11c CSV) ==")

# drop the transient (first 2 s) + the giant respawn/teleport snaps
rows = [r for r in rows if r["t"] > 2.0 and r["snap"] is not None and r["snap"] < 100]
print(f"rows (steady, snap<100): {len(rows)}")

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs)-1, int(len(xs)*p))] if xs else 0.0

# ---- H1: snap vs sub-tick remainder ----
print("\n== H1: snap vs sub-tick remainder (rem = span - rt*TICK) ==")
# bucket by remainder in tick fractions
buckets = {}
for r in rows:
    if r["rem"] is None:
        continue
    b = round(r["rem"] / TICK, 2)   # remainder in ticks (0..1)
    buckets.setdefault(b, []).append(r["snap"])
print("  rem(ticks)  n     mean_snap  p90_snap  max_snap")
for b in sorted(buckets):
    s = buckets[b]
    print(f"  {b:6.2f}   {len(s):4d}  {st.mean(s):8.2f}  {pct(s,0.9):8.2f}  {max(s):8.2f}")

# correlation
remv = [r["rem"] for r in rows if r["rem"] is not None]
snapv = [r["snap"] for r in rows if r["rem"] is not None]
print(f"  corr(rem, snap) = {corr(remv, snapv):.3f}")

# ---- H2: snap vs input CHANGE ----
print("\n== H2: snap vs input change ==")
chg, nochg = [], []
for i, r in enumerate(rows):
    prev = rows[i-1]["input"] if i > 0 else ""
    (chg if r["input"] != prev else nochg).append(r["snap"])
print(f"  input CHANGED:   n={len(chg):4d} mean {st.mean(chg):6.2f} p90 {pct(chg,0.9):6.2f} max {max(chg):7.2f}")
print(f"  input UNCHANGED: n={len(nochg):4d} mean {st.mean(nochg):6.2f} p90 {pct(nochg,0.9):6.2f} max {max(nochg):7.2f}")
# and within unchanged: thrusting vs coasting
thrust = [r["snap"] for i, r in enumerate(rows)
          if i > 0 and r["input"] == rows[i-1]["input"] and r["input"]]
coast = [r["snap"] for i, r in enumerate(rows)
         if i > 0 and r["input"] == rows[i-1]["input"] and not r["input"]]
print(f"  unchanged+thrust: n={len(thrust):4d} mean {st.mean(thrust):6.2f} p90 {pct(thrust,0.9):6.2f} max {max(thrust):7.2f}")
print(f"  unchanged+coast:  n={len(coast):4d} mean {st.mean(coast):6.2f} p90 {pct(coast,0.9):6.2f} max {max(coast):7.2f}")

# ---- H3: drift over the snapshot interval ----
print("\n== H3: drift (snap vs time-within-interval) ==")
# the snapshot interval is ~33ms; the reconcile happens at a random phase
# within it. If the ghost drifts, snap should correlate with the phase.
# We don't have the phase directly, but the 'n' (frames since last line)
# is a proxy for the frame rate, not the phase. Skip H3 (no phase in CSV).
print("  (no phase column in the CSV — H3 not testable offline;")
print("   would need the snapshot arrival time vs the reconcile time)")

# ---- summary: what fraction of the snap is the remainder? ----
print("\n== summary ==")
# model: snap ~ v * rem. Estimate v from the slope of snap vs rem.
# use only thrusting rows (coasting has v~0 so rem doesn't matter)
tr = [(r["rem"], r["snap"]) for r in rows if r["rem"] is not None and r["input"]]
if len(tr) > 10:
    v_est = corr([x for x, _ in tr], [y for _, y in tr])
    print(f"  thrusting rows: corr(rem, snap) = {v_est:.3f} "
          f"(n={len(tr)})")
    # the mean remainder
    print(f"  mean remainder = {st.mean([x for x, _ in tr])*1000:.1f}ms "
          f"({st.mean([x for x, _ in tr])/TICK:.2f} ticks)")
    print(f"  mean snap (thrusting) = {st.mean([y for _, y in tr]):.2f}px")
    # implied v = snap / rem
    pairs = [(y/x) for x, y in tr if x > 0.002]
    if pairs:
        print(f"  implied v = snap/rem: mean {st.mean(pairs):.0f} px/s "
              f"median {st.median(pairs):.0f} px/s")