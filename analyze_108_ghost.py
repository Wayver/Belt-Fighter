"""10.8 follow-up: is the client's LOCAL ship (ghost) jagged because it is
drawn at its raw 60 Hz step position with NO sub-step interpolation?

This analyzes the fresh F3 CSVs (MAC host / 4090 client) from the 30 Hz
live 2P run. It does NOT change any game code — it only reads the CSVs and
reports the numbers that discriminate between the two candidate causes of
the 'jagged local ship':

  (A) GHOST STEP QUANTIZATION — the ghost is drawn at its post-step pos
      (60 Hz updates) on a ~62 Hz display, with no lerp between steps.
      At high speed each step is MAX_SPEED*TICK ~ 8.7 px, so the ship
      moves in visible 60 Hz jumps. This is independent of the snapshot
      rate (the ghost steps at the SIM rate, not the wire rate), which is
      why the 30 Hz bump did NOT smooth the local ship.

  (B) RECONCILE SNAPBACK — the ghost is yanked to the authoritative pose
      on each snapshot (snap_px). At high speed the in-flight-input
      divergence (MAX_SPEED * one-way latency) is larger, so the snap is
      bigger. This is the 'snapback worse at high speed' report.

The data can't directly measure (A) (the CSV logs per-reconcile, not
per-frame ghost pos), but it CAN confirm the conditions that make (A)
visible: client display rate vs sim rate, and the per-step displacement
magnitude at the ship's speed. It measures (B) directly (snap_px).
"""
import csv, os, statistics as st

def load(name):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    with open(p) as f:
        return list(csv.DictReader(f))

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def pct(vals, q):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

net = load("net_debug.csv")
host = load("host_debug.csv")
sim = load("host_sim_debug.csv")

print("=" * 70)
print("CLIENT (net_debug.csv)  rows=%d" % len(net))
print("=" * 70)
fps = [f(r["fps"]) for r in net if f(r["fps"])]
dtmax = [f(r["dt_max"]) for r in net if f(r["dt_max"])]
hic = [f(r["hic"]) for r in net if f(r["hic"])]
snap = [f(r["snap_px"]) for r in net if f(r["snap_px"])]
delay = [f(r["delay"]) for r in net if f(r["delay"])]
jit = [f(r["jitter_ema"]) for r in net if f(r["jitter_ema"])]
buf = [f(r["buf_depth"]) for r in net if f(r["buf_depth"])]
inp = [r["input"] for r in net]

print("client fps        mean %.1f  min %.1f  (display rate)" %
      (st.mean(fps), min(fps)))
print("client dt_max     mean %.1f ms  max %.1f ms" %
      (st.mean(dtmax) * 1000, max(dtmax) * 1000))
print("client hic frames %d / %d" % (sum(1 for h in hic if h > 0), len(hic)))
print("snap_px           mean %.2f  p50 %.2f  p95 %.2f  max %.2f" %
      (st.mean(snap), pct(snap, .5), pct(snap, .95), max(snap)))
print("  snap_px > 20 px : %d  |  > 50 px : %d  |  > 100 px : %d" %
      (sum(1 for s in snap if s > 20), sum(1 for s in snap if s > 50),
       sum(1 for s in snap if s > 100)))
print("delay             mean %.3f  max %.3f  (adaptive interp delay)" %
      (st.mean(delay), max(delay)))
print("jitter_ema        mean %.4f  max %.4f" % (st.mean(jit), max(jit)))
print("buf_depth         mean %.1f  min %d  max %d  (want ~16, full)" %
      (st.mean(buf), min(buf), max(buf)))
print("input col values  %s" % sorted(set(inp)))

# Does snap_px correlate with input being active/changed?
if "1" in set(inp) or "0" in set(inp):
    s_on = [f(r["snap_px"]) for r in net if r["input"] == "1" and f(r["snap_px"])]
    s_off = [f(r["snap_px"]) for r in net if r["input"] == "0" and f(r["snap_px"])]
    if s_on and s_off:
        print("snap_px | input=1 mean %.2f (n=%d)  vs  input=0 mean %.2f (n=%d)" %
              (st.mean(s_on), len(s_on), st.mean(s_off), len(s_off)))

print()
print("=" * 70)
print("HOST (host_debug.csv)  rows=%d" % len(host))
print("=" * 70)
hfps = [f(r["fps"]) for r in host if f(r["fps"])]
hdt = [f(r["dt_max"]) for r in host if f(r["dt_max"])]
hhic = [f(r["hic"]) for r in host if f(r["hic"])]
print("host fps          mean %.1f  min %.1f" % (st.mean(hfps), min(hfps)))
print("host dt_max       mean %.1f ms  max %.1f ms" %
      (st.mean(hdt) * 1000, max(hdt) * 1000))
print("host hic frames   %d / %d" % (sum(1 for h in hhic if h > 0), len(hhic)))

print()
print("=" * 70)
print("HOST SIM CLOCK (host_sim_debug.csv)  rows=%d" % len(sim))
print("=" * 70)
# sim clock rate = (last sim_time - first sim_time) / (last t - first t)
ts = [f(r["t"]) for r in sim if f(r["t"])]
stimes = [f(r["sim_time"]) for r in sim if f(r["sim_time"])]
if len(ts) > 10 and len(stimes) > 10:
    rate = (stimes[-1] - stimes[0]) / (ts[-1] - ts[0])
    print("sim-clock rate    %.4fx  (1.0 = sim keeps real time)" % rate)
    # per-10s windows
    n = len(ts)
    w = max(1, n // 10)
    print("  per-window rate (steady?):")
    for i in range(0, n - w, w):
        a, b = i, min(i + w, n - 1)
        if ts[b] > ts[a]:
            print("    [%.1f-%.1f s]  %.4fx" %
                  (ts[a], ts[b], (stimes[b] - stimes[a]) / (ts[b] - ts[a])))
alphas = [f(r["alpha"]) for r in sim if f(r["alpha"])]
if alphas:
    print("render alpha      mean %.3f  min %.3f  max %.3f  (host local ship interp)" %
          (st.mean(alphas), min(alphas), max(alphas)))

print()
print("=" * 70)
print("DIAGNOSIS HELPERS")
print("=" * 70)
# Per-step displacement at the ship's top speed (the size of each 60 Hz jump
# the ghost makes if drawn without sub-step interpolation).
MAX_SPEED, TICK = 520.0, 1.0 / 60.0   # config.py values (avoid import path)
print("MAX_SPEED*TICK    = %.2f px  (one ghost step at top speed = one" %
      (MAX_SPEED * TICK))
print("                    visible jump if the ghost is NOT interpolated)")
print("client fps ~%.0f vs sim 60 Hz -> the ghost's 60 Hz position updates" %
      st.mean(fps))
print("                    land on a %.0f Hz display with a drifting phase" %
      st.mean(fps))
print("                    (no sub-step lerp) = the 'jagged' local ship.")