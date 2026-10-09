"""Analyze 10.10/10.11 netcode CSVs: client net_debug.csv + host sim/debug CSVs.

Goal: understand the snap_px sawtooth (0 -> ~8 -> ~16 -> 0, fast cycle) during coasting.
"""
import csv, statistics as st

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows

net = load("net_debug.csv")
hsim = load("host_sim_debug.csv")
hdbg = load("host_debug.csv")

def f(row, k):
    v = row.get(k, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None

print(f"client rows: {len(net)}, span {net[0]['t']}..{net[-1]['t']}")
print(f"host sim rows: {len(hsim)}, span {hsim[0]['t']}..{hsim[-1]['t']}")
print(f"host dbg rows: {len(hdbg)}, span {hdbg[0]['t']}..{hdbg[-1]['t']}")

# ---- client: time base ----
t0 = float(net[0]["t"])
rows = []
for r in net:
    rows.append({
        "t": float(r["t"]) - t0,
        "snap_px": f(r, "snap_px"),
        "input": r.get("input", ""),
        "delay": f(r, "delay"),
        "jitter_ema": f(r, "jitter_ema"),
        "buf_depth": f(r, "buf_depth"),
        "newest": f(r, "newest_stamp"),
        "render_t": f(r, "render_t"),
        "host_est": f(r, "host_time_est"),
        "fps": f(r, "fps"),
        "dt_max": f(r, "dt_max"),
        "hic": f(r, "hic"),
        "n": f(r, "n"),
        "replay_ticks": f(r, "replay_ticks"),
        "replay_span": f(r, "replay_span"),
    })

# session stats
print("\n== client session ==")
print(f"duration: {rows[-1]['t']:.1f}s")
fps = [r["fps"] for r in rows if r["fps"] and r["fps"] > 0]
print(f"fps: mean {st.mean(fps):.1f} median {st.median(fps):.1f} min {min(fps):.1f} max {max(fps):.1f}")
dtm = [r["dt_max"] for r in rows if r["dt_max"] is not None]
print(f"dt_max: mean {st.mean(dtm)*1000:.1f}ms p95 {sorted(dtm)[int(len(dtm)*0.95)]*1000:.1f}ms max {max(dtm)*1000:.1f}ms")
hics = [r["hic"] for r in rows if r["hic"] is not None]
print(f"hic frames: {sum(1 for h in hics if h and h>0)} of {len(hics)}")

# snap_px distribution
snaps = [r["snap_px"] for r in rows if r["snap_px"] is not None]
snaps_sorted = sorted(snaps)
def pct(xs, p): return xs[int(len(xs)*p)]
print(f"\nsnap_px: n={len(snaps)} mean {st.mean(snaps):.2f} p50 {pct(snaps_sorted,0.5):.2f} "
      f"p90 {pct(snaps_sorted,0.9):.2f} p99 {pct(snaps_sorted,0.99):.2f} max {max(snaps):.2f}")
nz = [s for s in snaps if s > 0.05]
print(f"snap_px > 0.05px: {len(nz)} ({100*len(nz)/len(snaps):.0f}%)")
big = [s for s in snaps if s > 4]
print(f"snap_px > 4px: {len(big)} ({100*len(big)/len(snaps):.0f}%)")

# reconcile frequency: n column = reconciles this frame?
n_recon = [r["n"] for r in rows if r["n"] is not None]
print(f"\nreconciles: total {sum(n_recon)}, frames with >=1: {sum(1 for x in n_recon if x and x>0)}")

# replay stats
rt = [r["replay_ticks"] for r in rows if r["replay_ticks"] is not None]
rs = [r["replay_span"] for r in rows if r["replay_span"] is not None]
print(f"replay_ticks: mean {st.mean(rt):.2f} min {min(rt)} max {max(rt)}")
print(f"replay_span:  mean {st.mean(rs)*1000:.1f}ms min {min(rs)*1000:.1f}ms max {max(rs)*1000:.1f}ms")
neg = [s for s in rs if s < -0.001]
print(f"replay_span < 0: {len(neg)} ({100*len(neg)/len(rs):.0f}%)  min {min(rs)*1000:.1f}ms")
zero = [s for s in rs if -0.001 <= s <= 0.001]
print(f"replay_span ~0: {len(zero)} ({100*len(zero)/len(rs):.0f}%)")

# ---- the sawtooth: snap_px over time, find cycles ----
print("\n== snap_px timeline (every 5th row, first 60 rows) ==")
for i, r in enumerate(rows[:300:5]):
    bar = "#" * int(r["snap_px"] // 2) if r["snap_px"] is not None else ""
    print(f"t={r['t']:7.2f} snap={r['snap_px']:6.2f} n={r['n']} rt={r['replay_ticks']} rs={r['replay_span']*1000:6.1f}ms {bar}")

# ---- clock analysis: host_time_est vs newest_stamp ----
print("\n== clock domains ==")
est_err = []
for r in rows:
    if r["host_est"] is not None and r["newest"] is not None:
        est_err.append(r["host_est"] - r["newest"])
est_err = [e for e in est_err if abs(e) < 5]
if est_err:
    print(f"host_time_est - newest_stamp: mean {st.mean(est_err)*1000:.1f}ms "
          f"std {st.pstdev(est_err)*1000:.1f}ms min {min(est_err)*1000:.1f} max {max(est_err)*1000:.1f}")
    # drift over time: compare first 10% vs last 10%
    n10 = max(1, len(est_err)//10)
    print(f"  first 10% mean: {st.mean(est_err[:n10])*1000:.1f}ms, last 10% mean: {st.mean(est_err[-n10:])*1000:.1f}ms")

# render_t vs newest - delay (render point lag)
lag = []
for r in rows:
    if r["render_t"] is not None and r["newest"] is not None and r["delay"] is not None:
        lag.append(r["newest"] - r["delay"] - r["render_t"])
lag = [l for l in lag if abs(l) < 5]
if lag:
    print(f"render lag (newest-delay-render_t): mean {st.mean(lag)*1000:.1f}ms "
          f"min {min(lag)*1000:.1f} max {max(lag)*1000:.1f}")

# delay stats
dels = [r["delay"] for r in rows if r["delay"] is not None]
print(f"interp delay: mean {st.mean(dels)*1000:.1f}ms min {min(dels)*1000:.1f} max {max(dels)*1000:.1f}")
pinned = sum(1 for d in dels if d > 0.34)
print(f"  pinned at max (>=0.34): {pinned}/{len(dels)}")

# buf depth
bd = [r["buf_depth"] for r in rows if r["buf_depth"] is not None]
print(f"buf_depth: mean {st.mean(bd):.1f} min {min(bd)} max {max(bd)}")

# ---- host sim clock rate ----
print("\n== host sim clock ==")
ht = [float(r["t"]) for r in hsim]
hs = [f(r, "sim_time") for r in hsim]
# sim_time advance per wall time
dts = [(ht[i+1]-ht[i], hs[i+1]-hs[i]) for i in range(len(ht)-1) if ht[i+1]-ht[i] > 0.001]
rates = [s/t for t, s in dts if t < 0.5]
print(f"host sim rate: mean {st.mean(rates):.4f} ticks/s per wall s (expect ~60)")
print(f"  min {min(rates):.2f} max {max(rates):.2f}")
# host fps
hf = [f(r, "fps") for r in hdbg if f(r, "fps") and f(r, "fps") > 0]
if hf:
    print(f"host fps: mean {st.mean(hf):.1f} median {st.median(hf):.1f} min {min(hf):.1f}")

# ---- correlate snap events with inputs ----
print("\n== inputs during 'coast' ==")
inputs = [r["input"] for r in rows if r["input"]]
print(f"frames with input: {len(inputs)} / {len(rows)}")
if inputs:
    print("sample inputs:", inputs[:10])