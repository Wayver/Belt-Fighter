"""Analyze the 10.11b live CSV: is the anchor working?"""
import csv, statistics as st

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

net = load("net_debug.csv")

def f(row, k):
    v = row.get(k, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None

t0 = float(net[0]["t"])
rows = []
for r in net:
    rows.append({
        "t": float(r["t"]) - t0,
        "snap_px": f(r, "snap_px"),
        "input": r.get("input", ""),
        "delay": f(r, "delay"),
        "newest": f(r, "newest_stamp"),
        "render_t": f(r, "render_t"),
        "host_est": f(r, "host_time_est"),
        "fps": f(r, "fps"),
        "n": f(r, "n"),
        "rt": f(r, "replay_ticks"),
        "rs": f(r, "replay_span"),
        "ow": f(r, "rtt_ow"),
    })

print(f"rows: {len(rows)}, duration {rows[-1]['t']:.1f}s")
fps = [r["fps"] for r in rows if r["fps"] and r["fps"] > 0]
print(f"fps: mean {st.mean(fps):.1f} median {st.median(fps):.1f}")

# rtt_ow trajectory
ows = [r["ow"] for r in rows if r["ow"] is not None]
print(f"\nrtt_ow: n={len(ows)} first {ows[0]*1000:.1f}ms last {ows[-1]*1000:.1f}ms "
      f"mean {st.mean(ows)*1000:.1f}ms min {min(ows)*1000:.1f} max {max(ows)*1000:.1f}")
# how long to settle
for frac in (0.1, 0.25, 0.5, 1.0):
    seg = ows[:max(1, int(len(ows)*frac))]
    print(f"  first {int(frac*100)}%: mean {st.mean(seg)*1000:.1f}ms std {st.pstdev(seg)*1000:.1f}ms")

# replay_span
rs = [r["rs"] for r in rows if r["rs"] is not None]
neg = [x for x in rs if x < -0.001]
pos = [x for x in rs if x > 0.001]
zero = [x for x in rs if -0.001 <= x <= 0.001]
print(f"\nreplay_span: mean {st.mean(rs)*1000:.1f}ms min {min(rs)*1000:.1f} max {max(rs)*1000:.1f}")
print(f"  negative: {len(neg)} ({100*len(neg)/len(rs):.0f}%)  zero: {len(zero)} ({100*len(zero)/len(rs):.0f}%)  positive: {len(pos)} ({100*len(pos)/len(rs):.0f}%)")
# span distribution in TICK units
from collections import Counter
c = Counter(round(x / (1/60), 1) for x in rs)
print("  span histogram (in ticks):", dict(sorted(c.items())))

# replay_ticks
rt = [r["rt"] for r in rows if r["rt"] is not None]
print(f"replay_ticks: mean {st.mean(rt):.2f} dist {dict(sorted(Counter(rt).items()))}")

# snap_px
snaps = [r["snap_px"] for r in rows if r["snap_px"] is not None]
ss = sorted(snaps)
def pct(xs, p): return xs[min(len(xs)-1, int(len(xs)*p))]
print(f"\nsnap_px: mean {st.mean(snaps):.2f} p50 {pct(ss,0.5):.2f} p90 {pct(ss,0.9):.2f} "
      f"p99 {pct(ss,0.99):.2f} max {max(snaps):.2f}")
nz = [s for s in snaps if s > 0.05]
big = [s for s in snaps if s > 4]
print(f"  >0.05px: {len(nz)} ({100*len(nz)/len(snaps):.0f}%)   >4px: {len(big)} ({100*len(big)/len(snaps):.0f}%)")

# correlation: snap_px vs span sign, and vs span magnitude
print("\n== snap vs span ==")
for label, sel in [
    ("span<0", lambda r: r["rs"] < -0.001),
    ("span~0", lambda r: -0.001 <= r["rs"] <= 0.001),
    ("span>0", lambda r: r["rs"] > 0.001),
]:
    ss_ = [r["snap_px"] for r in rows if sel(r) and r["snap_px"] is not None]
    if ss_:
        print(f"  {label:8s} n={len(ss_):4d} mean snap {st.mean(ss_):6.2f}  p90 {pct(sorted(ss_),0.9):6.2f}  max {max(ss_):7.2f}")

# input frames vs coast
inp_frames = [r for r in rows if r["input"]]
coast = [r for r in rows if not r["input"]]
for label, sel in [("coast", coast), ("input", inp_frames)]:
    ss_ = [r["snap_px"] for r in sel if r["snap_px"] is not None]
    if ss_:
        print(f"  {label:6s} n={len(ss_):4d} mean snap {st.mean(ss_):6.2f}  p90 {pct(sorted(ss_),0.9):6.2f}  max {max(ss_):7.2f}")

# big snaps: what do they look like in context?
print("\n== frames with snap_px > 4 (first 25) ==")
cnt = 0
for i, r in enumerate(rows):
    if r["snap_px"] is not None and r["snap_px"] > 4:
        prev = rows[i-1] if i > 0 else None
        print(f"t={r['t']:7.2f} snap={r['snap_px']:7.2f} span={r['rs']*1000:6.1f}ms rt={r['rt']} "
              f"ow={r['ow']*1000:5.1f}ms in={r['input'] or '-'} prev_in={prev['input'] or '-' if prev else '-'}")
        cnt += 1
        if cnt >= 25:
            break

# snap direction can't be seen from csv (only magnitude). But check
# snap vs (ow - span): if ghost is at snap_time+ow and replay goes to
# snap_time + n*TICK, the residual = ow - n*TICK... print that.
print("\n== residual = rtt_ow - replay_span (sub-tick remainder) ==")
res = [r["ow"] - r["rs"] for r in rows if r["ow"] is not None and r["rs"] is not None]
res = [x for x in res if abs(x) < 0.5]
if res:
    print(f"  mean {st.mean(res)*1000:.1f}ms min {min(res)*1000:.1f} max {max(res)*1000:.1f}")
    c2 = Counter(round(x / (1/60), 2) for x in res)
    print("  histogram (ticks):", dict(sorted(c2.items())))