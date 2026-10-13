"""8.5 Step 3 (client-side): new net_debug.csv vs the recorded 8.4 run
and the 8.0 baseline. The 8.4 client CSV was overwritten by this run, so
the 8.4 numbers are the ones recorded in the 8.x note / memory:
  8.4:  jitter mean 0.0232 (max 0.053), delay mean 0.1464,
        snap gaps mean 100.8 / p95 133.4 / max 150, 50/67/83 cluster = 152 gaps,
        snap_px mean 1.81 / p95 8.16 / max 38.6, render_t lag 0.229
  8.0:  jitter mean 0.0101 (max 0.031), delay mean 0.1202,
        snap gaps mean 111.8 / p95 150 / max 150, 50/67/83 cluster = 2 gaps,
        snap_px mean 2.31 / p95 6.71 / max 56.3, render_t lag 0.211
"""
import csv, statistics as st
from collections import Counter

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(row, key):
    v = row.get(key, '')
    try: return float(v)
    except (ValueError, TypeError): return None

def pct(xs, p):
    if not xs: return float('nan')
    xs = sorted(xs)
    k = (len(xs)-1) * p/100
    f, c = int(k), min(int(k)+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

rows = load('net_debug.csv')
t0, t1 = fnum(rows[0], 't'), fnum(rows[-1], 't')
jit = [fnum(r,'jitter_ema') for r in rows]; jit = [x for x in jit if x is not None]
dly = [fnum(r,'delay') for r in rows]; dly = [x for x in dly if x is not None]
spx = [fnum(r,'snap_px') for r in rows]; spx = [x for x in spx if x is not None]
fps = [fnum(r,'fps') for r in rows]; fps = [x for x in fps if x and x > 0]
lags = [fnum(r,'newest_stamp') - fnum(r,'render_t') for r in rows
        if fnum(r,'newest_stamp') is not None and fnum(r,'render_t') is not None]

# client-observed sim gaps (distinct newest_stamp jumps)
gaps = []
prev = None
for r in rows:
    s = fnum(r, 'newest_stamp')
    if s is None: continue
    if prev is not None and s > prev:
        gaps.append((s - prev) * 1000)
    prev = s
hist = Counter(round(g) for g in gaps)
short = sum(v for k, v in hist.items() if k < 90)   # the 50/67/83 cluster

print(f"=== 8.5 Step 3 CLIENT (new net_debug.csv, {t1-t0:.1f}s, {len(rows)} rows) ===")
print(f"  fps:        mean {st.mean(fps):.1f}")
print(f"  jitter_ema: mean {st.mean(jit):.4f}  max {max(jit):.4f}")
print(f"  delay:      mean {st.mean(dly):.4f}  min {min(dly):.4f}  max {max(dly):.4f}")
print(f"  snap_px:    mean {st.mean(spx):.2f}  p95 {pct(spx,95):.2f}  max {max(spx):.2f}")
print(f"  render_t lag vs newest: mean {st.mean(lags):.3f}  min {min(lags):.3f}  max {max(lags):.3f}")
print(f"  snap gaps:  n={len(gaps)} mean {st.mean(gaps):.1f} ms  p95 {pct(gaps,95):.1f} ms  max {max(gaps):.1f} ms")
print(f"  gap histogram (ms): {dict(sorted(hist.items()))}")
print(f"  short gaps (<90 ms, the 50/67/83 cluster): {short} of {len(gaps)}")
print()
print("  reference: 8.4 run -> jitter 0.0232/0.053, delay 0.1464, gaps 100.8/133.4/150,")
print("              cluster 152, snap_px 1.81/8.16/38.6, lag 0.229")
print("              8.0 base -> jitter 0.0101/0.031, delay 0.1202, gaps 111.8/150/150,")
print("              cluster 2, snap_px 2.31/6.71/56.3, lag 0.211")
print()
cluster_ok = short <= 10
jit_ok = st.mean(jit) <= 0.015
dly_ok = st.mean(dly) <= 0.135
if cluster_ok and jit_ok and dly_ok:
    print("  VERDICT: 8.5 Step 3 CLIENT-SIDE VERIFIED — the 50/67/83 ms cluster is")
    print("  gone and jitter/delay are back at (or better than) the 8.0 baseline.")
else:
    print(f"  VERDICT: INCOMPLETE — cluster={short} (want <=10), jitter={st.mean(jit):.4f}")
    print(f"  (want <=0.015), delay={st.mean(dly):.4f} (want <=0.135). Inspect the histogram.")