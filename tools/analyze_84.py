"""8.4: two-machine CSV diff — 8.x worker run vs 8.0 baseline (7.10c).

Client CSV: t,snap_px,input,delay,jitter_ema,buf_depth,newest_stamp,render_t,host_time_est,fps,dt_max,hic,n
Host CSV:   t,sim_time,fps,dt_max,hic,n
"""
import csv, statistics as st

def pct(xs, p):
    if not xs: return float('nan')
    xs = sorted(xs)
    k = (len(xs)-1) * p/100
    f, c = int(k), min(int(k)+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows

def fnum(row, key):
    v = row.get(key, '')
    if v in ('', None): return None
    try: return float(v)
    except ValueError: return None

def client_stats(path, label):
    rows = load(path)
    fps = [fnum(r,'fps') for r in rows]; fps = [x for x in fps if x is not None and x > 0]
    dt = [fnum(r,'dt_max') for r in rows]; dt = [x for x in dt if x is not None]
    jit = [fnum(r,'jitter_ema') for r in rows]; jit = [x for x in jit if x is not None]
    dly = [fnum(r,'delay') for r in rows]; dly = [x for x in dly if x is not None]
    spx = [fnum(r,'snap_px') for r in rows]; spx = [x for x in spx if x is not None]
    buf = [fnum(r,'buf_depth') for r in rows]; buf = [x for x in buf if x is not None]
    hic = sum(1 for r in rows if fnum(r,'hic') == 1)
    t0, t1 = fnum(rows[0],'t'), fnum(rows[-1],'t')
    # client-observed snap gaps from newest_stamp sequence
    stamps = [fnum(r,'newest_stamp') for r in rows]
    gaps = []
    prev = None
    for s in stamps:
        if s is None: continue
        if prev is not None and s > prev:
            gaps.append(s - prev)
        prev = s
    # render_t lag behind newest
    lags = []
    for r in rows:
        ns, rt = fnum(r,'newest_stamp'), fnum(r,'render_t')
        if ns is not None and rt is not None:
            lags.append(ns - rt)
    print(f"--- CLIENT {label} ({t1-t0:.1f}s, {len(rows)} rows, {hic} hic frames) ---")
    print(f"  fps:      mean {st.mean(fps):.1f}  p95 {pct(fps,95):.1f}  max {max(fps):.1f}")
    print(f"  dt_max:   mean {st.mean(dt):.4f}  max {max(dt):.4f}")
    print(f"  jitter:   mean {st.mean(jit):.4f}  max {max(jit):.4f}")
    print(f"  delay:    mean {st.mean(dly):.4f}  min {min(dly):.4f}  max {max(dly):.4f}")
    print(f"  buf_depth:mean {st.mean(buf):.1f}  min {min(buf):.0f}  max {max(buf):.0f}")
    print(f"  snap_px:  mean {st.mean(spx):.2f}  p95 {pct(spx,95):.2f}  max {max(spx):.2f}")
    if gaps:
        print(f"  snap gaps(client-observed): n={len(gaps)} mean {st.mean(gaps)*1000:.1f} ms  "
              f"p95 {pct(gaps,95)*1000:.1f} ms  max {max(gaps)*1000:.1f} ms")
    if lags:
        print(f"  render_t lag vs newest: mean {st.mean(lags):.3f}  min {min(lags):.3f}  max {max(lags):.3f}")

def host_stats(path, label):
    rows = load(path)
    fps = [fnum(r,'fps') for r in rows]; fps = [x for x in fps if x is not None and x > 0]
    dt = [fnum(r,'dt_max') for r in rows]; dt = [x for x in dt if x is not None]
    hic = sum(1 for r in rows if fnum(r,'hic') == 1)
    t0, t1 = fnum(rows[0],'t'), fnum(rows[-1],'t')
    print(f"--- HOST {label} ({t1-t0:.1f}s, {len(rows)} rows, {hic} hic frames) ---")
    print(f"  fps:    mean {st.mean(fps):.1f}  min {min(fps):.1f}  p95 {pct(fps,95):.1f}  max {max(fps):.1f}")
    print(f"  dt_max: mean {st.mean(dt):.4f}  max {max(dt):.4f}")

print("========== 8.0 BASELINE (7.10c, in-loop I/O) ==========")
client_stats('net_debug_latest.csv', 'baseline')
host_stats('host_debug_latest.csv', 'baseline')
print()
print("========== 8.x WORKER RUN (8.1-8.3) ==========")
client_stats('net_debug.csv', '8.x')
host_stats('host_debug.csv', '8.x')