"""Deeper 8.4 analysis: gap histograms, host sim-clock rate, stall/burst detection."""
import csv, statistics as st
from collections import Counter

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(row, key):
    v = row.get(key, '')
    try: return float(v)
    except (ValueError, TypeError): return None

def analyze_client(path, label):
    rows = load(path)
    print(f"===== CLIENT {label} =====")
    # gap histogram (sim-time gaps between distinct newest_stamp values)
    gaps = []
    prev = None
    for r in rows:
        s = fnum(r, 'newest_stamp')
        if s is None: continue
        if prev is not None and s > prev:
            gaps.append((s - prev) * 1000)
        prev = s
    hist = Counter(round(g) for g in gaps)
    print("  sim-time gap histogram (ms):", dict(sorted(hist.items())))
    # per-second host sim clock rate from host CSV? no — client sees stamps vs arrival t
    # arrival clock rate: (last_stamp - first_stamp) / (last_t - first_t)
    t0, s0 = fnum(rows[0], 't'), fnum(rows[0], 'newest_stamp')
    t1, s1 = fnum(rows[-1], 't'), fnum(rows[-1], 'newest_stamp')
    print(f"  apparent host sim-clock rate (stamps vs wall): {(s1-s0)/(t1-t0):.4f} x real time")
    # hic frames
    hic = [(fnum(r,'t'), fnum(r,'dt_max')) for r in rows if fnum(r,'hic') == 1]
    print(f"  hic frames: {len(hic)}")
    # delay trend: first 10s vs last 10s
    dly = [(fnum(r,'t'), fnum(r,'delay')) for r in rows if fnum(r,'delay') is not None]
    t_first = dly[0][0]
    early = [d for t, d in dly if t < t_first + 10]
    late = [d for t, d in dly if t > t_first + (dly[-1][0]-t_first) - 10]
    print(f"  delay first-10s mean {st.mean(early):.4f} | last-10s mean {st.mean(late):.4f}")
    # jitter trend
    jit = [(fnum(r,'t'), fnum(r,'jitter_ema')) for r in rows if fnum(r,'jitter_ema') is not None]
    early = [j for t, j in jit if t < t_first + 10]
    late = [j for t, j in jit if t > t_first + (jit[-1][0]-t_first) - 10]
    print(f"  jitter first-10s mean {st.mean(early):.4f} | last-10s mean {st.mean(late):.4f}")
    # snap_px outliers
    spx = [(fnum(r,'t'), fnum(r,'snap_px')) for r in rows if fnum(r,'snap_px') is not None]
    big = [(t, s) for t, s in spx if s > 10]
    print(f"  snap_px > 10px: {len(big)} events; top5: {sorted([s for _, s in big], reverse=True)[:5]}")
    return rows

def analyze_host(path, label):
    rows = load(path)
    print(f"===== HOST {label} =====")
    t0, s0 = fnum(rows[0], 't'), fnum(rows[0], 'sim_time')
    t1, s1 = fnum(rows[-1], 't'), fnum(rows[-1], 'sim_time')
    print(f"  host sim-clock rate (sim_time vs wall): {(s1-s0)/(t1-t0):.4f} x real time")
    fps = [fnum(r,'fps') for r in rows]; fps = [x for x in fps if x and x > 0]
    # fps per 10s bucket
    t_start = fnum(rows[0], 't')
    buckets = {}
    for r in rows:
        f = fnum(r, 'fps')
        if not f: continue
        b = int((fnum(r,'t') - t_start) // 10)
        buckets.setdefault(b, []).append(f)
    print("  fps per 10s bucket:", {k: round(st.mean(v),1) for k, v in sorted(buckets.items())})
    dt = [fnum(r,'dt_max') for r in rows if fnum(r,'dt_max') is not None]
    worst = sorted(((fnum(r,'dt_max'), fnum(r,'t')) for r in rows), reverse=True)[:5]
    print(f"  worst dt_max frames: {[(round(d,3), round(t-t_start,1)) for d, t in worst]}")

analyze_client('net_debug_latest.csv', 'BASELINE 8.0')
analyze_host('host_debug_latest.csv', 'BASELINE 8.0')
print()
analyze_client('net_debug.csv', '8.x WORKER')
analyze_host('host_debug.csv', '8.x WORKER')