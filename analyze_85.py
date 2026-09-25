"""8.5 Step 1: analyze host_sim_debug.csv — prove the dt clamp is losing time.

Columns: t,raw_dt,clamped_dt,sim_before,sim_after,advance,acc_after

Hypothesis to confirm:
  - On hiccup frames (raw_dt > 0.05), advance <= 0.05 < raw_dt -> time DROPPED.
  - sum(advance) < sum(raw_dt)  ->  sim-clock rate < 1.0 (the 0.976x drift).
  - The sim advances in ~0.05 s chunks on hiccup frames (the 50/67/83 ms
    client-observed gap cluster).

Run:  python3 analyze_85.py
"""
import csv, statistics as st

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(row, key):
    v = row.get(key, '')
    try: return float(v)
    except (ValueError, TypeError): return None

rows = load('host_sim_debug.csv')
# Column 3 was clamped_dt pre-fix (what the sim saw) and draw_dt post-fix
# (the clamped dt that drives draw() — the sim now sees raw_dt). Accept both.
dt_col = 'clamped_dt' if 'clamped_dt' in rows[0] else 'draw_dt'
PRE_FIX = (dt_col == 'clamped_dt')
# The user's run had a death: after game_over the host sim FREEZES
# (advance == 0 forever). Truncate at the last frame that advanced the sim
# so the frozen tail doesn't dilute the rate / drop calculations.
last_live = max(i for i, r in enumerate(rows) if (fnum(r, 'advance') or 0) > 0)
frozen = len(rows) - 1 - last_live
rows = rows[:last_live + 1]
print(f"=== 8.5 {'Step 1 (pre-fix)' if PRE_FIX else 'Step 3 (post-fix verify)'}: "
      f"host_sim_debug.csv ({len(rows)} live frames"
      + (f", {frozen} post-death frozen frames excluded" if frozen else "")
      + ") ===")

raw = [fnum(r, 'raw_dt') for r in rows]
raw = [x for x in raw if x is not None]
adv = [fnum(r, 'advance') for r in rows]
adv = [x for x in adv if x is not None]
t0, t1 = fnum(rows[0], 't'), fnum(rows[-1], 't')
s0, s1 = fnum(rows[0], 'sim_before'), fnum(rows[-1], 'sim_after')

print(f"  wall time:          {t1-t0:.2f} s")
print(f"  sum(raw_dt):        {sum(raw):.3f} s  (real time elapsed)")
print(f"  sum(advance):       {sum(adv):.3f} s  (sim time gained)")
print(f"  TIME DROPPED:       {sum(raw)-sum(adv):.3f} s  "
      f"({100*(sum(raw)-sum(adv))/(t1-t0):.2f}% of wall time lost)")
print(f"  sim-clock rate:     {(s1-s0)/(t1-t0):.4f} x real time  "
      f"(target ~1.0; 8.4 measured 0.9762)")

# Hiccup frames: raw_dt > 0.05 (the old clamp threshold)
hic = [r for r in rows if (fnum(r, 'raw_dt') or 0) > 0.05]
print(f"\n  hiccup frames (raw_dt > 0.05): {len(hic)} of {len(rows)}")
if hic:
    h_raw = [fnum(r, 'raw_dt') for r in hic]
    h_adv = [fnum(r, 'advance') for r in hic]
    h_lost = [fnum(r, 'raw_dt') - fnum(r, 'advance') for r in hic]
    print(f"    raw_dt:   mean {st.mean(h_raw)*1000:.1f} ms  max {max(h_raw)*1000:.1f} ms")
    print(f"    advance:  mean {st.mean(h_adv)*1000:.1f} ms  max {max(h_adv)*1000:.1f} ms")
    print(f"    LOST:     mean {st.mean(h_lost)*1000:.1f} ms  total {sum(h_lost):.3f} s")
    # Pre-fix: advance clusters at ~0.05 (the clamp). Post-fix: advance
    # tracks raw_dt (within STEP quantization + backlog-cap slack).
    from collections import Counter
    hist = Counter(round(a*1000) for a in h_adv)
    print(f"    advance histogram (ms): {dict(sorted(hist.items()))}")

# Non-hiccup frames: advance should ~ raw_dt (no loss)
ok = [r for r in rows if (fnum(r, 'raw_dt') or 0) <= 0.05]
if ok:
    o_raw = [fnum(r, 'raw_dt') for r in ok]
    o_adv = [fnum(r, 'advance') for r in ok]
    o_diff = [a - b for a, b in zip(o_adv, o_raw)]
    print(f"\n  normal frames (raw_dt <= 0.05): {len(ok)}")
    print(f"    |advance - raw_dt|: mean {st.mean(abs(d) for d in o_diff)*1000:.2f} ms  "
          f"max {max(abs(d) for d in o_diff)*1000:.2f} ms  (should be ~0: no loss)")

# acc_after should always be < STEP (1/60 = 16.67 ms)
acc = [fnum(r, 'acc_after') for r in rows if fnum(r, 'acc_after') is not None]
print(f"\n  acc_after: max {max(acc)*1000:.2f} ms  (must be < 16.67 ms = STEP)")

# Verdict
dropped = sum(raw) - sum(adv)
rate = (s1 - s0) / (t1 - t0)
print("\n=== VERDICT ===")
if PRE_FIX:
    if dropped > 0.5 and rate < 0.99:
        print(f"  CONFIRMED: the dt clamp drops {dropped:.2f} s over {t1-t0:.0f} s "
              f"({100*dropped/(t1-t0):.1f}% of wall time). Sim-clock rate {rate:.4f}x.")
        print("  -> Proceed to 8.5 Step 2 (feed unclamped dt + cap steps/frame).")
    else:
        print(f"  NOT CONFIRMED: dropped={dropped:.2f}s, rate={rate:.4f}x. "
              "Re-examine the mechanism before Step 2.")
else:
    # Post-fix success bar: sim-clock rate ~1.0 (>= 0.995), hiccup frames no
    # longer pinned at 50 ms (advance tracks raw_dt), no backlog-cap drops
    # (lost ~0 unless a pathological stall hit the 0.25 s cap).
    hic_lost = sum(fnum(r, 'raw_dt') - fnum(r, 'advance') for r in hic)
    if rate >= 0.995 and hic_lost < 0.05:
        print(f"  FIX VERIFIED: sim-clock rate {rate:.4f}x (target ~1.0), "
              f"hiccup-frame loss {hic_lost*1000:.0f} ms total "
              f"(was pinned at 50 ms pre-fix).")
        print("  -> 8.5 Step 3: fresh two-machine CSV run; check the client's")
        print("     50/67/83 ms gap cluster is gone + jitter/delay back to baseline.")
    else:
        print(f"  FIX INCOMPLETE: rate={rate:.4f}x, hiccup-frame loss "
              f"{hic_lost*1000:.0f} ms. Inspect the hiccup advance histogram.")