"""Analyze the client F3 logs from the 2026-10-21 live 2P run for the
'hang moments' the user felt.

The client render loop is SMOOTH (net_debug: hic=0, frame_dt max 30ms), so
the hang is NOT a client render stall. The two remaining suspects:
  1. WIRE / HOST STALL — snapshot arrivals gapped on the client
     (client_snap_debug wall gaps; the interpolated remote world freezes
     while the buffer holds its last window, and the ghost free-runs then
     snaps back on the next arrival).
  2. RECONCILE SNAP — the ghost got yanked (net_debug snap_px spikes). A
     huge snap (hundreds-thousands of px) is a teleport-scale event:
     either a death->respawn (the host's ship is elsewhere) or the ghost
     free-ran across a snapshot gap.

Clocks: net_debug.t and client_snap.wall are time.time() (wall).
ghost_clock.real_time is pygame.time.get_ticks()/1000 (process-relative).
The offset (wall - get_ticks/1000) is recovered from the REC rows, which
carry both real_time and snap_time (a host sim time that client_snap also
carries as sim_time with its wall).

Run:  python -m ship5.tools._analyze_hangs
"""
import bisect
import csv
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(name):
    with open(os.path.join(HERE, name), newline="") as f:
        return list(csv.DictReader(f))


def main():
    # --- client snapshot arrivals (wall + sim_time) ------------------------
    cs = read_csv("client_snap_debug.csv")
    arr = []
    for r in cs:
        try:
            arr.append((float(r["wall"]), float(r["sim_time"]), int(r["seq"])))
        except (ValueError, KeyError):
            continue
    arr.sort(key=lambda a: a[2])
    t0 = arr[0][0]
    print("client_snap: %d snapshots over %.1f s wall"
          % (len(arr), arr[-1][0] - t0))

    # wall gaps between consecutive seqs
    gaps = []
    for i in range(1, len(arr)):
        dw = arr[i][0] - arr[i - 1][0]
        ds = arr[i][1] - arr[i - 1][1]
        if dw > 0.04:
            gaps.append((arr[i][0] - t0, dw, ds, arr[i][2]))
    print("\n--- SNAPSHOT ARRIVAL GAPS > 40ms (a stall between host send and "
          "client receive) ---")
    if not gaps:
        print("none")
    for (t, dw, ds, seq) in sorted(gaps, key=lambda g: -g[1]):
        print("  t=%+7.2fs  wall_gap=%6.3fs  sim_delta=%5.3fs  seq=%d"
              % (t, dw, ds, seq))

    # sim_time delta histogram (host stamping cadence / dropped snaps)
    ds_hist = {}
    for i in range(1, len(arr)):
        ds = round(arr[i][1] - arr[i - 1][1], 3)
        ds_hist[ds] = ds_hist.get(ds, 0) + 1
    print("\nsim_time delta histogram: %s"
          % sorted(ds_hist.items(), key=lambda kv: -kv[1])[:6])

    # --- net_debug: reconcile snaps + frame hitches ------------------------
    nd = read_csv("net_debug.csv")
    rows = []
    for r in nd:
        try:
            rows.append({
                "t": float(r["t"]), "snap_px": float(r["snap_px"]),
                "delay": float(r["delay"]), "buf_depth": int(r["buf_depth"]),
                "frame_dt": float(r["frame_dt"]), "hic": int(r["hic"]),
                "host_rate": float(r["host_rate"]),
                "newest": float(r["newest_stamp"]),
                "render_t": float(r["render_t"]),
            })
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda r: r["t"])
    nt0 = rows[0]["t"]
    print("\nnet_debug: %d frames over %.1f s" % (len(rows), rows[-1]["t"] - nt0))
    hics = [r for r in rows if r["hic"] == 1]
    print("client frame hitches (hic=1): %d" % len(hics))
    big = sorted((r for r in rows if r["snap_px"] > 8), key=lambda r: r["t"])
    print("reconcile snaps > 8px: %d" % len(big))

    # --- ghost_clock: recover the wall offset, then zero-step runs --------
    gc = read_csv("ghost_clock.csv")
    recs, advs = [], []
    for r in gc:
        try:
            rt = float(r["real_time"])
        except (ValueError, KeyError):
            continue
        if r["kind"] == "REC":
            recs.append((rt, float(r["snap_time"] or 0.0)))
        elif r["kind"] == "ADV":
            advs.append((rt, int(r["steps"]), float(r["post_sim"] or 0.0)))
    advs.sort(key=lambda a: a[0])

    # offset = wall - get_ticks/1000, from REC rows matched to client_snap
    sim_sorted = [a[1] for a in arr]
    offs = []
    for (rt, st) in recs:
        if st <= 0:
            continue
        i = bisect.bisect_left(sim_sorted, st)
        if i < len(arr) and abs(arr[i][1] - st) < 0.05:
            offs.append(arr[i][0] - rt)
    offset = sorted(offs)[len(offs) // 2] if offs else 0.0
    print("\nghost_clock: %d ADV, %d REC; wall offset (wall - get_ticks/1000) "
          "= %.3f s (from %d REC matches)" % (len(advs), len(recs), offset,
                                               len(offs)))

    # zero-step runs (the ghost not advancing = a frozen prediction)
    zero_runs = []
    run = 0
    for i in range(1, len(advs)):
        if advs[i][1] == 0:
            run += 1
        else:
            if run >= 4:
                zero_runs.append((advs[i - 1][0] + offset - t0, run))
            run = 0
    if run >= 4:
        zero_runs.append((advs[-1][0] + offset - t0, run))
    print("ghost zero-step runs >=4 frames: %d" % len(zero_runs))
    for (t, run) in zero_runs[:12]:
        print("  t=%+7.2fs  %d consecutive zero-step frames" % (t, run))

    # --- context around each big reconcile snap ---------------------------
    print("\n--- CONTEXT AROUND EACH BIG RECONCILE SNAP (>8px) ---")
    for r in big:
        t_abs = r["t"]
        # snapshot arrivals in [t-2s, t+0.2s] (wall based)
        seg = [a for a in arr if t_abs - 2.0 <= a[0] <= t_abs + 0.2]
        seg_gaps = []
        for i in range(1, len(seg)):
            dw = seg[i][0] - seg[i - 1][0]
            if dw > 0.03:
                seg_gaps.append((seg[i][0] - t0, dw, seg[i][1] - seg[i - 1][1]))
        # ghost ADV in [t-1s, t] (convert to wall)
        gseg = [a for a in advs if t_abs - 1.0 <= a[0] + offset <= t_abs]
        nz = sum(1 for a in gseg if a[1] > 0)
        print("\n  SNAP t=%+7.2fs  snap_px=%6.0f  host_rate=%.3f  delay=%.3f"
              % (r["t"] - nt0, r["snap_px"], r["host_rate"], r["delay"]))
        print("    arrivals in [t-2s,t+0.2s]: %d" % len(seg))
        if seg_gaps:
            for (gt, dw, ds) in seg_gaps:
                print("      arrival gap t=%+7.2fs wall=%5.3fs sim_delta=%5.3fs"
                      % (gt, dw, ds))
        else:
            print("      (no arrival gap > 30ms in the preceding 2s)")
        if gseg:
            print("    ghost ADV in [t-1s,t]: %d frames, %d stepped, "
                  "max steps=%d" % (len(gseg), nz, max(a[1] for a in gseg)))
        else:
            print("    ghost ADV in [t-1s,t]: (none)")


if __name__ == "__main__":
    main()