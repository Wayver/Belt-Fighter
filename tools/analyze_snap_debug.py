"""Decompose the client's 33 ms snapshot-stamp gaps (plan_13 follow-up).

The client's 8 px snaps track its `newest_stamp` two-tick (33 ms) gaps.
A 33 ms gap can come from three places, and the per-snapshot CSVs
(localize it):

  sim_snap_debug.csv    (host machine, sim process)
      wall,mono,sim_time,seq,drops
      mono deltas  -> did the SIM batch (descheduled / slow step)?
      drops deltas -> did the SIM DROP snaps (net process too slow)?
  net_snap_debug.csv    (host machine, net process)
      wall,mono,sim_time,seq
      wall deltas  -> is the NET process sending smoothly or bursty?
  client_snap_debug.csv (client machine, client process)
      wall,mono,sim_time,seq
      wall deltas  -> is the CLIENT RECEIVE smooth (worker + wire)?

Correlation is by sim_time (the join key — the host's sim clock, carried
in every row). The cross-machine wall-clock latency (client_wall -
net_wall per sim_time) is the wire + client-receive one-way latency; its
VARIANCE (not its mean) is what causes snaps.

Usage:
  python tools/analyze_snap_debug.py [sim.csv net.csv client.csv]
(defaults: the repo-root CSVs)
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if not os.path.exists(path):
        return None
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({
            "wall": float(r["wall"]),
            "mono": float(r["mono"]),
            "sim_time": float(r["sim_time"]),
            "seq": int(r["seq"]),
            "drops": int(r["drops"]) if "drops" in r else None,
        })
    return rows


def cadence(rows, key, label):
    """Histogram of the deltas of `key` (ms) + the big-gap count."""
    if not rows or len(rows) < 2:
        print("  %s: not enough rows" % label)
        return
    d = [(b[key] - a[key]) * 1000.0
         for a, b in zip(rows, rows[1:]) if b[key] >= a[key]]
    if not d:
        print("  %s: no deltas" % label)
        return
    hist = {}
    for x in d:
        k = int(round(x / 16.7))   # bucket by tick count
        hist[k] = hist.get(k, 0) + 1
    n = len(d)
    one = sum(v for k, v in hist.items() if k == 1)
    two = sum(v for k, v in hist.items() if k == 2)
    big = sum(v for k, v in hist.items() if k >= 3)
    print("  %s: n=%d  1-tick %.1f%%  2-tick %.1f%%  >=3-tick %.1f%%  "
          "max=%.0f ms" % (label, n, 100.0 * one / n, 100.0 * two / n,
                           100.0 * big / n, max(d)))
    if big:
        # Show a few of the big gaps with their sim_time (the join key).
        shown = 0
        for a, b in zip(rows, rows[1:]):
            if (b[key] - a[key]) * 1000.0 >= 3 * 16.7 and shown < 8:
                print("      gap %.0f ms before sim_time=%.3f (seq %d)"
                      % ((b[key] - a[key]) * 1000.0, b["sim_time"],
                         b["seq"]))
                shown += 1


def main():
    args = sys.argv[1:]
    sim_p = args[0] if len(args) > 0 else os.path.join(REPO, "sim_snap_debug.csv")
    net_p = args[1] if len(args) > 1 else os.path.join(REPO, "net_snap_debug.csv")
    cli_p = args[2] if len(args) > 2 else os.path.join(REPO, "client_snap_debug.csv")

    sim = load(sim_p)
    net = load(net_p)
    cli = load(cli_p)

    print("== SIM (host machine, sim process) ==")
    if sim:
        print("  rows=%d  span=%.1f s" % (len(sim),
                                          sim[-1]["mono"] - sim[0]["mono"]))
        cadence(sim, "mono", "sim step cadence (mono)")
        cadence(sim, "sim_time", "sim stamp cadence (sim_time)")
        total_drops = sim[-1]["drops"] if sim[-1]["drops"] is not None else 0
        print("  snaps DROPPED by the sim (net queue full): %d" % total_drops)
        if total_drops:
            shown = 0
            for a, b in zip(sim, sim[1:]):
                if b["drops"] is not None and a["drops"] is not None \
                        and b["drops"] > a["drops"] and shown < 8:
                    print("      +%d drop(s) at sim_time=%.3f (seq %d)"
                          % (b["drops"] - a["drops"], b["sim_time"], b["seq"]))
                    shown += 1
    else:
        print("  missing: %s" % sim_p)

    print("\n== NET (host machine, net process) ==")
    if net:
        print("  rows=%d  span=%.1f s" % (len(net),
                                          net[-1]["wall"] - net[0]["wall"]))
        cadence(net, "wall", "net send cadence (wall)")
        cadence(net, "sim_time", "net send stamp cadence (sim_time)")
    else:
        print("  missing: %s" % net_p)

    print("\n== CLIENT (client machine, client process) ==")
    if cli:
        print("  rows=%d  span=%.1f s" % (len(cli),
                                          cli[-1]["wall"] - cli[0]["wall"]))
        cadence(cli, "wall", "client receive cadence (wall)")
        cadence(cli, "sim_time", "client receive stamp cadence (sim_time)")
    else:
        print("  missing: %s" % cli_p)

    # Cross-machine: wire + client-receive one-way latency per sim_time.
    if net and cli:
        net_by_sim = {round(r["sim_time"], 3): r for r in net}
        lat = []
        for r in cli:
            n = net_by_sim.get(round(r["sim_time"], 3))
            if n is not None:
                lat.append((r["sim_time"], (r["wall"] - n["wall"]) * 1000.0))
        if lat:
            vals = [v for _, v in lat]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            print("\n== WIRE + CLIENT RECEIVE (client_wall - net_wall, per sim_time) ==")
            print("  matched=%d  mean=%.1f ms  stddev=%.1f ms  min=%.1f  max=%.1f"
                  % (len(lat), mean, var ** 0.5, min(vals), max(vals)))
            worst = sorted(lat, key=lambda x: -x[1])[:8]
            for st, v in worst:
                print("      sim_time=%.3f  %.1f ms" % (st, v))
        else:
            print("\n== WIRE + CLIENT RECEIVE == no matching sim_times "
                  "(clocks or cadence misaligned?)")

    print("\nHOW TO READ IT")
    print("  * sim mono 2-tick high + drops 0  -> the SIM batches (Mac OS")
    print("    descheduled the sim process, or a step took >16.7 ms).")
    print("  * sim mono smooth + drops > 0     -> the NET process was too")
    print("    slow to drain the queue; the sim dropped snaps.")
    print("  * sim smooth + drops 0 + net wall bursty -> the NET process")
    print("    sends in bursts (descheduled / GIL-free but OS-scheduled).")
    print("  * all three smooth except client wall -> the WIRE or the")
    print("    CLIENT's NetWorker thread (Phase 3 territory).")
    print("  * wire+client latency variance high -> the LAN is jittery")
    print("    (a smooth 60 Hz stream arriving in clumps).")


if __name__ == "__main__":
    main()