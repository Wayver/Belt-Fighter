"""Confirm the descheduling mechanism (plan_20, step 1 — saturation arm).

Spawns the REAL net process (loop log enabled) + a fake client, then spawns
N separate BUSY PROCESSES (each burning one core in a tight CPU loop —
mirroring the live host's render + sim processes, which are SEPARATE
processes, not threads). If the net loop's 1 ms sleep starts to overshoot
under this CPU contention, that confirms the 18.56 ms on the live host is
descheduling (option a), not a slow loop body (b) or timer imprecision (c).

Run from the repo root:  python -m ship5.diag_net_saturation
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import threading
import multiprocessing

from .hulls import PLAYER_HULLS, default_loadout
from .net import (connect, do_handshake_client, serialize_hull,
                  serialize_loadout)
from .net_process import net_process_main

RUN_SECONDS = 6.0
TICK = 1.0 / 60.0


def _fake_snapshot():
    return [0.0, [], [], [], [],
            [(1, 10.0, 20.0, 0.0, 0.0, 'small', 0.0, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))],
            None, False, 0.0, 0, 0]


def _busy_worker(stop_pipe):
    """Burn one core in a tight CPU loop until stop_pipe has data."""
    x = 0
    while not stop_pipe.poll():
        x += 1
        if x % 100000 == 0:
            # yield a hair so the OS can schedule (and so we don't wedge
            # the box); still ~99% CPU.
            time.sleep(0.0001)


def _spawn_net(ctx, port, host_hull, host_loadout):
    status_parent, status_child = ctx.Pipe()
    net_queue = ctx.Queue(maxsize=16)
    ri_sim_end, ri_net_end = ctx.Pipe()
    p = ctx.Process(target=net_process_main,
                    args=(status_child, net_queue, ri_net_end,
                          host_hull, host_loadout, port))
    p.start()
    status_child.close()
    ri_net_end.close()
    return p, status_parent, net_queue, ri_sim_end


def _run(tag, n_busy, log_path):
    ctx = multiprocessing.get_context("spawn")
    import socket as _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, status_parent, net_queue, ri_sim_end = _spawn_net(
        ctx, port, host_hull, host_loadout)

    client = {}
    client_err = []
    client_stop = threading.Event()

    def fake_client():
        conn = None
        try:
            time.sleep(0.3)
            conn = connect("127.0.0.1", port)
            do_handshake_client(conn, serialize_hull(client_hull),
                                serialize_loadout(client_loadout))
            conn.set_nonblocking()
            client["conn"] = conn
            while not client_stop.is_set():
                conn.poll()
                time.sleep(0.001)
        except Exception as e:
            client_err.append(repr(e))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    th = threading.Thread(target=fake_client, daemon=True)
    th.start()

    deadline = time.monotonic() + 5.0
    connected = False
    while time.monotonic() < deadline:
        if status_parent.poll():
            m = status_parent.recv()
            if m[0] == "connected":
                connected = True
                break
    if not connected:
        print("  [%s] no connection" % tag)
        p.terminate()
        return None
    status_parent.send(("loop_log", log_path))
    th.join(timeout=5)
    if client_err:
        print("  [%s] client error: %r" % (tag, client_err))
        p.terminate()
        return None

    # Spawn N busy PROCESSES (separate GILs, each burning a core).
    busy = []
    for _ in range(n_busy):
        stop_c, stop_k = ctx.Pipe()
        w = ctx.Process(target=_busy_worker, args=(stop_k,))
        w.start()
        stop_k.close()
        busy.append((w, stop_c))
    time.sleep(0.5)   # let the busy workers ramp up

    # Feed snaps at 60 Hz for RUN_SECONDS.
    snap = _fake_snapshot()
    t0 = time.monotonic()
    sim_t = 0.0
    next_pub = t0
    n_pub = 0
    while time.monotonic() - t0 < RUN_SECONDS:
        now = time.monotonic()
        while next_pub < now:
            try:
                net_queue.put_nowait(("snap", sim_t, snap))
                n_pub += 1
            except Exception:
                pass
            sim_t += TICK
            next_pub += TICK
        while ri_sim_end.poll():
            try:
                ri_sim_end.recv()
            except Exception:
                break
        time.sleep(0.001)

    # Stop the busy workers.
    for w, stop_c in busy:
        try:
            stop_c.send("stop")
        except Exception:
            pass
    for w, _ in busy:
        w.join(timeout=2)
        if w.is_alive():
            w.terminate()

    status_parent.send(("loop_log", None))
    status_parent.send(("quit",))
    p.join(timeout=3)
    if p.is_alive():
        p.terminate()
        p.join()
    status_parent.close()
    ri_sim_end.close()
    client_stop.set()
    print("  [%s] published %d snaps over %.1f s (n_busy=%d)"
          % (tag, n_pub, RUN_SECONDS, n_busy))
    return log_path


def _analyze(tag, path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("  [%s] no loop log rows" % tag)
        return
    rows = rows[1:]
    n = len(rows)
    sleep_act = [float(r["sleep_actual"]) for r in rows]
    work = [float(r["work_time"]) for r in rows]
    period = [float(r["total_period"]) for r in rows]

    def _stats(xs):
        xs = sorted(xs)
        m = len(xs)
        return (min(xs), xs[m // 2], xs[int(m * 0.9)], xs[int(m * 0.99)],
                max(xs), sum(xs) / m)

    def _line(name, xs):
        lo, med, p90, p99, hi, mean = _stats(xs)
        print("    %-14s min=%7.3f med=%7.3f p90=%7.3f p99=%7.3f "
              "max=%7.3f mean=%7.3f ms"
              % (name, lo * 1000, med * 1000, p90 * 1000, p99 * 1000,
                 hi * 1000, mean * 1000))

    span = float(rows[-1]["mono"]) - float(rows[0]["mono"])
    print("  [%s] %d iterations over %.1f s -> %.1f Hz"
          % (tag, n, span, n / span if span > 0 else 0.0))
    _line("total_period", period)
    _line("work_time", work)
    _line("sleep_actual", sleep_act)
    over = [a - 0.001 for a in sleep_act]
    _line("sleep_overshoot", over)
    n_over_5 = sum(1 for x in over if x > 0.005)
    n_over_10 = sum(1 for x in over if x > 0.010)
    print("    sleep overshoot >5ms: %d (%.1f%%)   >10ms: %d (%.1f%%)"
          % (n_over_5, 100.0 * n_over_5 / n, n_over_10,
             100.0 * n_over_10 / n))
    n_per_30 = sum(1 for x in period if x > 0.030)
    print("    total_period >30ms: %d (%.1f%%)"
          % (n_per_30, 100.0 * n_per_30 / n))


def main():
    nproc = os.cpu_count() or 4
    print("Net cadence SATURATION diagnostic (plan_20, step 1)")
    print("  nproc=%d, run %.1f s per arm\n" % (nproc, RUN_SECONDS))
    # Baseline: no busy workers.
    print("BASELINE (no busy workers):")
    path = _run("base", 0, "net_loop_sat_base.csv")
    if path:
        _analyze("base", path)
    # Saturated: nproc busy workers (every core busy).
    print("\nSATURATED (%d busy workers, all cores):" % nproc)
    path = _run("sat", nproc, "net_loop_sat_full.csv")
    if path:
        _analyze("sat", path)


if __name__ == "__main__":
    main()