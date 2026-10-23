"""Diagnose WHY the host's NET process sends at ~54 Hz (plan_20, step 1).

Spawns the REAL net process (net_process_main), enables its per-iteration
loop log (sleep_requested, sleep_actual, work_time, total_period), feeds it
snaps at a clean 60 Hz (the sim's measured publish rate), and decomposes the
send period:

  total_period ~= work_time + sleep_actual + wake_latency

  * work_time    = the loop body (drain + serialize + send + receive).
                   If this is large, the loop body is eating the tick.
  * sleep_actual = the real time the 1 ms sleep took. If a 1 ms request
                   returns after ~17 ms, the OS is descheduling the process
                   (sleep overshoot) — the working hypothesis.
  * wake_latency = total_period - work_time - sleep_actual (the remainder).

Two arms:
  IDLE  — the parent does nothing (baseline OS jitter).
  BUSY  — the parent holds the GIL with a render-like load (CPU
          contention), mirroring the live host where render + sim + net
          run concurrently.

The fake client drains the socket (poll loop) so the send buffer never
fills — exactly what the real client does.

Run from the repo root:  python -m ship5.diag_net_cadence
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import threading
import multiprocessing

from .hulls import PLAYER_HULLS, default_loadout
from .net import (connect, do_handshake_client, serialize_hull,
                  serialize_loadout, T_SNAP)
from .net_process import net_process_main

RUN_SECONDS = 6.0
GIL_HOLD_MS = 20.0
FRAME_MS = 20.0
TICK = 1.0 / 60.0


def _fake_snapshot():
    """A minimal 11-tuple snapshot with 2 rocks (the SIM form) — enough
    for serialize_snapshot (same shape as test_net_process._fake_snapshot)."""
    return [0.0, [], [], [], [],
            [(1, 10.0, 20.0, 0.0, 0.0, 'small', 0.0, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))),
             (2, 30.0, 40.0, 0.0, 0.0, 'small', 0.5, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))],
            None, False, 0.0, 0, 0]


def _gil_hold(ms):
    t0 = time.monotonic()
    x = 0
    while (time.monotonic() - t0) * 1000.0 < ms:
        x += 1
    return x


def _render_load(stop_event, hold_ms, frame_ms):
    """A render-like load: hold the GIL for `hold_ms` every `frame_ms`
    (the draw holds the GIL for its whole frame)."""
    frame_s = frame_ms / 1000.0
    while not stop_event.is_set():
        t0 = time.monotonic()
        _gil_hold(hold_ms)
        elapsed = time.monotonic() - t0
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)


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


def _run_arm(tag, busy, log_path):
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

    # The fake client: connect + handshake, then drain the socket (poll)
    # for the whole run — exactly what the real client does.
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

    # Wait for the connection, then enable the loop log.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if status_parent.poll():
            m = status_parent.recv()
            if m[0] == "connected":
                break
    else:
        print("  [%s] no connection" % tag)
        p.terminate()
        return None
    status_parent.send(("loop_log", log_path))
    th.join(timeout=5)
    if client_err:
        print("  [%s] client error: %r" % (tag, client_err))
        p.terminate()
        return None

    stop = threading.Event()
    loader = None
    if busy:
        loader = threading.Thread(target=_render_load,
                                  args=(stop, GIL_HOLD_MS, FRAME_MS))
        loader.start()

    # Feed snaps at a clean 60 Hz (the sim's measured publish rate) for
    # RUN_SECONDS, draining the remote_input Pipe (the net process may
    # reflect pongs — there are none here, but keep the Pipe drained).
    snap = _fake_snapshot()
    t0 = time.monotonic()
    sim_t = 0.0
    next_pub = t0
    n_pub = 0
    while time.monotonic() - t0 < RUN_SECONDS:
        now = time.monotonic()
        while next_pub < now:
            # put_nowait drop-when-full: the sim's exact behavior (a
            # bounded Queue, put_nowait) — the feed never blocks.
            try:
                net_queue.put_nowait(("snap", sim_t, snap))
                n_pub += 1
            except Exception:
                pass   # dropped (queue full) — the sim would drop too
            sim_t += TICK
            next_pub += TICK
        while ri_sim_end.poll():
            try:
                ri_sim_end.recv()
            except Exception:
                break
        time.sleep(0.001)

    if loader is not None:
        stop.set()
        loader.join()

    status_parent.send(("loop_log", None))   # close the log
    status_parent.send(("quit",))
    p.join(timeout=3)
    if p.is_alive():
        p.terminate()
        p.join()
    status_parent.close()
    ri_sim_end.close()
    client_stop.set()
    print("  [%s] published %d snaps over %.1f s" % (tag, n_pub, RUN_SECONDS))
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
    # Skip the first row (total_period is 0 there).
    rows = rows[1:]
    n = len(rows)
    sleep_req = [float(r["sleep_requested"]) for r in rows]
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
    # sleep overshoot: sleep_actual - sleep_requested
    over = [a - r for a, r in zip(sleep_act, sleep_req)]
    _line("sleep_overshoot", over)
    n_over_5 = sum(1 for x in over if x > 0.005)
    n_over_10 = sum(1 for x in over if x > 0.010)
    print("    sleep overshoot >5ms: %d (%.1f%%)   >10ms: %d (%.1f%%)"
          % (n_over_5, 100.0 * n_over_5 / n, n_over_10,
             100.0 * n_over_10 / n))
    n_work_5 = sum(1 for x in work if x > 0.005)
    print("    work_time >5ms: %d (%.1f%%)"
          % (n_work_5, 100.0 * n_work_5 / n))
    n_per_30 = sum(1 for x in period if x > 0.030)
    print("    total_period >30ms: %d (%.1f%%)"
          % (n_per_30, 100.0 * n_per_30 / n))


def main():
    print("Net cadence diagnostic (plan_20, step 1)")
    print("  run %.1f s per arm, GIL hold %.0f ms/frame\n"
          % (RUN_SECONDS, GIL_HOLD_MS))
    print("IDLE arm (parent does nothing):")
    path = _run_arm("idle", busy=False, log_path="net_loop_diag_idle.csv")
    if path:
        _analyze("idle", path)
    print("\nBUSY arm (parent holds the GIL, render-like load):")
    path = _run_arm("busy", busy=True, log_path="net_loop_diag_busy.csv")
    if path:
        _analyze("busy", path)


if __name__ == "__main__":
    main()