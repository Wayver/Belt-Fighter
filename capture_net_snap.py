"""Capture net_snap_debug.csv from the REAL net process (plan_20, step 3).

Spawns the real net process, enables its per-send snap log (the same F3
telemetry the live host writes to net_snap_debug.csv), feeds it snaps at a
clean 60 Hz (the sim's measured publish rate) for a few seconds, and
analyzes the resulting send cadence. This is the headless equivalent of the
live 2P test: the net process is the real one, the client is a real loopback
TCP connection, and the send cadence is the net process's own (wall, mono,
sim_time, seq) rows.

Run from the repo root:  python -m ship5.capture_net_snap
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import csv
import time
import threading
import multiprocessing

from .hulls import PLAYER_HULLS, default_loadout
from .net import (connect, do_handshake_client, serialize_hull,
                  serialize_loadout)
from .net_process import net_process_main

RUN_SECONDS = 8.0
TICK = 1.0 / 60.0
OUT = "net_snap_debug.csv"


def _fake_snapshot():
    return [0.0, [], [], [], [],
            [(1, 10.0, 20.0, 0.0, 0.0, 'small', 0.0, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))),
             (2, 30.0, 40.0, 0.0, 0.0, 'small', 0.5, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))],
            None, False, 0.0, 0, 0]


def main():
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

    status_parent, status_child = ctx.Pipe()
    net_queue = ctx.Queue(maxsize=16)
    ri_sim_end, ri_net_end = ctx.Pipe()
    p = ctx.Process(target=net_process_main,
                    args=(status_child, net_queue, ri_net_end,
                          host_hull, host_loadout, port))
    p.start()
    status_child.close()
    ri_net_end.close()

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
        print("no connection")
        p.terminate()
        return
    # Enable the per-send snap log (the live host's F3 net_snap_debug.csv).
    status_parent.send(("snap_log", OUT))
    th.join(timeout=5)
    if client_err:
        print("client error: %r" % client_err)
        p.terminate()
        return

    # Feed snaps at a clean 60 Hz for RUN_SECONDS.
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

    status_parent.send(("snap_log", None))   # close the log
    status_parent.send(("quit",))
    p.join(timeout=3)
    if p.is_alive():
        p.terminate()
        p.join()
    status_parent.close()
    ri_sim_end.close()
    client_stop.set()
    print("published %d snaps over %.1f s" % (n_pub, RUN_SECONDS))

    # Analyze the send cadence from the net process's own log.
    mono = []
    with open(OUT) as f:
        for r in csv.DictReader(f):
            mono.append(float(r["mono"]))
    gaps = [b - a for a, b in zip(mono, mono[1:])]
    gaps.sort()
    n = len(gaps)
    import statistics
    print("\nnet_snap_debug.csv send cadence (%d sends):" % len(mono))
    print("  Hz = %.1f" % (n / (mono[-1] - mono[0])))
    print("  gaps ms: mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f"
          % (statistics.mean(gaps) * 1000, gaps[n // 2] * 1000,
             gaps[int(n * 0.9)] * 1000, gaps[int(n * 0.99)] * 1000,
             gaps[-1] * 1000))
    print("  gaps >20ms: %d (%.1f%%)   >30ms: %d (%.1f%%)"
          % (sum(1 for g in gaps if g > 0.020),
             100.0 * sum(1 for g in gaps if g > 0.020) / n,
             sum(1 for g in gaps if g > 0.030),
             100.0 * sum(1 for g in gaps if g > 0.030) / n))


if __name__ == "__main__":
    main()