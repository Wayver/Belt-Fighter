"""Smoke test for the plan_13 follow-up snap logs (sim + net process).

Run: python tools/_smoke_snap_log.py   (from the repo root)
"""
import os
import sys
import time
import socket
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from ship5.net import (connect, do_handshake_client, serialize_hull,
                       serialize_loadout, T_SNAP)
from ship5.hulls import PLAYER_HULLS, default_loadout
from ship5.net_process import net_process_main


def _fake_snapshot():
    """A minimal 11-tuple snapshot (enough for serialize_snapshot)."""
    return ([], [], [], [], [], [[1, 10.0, 20.0], [2, 30.0, 40.0]],
            None, False, 0.0, 0, 0)


def main():
    ctx = multiprocessing.get_context("spawn")
    status_p, status_c = ctx.Pipe()
    net_q = ctx.Queue(maxsize=16)
    ri_sim, ri_net = ctx.Pipe()
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    hh = PLAYER_HULLS[0]
    p = ctx.Process(target=net_process_main,
                    args=(status_c, net_q, ri_net, hh,
                          default_loadout(hh), port))
    p.start()
    status_c.close()
    ri_net.close()
    time.sleep(0.5)
    conn = connect("127.0.0.1", port, timeout=5.0)
    ch, cl = do_handshake_client(
        conn, serialize_hull(PLAYER_HULLS[1]),
        serialize_loadout(default_loadout(PLAYER_HULLS[1])))
    m = status_p.recv()
    assert m[0] == "connected", m
    nlog = "/tmp/net_snap_test.csv"
    status_p.send(("snap_log", nlog))
    time.sleep(0.2)
    net_q.put(("snap", 1.5, _fake_snapshot()))
    conn.set_nonblocking()
    got = None
    for _ in range(50):
        for msg in conn.poll():
            if msg.get("type") == T_SNAP:
                got = msg
        if got:
            break
        time.sleep(0.02)
    assert got is not None, "no snap received"
    status_p.send(("quit",))
    p.join(timeout=3)
    assert not p.is_alive()
    rows = open(nlog).read().strip().splitlines()
    print("net log rows:", rows)
    assert rows[0] == "wall,mono,sim_time,seq"
    assert len(rows) == 2 and rows[1].split(",")[2] == "1.5000"
    print("net process snap_log + per-send log: OK")
    print("SMOKE TEST: ALL OK")


if __name__ == "__main__":
    main()