"""Chunk 1a test: the host's dedicated NET process (plan_13, Phase 1).

Spawns the REAL `net_process_main` (a separate process, its own GIL) and
drives it with a FAKE CLIENT over a real loopback TCP connection. Verifies
the net process:
  * owns the socket: listen + accept + the join/welcome handshake, and
    reports ("connected", client_hull_id, client_loadout) on the status Pipe.
  * forwards the sim's net Queue to the socket: a ("snap", t, snap) becomes a
    T_SNAP (pruned on the wire); a ("beam", msg) is forwarded verbatim.
  * forwards the client's input to the sim: a T_INPUT becomes
    ("remote_input", ShipInput) on the remote_input Pipe, and the T_PONG is
    reflected (the tag intact).
  * forwards a T_RESPAWN as ("command", "respawn", 1) on the remote_input Pipe.
  * reports ("disconnected",) when the client goes away, then exits.
  * exits cleanly on ("quit",) (the render process's ESC/QUIT).

Headless: SDL_VIDEODRIVER/AUDIODRIVER=dummy (the net import chain pulls in
pygame via net -> intent -> pygame).

Run:  python -m ship5.test_net_process
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import socket
import sys
import time
import threading
import multiprocessing

from .hulls import PLAYER_HULLS, default_loadout
from .intent import ShipInput
from .net import (connect, do_handshake_client, serialize_hull,
                  serialize_loadout, serialize_input, deserialize_snapshot,
                  T_SNAP, T_BEAM, T_INPUT, T_RESPAWN, T_PONG)
from .net_process import net_process_main


def check(label, cond, extra=""):
    print(("PASS: " if cond else "FAIL: ") + label
          + (("  " + extra) if extra else ""))
    return bool(cond)


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fake_snapshot():
    """A minimal 11-tuple snapshot with 2 rocks at index 5 (the SIM form).

    serialize_snapshot reads the SIM 9-tuple rock (id, x, y, vx, vy, size,
    angle, spin, verts) and prunes it to the wire [id, x, y, angle, verts]
    (plan_17). Enough for serialize_snapshot (list(snap); prune snap[5];
    snap[6]=None) and for the client-side deserialize_snapshot to
    round-trip."""
    return [0.0, [], [], [], [],
            [(1, 10.0, 20.0, 0.0, 0.0, 'small', 0.0, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))),
             (2, 30.0, 40.0, 0.0, 0.0, 'small', 0.5, 0.0,
              ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))],
            None, False, 0.0, 0, 0]


def _wait_for(conn, want_type, timeout=3.0):
    """Poll the client's Connection until a message of `want_type` arrives
    (or timeout). Returns the message or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for m in conn.poll():
            if m.get("type") == want_type:
                return m
        time.sleep(0.002)
    return None


def _wait_for_pipe(pipe, timeout=3.0):
    """Recv one message from a Pipe (or None on timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll():
            return pipe.recv()
        time.sleep(0.002)
    return None


def _wait_for_status(pipe, want, timeout=5.0):
    """Poll the status Pipe until a message whose kind == `want` (or timeout).

    Skips any other status messages (there should be none, but be safe).
    Returns the message or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll():
            m = pipe.recv()
            if m[0] == want:
                return m
        time.sleep(0.002)
    return None


def _spawn_net(ctx, port, host_hull, host_loadout):
    """Spawn the net process; return (proc, status_parent, net_queue,
    ri_sim_end). The parent closes the child ends it does not use."""
    status_parent, status_child = ctx.Pipe()
    net_queue = ctx.Queue(maxsize=16)
    ri_sim_end, ri_net_end = ctx.Pipe()   # net(child) -> sim(parent)
    p = ctx.Process(target=net_process_main,
                    args=(status_child, net_queue, ri_net_end,
                          host_hull, host_loadout, port))
    p.start()
    status_child.close()   # the parent uses status_parent for both directions
    ri_net_end.close()     # the parent only recvs on ri_sim_end
    return p, status_parent, net_queue, ri_sim_end


def test_net_process():
    """The full net-process functional test (connect + send + receive +
    disconnect). Returns True when all checks pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, status_parent, net_queue, ri_sim_end = _spawn_net(
        ctx, port, host_hull, host_loadout)

    # --- the fake client (a real TCP connection + the client-side handshake)
    client = {}
    client_err = []
    def fake_client():
        try:
            time.sleep(0.3)   # let the net process bind + listen
            conn = connect("127.0.0.1", port)
            hh, hl = do_handshake_client(
                conn, serialize_hull(client_hull),
                serialize_loadout(client_loadout))
            conn.set_nonblocking()   # the game-loop mode (poll returns fast)
            client["conn"] = conn
            client["host_hull"] = hh
            client["host_loadout"] = hl
        except Exception as e:
            client_err.append(repr(e))
    th = threading.Thread(target=fake_client, daemon=True)
    th.start()

    try:
        # --- 1. the net process reported the client is joined -------------
        status = _wait_for_status(status_parent, "connected")
        ok &= check("connected: the net process reported the client joined",
                    status is not None, repr(status))
        if status is not None:
            ok &= check("connected: the client's hull id is in the status",
                        status[1] == client_hull.id, repr(status[1]))
            # The status carries the client's loadout in WIRE form (a dict of
            # component ids — exactly what do_handshake_host returns); the
            # render process (Chunk 1b) deserializes it. Compare against the
            # serialized form, not the ComponentType objects.
            ok &= check("connected: the client's loadout is in the status",
                        status[2] == serialize_loadout(client_loadout),
                        "got %r" % (status[2],))
        th.join(timeout=5)
        ok &= check("fake client: no error", not client_err, repr(client_err))
        ok &= check("fake client: the handshake returned the host's hull",
                    client.get("host_hull") == host_hull.id,
                    repr(client.get("host_hull")))
        conn = client.get("conn")
        if conn is None:
            print("  (no client connection — aborting the rest)")
            return ok

        # --- 2. the net process forwards the sim's snapshot to the client --
        snap = _fake_snapshot()
        net_queue.put(("snap", 1.5, snap))
        got_snap = _wait_for(conn, T_SNAP)
        ok &= check("snap: the net process forwarded the snapshot to the "
                    "client", got_snap is not None, repr(got_snap))
        if got_snap is not None:
            ok &= check("snap: the snapshot's sim_time is intact",
                        got_snap["sim_time"] == 1.5,
                        repr(got_snap.get("sim_time")))
            d = deserialize_snapshot(got_snap["snap"])
            # plan_17: the wire rock is [id, x, y, angle, verts]; the net
            # process's tracker is fresh, so first sight of both rocks ->
            # full verts (a list, not None).
            pruned = (len(d) == 11 and d[6] is None
                      and all(len(r) == 5 and isinstance(r[4], list)
                              for r in d[5]))
            ok &= check("snap: the snapshot is pruned on the wire",
                        pruned, "len=%d rng=%r rocks=%r"
                        % (len(d), d[6], d[5]))

        # --- 3. the net process forwards a beam verbatim -------------------
        beam = {"type": T_BEAM, "sim_time": 1.6, "sx": 0, "sy": 0,
                "ex": 10, "ey": 20}
        net_queue.put(("beam", beam))
        got_beam = _wait_for(conn, T_BEAM)
        ok &= check("beam: the net process forwarded the beam to the client",
                    got_beam is not None and got_beam.get("ex") == 10,
                    repr(got_beam))

        # --- 4. the net process forwards the client's input to the sim -----
        inp = serialize_input(ShipInput(turn=1.0, thrust_fwd=1.0))
        conn.send({"type": T_INPUT, "inp": inp, "tag": 7})
        conn.drain_send()
        ri = _wait_for_pipe(ri_sim_end)
        ok &= check("input: the net process forwarded the client's input to "
                    "the sim", ri is not None and ri[0] == "remote_input",
                    repr(ri))
        if ri is not None and ri[0] == "remote_input":
            ok &= check("input: the forwarded input is a ShipInput (turn=1)",
                        isinstance(ri[1], ShipInput) and ri[1].turn == 1.0,
                        repr(ri[1]))
        got_pong = _wait_for(conn, T_PONG)
        ok &= check("input: the net process reflected the T_PONG (tag=7)",
                    got_pong is not None and got_pong.get("tag") == 7,
                    repr(got_pong))

        # --- 5. the net process forwards a respawn as a command ------------
        conn.send({"type": T_RESPAWN})
        conn.drain_send()
        ri2 = _wait_for_pipe(ri_sim_end)
        ok &= check("respawn: the net process forwarded the respawn command "
                    "to the sim", ri2 == ("command", "respawn", 1),
                    repr(ri2))

        # --- 6. the net process reports the disconnect + exits -------------
        conn.close()
        status2 = _wait_for_status(status_parent, "disconnected")
        ok &= check("disconnect: the net process reported the client left",
                    status2 is not None, repr(status2))
        p.join(timeout=3)
        ok &= check("disconnect: the net process exited", not p.is_alive())
    finally:
        if p.is_alive():
            p.terminate()
            p.join()
        ri_sim_end.close()
        status_parent.close()

    print("\nNET PROCESS (connect + send + receive + disconnect):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_net_process_latest_wins():
    """The snapshot LATEST-WINS: when the net loop is slower than the sim's
    publish rate (the loop runs at ~54Hz, the sim at 60Hz), multiple snaps
    can land in the queue between iterations. A FIFO (oldest-first) would
    queue stale snaps behind the newest — the client would render seconds-
    old state and the ship would lag far behind the input. Instead, each
    drained snap REPLACES the pending one (latest-wins) and only the
    NEWEST is sent. This test verifies:
      1. A burst of 2 snaps back-to-back -> exactly 1 arrives, with the
         NEWER sim_time (the older one is dropped, not held).
      2. A burst of 3 snaps back-to-back -> exactly 1 arrives, with the
         NEWEST sim_time.
      3. No staleness: the received sim_time is always the max of what
         was put (the client never renders an older snap).

    Returns True when all checks pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, status_parent, net_queue, ri_sim_end = _spawn_net(
        ctx, port, host_hull, host_loadout)

    # The fake client records (arrival_wall, sim_time) for each T_SNAP.
    snap_arrivals = []
    client = {}
    client_err = []
    client_stop = threading.Event()
    def fake_client():
        conn = None
        try:
            time.sleep(0.3)   # let the net process bind + listen
            conn = connect("127.0.0.1", port)
            do_handshake_client(conn, serialize_hull(client_hull),
                                serialize_loadout(client_loadout))
            conn.set_nonblocking()
            client["conn"] = conn
            while not client_stop.is_set():
                for m in conn.poll():
                    if m.get("type") == T_SNAP:
                        snap_arrivals.append((time.monotonic(), m["sim_time"]))
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

    try:
        status = _wait_for_status(status_parent, "connected")
        ok &= check("latestwins: the net process reported the client joined",
                    status is not None, repr(status))
        if status is None:
            return ok
        th.join(timeout=5)
        ok &= check("latestwins: fake client has no error",
                    not client_err, repr(client_err))

        # Burst of 2 snaps back-to-back (the sim batched 2 ticks into one
        # wake). Latest-wins: if both are drained in the same loop
        # iteration, only the newer is sent (the older is dropped). If
        # they're drained in separate iterations (a timing race), the
        # client sees 2 snaps, but the LAST one is the newer (no
        # staleness). The robust assertion: the LAST arrival's sim_time
        # is the newer one.
        snap = _fake_snapshot()
        t1 = 1.0
        t2 = 1.0 + 1 / 60.0
        net_queue.put(("snap", t1, snap))
        net_queue.put(("snap", t2, snap))

        # Wait for at least 1 arrival (allow a generous window).
        deadline = time.monotonic() + 1.0
        while len(snap_arrivals) < 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        # Give a short grace period to catch any 2nd arrival.
        time.sleep(0.05)

        ok &= check("latestwins: a 2-snap burst -> at least 1 arrived",
                    len(snap_arrivals) >= 1,
                    "n=%d" % len(snap_arrivals))
        if snap_arrivals:
            last_t = snap_arrivals[-1][1]
            ok &= check("latestwins: the LAST arrival is the NEWER snap "
                        "(sim_time=%.4f, want %.4f — no staleness)"
                        % (last_t, t2),
                        abs(last_t - t2) < 1e-6,
                        "last=%.4f" % last_t)
            print("  [latestwins] burst of 2 -> %d arrived, last "
                  "sim_time=%.4f (the newer)" % (len(snap_arrivals),
                                                 last_t))

        # Burst of 3 snaps back-to-back. Same logic: the LAST arrival's
        # sim_time must be the newest (no staleness).
        snap_arrivals.clear()
        t1 = 2.0
        t2 = 2.0 + 1 / 60.0
        t3 = 2.0 + 2 / 60.0
        net_queue.put(("snap", t1, snap))
        net_queue.put(("snap", t2, snap))
        net_queue.put(("snap", t3, snap))
        deadline = time.monotonic() + 1.0
        while len(snap_arrivals) < 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        time.sleep(0.05)

        ok &= check("latestwins: a 3-snap burst -> at least 1 arrived",
                    len(snap_arrivals) >= 1,
                    "n=%d" % len(snap_arrivals))
        if snap_arrivals:
            last_t = snap_arrivals[-1][1]
            ok &= check("latestwins: the LAST arrival is the NEWEST snap "
                        "(sim_time=%.4f, want %.4f — no staleness)"
                        % (last_t, t3),
                        abs(last_t - t3) < 1e-6,
                        "last=%.4f" % last_t)
            print("  [latestwins] burst of 3 -> %d arrived, last "
                  "sim_time=%.4f (the newest)" % (len(snap_arrivals),
                                                  last_t))

        # Shutdown: the client goes away -> the net process reports
        # disconnected + exits.
        client_stop.set()
        th.join(timeout=3)
        status2 = _wait_for_status(status_parent, "disconnected")
        ok &= check("latestwins: the net process reported the disconnect",
                    status2 is not None, repr(status2))
        p.join(timeout=3)
        ok &= check("latestwins: the net process exited", not p.is_alive())
    finally:
        client_stop.set()
        if p.is_alive():
            p.terminate()
            p.join()
        ri_sim_end.close()
        status_parent.close()

    print("\nNET PROCESS (snapshot latest-wins):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_net_process_quit():
    """The net process exits cleanly on ("quit",) (the render process's
    ESC/QUIT) — tested in the accept loop (no client connected). Returns
    True when the process exits."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    p, status_parent, net_queue, ri_sim_end = _spawn_net(
        ctx, port, host_hull, host_loadout)
    try:
        time.sleep(0.3)   # let the net process reach the accept loop
        status_parent.send(("quit",))
        p.join(timeout=3)
        ok &= check("quit: the net process exited cleanly on ('quit',)",
                    not p.is_alive())
    finally:
        if p.is_alive():
            p.terminate()
            p.join()
        ri_sim_end.close()
        status_parent.close()
    print("\nNET PROCESS (quit):", "ALL PASS" if ok else "FAILURES")
    return ok


def test_net_process_cadence():
    """The plan_20 SEND CADENCE: the net process paces its sends to a
    steady 60 Hz (16.7 ms) with a deadline + busy-wait hybrid, so the
    client receives ~one snap per frame. This test feeds the net process
    snaps at a clean 60 Hz (the sim's measured publish rate) for a few
    seconds and asserts the SEND gaps are ~16.7 ms:
      * p90 gap < 20 ms (a steady 16.7 ms cadence, not the old ragged
        17-26 ms),
      * NO gap > 30 ms (the old 53.9 Hz loop produced 315 such gaps),
      * the mean gap is within [15, 20] ms (the 16.7 ms target).
    Returns True when all checks pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, status_parent, net_queue, ri_sim_end = _spawn_net(
        ctx, port, host_hull, host_loadout)

    # The fake client records (arrival_mono, sim_time) for each T_SNAP.
    snap_arrivals = []
    client = {}
    client_err = []
    client_stop = threading.Event()
    def fake_client():
        conn = None
        try:
            time.sleep(0.3)   # let the net process bind + listen
            conn = connect("127.0.0.1", port)
            do_handshake_client(conn, serialize_hull(client_hull),
                                serialize_loadout(client_loadout))
            conn.set_nonblocking()
            client["conn"] = conn
            while not client_stop.is_set():
                for m in conn.poll():
                    if m.get("type") == T_SNAP:
                        snap_arrivals.append((time.monotonic(), m["sim_time"]))
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

    try:
        status = _wait_for_status(status_parent, "connected")
        ok &= check("cadence: the net process reported the client joined",
                    status is not None, repr(status))
        if status is None:
            return ok
        th.join(timeout=5)
        ok &= check("cadence: fake client has no error",
                    not client_err, repr(client_err))

        # Feed snaps at a clean 60 Hz (the sim's measured publish rate)
        # for a few seconds. The net process paces its SENDS to 60 Hz
        # (SEND_TICK), so the client should receive ~one snap per frame.
        snap = _fake_snapshot()
        TICK = 1.0 / 60.0
        RUN_SECONDS = 3.0
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
                    pass   # dropped (queue full) — the sim would drop too
                sim_t += TICK
                next_pub += TICK
            while ri_sim_end.poll():
                try:
                    ri_sim_end.recv()
                except Exception:
                    break
            time.sleep(0.001)

        # Let the last few snaps drain through the net process.
        time.sleep(0.2)

        # Analyze the SEND gaps (the client's arrival times are the net
        # process's send times + a constant loopback latency, so the gaps
        # are the send gaps).
        if len(snap_arrivals) < 2:
            ok &= check("cadence: the client received snaps",
                        len(snap_arrivals) >= 2,
                        "n=%d" % len(snap_arrivals))
            return ok
        gaps = [b[0] - a[0] for a, b in zip(snap_arrivals, snap_arrivals[1:])]
        gaps.sort()
        n = len(gaps)
        mean = sum(gaps) / n
        p90 = gaps[int(n * 0.9)]
        p99 = gaps[int(n * 0.99)]
        max_gap = gaps[-1]
        n_over_30 = sum(1 for g in gaps if g > 0.030)
        n_over_20 = sum(1 for g in gaps if g > 0.020)
        print("  [cadence] %d snaps over %.1f s -> %.1f Hz"
              % (len(snap_arrivals), RUN_SECONDS,
                 len(snap_arrivals) / RUN_SECONDS))
        print("  [cadence] gaps ms: mean=%.2f p90=%.2f p99=%.2f max=%.2f"
              % (mean * 1000, p90 * 1000, p99 * 1000, max_gap * 1000))
        print("  [cadence] gaps >20ms: %d (%.1f%%)   >30ms: %d (%.1f%%)"
              % (n_over_20, 100.0 * n_over_20 / n,
                 n_over_30, 100.0 * n_over_30 / n))

        ok &= check("cadence: the mean gap is ~16.7 ms (in [15, 20] ms)",
                    0.015 <= mean <= 0.020,
                    "mean=%.2f ms" % (mean * 1000))
        ok &= check("cadence: p90 gap < 20 ms (a steady 16.7 ms cadence)",
                    p90 < 0.020,
                    "p90=%.2f ms" % (p90 * 1000))
        ok &= check("cadence: NO gap > 30 ms (the old 53.9 Hz loop had 315)",
                    n_over_30 == 0,
                    "n_over_30=%d" % n_over_30)

        # Shutdown: the client goes away -> the net process reports
        # disconnected + exits.
        client_stop.set()
        th.join(timeout=3)
        status2 = _wait_for_status(status_parent, "disconnected")
        ok &= check("cadence: the net process reported the disconnect",
                    status2 is not None, repr(status2))
        p.join(timeout=3)
        ok &= check("cadence: the net process exited", not p.is_alive())
    finally:
        client_stop.set()
        if p.is_alive():
            p.terminate()
            p.join()
        ri_sim_end.close()
        status_parent.close()

    print("\nNET PROCESS (send cadence, plan_20):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def main():
    ok = True
    print("NET PROCESS (plan_13, Phase 1, Chunk 1a)")
    ok &= test_net_process()
    ok &= test_net_process_latest_wins()
    ok &= test_net_process_cadence()
    ok &= test_net_process_quit()
    print("\nNET PROCESS (1a):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()