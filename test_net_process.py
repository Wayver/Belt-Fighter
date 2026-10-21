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


def main():
    ok = True
    print("NET PROCESS (plan_13, Phase 1, Chunk 1a)")
    ok &= test_net_process()
    ok &= test_net_process_quit()
    print("\nNET PROCESS (1a):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()