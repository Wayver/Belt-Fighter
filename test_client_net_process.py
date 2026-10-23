"""Test: the client's dedicated NET process (the client-side mirror of
net_process.py — the GIL fix).

Spawns the REAL `client_net_process_main` (a separate process, its own GIL)
and drives it with a FAKE HOST over a real loopback TCP connection. Verifies
the net process:
  * owns the socket: connect + the join/welcome handshake, and reports
    ("connected", host_hull_id, host_loadout) on the msg Pipe.
  * forwards the host's snapshots to the render: a T_SNAP becomes
    ("snap", sim_time, snap, wire_mono, wire_wall) — the wire-arrival stamp
    is a float (the true wire cadence, stamped on the net process's OWN GIL).
  * forwards a T_BEAM as ("beam", msg), a T_ECHO as ("echo", msg), and a
    T_PONG as ("pong", tag).
  * forwards the render's input to the socket: ("input", inp_dict, tag)
    becomes a T_INPUT (the tag intact); ("respawn",) becomes a T_RESPAWN.
  * reports ("disconnected",) when the host goes away, then exits.
  * exits cleanly on ("quit",) (the render process's ESC/QUIT).

Headless: SDL_VIDEODRIVER/AUDIODRIVER=dummy (the net import chain pulls in
pygame via net -> intent -> pygame).

Run:  python -m ship5.test_client_net_process
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import socket
import time
import threading
import multiprocessing

from .hulls import PLAYER_HULLS, default_loadout
from .intent import ShipInput
from .net import (Host, connect, do_handshake_host, do_handshake_client,
                  serialize_hull, serialize_loadout, serialize_input,
                  T_SNAP, T_BEAM, T_ECHO, T_PONG, T_INPUT, T_RESPAWN)
from .client_net_process import client_net_process_main


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


def _wait_for_pipe(pipe, timeout=3.0):
    """Recv one message from a Pipe (or None on timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll():
            return pipe.recv()
        time.sleep(0.002)
    return None


def _wait_for_msg(pipe, want, timeout=5.0):
    """Poll the msg Pipe until a message whose kind == `want` (or timeout).

    Skips any other messages (there should be none, but be safe). Returns
    the message or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll():
            m = pipe.recv()
            if m[0] == want:
                return m
        time.sleep(0.002)
    return None


def _wait_for(conn, want_type, timeout=3.0):
    """Poll the host's Connection until a message of `want_type` arrives
    (or timeout). Returns the message or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for m in conn.poll():
            if m.get("type") == want_type:
                return m
        time.sleep(0.002)
    return None


def _spawn_client_net(ctx, ip, port, client_hull, client_loadout):
    """Spawn the client net process; return (proc, cmd_parent, msg_parent).
    The parent closes the child ends it does not use."""
    cmd_parent, cmd_child = ctx.Pipe()   # render(parent) -> net(child)
    msg_parent, msg_child = ctx.Pipe()   # net(child) -> render(parent)
    p = ctx.Process(target=client_net_process_main,
                    args=(cmd_child, msg_child, ip, port,
                          client_hull, client_loadout))
    p.start()
    cmd_child.close()   # the parent only SENDS on cmd_parent
    msg_child.close()   # the parent only RECEIVES on msg_parent
    return p, cmd_parent, msg_parent


def _fake_host(port, host_hull, host_loadout, host, host_err):
    """The fake host: accept the client's connection + do the handshake,
    then yield the (non-blocking) Connection to the test (which drives it
    directly). Runs in a thread (the accept + handshake are blocking)."""
    h = None
    try:
        h = Host(port)
        conn, _addr = h.accept_one(timeout=10.0)
        ch, cl = do_handshake_host(conn, serialize_hull(host_hull),
                                   serialize_loadout(host_loadout))
        conn.set_nonblocking()   # the game-loop mode (poll returns fast)
        host["conn"] = conn
        host["client_hull"] = ch
        host["client_loadout"] = cl
        host["h"] = h
    except Exception as e:
        host_err.append(repr(e))
        if h is not None:
            try:
                h.close()
            except Exception:
                pass


def test_client_net_process():
    """The full client net-process functional test (connect + receive +
    send + disconnect). Returns True when all checks pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, cmd_parent, msg_parent = _spawn_client_net(
        ctx, "127.0.0.1", port, client_hull, client_loadout)

    host = {}
    host_err = []
    th = threading.Thread(target=_fake_host,
                          args=(port, host_hull, host_loadout, host,
                                host_err), daemon=True)
    th.start()

    try:
        # --- 1. the net process reported the join (the handshake) --------
        status = _wait_for_msg(msg_parent, "connected")
        ok &= check("connected: the net process reported the join",
                    status is not None, repr(status))
        if status is not None:
            ok &= check("connected: the host's hull id is in the status",
                        status[1] == host_hull.id, repr(status[1]))
            # The status carries the host's loadout in WIRE form (a dict of
            # component ids — exactly what do_handshake_client returns); the
            # render process deserializes it. Compare against the serialized
            # form, not the ComponentType objects.
            ok &= check("connected: the host's loadout is in the status",
                        status[2] == serialize_loadout(host_loadout),
                        "got %r" % (status[2],))
        th.join(timeout=5)
        ok &= check("fake host: no error", not host_err, repr(host_err))
        ok &= check("fake host: the handshake returned the client's hull",
                    host.get("client_hull") == client_hull.id,
                    repr(host.get("client_hull")))
        conn = host.get("conn")
        if conn is None:
            print("  (no host connection — aborting the rest)")
            return ok

        # --- 2. the net process forwards the host's snapshot to the render
        snap = {"players": [[1.0, 2.0, 0.0]], "asteroids": []}
        conn.send({"type": T_SNAP, "sim_time": 1.5, "snap": snap})
        conn.drain_send()
        got = _wait_for_msg(msg_parent, "snap")
        ok &= check("snap: the net process forwarded the snapshot to the "
                    "render", got is not None, repr(got))
        if got is not None:
            # got = ("snap", sim_time, snap, wire_mono, wire_wall)
            ok &= check("snap: the snapshot's sim_time is intact",
                        got[1] == 1.5, repr(got[1]))
            ok &= check("snap: the snapshot payload is intact",
                        got[2] == snap, repr(got[2]))
            ok &= check("snap: the wire-arrival stamp is a float "
                        "(stamped on the net process's own GIL)",
                        isinstance(got[3], float) and isinstance(got[4],
                                                                 float),
                        "mono=%r wall=%r" % (got[3], got[4]))

        # --- 3. the net process forwards a beam / echo / pong -------------
        beam = {"type": T_BEAM, "sim_time": 1.6, "ex": 10, "ey": 20,
                "owner": 0}
        conn.send(beam)
        conn.drain_send()
        got_beam = _wait_for_msg(msg_parent, "beam")
        ok &= check("beam: the net process forwarded the beam to the render",
                    got_beam is not None and got_beam[1].get("ex") == 10,
                    repr(got_beam))

        echo = {"type": T_ECHO, "entries": [[1.7,
                                             serialize_input(
                                                 ShipInput(turn=1.0))]]}
        conn.send(echo)
        conn.drain_send()
        got_echo = _wait_for_msg(msg_parent, "echo")
        ok &= check("echo: the net process forwarded the echo to the render",
                    got_echo is not None
                    and len(got_echo[1].get("entries", [])) == 1,
                    repr(got_echo))

        conn.send({"type": T_PONG, "tag": 9})
        conn.drain_send()
        got_pong = _wait_for_msg(msg_parent, "pong")
        ok &= check("pong: the net process forwarded the pong tag to the "
                    "render", got_pong is not None and got_pong[1] == 9,
                    repr(got_pong))

        # --- 4. the net process forwards the render's input to the host ---
        inp = serialize_input(ShipInput(turn=1.0, thrust_fwd=1.0))
        cmd_parent.send(("input", inp, 7))
        got_inp = _wait_for(conn, T_INPUT)
        ok &= check("input: the net process forwarded the render's input to "
                    "the host", got_inp is not None, repr(got_inp))
        if got_inp is not None:
            ok &= check("input: the forwarded input is intact (turn=1)",
                        got_inp.get("inp", {}).get("turn") == 1.0,
                        repr(got_inp.get("inp")))
            ok &= check("input: the tag is intact (tag=7)",
                        got_inp.get("tag") == 7, repr(got_inp.get("tag")))

        # --- 5. the net process forwards a respawn to the host ------------
        cmd_parent.send(("respawn",))
        got_respawn = _wait_for(conn, T_RESPAWN)
        ok &= check("respawn: the net process forwarded the respawn to the "
                    "host", got_respawn is not None, repr(got_respawn))

        # --- 6. the net process reports the disconnect + exits ------------
        conn.close()
        status2 = _wait_for_msg(msg_parent, "disconnected")
        ok &= check("disconnect: the net process reported the host left",
                    status2 is not None, repr(status2))
        p.join(timeout=3)
        ok &= check("disconnect: the net process exited", not p.is_alive())
    finally:
        if p.is_alive():
            p.terminate()
            p.join()
        cmd_parent.close()
        msg_parent.close()
        h = host.get("h")
        if h is not None:
            try:
                h.close()
            except Exception:
                pass

    print("\nCLIENT NET PROCESS (connect + receive + send + disconnect):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_client_net_process_wire_cadence():
    """The GIL-fix signal: the net process stamps the wire-arrival time on
    its OWN GIL (uncontended by the render thread's draw), so the receive
    cadence is the TRUE wire cadence — NOT quantized to the render frame.

    This test feeds the client net process snaps at a clean 60 Hz (the
    host's measured send rate) for a few seconds and asserts the WIRE-STAMP
    gaps (the "receive" times the render process would log) are ~16.7 ms:
      * p90 gap < 20 ms (a steady 16.7 ms cadence),
      * NO gap > 30 ms (the old NetWorker THREAD produced a bimodal 16/32 ms
        artifact — ~13% of gaps at 30-36 ms — from the GIL re-acquisition),
      * the mean gap is within [15, 20] ms (the 16.7 ms target).

    The render process (the "consumer") drains the msg Pipe in a tight loop
    (NOT once per frame) so the test measures the net process's stamp
    cadence, not a poll-quantization artifact. Returns True when all checks
    pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)

    p, cmd_parent, msg_parent = _spawn_client_net(
        ctx, "127.0.0.1", port, client_hull, client_loadout)

    host = {}
    host_err = []
    th = threading.Thread(target=_fake_host,
                          args=(port, host_hull, host_loadout, host,
                                host_err), daemon=True)
    th.start()

    # The "render process" consumer: drains the msg Pipe in a tight loop and
    # records the wire-arrival stamp (got[3]) of each snap. A tight loop (not
    # once per frame) so the recorded cadence is the net process's stamp
    # cadence, not a poll-quantization artifact.
    snap_stamps = []
    consumer_stop = threading.Event()
    def consumer():
        while not consumer_stop.is_set():
            if msg_parent.poll():
                m = msg_parent.recv()
                if m[0] == "snap":
                    snap_stamps.append(m[3])   # the wire-arrival stamp
            else:
                time.sleep(0.0005)
    cth = threading.Thread(target=consumer, daemon=True)

    try:
        status = _wait_for_msg(msg_parent, "connected")
        ok &= check("wirecadence: the net process reported the join",
                    status is not None, repr(status))
        if status is None:
            return ok
        th.join(timeout=5)
        ok &= check("wirecadence: fake host has no error",
                    not host_err, repr(host_err))
        conn = host.get("conn")
        if conn is None:
            print("  (no host connection — aborting)")
            return ok
        cth.start()

        # Feed snaps at a clean 60 Hz (the host's measured send rate) for a
        # few seconds. The net process stamps each on its own GIL as it
        # lands, so the stamps should be ~16.7 ms apart.
        snap = {"players": [[1.0, 2.0, 0.0]], "asteroids": []}
        TICK = 1.0 / 60.0
        RUN_SECONDS = 3.0
        t0 = time.monotonic()
        sim_t = 0.0
        next_pub = t0
        n_pub = 0
        while time.monotonic() - t0 < RUN_SECONDS:
            now = time.monotonic()
            while next_pub < now:
                conn.send({"type": T_SNAP, "sim_time": sim_t, "snap": snap})
                conn.drain_send()
                n_pub += 1
                sim_t += TICK
                next_pub += TICK
            time.sleep(0.001)

        # Let the last few snaps drain through the net process.
        time.sleep(0.2)
        consumer_stop.set()
        cth.join(timeout=3)

        if len(snap_stamps) < 2:
            ok &= check("wirecadence: the render received snaps",
                        len(snap_stamps) >= 2,
                        "n=%d" % len(snap_stamps))
            return ok
        gaps = [b - a for a, b in zip(snap_stamps, snap_stamps[1:])]
        gaps.sort()
        n = len(gaps)
        mean = sum(gaps) / n
        p90 = gaps[int(n * 0.9)]
        p99 = gaps[int(n * 0.99)]
        max_gap = gaps[-1]
        n_over_30 = sum(1 for g in gaps if g > 0.030)
        n_over_20 = sum(1 for g in gaps if g > 0.020)
        print("  [wirecadence] %d snaps over %.1f s -> %.1f Hz"
              % (len(snap_stamps), RUN_SECONDS,
                 len(snap_stamps) / RUN_SECONDS))
        print("  [wirecadence] wire-stamp gaps ms: mean=%.2f p90=%.2f "
              "p99=%.2f max=%.2f"
              % (mean * 1000, p90 * 1000, p99 * 1000, max_gap * 1000))
        print("  [wirecadence] gaps >20ms: %d (%.1f%%)   >30ms: %d (%.1f%%)"
              % (n_over_20, 100.0 * n_over_20 / n,
                 n_over_30, 100.0 * n_over_30 / n))

        ok &= check("wirecadence: the mean gap is ~16.7 ms (in [15, 20] ms)",
                    0.015 <= mean <= 0.020,
                    "mean=%.2f ms" % (mean * 1000))
        ok &= check("wirecadence: p90 gap < 20 ms (a steady 16.7 ms cadence)",
                    p90 < 0.020,
                    "p90=%.2f ms" % (p90 * 1000))
        ok &= check("wirecadence: NO gap > 30 ms (the old NetWorker THREAD "
                    "had ~13%% at 30-36 ms from the GIL re-acquisition)",
                    n_over_30 == 0,
                    "n_over_30=%d" % n_over_30)

        # Shutdown: the host goes away -> the net process reports
        # disconnected + exits.
        conn.close()
        status2 = _wait_for_msg(msg_parent, "disconnected")
        ok &= check("wirecadence: the net process reported the disconnect",
                    status2 is not None, repr(status2))
        p.join(timeout=3)
        ok &= check("wirecadence: the net process exited", not p.is_alive())
    finally:
        consumer_stop.set()
        if p.is_alive():
            p.terminate()
            p.join()
        cmd_parent.close()
        msg_parent.close()
        h = host.get("h")
        if h is not None:
            try:
                h.close()
            except Exception:
                pass

    print("\nCLIENT NET PROCESS (wire-arrival cadence, the GIL fix):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_client_net_process_quit():
    """The net process exits cleanly on ("quit",) (the render process's
    ESC/QUIT) — tested in the GAME LOOP (a host is present, so the net
    process has connected + handshaken and is draining the cmd Pipe). The
    connect phase is BLOCKING (unlike the host's poll-based accept loop), so
    a quit sent mid-connect is only seen once connect returns — the game-loop
    quit is the meaningful case. Returns True when the process exits."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    host_hull = PLAYER_HULLS[0]
    host_loadout = default_loadout(host_hull)
    client_hull = PLAYER_HULLS[1]
    client_loadout = default_loadout(client_hull)
    p, cmd_parent, msg_parent = _spawn_client_net(
        ctx, "127.0.0.1", port, client_hull, client_loadout)
    host = {}
    host_err = []
    th = threading.Thread(target=_fake_host,
                          args=(port, host_hull, host_loadout, host,
                                host_err), daemon=True)
    th.start()
    try:
        status = _wait_for_msg(msg_parent, "connected")
        ok &= check("quit: the net process reached the game loop",
                    status is not None, repr(status))
        th.join(timeout=5)
        # The host stays connected (the conn is open) — the net process is in
        # its game loop, draining the cmd Pipe. Send ("quit",): the net
        # process sees it, closes the conn itself, and exits cleanly.
        cmd_parent.send(("quit",))
        p.join(timeout=3)
        ok &= check("quit: the net process exited cleanly on ('quit',)",
                    not p.is_alive())
    finally:
        if p.is_alive():
            p.terminate()
            p.join()
        cmd_parent.close()
        msg_parent.close()
        h = host.get("h")
        if h is not None:
            try:
                h.close()
            except Exception:
                pass
    print("\nCLIENT NET PROCESS (quit):", "ALL PASS" if ok else "FAILURES")
    return ok


def main():
    ok = True
    print("CLIENT NET PROCESS (the client-side GIL fix)")
    ok &= test_client_net_process()
    ok &= test_client_net_process_wire_cadence()
    ok &= test_client_net_process_quit()
    print("\nCLIENT NET PROCESS:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()