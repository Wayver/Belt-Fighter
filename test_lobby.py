"""plan_23 M1 test: the lobby relay (server + two spawned lobby clients).

Spawns a REAL `LobbyServer` (in-process, on a free loopback port) and TWO
real `lobby_client_main` child processes (spawn context, their own GILs —
the same pattern as test_net_process.py) over real loopback TCP. Verifies:

  * create/join/match relay: the joiner's match carries the HOST's reported
    game address (the load-bearing field — what the client dials); the
    creator's match carries the joiner's source address (informational).
  * join with no open room -> ("full",).
  * create while a room is open -> ("error", "room already open").
  * after a match, the room is closed: a later join -> ("full",).
  * a malformed create (no addr) -> ("error", ...).
  * an unknown message type -> ("error", ...).
  * the server's standalone smoke: binds a real port, close() is clean.

Headless: SDL_VIDEODRIVER/AUDIODRIVER=dummy (the net import chain pulls in
pygame via net -> intent -> pygame).

Run:  python -m ship5.test_lobby
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import socket
import time
import multiprocessing

from .lobby import LobbyServer, lobby_client_main


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


def _wait_for_status(pipe, timeout=10.0):
    """Recv one message from the status Pipe (or None on timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll():
            return pipe.recv()
        time.sleep(0.002)
    return None


def _spawn_lobby_client(ctx, lobby_port, action, addr):
    """Spawn one lobby_client_main child; return (proc, status_parent)."""
    status_parent, status_child = ctx.Pipe()
    p = ctx.Process(target=lobby_client_main,
                    args=(status_child, "127.0.0.1", lobby_port,
                          action, addr))
    p.start()
    status_child.close()   # the parent uses status_parent (recv only)
    return p, status_parent


def test_match_relay():
    """The core relay: create + join -> both sides get a match."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    lobby_port = _free_port()
    host_game_addr = "127.0.0.1:%d" % _free_port()   # the host's game ip:port

    server = LobbyServer(lobby_port)
    server.start()
    try:
        # The host (creator) opens a room first.
        creator, creator_status = _spawn_lobby_client(
            ctx, lobby_port, "create", host_game_addr)
        # The client (joiner) joins after a beat (the room must be open).
        time.sleep(0.5)
        joiner, joiner_status = _spawn_lobby_client(
            ctx, lobby_port, "join", None)

        # The JOINER's match carries the HOST's game address — the address
        # the client dials to start the P2P game (the load-bearing field).
        jm = _wait_for_status(joiner_status)
        ok &= check("joiner: got a match status",
                    jm is not None and jm[0] == "match", repr(jm))
        if jm is not None and jm[0] == "match":
            ok &= check("joiner: the match carries the host's game address",
                        jm[1] == host_game_addr, repr(jm[1]))
        # The CREATOR's match carries the joiner's source address
        # (informational — the host is the listener, the client dials it).
        cm = _wait_for_status(creator_status)
        ok &= check("creator: got a match status",
                    cm is not None and cm[0] == "match", repr(cm))
        if cm is not None and cm[0] == "match":
            ok &= check("creator: the match carries the joiner's source "
                        "address (127.0.0.1:port)",
                        isinstance(cm[1], str)
                        and cm[1].startswith("127.0.0.1:"), repr(cm[1]))
        # Both children exit after reporting their outcome.
        creator.join(timeout=5)
        joiner.join(timeout=5)
        ok &= check("both lobby clients exited after the match",
                    not creator.is_alive() and not joiner.is_alive())
        # The room is closed once matched.
        ok &= check("room closed after the match",
                    server.room_addr is None, repr(server.room_addr))
    finally:
        server.close()
        for p in (creator, joiner):
            if p.is_alive():
                p.terminate()
                p.join()
        creator_status.close()
        joiner_status.close()

    print("\nLOBBY (create/join/match relay):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_error_paths():
    """full / room-already-open / post-match full / malformed create /
    unknown type."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    lobby_port = _free_port()
    host_game_addr = "127.0.0.1:%d" % _free_port()

    server = LobbyServer(lobby_port)
    server.start()
    procs = []
    pipes = []
    try:
        # 1. join with no open room -> full.
        p, s = _spawn_lobby_client(ctx, lobby_port, "join", None)
        procs.append(p); pipes.append(s)
        m = _wait_for_status(s)
        ok &= check("join with no room: got full",
                    m is not None and m[0] == "full", repr(m))

        # 2. create opens the room; a second create -> error.
        p, s = _spawn_lobby_client(ctx, lobby_port, "create", host_game_addr)
        procs.append(p); pipes.append(s)
        time.sleep(0.5)
        ok &= check("create: the room is open",
                    server.room_addr == host_game_addr,
                    repr(server.room_addr))
        p2, s2 = _spawn_lobby_client(ctx, lobby_port, "create",
                                     "127.0.0.1:1")
        procs.append(p2); pipes.append(s2)
        m2 = _wait_for_status(s2)
        ok &= check("second create: got error (room already open)",
                    m2 is not None and m2[0] == "error"
                    and "already open" in m2[1], repr(m2))

        # 3. a join now matches the open room -> the room closes; a LATER
        #    join gets full again.
        p3, s3 = _spawn_lobby_client(ctx, lobby_port, "join", None)
        procs.append(p3); pipes.append(s3)
        m3 = _wait_for_status(s3)
        ok &= check("join with an open room: got match (host addr)",
                    m3 is not None and m3[0] == "match"
                    and m3[1] == host_game_addr, repr(m3))
        p4, s4 = _spawn_lobby_client(ctx, lobby_port, "join", None)
        procs.append(p4); pipes.append(s4)
        m4 = _wait_for_status(s4)
        ok &= check("join after the match: got full (room closed)",
                    m4 is not None and m4[0] == "full", repr(m4))

        # 4. a malformed create (no addr) -> error.
        p5, s5 = _spawn_lobby_client(ctx, lobby_port, "create", "")
        procs.append(p5); pipes.append(s5)
        m5 = _wait_for_status(s5)
        ok &= check("create without addr: got error",
                    m5 is not None and m5[0] == "error", repr(m5))
    finally:
        server.close()
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
        for s in pipes:
            s.close()

    print("\nLOBBY (error paths):", "ALL PASS" if ok else "FAILURES")
    return ok


def test_unknown_type():
    """A raw connection sending an unknown message type gets an error
    (the server never crashes on a hostile/malformed client)."""
    ok = True
    lobby_port = _free_port()
    server = LobbyServer(lobby_port)
    server.start()
    try:
        from .net import connect, T_SNAP
        conn = connect("127.0.0.1", lobby_port, timeout=5.0)
        conn.send({"type": T_SNAP, "sim_time": 1.0, "snap": [1, 2, 3]})
        conn.drain_send()
        conn.set_timeout(0.05)
        got = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and got is None:
            for m in conn.poll():
                if m.get("type") == "error":
                    got = m
        ok &= check("unknown type: the server replied with an error",
                    got is not None, repr(got))
        conn.close()
    finally:
        server.close()

    print("\nLOBBY (unknown message type):",
          "ALL PASS" if ok else "FAILURES")
    return ok


def test_server_smoke():
    """The standalone server binds a real port and closes cleanly."""
    ok = True
    lobby_port = _free_port()
    server = LobbyServer(lobby_port)
    ok &= check("server: bound to the requested port",
                server.bound_port == lobby_port,
                repr(server.bound_port))
    server.start()
    time.sleep(0.1)
    server.close()
    ok &= check("server: close() is clean (no exception, room cleared)",
                server.room_addr is None)
    # close() is idempotent.
    server.close()
    ok &= check("server: close() is idempotent", True)

    print("\nLOBBY (server smoke):", "ALL PASS" if ok else "FAILURES")
    return ok


def main():
    ok = True
    ok &= test_server_smoke()
    ok &= test_match_relay()
    ok &= test_error_paths()
    ok &= test_unknown_type()
    print("\nLOBBY SELF-TEST:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()