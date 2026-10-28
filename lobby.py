"""Lobby: find a match, then play P2P (plan_23).

Two players connect to this lobby server; it pairs one "create" (the host,
who reports its own game ip:port) with one "join" (the client) and hands
each side the other's address. After that the two players play the EXISTING
host-authoritative 2P game DIRECTLY (the host's NET_PORT socket, the
join/welcome handshake, snapshots, input, echoes) — the lobby is a thin
one-shot relay, never on the game's critical path, and carries no game
state or traffic.

Transport: the same framing as the game wire (4-byte big-endian length +
JSON, via net.encode_frame / extract_frames / net.Connection), so the
partial-frame send + receive-resync behavior comes for free.

Lobby wire messages (the 'type' field):
    create  (client->lobby)  {"type":"create","addr":"ip:port"}
    join    (client->lobby)  {"type":"join"}
    match   (lobby->client)  {"type":"match","addr":"ip:port"}
    full    (lobby->client)  {"type":"full"}
    closed  (lobby->client)  {"type":"closed"}
    error   (lobby->client)  {"type":"error","msg":"..."}

`match` is asymmetric by design:
  * the JOINER's match carries the HOST's reported game address — the
    address the client dials to start the P2P game (the load-bearing field);
  * the CREATOR's match carries the joiner's source address — informational
    only (the host is the listener; the client connects TO it).

Run the server standalone (any machine on the LAN, or a VPS):

    python -m ship5.lobby [port]        # default port: config.LOBBY_PORT

The client side is a SPAWN TARGET (lobby_client_main) — the game spawns it
as a child process (its own GIL, off the render thread) exactly like
net_process.py / client_net_process.py, and talks to it through a status
Pipe. See plan_23_lobby.md for the full design.
"""
import os
import socket
import threading
import time
import traceback

from .config import LOBBY_PORT
from .net import Connection, connect, encode_frame, extract_frames

__all__ = [
    "LobbyServer", "lobby_client_main",
    "T_CREATE", "T_JOIN", "T_MATCH", "T_FULL", "T_CLOSED", "T_ERROR",
]

T_CREATE = "create"
T_JOIN = "join"
T_MATCH = "match"
T_FULL = "full"
T_CLOSED = "closed"
T_ERROR = "error"

# How long a lobby client waits for its match before giving up (the host
# waits here for a friend to join; 10 minutes is generous but bounded, so
# a leaked lobby process can't linger forever).
LOBBY_MATCH_TIMEOUT = 600.0


def _peer_addr(conn):
    """The peer's 'ip:port' as seen by this end (for the match relay)."""
    ip, port = conn.sock.getpeername()
    return "%s:%d" % (ip, port)


class LobbyServer:
    """A single-room lobby: pairs one create with one join.

    Thread-per-connection (stdlib socket + threading — no pygame, no
    third-party deps). The only shared state is the single open room
    (self._room), guarded by self._lock. One open room at a time keeps the
    server trivially correct (a 2P game has one host at a time); multiple
    concurrent rooms is a small future extension (a dict of rooms).

    Lifecycle: `LobbyServer(port)` binds + listens immediately; `start()`
    launches the acceptor thread; `close()` stops everything. `port=0`
    lets the OS pick a free ephemeral port (read it back via `bound_port`)
    — the tests use this.
    """

    def __init__(self, port=LOBBY_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.listen(8)
        self._lock = threading.Lock()
        self._room = None      # {"addr": str, "conn": Connection} | None
        self._conns = set()    # live client Connections (for close())
        self._stop = threading.Event()
        self._acceptor = None

    @property
    def bound_port(self):
        return self.sock.getsockname()[1]

    @property
    def room_addr(self):
        """The open room's host address, or None (tests + diagnostics)."""
        with self._lock:
            return self._room["addr"] if self._room is not None else None

    def start(self):
        self._acceptor = threading.Thread(target=self._accept_loop,
                                          daemon=True)
        self._acceptor.start()

    def close(self):
        """Stop the acceptor + close every client connection (idempotent)."""
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        with self._lock:
            self._room = None
            conns = list(self._conns)
            self._conns.clear()
        for c in conns:
            c.close()
        if self._acceptor is not None:
            self._acceptor.join(timeout=2.0)
            self._acceptor = None

    # -- internals ----------------------------------------------------------

    def _log(self, line):
        # Standalone-operator log (the server is the only thing a human
        # watches while it runs). Quiet enough to leave on.
        print("[lobby] %s" % line, flush=True)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn_sock, _addr = self.sock.accept()
            except OSError:
                break               # the listen socket was closed
            conn = Connection(conn_sock)
            try:
                # A short recv timeout so poll() returns quickly (the
                # worker loop below also sleeps between polls).
                conn.sock.settimeout(0.05)
            except OSError:
                pass
            t = threading.Thread(target=self._serve, args=(conn,),
                                 daemon=True)
            t.start()

    def _serve(self, conn):
        """One client connection: read messages, respond, until it drops."""
        with self._lock:
            self._conns.add(conn)
        try:
            while not self._stop.is_set() and not conn.closed:
                for m in conn.poll():
                    self._handle(conn, m)
                time.sleep(0.002)
        except OSError:
            pass                    # the socket went away (close/RST)
        finally:
            self._drop_room(conn)
            with self._lock:
                self._conns.discard(conn)
            conn.close()

    def _drop_room(self, conn):
        """If `conn` was the room's host and it just dropped, clear the room
        (a later join then gets `full` instead of matching a dead host)."""
        with self._lock:
            if self._room is not None and self._room["conn"] is conn:
                self._room = None
                self._log("room cancelled (host connection dropped)")

    def _send(self, conn, msg):
        conn.send(msg)
        conn.drain_send()

    def _handle(self, conn, m):
        t = m.get("type")
        if t == T_CREATE:
            addr = m.get("addr")
            if not isinstance(addr, str) or not addr.strip():
                self._send(conn, {"type": T_ERROR,
                                  "msg": "create needs an 'addr' string"})
                return
            with self._lock:
                if self._room is not None:
                    resp = "room already open"
                else:
                    self._room = {"addr": addr.strip(), "conn": conn}
                    resp = None
            if resp is not None:
                self._send(conn, {"type": T_ERROR, "msg": resp})
            else:
                self._log("room open (host addr %s)" % addr.strip())
        elif t == T_JOIN:
            with self._lock:
                room = self._room
                if room is None:
                    resp = None
                else:
                    self._room = None
                    resp = (room["conn"], room["addr"])
            if resp is None:
                self._send(conn, {"type": T_FULL})
            else:
                host_conn, host_addr = resp
                self._log("match: host %s <-> client %s"
                          % (host_addr, _peer_addr(conn)))
                # The joiner gets the HOST's game address (what it dials);
                # the creator gets the joiner's source address (informational
                # — the host is the listener, the client connects to it).
                self._send(host_conn, {"type": T_MATCH,
                                       "addr": _peer_addr(conn)})
                self._send(conn, {"type": T_MATCH, "addr": host_addr})
        else:
            self._send(conn, {"type": T_ERROR,
                              "msg": "unknown message type %r" % t})


def lobby_client_main(status_conn, lobby_ip, lobby_port, action, addr):
    """The SPAWN TARGET for the game's lobby connection (MUST be a top-level
    function: spawn re-imports this module by name to find it).

    Connects to the lobby at lobby_ip:lobby_port, sends ONE action
    ("create" with `addr` = our game ip:port, or "join" with addr ignored),
    waits for the outcome, reports it on the status Pipe, and exits.

    Status messages to the render process (the render polls them
    non-blocking, like the other net processes):
      ("match", peer_addr)   matched — for a joiner, peer_addr is the HOST's
                             game address to dial; for a creator, the
                             joiner's source address (informational)
      ("full",)              no open room (join)
      ("closed",)            the room was cancelled
      ("error", text)        connect failed / the lobby said error / timeout
      ("crash", tb)          this process died

    Exits on ("quit",) (the render process's shutdown) or when the outcome
    is reported. The lobby connection is short-lived (seconds), so this
    process never coexists with the game's net process.
    """
    # Headless: the net import chain (net -> intent -> pygame) pulls in
    # pygame, so set the dummy drivers BEFORE importing net (inside the
    # function, so the PARENT keeps its real video driver). Mirrors
    # net_process_main.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    def _status(msg):
        """Send a status message to the render process. A failure means the
        render process is gone — there is no one left to report to, so
        return False and the caller exits."""
        try:
            status_conn.send(msg)
            return True
        except (OSError, EOFError, ConnectionError, BrokenPipeError):
            return False

    conn = None
    try:
        conn = connect(lobby_ip, lobby_port, timeout=10.0)
        # The request is written fully (blocking) before we wait — a clean
        # request/response, like do_handshake_client.
        if action == "create":
            conn.send({"type": T_CREATE, "addr": addr})
        else:
            conn.send({"type": T_JOIN})
        conn.drain_send()
        conn.set_timeout(0.05)
        deadline = time.monotonic() + LOBBY_MATCH_TIMEOUT
        while not conn.closed:
            # Quit from the render process (its shutdown sends ("quit",) on
            # every exit path — the established pattern).
            if status_conn.poll():
                m = status_conn.recv()
                if m[0] == "quit":
                    return
            for m in conn.poll():
                t = m.get("type")
                if t == T_MATCH:
                    if not _status(("match", m.get("addr"))):
                        return
                    return
                elif t == T_FULL:
                    if not _status(("full",)):
                        return
                    return
                elif t == T_CLOSED:
                    if not _status(("closed",)):
                        return
                    return
                elif t == T_ERROR:
                    if not _status(("error", m.get("msg", "lobby error"))):
                        return
                    return
            if time.monotonic() >= deadline:
                if not _status(("error",
                                "timed out waiting for a match")):
                    return
                return
    except BaseException:
        try:
            status_conn.send(("crash", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if conn is not None:
            conn.close()


def main(argv=None):
    """Run the lobby server standalone:  python -m ship5.lobby [port]"""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    try:
        port = int(argv[0]) if argv else LOBBY_PORT
    except ValueError:
        print("usage: python -m ship5.lobby [port]")
        raise SystemExit(2)
    # The net import chain pulls in pygame; the server never creates a
    # display, so the dummy drivers are harmless (and required).
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    server = LobbyServer(port)
    server.start()
    print("lobby listening on 0.0.0.0:%d (Ctrl-C to stop)"
          % server.bound_port, flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()