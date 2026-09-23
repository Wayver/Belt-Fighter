"""Network transport (Session 6.3): TCP, non-blocking, length-prefixed JSON.

This is the top of the netcode stack. Everything below it (the deterministic
sim, the 11-tuple `Game.snapshot()`, `SnapshotBuffer`, `PredictedShip`,
`Game.push_snapshot`, `intent.ShipInput`) is already done and verified; this
module is the wire that carries them between the two peers.

Pinned decisions (see the pinned plan note — do NOT re-derive):
  * Transport: TCP, non-blocking. Reliable + ordered, so there is no loss or
    reorder to handle. 10 Hz snapshots + 60 Hz input are tiny. The socket is
    polled in the game loop (no threads, no shared state). Outgoing messages
    go to a send buffer that is drained each frame. Connection setup
    (connect/accept + the join/welcome handshake) is a separate, BLOCKING
    phase that happens BEFORE the game loop; only then does the socket flip
    to non-blocking.
  * Topology: 2P, host-authoritative. The host runs the sim and broadcasts a
    snapshot every `SNAPSHOT_INTERVAL` ticks; the client predicts its own
    ship and renders remote entities from the buffer.
  * Protocol: 4-byte big-endian length + JSON payload. Four message types:
        join    (client->host)  {'type':'join','hull':h,'loadout':{...}}
        welcome (host->client)  {'type':'welcome','hull':h,'loadout':{...}}
        input   (client->host)  {'type':'input','inp':{...}}
        snap    (host->client)  {'type':'snap','sim_time':t,'snap':[...]}
  * `ShipInput` serializes via `dataclasses.asdict` / `ShipInput(**dict)`.
  * The snapshot serializes directly (tuples -> JSON arrays). On deserialize
    ONLY the rng state (index 6) is converted back: JSON `[v,[ints],null]`
    -> `(v, tuple(ints), None)` for `random.Random.setstate()`. Everything
    else stays a list — Python unpacking/indexing work identically on lists,
    and `apply_snapshot` never type-checks them. This conversion lives HERE,
    in the protocol layer, NOT in `apply_snapshot`.
  * Stream integrity (Session 6.9): the send side sends frames in PARTS and
    keeps only the UNSENT remainder when the buffer fills (re-sending a
    partially-sent frame would duplicate bytes and corrupt the stream —
    invisible on loopback, real on a congested LAN). The receive side
    RESYNCS past a corrupted region instead of raising, so a stray bad byte
    costs a few frames, not a crash.

The framing is self-contained and testable in isolation: `encode_frame` /
`extract_frames` are pure byte functions with no socket involved.

Run the self-test (headless, no display needed):

    python -m ship5.net
"""
import json
import socket
import struct
import time
from dataclasses import asdict

from .intent import ShipInput
from .hulls import (PLAYER_HULLS, COMPONENT_CATALOG, DEFAULT_HULL,
                    default_loadout, validate_loadout)

__all__ = [
    "encode_frame", "extract_frames",
    "serialize_input", "deserialize_input",
    "serialize_snapshot", "deserialize_snapshot",
    "serialize_hull", "deserialize_hull",
    "serialize_loadout", "deserialize_loadout",
    "Connection", "Host", "connect",
    "do_handshake_client", "do_handshake_host",
    "T_JOIN", "T_WELCOME", "T_INPUT", "T_SNAP",
]

# 4-byte big-endian unsigned length prefix.
_LEN = struct.Struct(">I")
# Sanity cap so a corrupt length can't make us allocate gigabytes.
_MAX_FRAME = 64 * 1024 * 1024   # 64 MiB

# Message type tags (the 'type' field of every JSON payload).
T_JOIN = "join"
T_WELCOME = "welcome"
T_INPUT = "input"
T_SNAP = "snap"


# --- framing: pure byte functions (no socket) -----------------------------

def encode_frame(obj):
    """Encode one message dict as a length-prefixed JSON frame (bytes).

    4-byte big-endian length + the UTF-8 JSON payload. The length is the
    payload byte count, NOT including the 4 prefix bytes.
    """
    payload = json.dumps(obj).encode("utf-8")
    return _LEN.pack(len(payload)) + payload


def _find_frame_start(buf, i):
    """Scan forward from `i` for the next position that looks like the start
    of a valid frame: a plausible length prefix, a COMPLETE payload already
    in the buffer, that decodes as UTF-8 JSON to a dict with a 'type' field.
    Returns the position, or -1 when none is found.

    Used by `extract_frames` to RESYNC after a corrupted region. A
    length-prefixed stream has no other framing — once a byte is duplicated
    or dropped, every later frame is misaligned, so the only recovery is to
    find the next frame that is actually valid. The candidate checks are
    strict enough that a false positive is effectively impossible (the
    length must be plausible AND the full payload present AND it must parse
    to a message dict).
    """
    n = len(buf)
    j = i
    while n - j >= 4:
        (length,) = _LEN.unpack_from(buf, j)
        if 4 <= length <= _MAX_FRAME and n - j >= 4 + length:
            try:
                msg = json.loads(bytes(buf[j + 4:j + 4 + length])
                                 .decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                msg = None
            if isinstance(msg, dict) and "type" in msg:
                return j
        j += 1
    return -1


def extract_frames(buf):
    """Pull every COMPLETE frame out of `buf` (a bytes/bytearray).

    Returns (messages, rest): `messages` is a list of decoded dicts in
    arrival order (TCP is ordered, so this is the send order); `rest` is the
    trailing partial frame (bytes) that must be kept for the next call. A
    frame is complete only once its full length-prefixed payload has arrived,
    so this is safe to call on a partial buffer.

    CORRUPTION RESYNC (Session 6.9): if a frame's bytes do not decode (a
    duplicated/dropped byte misaligned the stream), the length prefix at
    that position is garbage. Instead of raising (which would crash the game
    loop), the scanner RESYNCS — it skips the corrupted region and resumes
    at the next position that parses as a valid frame. The cost is losing
    the frames inside the corrupted region (a few, at most); the stream
    re-aligns and the game continues. The send side (drain_send) no longer
    corrupts the stream — this is a safety net for any other stray bad byte.
    """
    out = []
    i = 0
    n = len(buf)
    while n - i >= 4:
        (length,) = _LEN.unpack_from(buf, i)
        start = i + 4
        end = start + length
        if length > _MAX_FRAME:
            # Implausible length: the stream is misaligned here. Resync.
            j = _find_frame_start(buf, i + 1)
            if j < 0:
                # No complete valid frame anywhere in the rest of the buffer
                # (the scanner checked every position): the tail is
                # unrecoverable, so drop it. We lose at most the one partial
                # frame at the end; the next complete frame re-aligns the
                # stream. (Dropping the whole tail is safe precisely because
                # _find_frame_start found NO complete valid frame in it.)
                return out, b""
            i = j
            continue
        if n < end:
            break                       # partial frame: wait for more bytes
        try:
            msg = json.loads(bytes(buf[start:end]).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            # Complete per the (garbage) length but undecodable: corrupted
            # region. Resync past it (same -1 handling as above).
            j = _find_frame_start(buf, i + 1)
            if j < 0:
                return out, b""
            i = j
            continue
        out.append(msg)
        i = end
    return out, bytes(buf[i:])


# --- payload (de)serialization --------------------------------------------

def serialize_input(inp):
    """ShipInput -> plain dict (the 'inp' field of an input message)."""
    return asdict(inp)


def deserialize_input(d):
    """dict -> ShipInput (inverse of serialize_input)."""
    return ShipInput(**d)


def serialize_snapshot(snap):
    """Game.snapshot() tuple -> JSON-safe structure.

    A no-op passthrough: tuples become JSON arrays on `json.dumps`, and every
    field is already a plain number / bool / None / tuple / list. The only
    field that needs a REVERSE conversion on the way back is the rng state,
    handled by `deserialize_snapshot`.
    """
    return snap


def deserialize_snapshot(snap):
    """JSON-decoded snapshot -> a structure `Game.apply_snapshot` accepts.

    After `json.loads`, every tuple is a list. That is fine for everything
    EXCEPT the rng state (index 6): `random.Random.setstate()` requires
    `(version, tuple_of_ints, None)`, but JSON gives `[version, [ints], None]`.
    Convert ONLY that field back to the tuple form; leave the rest as lists
    (unpacking/indexing work identically on lists, and apply_snapshot never
    type-checks them).
    """
    s = list(snap)
    version, ints, gauss = s[6]
    s[6] = (version, tuple(ints), gauss)
    return s


# --- hull/loadout wire mapping (Session 6.5) ------------------------------
#
# The join/welcome messages carry each peer's hull + loadout, but the menu
# produces HullType/ComponentType OBJECTS while the wire is JSON. The
# mapping is by STABLE ID: every player hull has a unique `id`, and every
# catalog component has a unique `id` (the net self-test asserts both).
# A loadout is slot_name -> component id.
#
# The DESERIALIZER is defensive (the host trusts no one): an unknown hull
# id falls back to the default hull, an unknown component id or a
# slot/part-type mismatch falls back to that slot's stock part, and a
# structurally broken payload falls back to the stock loadout for the
# resolved hull. A hostile client can therefore only ever fly a valid
# catalog fit — never crash the host's sim.

def serialize_hull(hull):
    """HullType -> its stable id (the 'hull' field of join/welcome)."""
    return hull.id


def deserialize_hull(hid):
    """hull id -> HullType; unknown ids fall back to the default hull."""
    for h in PLAYER_HULLS:
        if h.id == hid:
            return h
    return DEFAULT_HULL


def serialize_loadout(loadout):
    """slot_name -> ComponentType dict -> slot_name -> component id."""
    return {slot: comp.id for slot, comp in loadout.items()}


def deserialize_loadout(hull, lod):
    """component-id dict -> a VALID slot_name -> ComponentType fit for
    `hull`. Every slot is filled: known, compatible parts are kept;
    anything else (unknown id, wrong slot type, missing slot, non-dict
    payload) falls back to the slot's stock part."""
    stock = default_loadout(hull)
    if not isinstance(lod, dict):
        return stock
    out = {}
    for slot in hull.slots:
        comp = lod.get(slot.name)
        ok = False
        if isinstance(comp, str):
            for opt in COMPONENT_CATALOG[slot.slot_type]:
                if opt.id == comp:
                    out[slot.name] = opt
                    ok = True
                    break
        if not ok:
            out[slot.name] = stock[slot.name]
    return out


# --- connection: non-blocking socket + send buffer + per-frame poll --------

class Connection:
    """A non-blocking TCP connection with a send buffer and a per-frame poll.

    Lifecycle:
      1. Built around a connected socket (from `Host.accept_one` or
         `connect`). It starts BLOCKING so the join/welcome handshake can use
         plain `send`/`recv`.
      2. `set_nonblocking()` flips it for the game loop.
      3. Each frame the game loop calls `drain_send()` (flush the send buffer)
         and `poll()` (drain the socket into the receive buffer and return
         every complete message).

    No threads, no shared state: the socket is only ever touched from the
    game-loop thread.
    """

    def __init__(self, sock):
        self.sock = sock
        self._send_buf = []        # encoded frames (bytes), oldest first
        self._recv_buf = b""       # trailing partial frame, carried across polls
        self.closed = False        # True once the peer went away (or we closed)

    # -- mode ---------------------------------------------------------------
    def set_nonblocking(self):
        """Flip the socket to non-blocking (call once, after the handshake)."""
        self.sock.setblocking(False)

    def set_timeout(self, timeout):
        """Set a socket timeout (seconds, or None for blocking). The
        handshake uses a short timeout so its wait loops can check their
        deadline; the game loop never sets one (non-blocking instead)."""
        self.sock.settimeout(timeout)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # -- send side ----------------------------------------------------------
    def send(self, msg):
        """Queue one message dict for sending (encoded into the send buffer).

        Does NOT touch the socket — the bytes go out on the next
        `drain_send()`, so a full socket buffer never blocks the game loop.
        """
        self._send_buf.append(encode_frame(msg))

    def drain_send(self):
        """Flush as much of the send buffer as the socket will take.

        In non-blocking mode a full kernel buffer raises BlockingIOError; the
        remaining frames stay queued for the next frame. Returns the number of
        frames sent this call.

        CRITICAL (Session 6.9): a frame larger than the free send-buffer
        space is sent in PARTS — `sendall` on a non-blocking socket sends
        what fits, raises BlockingIOError, and the UNSENT REMAINDER must be
        kept (not the whole frame). Re-sending the whole frame duplicates the
        bytes already on the wire and corrupts the length-prefixed stream
        (the client then fails to decode a frame). This is invisible on
        loopback (the buffer rarely fills) but happens on a real LAN when a
        burst of ~30 KB snapshots outruns the client's reads — e.g. the
        heavy explosion/game-over frame at a ship's death.
        """
        sent = 0
        while self._send_buf:
            frame = self._send_buf[0]
            try:
                # Send the frame in parts until it is fully out or the
                # buffer is full. `send` returns the number of bytes written;
                # the unsent tail stays at the front of the queue.
                view = memoryview(frame)
                while view:
                    n = self.sock.send(view)
                    view = view[n:]
                del self._send_buf[0]
                sent += 1
            except (BlockingIOError, socket.timeout):
                # Buffer full (or timed out): keep the UNSENT REMAINDER of
                # this frame (view now points at it) and stop — the rest of
                # the queue is untouched.
                if view:
                    self._send_buf[0] = bytes(view)
                break
            except ConnectionError:
                # The peer is gone (RST / reset / broken pipe): mark the
                # connection closed so the game loop notices via self.closed
                # and stop trying to send. (Session 6.7 — see poll().)
                self.closed = True
                break
        return sent

    # -- receive side -------------------------------------------------------
    def poll(self):
        """Drain the socket into the receive buffer and return every complete
        message (a list of dicts, in arrival order).

        On a BLOCKING socket (the handshake phase) `recv` blocks until a byte
        arrives, so `poll` blocks until at least one complete message is in
        hand. On a NON-BLOCKING socket (the game loop) `recv` raises
        BlockingIOError when nothing is ready — that means "no more right
        now", NOT a close, so `poll` returns whatever complete frames are
        buffered. A b"" return (either mode) is the peer's FIN: the
        connection is closed and `self.closed` is set so the game loop can
        notice without inspecting the socket itself.
        """
        try:
            data = self.sock.recv(65536)
        except BlockingIOError:
            # Nothing ready right now (non-blocking mode): not a close.
            msgs, self._recv_buf = extract_frames(self._recv_buf)
            return msgs
        except socket.timeout:
            # A recv TIMEOUT is not a close (the handshake sets a short
            # timeout so its wait loops can check their deadline): just
            # return whatever complete frames are already buffered.
            msgs, self._recv_buf = extract_frames(self._recv_buf)
            return msgs
        except ConnectionError:
            # A real socket error (RST / reset / aborted): the peer is gone.
            # Treat it as a close (like a FIN) so the game loop notices via
            # self.closed instead of crashing. (Session 6.7: the e2e test
            # surfaced this — a peer that closes while the other side still
            # has unread data makes the kernel send a RST, and the other
            # side's recv raises ConnectionResetError rather than returning
            # b"".)
            self.closed = True
            msgs, self._recv_buf = extract_frames(self._recv_buf)
            return msgs
        if data:
            self._recv_buf += data
        else:
            # b"" from recv means the peer closed the connection.
            self.closed = True
        msgs, self._recv_buf = extract_frames(self._recv_buf)
        return msgs


# --- connection setup (blocking phase, BEFORE the game loop) ---------------

class Host:
    """The host's listening socket: bind 0.0.0.0:port, wait for ONE client.

    `accept_one` blocks until a client connects (this is the connection-setup
    phase, which happens before the game loop), then returns a `Connection`
    wrapped around the accepted socket. The host is 2P-only for v1, so it
    serves exactly one client and then stops listening. Pass port=0 to let the
    OS pick a free ephemeral port (read it back via `sock.getsockname()[1]`).
    """

    def __init__(self, port):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.listen(1)

    def accept_one(self, timeout=None):
        """Block until a client connects; return (Connection, (ip, port)).

        With `timeout` (seconds), return (None, None) when no client
        connects in time — the host loop calls this in a loop, pumping
        pygame events between attempts, so ESC/QUIT can cancel the wait
        (the connection phase still happens before the game loop; only
        the LISTEN socket is timed, the accepted connection is not).
        """
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            conn_sock, addr = self.sock.accept()
        except socket.timeout:
            return None, None
        finally:
            self.sock.settimeout(None)
        return Connection(conn_sock), addr

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def connect(ip, port, timeout=10.0):
    """The client's connect: return a `Connection` to the host at ip:port.

    Blocking (connection-setup phase). Raises socket.error on refusal.
    """
    s = socket.create_connection((ip, port), timeout=timeout)
    return Connection(s)


# --- the join/welcome handshake (blocking, before the game loop) -----------

def do_handshake_client(conn, hull, loadout, timeout=30.0):
    """Client side: send `join` (the client's hull/loadout -> player 1) and
    block until the host's `welcome` (the host's hull/loadout -> player 0)
    arrives. Returns the host's (hull, loadout), or (None, None) when the
    host goes away or `timeout` seconds pass (the caller bails to the menu).

    The socket is still blocking here, so the join is fully written (via
    `drain_send`'s blocking sendall) before we wait on the welcome — a clean
    request/response with no deadlock. A short recv timeout lets the wait
    loop check its deadline without a thread.
    """
    conn.send({"type": T_JOIN, "hull": hull, "loadout": loadout})
    conn.drain_send()
    conn.set_timeout(0.05)
    deadline = time.monotonic() + timeout
    while not conn.closed:
        for m in conn.poll():
            if m.get("type") == T_WELCOME:
                return m["hull"], m["loadout"]
        if time.monotonic() >= deadline:
            return None, None
    return None, None


def do_handshake_host(conn, host_hull, host_loadout, timeout=30.0):
    """Host side: block until the client's `join` arrives, send the `welcome`
    (the host's hull/loadout -> player 0), and return the client's
    (hull, loadout) so the host can build player 1.

    Returns (None, None) when the client goes away before joining, or when
    no join arrives within `timeout` seconds (a client that connected but
    never sent a join must not hold the host's game forever).
    """
    conn.set_timeout(0.05)
    deadline = time.monotonic() + timeout
    while not conn.closed:
        for m in conn.poll():
            if m.get("type") == T_JOIN:
                conn.send({"type": T_WELCOME,
                           "hull": host_hull, "loadout": host_loadout})
                conn.drain_send()
                return m["hull"], m["loadout"]
        if time.monotonic() >= deadline:
            return None, None
    return None, None


# ---------------------------------------------------------------------------
# Self-test (headless). Exercises the framing in isolation AND a real
# loopback TCP round-trip of the handshake + one snapshot. Run:
#     python -m ship5.net
# ---------------------------------------------------------------------------

class _Keys:
    """Minimal stand-in for pygame.key.get_pressed() (all keys released)."""
    def __init__(self, pressed=None):
        self.p = pressed or {}
    def __getitem__(self, k):
        return self.p.get(k, 0)


def _self_test():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import time
    import threading
    import pygame
    from .config import WIDTH, HEIGHT
    from .fog import make_light_texture
    from .game import Game, STEP
    from .ai_enemy import AIEnemy
    from .asteroid import Asteroid

    ok = True

    # --- 1. framing round-trip (pure, no socket) ---
    m1 = {"type": T_SNAP, "sim_time": 0.5, "snap": [1, 2, 3]}
    m2 = {"type": T_INPUT, "inp": {"turn": 1.0, "fire": True}}
    frames, rest = extract_frames(encode_frame(m1) + encode_frame(m2))
    if frames == [m1, m2] and rest == b"":
        print("PASS: framing round-trip (two frames, one buffer)")
    else:
        ok = False
        print("FAIL: framing round-trip ->", frames, rest)

    # A frame split across two buffers must yield nothing until complete.
    full = encode_frame(m1)
    partial, rest2 = extract_frames(full[:5])
    if partial != [] or rest2 != full[:5]:
        ok = False
        print("FAIL: partial frame should yield nothing")
    frames2, rest3 = extract_frames(rest2 + full[5:])
    if frames2 == [m1] and rest3 == b"":
        print("PASS: framing reassembly across partial buffers")
    else:
        ok = False
        print("FAIL: framing reassembly ->", frames2, rest3)

    # --- 1b. congested send: a frame larger than the free buffer space must
    # be sent in PARTS without duplicating bytes (Session 6.9). A fake
    # socket whose send buffer holds only `capacity` bytes; recv() frees
    # space, simulating the peer reading. The whole stream must arrive
    # byte-identical and decode to the exact messages sent.
    class _Congested:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buf = b""
            self.rpos = 0
        def setblocking(self, b):
            pass
        def send(self, data):
            space = self.capacity - (len(self.buf) - self.rpos)
            if space <= 0:
                raise BlockingIOError
            take = min(len(data), space)
            self.buf += bytes(data[:take])
            return take
        def recv(self, n):
            end = min(self.rpos + n, len(self.buf))
            out = self.buf[self.rpos:end]
            self.rpos = end
            return out
        def close(self):
            pass

    payload = "x" * 50000            # ~50 KB frame, like a real snapshot
    frames_sent = [{"type": T_SNAP, "data": payload} for _ in range(3)]
    intact = True
    for cap in (200, 500, 1000):     # all smaller than one frame
        sock = _Congested(cap)
        conn = Connection(sock)
        conn.set_nonblocking()
        for f in frames_sent:
            conn.send(f)
        expected = b"".join(encode_frame(f) for f in frames_sent)
        received = b""
        for _ in range(100000):
            conn.drain_send()
            received += sock.recv(65536)
            if len(received) >= len(expected):
                break
        if received != expected:
            intact = False
            print("FAIL: congested send (capacity %d) corrupted the stream: "
                  "received %d bytes, expected %d"
                  % (cap, len(received), len(expected)))
    if intact:
        print("PASS: congested send — frames larger than the buffer arrive "
              "byte-identical (no duplication)")
    else:
        ok = False

    # --- 1c. receive resync: a corrupted region (duplicated bytes) must be
    # skipped, not raised — the stream re-aligns at the next valid frame
    # (Session 6.9 safety net).
    good = [{"type": T_SNAP, "data": "a" * 100},
            {"type": T_SNAP, "data": "b" * 100},
            {"type": T_SNAP, "data": "c" * 100}]
    stream = b"".join(encode_frame(f) for f in good)
    # Corrupt the middle: duplicate a 50-byte chunk inside frame 2.
    corrupted = stream[:120] + stream[120:170] + stream[170:]
    msgs, rest = extract_frames(corrupted)
    if (rest == b"" and len(msgs) >= 2
            and all("type" in m for m in msgs)):
        print("PASS: receive resync — corrupted region skipped, stream "
              "re-aligned (%d frames recovered)" % len(msgs))
    else:
        ok = False
        print("FAIL: receive resync -> %d msgs, rest %d bytes"
              % (len(msgs), len(rest)))

    # --- 2. ShipInput (de)serialization ---
    inp = ShipInput(turn=1.0, thrust_fwd=1.0, fire=True, laser_fire=True)
    back = deserialize_input(serialize_input(inp))
    if back == inp:
        print("PASS: ShipInput round-trip")
    else:
        ok = False
        print("FAIL: ShipInput round-trip ->", back)

    # --- 2b. hull/loadout wire mapping (Session 6.5) ---
    from .hulls import (PLAYER_HULLS, COMPONENT_CATALOG, DEFAULT_HULL,
                        default_loadout)
    # Ids must be unique for the mapping to be well-defined.
    hull_ids = [h.id for h in PLAYER_HULLS]
    if len(hull_ids) != len(set(hull_ids)):
        ok = False
        print("FAIL: duplicate hull ids:", hull_ids)
    for st, opts in COMPONENT_CATALOG.items():
        ids = [c.id for c in opts]
        if len(ids) != len(set(ids)):
            ok = False
            print("FAIL: duplicate component ids in", st, ids)
    # Round-trip: every hull + its stock loadout survives the wire.
    wire_ok = True
    for h in PLAYER_HULLS:
        if deserialize_hull(serialize_hull(h)) is not h:
            wire_ok = False
        stock = default_loadout(h)
        if deserialize_loadout(h, serialize_loadout(stock)) != stock:
            wire_ok = False
    if wire_ok:
        print("PASS: hull/loadout wire round-trip (all %d hulls)"
              % len(PLAYER_HULLS))
    else:
        ok = False
        print("FAIL: hull/loadout wire round-trip")
    # Defensive: a hostile/broken payload must resolve to a valid fit.
    if deserialize_hull("no_such_hull") is not DEFAULT_HULL:
        ok = False
        print("FAIL: unknown hull id should fall back to default")
    h0 = PLAYER_HULLS[0]
    bad = deserialize_loadout(h0, {"gun": "no_such_part",
                                   "reactor": "main_engine"})  # wrong slot
    if bad != default_loadout(h0):
        ok = False
        print("FAIL: bad loadout should fall back to stock")
    if deserialize_loadout(h0, "not a dict") != default_loadout(h0):
        ok = False
        print("FAIL: non-dict loadout should fall back to stock")
    if ok:
        print("PASS: hostile payload falls back to a valid stock fit")

    # --- 3. snapshot (de)serialization incl. the rng-state tuple ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=1234)
    for _ in range(60):
        g.update(STEP, _Keys())
    snap = g.snapshot()

    # The rng state must survive a real JSON round-trip as a proper tuple.
    as_json = json.loads(json.dumps(serialize_snapshot(snap)))
    d = deserialize_snapshot(as_json)
    rng = d[6]
    if (isinstance(rng, tuple) and len(rng) == 3
            and isinstance(rng[1], tuple) and rng[2] is None
            and rng == snap[6]):
        print("PASS: snapshot rng-state tuple survives JSON round-trip")
    else:
        ok = False
        print("FAIL: snapshot rng-state ->", type(rng))

    # A FRESH game restored from the deserialized snapshot must match.
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    g2 = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed=1234)
    g2.apply_snapshot(d)
    if g2.snapshot() == snap:
        print("PASS: snapshot -> JSON -> apply_snapshot is lossless")
    else:
        ok = False
        print("FAIL: snapshot JSON round-trip diverges")

    # --- 4. loopback TCP: handshake + one snapshot over a real socket ---
    host = Host(0)                      # OS picks a free ephemeral port
    port = host.sock.getsockname()[1]
    try:
        client = connect("127.0.0.1", port)
        hconn, _addr = host.accept_one()

        # One-process handshake: the client's join is written (blocking
        # sendall) before it blocks on the welcome, and the host blocks on the
        # join before it writes the welcome, so the two blocking calls need to
        # run concurrently — a single test-only thread (the game loop itself
        # stays thread-free).
        result = {}

        def client_side():
            result["host"] = do_handshake_client(
                client, hull="client_hull", loadout={"a": 1})

        t = threading.Thread(target=client_side)
        t.start()
        ch, cl = do_handshake_host(hconn, "host_hull", {"b": 2})
        t.join()

        if (ch == "client_hull" and cl == {"a": 1}
                and result.get("host") == ("host_hull", {"b": 2})):
            print("PASS: loopback join/welcome handshake")
        else:
            ok = False
            print("FAIL: handshake -> host saw", (ch, cl),
                  "client saw", result.get("host"))

        # One snapshot host->client over the (now non-blocking) socket.
        hconn.set_nonblocking()
        client.set_nonblocking()
        hconn.send({"type": T_SNAP, "sim_time": g.sim_time,
                    "snap": serialize_snapshot(snap)})
        hconn.drain_send()

        got = None
        for _ in range(400):
            for m in client.poll():
                if m.get("type") == T_SNAP:
                    got = m
            if got:
                break
            time.sleep(0.005)
        if got is not None:
            d2 = deserialize_snapshot(got["snap"])
            if got["sim_time"] == g.sim_time and d2[6] == snap[6]:
                print("PASS: loopback snapshot host->client round-trip")
            else:
                ok = False
                print("FAIL: loopback snapshot mismatch")
        else:
            ok = False
            print("FAIL: no snapshot arrived over loopback")

        hconn.close()
        client.close()
    finally:
        host.close()

    pygame.quit()
    print("NET SELF-TEST:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _self_test()