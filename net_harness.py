"""Network-impairment harness (Session 7.4).

Test infrastructure that makes "wifi" testable in CI: it wraps the loopback
e2e's connection with a `ProxyConnection` + a relay thread that injects
latency, jitter, loss, and stalls — so the REAL `run_host` / `run_client`
run UNCHANGED over a deliberately degraded link.

Design (keeps the clean e2e checks clean):
  * The proxy PASSES THROUGH to the real socket during the blocking
    join/welcome handshake (before `set_nonblocking`), so the handshake is
    exactly the clean one.
  * When the game loop flips the socket to non-blocking (`set_nonblocking`),
    the proxy switches to RELAY mode: `send()` hands encoded frames to a
    shared relay thread, and `poll()` reads the frames the relay has
    delivered (after impairment). The real socket is then idle — the relay
    IS the transport.
  * The relay applies, per frame, per direction: a base latency + uniform
    jitter, an independent drop roll (direction-selectable), and a periodic
    stall window (no delivery while a stall is active).

The harness is OPT-IN per test: a clean e2e (no impairment) runs the real
`connect` / `Host` unchanged, so the clean-tuned checks (sim_time tracking,
the 7.2 angle bounds) keep running clean. Impaired batteries run their own,
impairment-appropriate checks.

This is pure test infrastructure: it depends on nothing from 7.5/7.6 and is
built FIRST so the "feel" sessions are tested against realistic conditions
from the start.
"""
import queue
import random
import threading
import time

from .net import encode_frame, extract_frames

__all__ = ["Impairment", "Relay", "ProxyConnection"]


class Impairment:
    """The impairment profile a relay applies to frames.

    latency     — base one-way delay, seconds (e.g. 0.04 for 40 ms).
    jitter      — uniform +/- jitter added to each frame's delay, seconds.
    loss        — per-frame drop probability in [0, 1] (independent).
    loss_dirs   — the set of SENDING roles the loss applies to: {0} =
                  snapshots (host->client), {1} = inputs (client->host),
                  {0, 1} = both (the harness's full capability).
    stall       — stall duration, seconds (0 = no stalls). While a stall is
                  active the relay delivers nothing (frames queue up and
                  burst out when it ends — a realistic wifi stall).
    stall_every — seconds between stall windows (ignored when stall == 0).
    seed        — RNG seed for a reproducible jitter/loss pattern.
    """
    def __init__(self, latency=0.0, jitter=0.0, loss=0.0, loss_dirs=(0, 1),
                 stall=0.0, stall_every=5.0, seed=None):
        self.latency = latency
        self.jitter = jitter
        self.loss = loss
        self.loss_dirs = set(loss_dirs)
        self.stall = stall
        self.stall_every = stall_every
        self.rng = random.Random(seed)

    def delay(self):
        """This frame's one-way delay (latency + uniform jitter, >= 0)."""
        d = self.latency
        if self.jitter > 0.0:
            d += self.rng.uniform(-self.jitter, self.jitter)
        return max(0.0, d)

    def should_drop(self, send_role):
        """Independent per-frame drop roll for a frame SENT by `send_role`."""
        return (self.loss > 0.0 and send_role in self.loss_dirs
                and self.rng.random() < self.loss)


class Relay:
    """A shared impairment relay between two `ProxyConnection`s.

    Role 0 is the host's proxy, role 1 the client's (assigned when the
    proxies are built). Each proxy's `send()` enqueues a frame into
    `pending[role]`; the relay thread moves DUE frames to the OTHER role's
    `outbox` (applying latency/jitter/loss) and the other proxy's `poll()`
    drains its outbox. A stall window pauses all delivery.

    Thread safety: `pending[role]` is guarded by a lock (written by the game
    thread via `send`, read by the relay thread); `outbox[role]` is a
    `queue.Queue` (written by the relay thread, read by the game thread via
    `poll`). The stats counters are written by the relay thread only and
    read by the test after `stop()` (which joins the relay thread).
    """
    def __init__(self, impairment):
        self.imp = impairment
        self.pending = [[], []]            # role -> [(due_time, frame)]
        self.outbox = [queue.Queue(), queue.Queue()]
        self.proxies = [None, None]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._start = None
        self._stall_active = False
        # stats (for the test's assertions + PASS lines)
        self.sent = [0, 0]                 # role -> frames enqueued by role
        self.delivered = [0, 0]            # role -> frames delivered to role
        self.dropped = [0, 0]              # role -> frames role sent, dropped
        self.delivered_times = [[], []]    # role -> [monotonic delivery time]
        self.stall_count = 0               # stall windows entered

    # -- proxy registration ------------------------------------------------
    def make_proxy(self, real_conn, role):
        p = ProxyConnection(real_conn, self, role)
        self.proxies[role] = p
        return p

    def start(self):
        """Start the relay thread (idempotent; called on the first flip)."""
        if self._thread is None:
            self._start = time.monotonic()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the relay thread and join it (call before reading stats)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- game-thread side ---------------------------------------------------
    def enqueue(self, role, frame):
        """Queue a frame SENT by `role` for impaired delivery to the peer."""
        with self._lock:
            self.pending[role].append(
                (time.monotonic() + self.imp.delay(), frame))
        self.sent[role] += 1

    def mark_closed(self, role):
        """The peer at `role` went away: flag the OTHER proxy closed."""
        other = 1 - role
        if self.proxies[other] is not None:
            self.proxies[other]._closed = True

    # -- relay thread -------------------------------------------------------
    def _in_stall(self, now):
        if self.imp.stall <= 0.0:
            return False
        phase = (now - self._start) % self.imp.stall_every
        return phase < self.imp.stall

    def _run(self):
        while not self._stop.is_set():
            now = time.monotonic()
            if self._in_stall(now):
                if not self._stall_active:
                    self.stall_count += 1
                    self._stall_active = True
                self._stop.wait(0.002)
                continue
            self._stall_active = False
            for role in (0, 1):
                with self._lock:
                    due = [f for (t, f) in self.pending[role] if t <= now]
                    self.pending[role] = [(t, f)
                                          for (t, f) in self.pending[role]
                                          if t > now]
                for frame in due:
                    if self.imp.should_drop(role):
                        self.dropped[role] += 1
                        continue
                    self.outbox[1 - role].put(frame)
                    self.delivered[1 - role] += 1
                    self.delivered_times[1 - role].append(time.monotonic())
            self._stop.wait(0.002)


class ProxyConnection:
    """Wraps a real `Connection`: pass-through during the blocking handshake,
    relay-routed once the game loop flips the socket to non-blocking.

    Exposes the same surface the game loop + handshake use: `send`,
    `drain_send`, `poll`, `set_nonblocking`, `set_timeout`, `close`,
    `closed`, `sock`. The real socket is used ONLY for the handshake; after
    the flip the relay is the transport and the real socket sits idle until
    `close()`.
    """
    def __init__(self, real_conn, relay, role):
        self._real = real_conn
        self._relay = relay
        self._role = role
        self._live = False
        self._closed = False
        self._recv_buf = b""

    # -- pass-through surface (handshake phase) ----------------------------
    @property
    def sock(self):
        return self._real.sock

    def set_timeout(self, timeout):
        if not self._live:
            self._real.set_timeout(timeout)

    def close(self):
        self._closed = True
        self._relay.mark_closed(self._role)
        if self._real is not None:
            self._real.close()

    # -- the flip -----------------------------------------------------------
    def set_nonblocking(self):
        """The game loop's flip: switch from pass-through to relay mode and
        start the relay thread (idempotent)."""
        self._live = True
        self._relay.start()

    # -- send side ----------------------------------------------------------
    def send(self, msg):
        if not self._live:
            self._real.send(msg)
            return
        self._relay.enqueue(self._role, encode_frame(msg))

    def drain_send(self):
        if not self._live:
            return self._real.drain_send()
        return 0          # frames already handed to the relay

    # -- receive side -------------------------------------------------------
    def poll(self):
        if not self._live:
            return self._real.poll()
        while True:
            try:
                frame = self._relay.outbox[self._role].get_nowait()
            except queue.Empty:
                break
            self._recv_buf += frame
        msgs, self._recv_buf = extract_frames(self._recv_buf)
        return msgs

    @property
    def closed(self):
        if not self._live:
            return self._real.closed
        return self._closed