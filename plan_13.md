# Plan 13 — Dedicated Net Process (Host-Side)

**Status:** blueprint (source of truth for implementation).
**Scope:** Phases 1–2 only (host-side Net process + relay-test rework).
Phase 3 (client-side Net process) is out of scope — noted at the end.

---

## 1. Why this plan (the problem Option B left behind)

Before the multiprocess work, the sim and render were **threads** in one
process. Snapshots were sent from the **sim thread**, but that thread shared
the GIL with the render thread, so a heavy draw frame could starve the send.
Moving the sim to its own process fixed the **sim's step rate** (the sim no
longer competes for the GIL), but along the way the **snapshot send** ended up
back in the render process:

```
Option B send path (today):
  sim process ──net Queue──> _NetPump thread ──> NetWorker thread ──> socket
                               (render process)    (render process)
```

`_NetPump` and `NetWorker` are **threads in the render process**, so they share
its GIL with the draw. On the Mac host, a heavy draw frame can starve the
pump/worker for up to a GIL-switch interval, and the snapshot **send cadence**
inherits that jitter. The sim steps smoothly (its own process), but the *send*
is still coupled to the render process's GIL. That is the residue the client
feels as snap.

**The fix:** a dedicated **Net process** for the host. It has its own GIL,
shared only with itself. The draw can hold the render GIL for as long as it
wants and the net process keeps draining the queue and writing to the socket.
The render thread is **completely out of the send path**.

```
Plan 13 send path (target):
  sim process ──net Queue──> NET PROCESS ──> socket ──> client
                              (own GIL, no draw)
```

---

## 2. The 3-process architecture (host)

### Process tree

The render process (the existing main process) spawns **two** children:

```
RENDER PROCESS (main, pygame display)
  ├─ spawns ──> SIM PROCESS   (sim_process.py, headless, authoritative Game)
  └─ spawns ──> NET PROCESS   (net_process.py, headless, owns the socket)
```

The client is unchanged in Phases 1–2: it runs `run_client` in its own process
with the socket in its render process (via `NetWorker`). Only the **host** gets
a dedicated net process.

### What each process owns

| Concern | Render process | Sim process | Net process |
|---|---|---|---|
| pygame display / draw | YES | no | no |
| authoritative `Game` + `SimThread` (60 Hz steps) | no | YES | no |
| the socket (`Host`/`Connection`) | **no** (was: yes) | no | **YES** |
| listen / accept / handshake | no (polls status) | no | **YES** |
| snapshot send (snap/beam/echo → socket) | **no** (was: pump+worker) | no | **YES** |
| client input (socket → sim) | **no** (was: worker sink) | applies it | **YES** (forwards) |
| T_PONG reflection | **no** (was: worker sink) | no | **YES** |
| local input / T-V-G-R-F commands → sim | **YES** (cmd Pipe) | applies | no |
| model drain (latest-wins) + sfx + draw | **YES** | publishes | no |

### Channels (all from the spawn context)

The render process creates every channel and hands the appropriate end to each
child.

**Render → Sim** (one Pipe, small + low-rate + ordered):
- `("setup", host_hull, host_loadout, client_hull, client_loadout, seed)`
- `("input", ShipInput)` — the host's local input, per frame.
- `("command", name, arg)` — T/V/G/R/F, per key.
- `("quit",)` — shutdown.

**Sim → Render** (bounded Queue, `put_nowait` drop-when-full):
- `("model", render_model_dict)` — 60 Hz, on a step.
- `("ready",)` / `("crash", tb)` / `("done", ...)` — lifecycle.
- (maxsize 4 — models + lifecycle only, as today.)

**Sim → Net** (bounded Queue, `put_nowait` drop-when-full):
- `("snap", sim_time, snapshot)` — 60 Hz, on a step.
- `("beam", msg)` / `("echo", msg)` — T_BEAM / T_ECHO wire messages.
- (maxsize 16 — ~2 steps of wire traffic. A dropped snapshot is a network
  loss the client's interpolation buffer already handles.)
- **Drained by the NET process** (not the render process). This is the key
  change: the render thread no longer touches this queue at all.

**Net → Sim** (one Pipe, small + low-rate + ordered):
- `("remote_input", ShipInput)` — the client's input, applied to player 1.
- (The net process forwards the client's input here the moment it lands off
  the socket — replacing the render process's worker input sink.)

**Render ↔ Net** (one bidirectional Pipe):
- Net → Render (status): `("connected", client_hull, client_loadout)` (after
  the handshake), `("disconnected",)`, `("error", text)`.
- Render → Net (command): `("quit",)` (on ESC/QUIT / port-in-use / sim crash).
- The render process polls the status end (`conn.poll()`) — cheap, non-blocking.

### Data flow (steady state)

```
HOST local input ──cmd Pipe──> SIM (applies to player 0)
CLIENT input ──socket──> NET ──remote_input Pipe──> SIM (applies to player 1)
SIM steps @60Hz ──model Queue──> RENDER (draws the shell)
SIM steps @60Hz ──net Queue──> NET ──socket──> CLIENT
CLIENT input ──socket──> NET ──T_PONG──> socket ──> CLIENT (RTT)
```

The render thread's per-frame work is now: send local input + commands to the
sim, drain the model Queue (latest-wins), play sfx, draw the shell, and poll
the net status Pipe. **It never touches the socket, the net Queue, or the
snapshot send.**

---

## 3. Phase 1 — Host Net Process

Goal: move the host's socket + snapshot send + client-input receive into a
dedicated net process. The render thread is completely out of the send path.

Three chunks, each independently testable and sized for one 64k session.

### Chunk 1a — Create `net_process.py` (the net process spawn target)

A new top-level module (spawn re-imports it by name). The spawn target is a
top-level function:

```
net_process_main(status_conn, net_queue, remote_input_conn,
                 host_hull, host_loadout, port)
```

**Behavior:**
1. **Listen + accept + handshake** (blocking phase, before the game loop):
   - `Host(port)` → bind 0.0.0.0:port, listen(1).
   - Loop `accept_one(timeout=0.25)` until a client connects (or a `("quit",)`
     arrives on `status_conn`, or `status_conn` hits EOF — the render process
     died). This mirrors the render process's current waiting-screen accept
     loop, but in the net process.
   - `do_handshake_host(conn, serialize_hull(host_hull),
     serialize_loadout(host_loadout))` → the client's (hull, loadout).
   - On success: `status_conn.send(("connected", client_hull_id,
     client_loadout_dict))`. On the client vanishing before joining:
     `status_conn.send(("error", "client vanished before joining"))` and exit.
2. **Game loop** (after the handshake):
   - `conn.set_nonblocking()`.
   - Each iteration (~1 ms pace, like `NetWorker.WAIT`):
     - **Send side:** drain `net_queue` (`get_nowait`) → for each
       `("snap", t, snap)` send `{"type": T_SNAP, "sim_time": t,
       "snap": serialize_snapshot(snap)}`; for `("beam", msg)` / `("echo",
       msg)` send `msg`. Then `conn.drain_send()`.
       - **Latest-wins on snaps:** if multiple snaps are queued, send only the
         newest (drop the stale ones) — a backlog must not burst out. (Beams /
         echoes are events; send all of them.)
     - **Receive side:** `for m in conn.poll():`
       - `T_INPUT` → `remote_input_conn.send(("remote_input",
         deserialize_input(m["inp"])))`; if `m` has a `"tag"`, reflect
         `conn.send({"type": T_PONG, "tag": m["tag"]})`.
       - `T_RESPAWN` → forward to the sim as a command. **Decision:** the net
         process does not have a direct command channel to the sim (only the
         remote_input Pipe). So it forwards respawn over the remote_input Pipe
         as a distinct message: `("command", "respawn", 1)`. The sim process's
         bridge loop handles both `("remote_input", ...)` and `("command", ...)`
         on that Pipe (see Chunk 1c). This keeps the net→sim channel to ONE
         Pipe.
     - **Close detection:** if `conn.closed`, `status_conn.send(
       ("disconnected",))` and exit.
   - **Shutdown:** on `("quit",)` from the render process (or `status_conn`
     EOF), close the socket and exit cleanly.

**Notes:**
- The net process is **headless** — no pygame display, no audio. It imports
  only `net.py` (which does not import pygame) + `socket`/`time`/`multiprocessing`.
  It does NOT need `SDL_VIDEODRIVER=dummy` (it never imports pygame), but
  setting it is harmless insurance if a transitive import ever pulls pygame in.
- The net process **owns the socket entirely** after the handshake. The render
  process never touches it (no `conn.poll()` / `drain_send()` / `closed` in
  `run_host`).
- The net process is a **daemon** child: if the render process dies, the
  `status_conn` EOF + the OS reaping take it down. (The render process also
  sends `("quit",)` + joins on every exit path, like the sim process.)

**Verification gate (1a):** a unit test (`test_net_process.py`, new) that:
- Spawns the net process with a fake `status_conn` Pipe + a real `net_queue`
  + a real `remote_input_conn` Pipe + a free port.
- Connects a fake client (a real TCP socket) + does the client-side handshake.
- Asserts: the status Pipe gets `("connected", client_hull, client_loadout)`.
- Puts a `("snap", t, snap)` on the `net_queue`; asserts the fake client
  receives a `T_SNAP` with the right `sim_time` + pruned snap.
- Sends a `T_INPUT` (with a tag) from the fake client; asserts the
  `remote_input_conn` Pipe gets `("remote_input", ShipInput)` AND the fake
  client receives a `T_PONG` with the same tag.
- Closes the fake client; asserts the status Pipe gets `("disconnected",)`.
- Sends `("quit",)`; asserts the net process exits cleanly (join < 2 s).

### Chunk 1b — Rework `run_host` to spawn the net process

Changes to `run_host` (in `__main__.py`):

1. **Spawn the net process EARLY** (before the waiting screen), because it
   owns the listen socket + accept + handshake:
   - Create the channels: `status_conn` (bidirectional Pipe), `sim_net_q`
     (bounded Queue, maxsize 16), `remote_input_conn` (Pipe, net→sim).
   - `net_proc = _sim_ctx.Process(target=net_process_main, args=(
     status_child_conn, sim_net_q, remote_input_child_conn, menu.hull,
     menu.loadout, NET_PORT))`. Start it.
   - The render process keeps `status_parent_conn` (polls status, sends
     `("quit",)`).
2. **Waiting screen** (phase 1): instead of `host.accept_one(...)`, the render
   process polls `status_parent_conn.poll()`:
   - `("connected", ch, cl)` → break out of the wait (the client is joined).
   - `("error", text)` → notice + return to the menu.
   - Otherwise → draw the waiting screen, pump events (ESC/QUIT → send
     `("quit",)` to the net process + return).
   - **Remove** the `Host(NET_PORT)` / `accept_one` / `do_handshake_host` from
     the render process (the net process does all of it).
3. **Phase 3 (spawn the sim + build the shell):**
   - The client's hull/loadout now come from the status message (`ch`, `cl`),
     not from `do_handshake_host`.
   - Spawn the sim process with the **4-arg** form: `sim_process_main(
     sim_child_conn, sim_model_q, sim_net_q, remote_input_parent_conn)`.
     (The sim's `net_queue` is now drained by the net process; the
     `remote_input_conn` is the net→sim Pipe.)
   - **Remove** the `NetWorker` + `_NetPump` + `_client_input_sink` +
     `_sim_cmd_lock` (the worker's input sink is gone — the net process
     forwards client input directly to the sim).
   - The render process's `_sim_cmd_send` no longer needs the send-side lock
     (only the render thread sends on the cmd Pipe now — the worker thread is
     gone). **Decision:** keep the lock (cheap insurance) or remove it.
     Recommend: remove it (single sender now) but note it in the plan.
4. **Game loop (phase 4):**
   - **Remove** the `for m in worker.poll():` block (T_INPUT / T_RESPAWN
     handling) — the net process handles client input + respawn.
   - **Replace** `if worker.closed:` with a poll of the status Pipe:
     `if status_parent_conn.poll(): m = status_parent_conn.recv(); if
     m[0] == "disconnected": notice + return`.
   - The rest (local input → cmd Pipe, model drain, sfx, draw, telemetry) is
     unchanged.
5. **Finally block:**
   - **Remove** `net_pump.stop()` + `worker.stop()`.
   - **Add** net-process shutdown: `status_parent_conn.send(("quit",))`,
     close, `net_proc.join(timeout=2)`, `terminate()` if still alive.
   - Keep the sim-process shutdown (unchanged).
   - **Ordering:** stop the net process BEFORE the sim process (the net
     process writes to the sim's remote_input Pipe; the Pipe must not be
     written after the sim is gone). Actually — the net process writes to the
     remote_input Pipe (net→sim), and the sim process reads it. If the sim
     dies first, the net process's write to the remote_input Pipe would hit
     EOF and the net process should exit cleanly. So the ordering is: stop the
     net process first (it stops writing to the sim's Pipe), then stop the sim
     process. This mirrors the current "stop the worker before the sim" rule.

**Verification gate (1b):** `test_sim_process.py`'s M3 e2e (the real
`run_host` + a fake client over loopback) passes:
- The fake client's input reaches the sim (player 1 turns in the host's
  models) — via the net process + remote_input Pipe.
- The fake client receives snapshots + input echoes over the wire — via the
  net process.
- The host loop exits cleanly (no hang); the net process + sim process are
  both gone after exit.
- **New:** assert the render thread never touched the socket (the `NetWorker`
  + `_NetPump` are gone — verified by their absence in `run_host`).

### Chunk 1c — Rework `sim_process.py` (remote_input Pipe)

Changes to `sim_process.py`:

1. **Signature:** `sim_process_main(cmd_conn, model_queue, net_queue,
   remote_input_conn)`.
2. **The `_Worker`** is unchanged (it writes to `net_queue` — now drained by
   the net process).
3. **The bridge loop** drains **two** channels:
   - `cmd_conn` (render→sim): `("input", ...)` / `("command", ...)` /
     `("quit",)`. (The `("remote_input", ...)` message is REMOVED from the cmd
     Pipe — client input now comes via `remote_input_conn`.)
   - `remote_input_conn` (net→sim): `("remote_input", ShipInput)` →
     `game.remote_input = ShipInput` (the reference swap, same as today).
     `("command", name, arg)` → `st.command(name, arg)` (the net process
     forwards respawn here).
   - Both are drained each iteration (a `poll()` on each Pipe). The
     `remote_input_conn` EOF (the net process died) is treated like the
     `cmd_conn` EOF (the render process died) — shut down cleanly.
4. **The `("done", ...)` report** is unchanged (sent on `("quit",)` from the
   cmd Pipe).

**Verification gate (1c):** `test_sim_process.py`'s M2 test (input + commands
+ remote input) passes with the 4-arg form:
- Local input (cmd Pipe) drives player 0.
- Commands (cmd Pipe) toggle targeting / reset.
- Remote input (remote_input Pipe) drives player 1.
- Respawn (remote_input Pipe, `("command", "respawn", 1)`) is applied.

**Phase 1 complete when:** all of `test_sim_process` (M1 A/B + M2 + M3 pipe +
M3 e2e) + `test_snapshot` + `test_determinism` + `test_render_model` +
`test_sim_thread` + `test_m2b_local` pass, AND a live 2P run (Mac host, 4090
client) shows the client's `newest_stamp` delta staying ~17 ms with a ~0
snap tail **even under a heavy draw** (the F3 capture re-run).

---

## 4. Phase 2 — Relay Test Rework

### The challenge

`test_network.py`'s 7.4/7.5b impairment batteries wrap the host's connection
with a `ProxyConnection` **in the render process** (by patching `M.Host` /
`M.connect`). The relay is a single in-memory object shared by two proxies
(host role 0, client role 1), both in the same process.

In the 3-process model, the host's socket is in the **net process** (a child
of the render process). The host's proxy would be in the net process; the
client's proxy is in the render process (the client thread). The relay must
now **span two processes**.

### The cross-process relay design

The relay's shared state today:
- `pending[role]` — a list of `(due_time, frame)` tuples, written by the game
  thread (via `send`), read by the relay thread.
- `outbox[role]` — a `queue.Queue`, written by the relay thread, read by the
  game thread (via `poll`).

**Rework:** make `pending[role]` and `outbox[role]` **multiprocessing
Queues**. The relay thread runs in the **test process** (the parent — where
both `run_host` and `run_client` are called). The game threads (in the net
process and the render process) communicate with the relay thread via the
multiprocessing Queues:

- `pending[role]` → `multiprocessing.Queue`. The game thread (net process for
  role 0, render process for role 1) enqueues `(due_time, frame)` tuples via
  `send()`.
- The relay thread (test process) reads from `pending[0]` + `pending[1]`,
  maintains a **local buffer** of `(due_time, frame)` tuples (the not-yet-due
  ones), and delivers the due ones to `outbox[1-role]` (applying loss/stall).
  - The local buffer handles the non-monotonic due times (jitter): the relay
    thread reads frames in enqueue order, adds them to the buffer, and each
    iteration delivers all buffer entries whose `due_time <= now` (sorted by
    due time). This is O(n log n) per iteration — fine for the test (small n).
- `outbox[role]` → `multiprocessing.Queue`. The relay thread enqueues
  delivered frames. The game thread reads from `outbox[role]` via `poll()`.

The relay's **stats** (`sent` / `delivered` / `dropped` / `stall_count`) live
in the relay thread's process (the test process), so the test reads them
directly after `stop()` — no cross-process stat aggregation needed.

**The `ProxyConnection`** is unchanged in shape (it still wraps a real
`Connection`, passes through during the handshake, and switches to relay mode
on `set_nonblocking`). The only change is that its `send()` / `poll()` now
talk to the relay's **multiprocessing Queues** instead of in-process lists /
queues. The proxy in the net process (role 0) and the proxy in the render
process (role 1) each hold references to the relay's Queues (passed to them).

### How the relay is injected

The test creates the relay (in the test process) and passes its Queues to:
- **The net process** (role 0): via the spawn args. `run_host` reads a
  module-level test hook `M._NET_RELAY` (None in production, set by the test).
  If set, `run_host` passes the relay's role-0 Queues to the net process (via
  the spawn args), and the net process wraps its accepted `Connection` in a
  `ProxyConnection` (role 0) on `set_nonblocking`.
- **The render process** (role 1, the client thread): via the patched
  `M.connect` (the test already patches `M.connect` to return a
  `ProxyConnection` (role 1) — now the proxy holds the relay's role-1 Queues).

The `M._NET_RELAY` hook is a test-only injection point in production code
(analogous to the existing `debug_net` hook). It is `None` in production, so
the net process uses a plain socket.

### Chunk 2a — Rework `net_harness.py` (cross-process relay)

Changes to `net_harness.py`:
- `Relay.__init__`: `pending[role]` + `outbox[role]` become
  `multiprocessing.Queue` (from the spawn context). The relay thread runs in
  the creating process.
- `Relay._run`: read from `pending[0]` + `pending[1]` (multiprocessing
  Queues), maintain a local buffer of `(due_time, frame)` tuples, deliver the
  due ones to `outbox[1-role]` (applying loss/stall). The stall logic is
  unchanged.
- `ProxyConnection.send`: enqueue `(due_time, frame)` into
  `self._relay.pending[self._role]` (a multiprocessing Queue).
- `ProxyConnection.poll`: read from `self._relay.outbox[self._role]` (a
  multiprocessing Queue).
- The stats counters are unchanged (written by the relay thread, read by the
  test after `stop()`).

**Verification gate (2a):** a unit test that creates a relay (in the parent
process), spawns a child process that acts as role 0 (enqueues frames into
`pending[0]`, reads from `outbox[0]`), and the parent acts as role 1
(enqueues into `pending[1]`, reads from `outbox[1]`). Asserts: frames flow
both directions with the impairment applied (latency/jitter/loss/stall), and
the stats are correct.

### Chunk 2b — Rework `test_network.py` (inject the relay into the net process)

Changes to `test_network.py`:
- In `run()`, when `impairment is not None`:
  - Create the relay (in the test process).
  - Set `M._NET_RELAY = relay` (the test hook) so `run_host` passes the
    relay's role-0 Queues to the net process.
  - Patch `M.connect` to return a `ProxyConnection` (role 1) holding the
    relay's role-1 Queues (the client thread).
  - **Remove** the `M.Host` patch (the host's socket is in the net process,
    not the render process — the net process wraps its own connection in the
    role-0 proxy).
- After the run: `M._NET_RELAY = None`; `relay.stop()`; read the stats.

**Verification gate (2b):** `test_network.py`'s full battery (clean + 7.4a
40ms jitter + 7.4b snap loss + 7.4c 100ms stall + 7.5b jitter) is GREEN
(exit 0). The relay's stats (delivered / dropped / stall_count) are correct
for each battery.

**Phase 2 complete when:** `test_network.py` is GREEN (all batteries) + the
clean e2e checks (7.1 sim_time tracking, 7.2 angle bounds) still pass clean.

---

## 5. Key design decisions (locked)

1. **Socket ownership moves to the net process.** The net process does
   `Host(port)` / `accept_one` / `do_handshake_host`. The render process never
   touches the socket (no `Host` / `Connection` / `conn.poll()` /
   `conn.drain_send()` / `conn.closed` in `run_host`).
2. **The net process is spawned EARLY** (before the waiting screen), because
   it owns the listen socket. The render process's waiting screen polls the
   status Pipe (not `accept_one`).
3. **The sim's net Queue is drained by the net process** (not the render
   process). The render thread never touches the net Queue.
4. **Client input flows net→sim via a dedicated Pipe** (`remote_input_conn`),
   replacing the render process's worker input sink. The net process forwards
   the client's input the moment it lands off the socket (~1 ms), so the input
   reaches the sim within ~1 ms (same as today's worker sink).
5. **T_PONG is reflected in the net process** (not the render process), so the
   measured RTT is honest (a render-thread reflection would add a full frame).
6. **The net→sim channel is ONE Pipe** (`remote_input_conn`), carrying both
   `("remote_input", ShipInput)` and `("command", name, arg)` (respawn). This
   keeps the channel count minimal.
7. **The render↔net channel is ONE bidirectional Pipe** (status + quit).
8. **The relay is cross-process** (multiprocessing Queues for `pending` /
   `outbox`; the relay thread runs in the test process). The relay is injected
   into the net process via the `M._NET_RELAY` test hook.
9. **The net process is headless** (no pygame). It imports only `net.py` +
   stdlib.
10. **Shutdown ordering:** stop the net process BEFORE the sim process (the
    net process writes to the sim's remote_input Pipe; the Pipe must not be
    written after the sim is gone).

---

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| The net process's accept loop races the render process's waiting screen (both polling). | The net process owns the accept; the render process only polls the status Pipe. No race — the net process is the single acceptor. |
| The `M._NET_RELAY` test hook leaks into production. | It is `None` in production (the net process uses a plain socket). The hook is read once at spawn time. |
| The cross-process relay's local buffer grows unbounded (frames never become due). | The relay thread delivers all due frames each iteration; the buffer only holds not-yet-due frames (bounded by the max delay + jitter, ~60 ms). A stall window pauses delivery (frames queue in the buffer) — bounded by the stall duration. |
| The net process dies mid-game (crash). | The render process polls the status Pipe; a `status_conn` EOF (the net process died) is treated as a disconnect (notice + return to the menu). The net process sends `("error", tb)` on a crash (like the sim process's `("crash", tb)`). |
| The remote_input Pipe EOF (the net process died) crashes the sim process. | The sim process's bridge loop treats a `remote_input_conn` EOF like a `cmd_conn` EOF (shut down cleanly). |
| The net process's latest-wins snap drop loses a beam/echo. | Snaps are latest-wins (drop stale); beams/echoes are events (send all). The net process drains the queue fully each iteration, so beams/echoes are not dropped unless the queue is full (a network loss the client handles). |

---

## 7. Out of scope (Phase 3 — client-side Net process)

Phase 3 would move the **client's** socket into a dedicated net process (for
symmetry). The client's net work (receive snapshots, send input) is light and
was not the bottleneck, so Phases 1–2 (host-side only) are expected to kill
the snap the user is feeling. Phase 3 is deferred until Phases 1–2 are
verified in a live 2P run. If Phase 3 is later desired, the client's
`run_client` would spawn a net process (analogous to the host's), and the
client's `NetWorker` would be removed. The relay test would need a third
proxy (the client's net process) — a further rework of `net_harness.py`.

---

## 8. Session plan (64k context chunks)

| Session | Chunk | Deliverable | Gate |
|---|---|---|---|
| 1 | 1a | `net_process.py` + `test_net_process.py` | 1a unit test GREEN |
| 2 | 1b | `run_host` rework (spawn net process, poll status, remove worker/pump) | `test_sim_process` M3 e2e GREEN |
| 3 | 1c | `sim_process.py` rework (remote_input Pipe) | `test_sim_process` M2 + M3 e2e GREEN |
| 4 | — | **Live 2P run** (Mac host, 4090 client) + F3 capture re-run | `newest_stamp` delta ~17 ms, snap tail ~0, under a heavy draw |
| 5 | 2a | `net_harness.py` cross-process relay + unit test | 2a unit test GREEN |
| 6 | 2b | `test_network.py` relay injection | `test_network` full battery GREEN |
| 7 | — | **Final live 2P run** + full regression suite | all tests GREEN + the snap is gone |

Each session is sized to fit a 64k context window (one chunk + its test + the
relevant existing files).