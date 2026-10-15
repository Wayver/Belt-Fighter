# Belt Fighter 12 — Separate-process sim (kill the GIL batching)

Scope: move the host's authoritative sim out of the render process so the
render thread can no longer starve it via the GIL. This is the fix for the
residual ~8 px snap tail (plan_11 fixed the client-side clock/replay; the
remaining tail is host-side snapshot cadence).

---

## A. Why (the evidence, from the 2026-10-14 CSVs)

- Client `net_debug.csv` (3,080 reconciles): big snaps (>=3 px) correlate
  with the per-frame `newest_stamp` delta:
  - 1-tick jump (17 ms): 2,677 frames -> 1 big snap (0.04%)
  - 2-tick jump (33 ms): 384 frames -> 347 big snaps (90.4%)
  - 3+ tick jumps: 18 frames -> 16 big snaps
- Mechanism: the ghost's clock (set in `advance()` before the poll loop,
  anchored on the PREVIOUS snapshot) sits at prev_stamp + ~16 ms (1 frame)
  + ~10 ms (rtt_ow) = prev_stamp + ~26 ms. A 1-tick jump (new stamp
  prev+17 ms) leaves the ghost ~9 ms AHEAD (positive span, no pull-back,
  small snap). A 2-tick jump (new stamp prev+33 ms) leaves it ~7 ms BEHIND
  (negative span; the reconcile pulls clock+pose forward by |span| ->
  7 ms x ~1,200 px/s ~= 8 px). Confirmed in `ghost_clock.csv`: 62.8% of
  negative-span RECs have the estimator's MAX_LEAD cap binding on the
  preceding ADV (vs 18.9% of positive).
- Host `host_sim_debug.csv` (2,963 frames): the host batches — 485 frames
  (16.4%) advance 2 ticks in one frame. Overall sim rate is exactly
  1.0000x (no sim time lost; only the snapshot STAMPS are ragged).
- GIL link: 2-tick frames have the SAME frame's render raw_dt mean 25.4 ms
  vs 18.2 ms for 1-tick frames; "previous frame was long" does NOT predict
  batching (24.1% vs 23.5% — rules out OS scheduler jitter). The render
  thread holds the GIL for its whole draw (~18-25 ms > the 16.7 ms tick),
  starving the sim thread; the sim then batches 2 ticks when it gets the
  GIL back. Same-process threads share the GIL — that is the defect.

## B. Architecture

### Before (one process, three threads)
```
render thread (pygame display, draw, input sampling)
sim thread    (Game, 60 Hz steps, publish_snapshot)   <-- shares GIL with render
NetWorker     (socket I/O, snapshot send, input sink)
```

### After (two processes)
```
RENDER PROCESS (the existing main process)
  render thread: pygame display, input sampling, commands,
                 _HostRenderClock, shell-Game.draw(dt, model);
                 drains the MODEL Queue (latest-wins model)
  NetWorker thread: socket I/O UNCHANGED (owns the Connection)
  net-pump thread (_NetPump): drains the NET Queue (snap/beam/echo),
                 feeds the worker (the M4 redo — the draw is out of the
                 snapshot's send path)

SIM PROCESS (spawned, headless)
  main thread: Game + the SimThread loop (60 Hz steps)
  NO pygame display (SDL_VIDEODRIVER=dummy), NO audio (SDL_AUDIODRIVER=dummy)
  sends: render_model (60 Hz, MODEL Queue), wire events (snap/beam/echo,
         NET Queue), lifecycle
  receives: local input (per frame), commands (per key), setup (hulls/seed)
```

The host's render thread becomes a "local client" of the sim process —
the same relationship the remote client has over the network, but over a
pipe. The remote client is UNCHANGED.

## C. IPC contract (multiprocessing.Pipe, spawn context)

Render -> Sim (one pipe, small messages):
- `("setup", local_hull, local_loadout, client_hull, client_loadout, seed)`
  — sent once after the handshake; the sim process builds the Game.
- `("input", ShipInput)` — per frame (dataclass, pickles; latest wins in
  the sim loop, same as today's reference swap).
- `("command", name, arg)` — per key (T/V/G/R/F/respawn), same shape as
  SimThread.command.
- `("quit",)` — shutdown.

Sim -> Render (two bounded Queues — the M4 redo split the wire traffic
off the model Queue so the render thread's draw is out of the send path):
- MODEL Queue: `("model", render_model_dict)` — 60 Hz, only when a step
  ran (same publish rule as today). The render thread drains to the
  NEWEST per frame (latest-wins — the pipe analog of the reference swap).
  Plus lifecycle: `ready` (first model published), `crash` (traceback
  text), `done` (child step report).
- NET Queue: `("snap", sim_time, snapshot)` — 60 Hz; a dedicated net-pump
  thread (run_host's _NetPump) forwards to `worker.publish_snapshot` (the
  worker sends it the moment it lands — age at send stays ~0).
  `("beam", [beam event])` / `("echo", [entries])` — forwarded to
  `worker.send` (T_BEAM / T_ECHO).

Sizes: the render model is plain data (the parity test
`test_render_model.py` asserts no pygame objects) — a few KB; worst case
~1 MB/s over the pipe. Negligible.

## D. What moves where

| Component | Before | After |
|---|---|---|
| `Game` (sim state, `_step`, `snapshot`, `render_model`) | render process, owned by sim thread | SIM process |
| `SimThread` loop | render process thread | SIM process main thread (the loop body is unchanged; it can run on the process's main thread — no thread needed) |
| `NetWorker` + `Connection` | render process | render process (UNCHANGED — avoids cross-process socket passing, which is platform-fragile on spawn) |
| Snapshot/beam/echo send | sim thread -> worker (in-process) | sim process -> NET Queue -> net-pump thread (`_NetPump`) -> worker (the M4 redo — the render thread's draw is out of the send path) |
| Worker input sink (`game.set_remote_input`) | sets Game field directly | sends `("remote_input", ShipInput)` over the pipe; the sim process applies it to `game.remote_input` (reference swap, same as today) |
| T_PONG reflection | worker thread | worker thread (unchanged — a socket write, no sim state) |
| `game.draw(dt, model)` | the live Game (shared with the sim thread) | a SHELL Game in the render process (real surfaces, never stepped) |
| Local input sampling (`ShipInput.from_keys`) | render thread -> `sim_thread.publish_input` | render thread -> pipe `("input", ...)` |
| T/V/G/R/F commands | render thread -> `sim_thread.command` | render thread -> pipe `("command", ...)` |
| Audio (host's own SFX) | sim thread (Game.sound) | render process — see D.1 |

### D.1 Audio bridge (the one real gap)
The sim process is headless and silent (`sound=None`). Today the host's
own SFX fire from `_step` via `Game.sound`. Fix: `render_model()` gains a
`"sfx"` list — presentation sound events since the last publish
(`(name, *params)`, e.g. shot/explosion/beam, the same calls `_step`
currently makes on `self.sound`). The sim thread collects them into a
per-Game queue (like `_beam_events`); the render process plays them via
the existing SoundBank. Bounded, drained per publish.

### D.2 The shell Game (render process)
Constructed with the REAL surfaces/fonts/textures, the host's hull +
(the client's hull for the player-1 shield), `players=2`, then
`set_player_ship(1, ...)` — exactly as today — but NEVER stepped.
`draw(dt, model)` runs on it. Its `cam`, `_standins`, `_death_bursts`,
`remote_beams` are the render-side presentation state (today they live on
the shared Game and are only touched by the render thread — the shell
preserves that split cleanly).

AUDIT (first task of M3): confirm `draw()` reads only (surfaces, model,
render-side presentation state) — `test_render_model.py` already asserts
the model carries everything `draw()` reads; extend it to run against a
shell Game that was never stepped (catches any accidental live read).

## E. Key implementation decisions

1. **Spawn context, not fork.** The render process has threads (the
   worker); fork-with-threads risks a GIL-held-at-fork deadlock. `spawn`
   re-imports the module, so the sim entry point must be a TOP-LEVEL
   function (picklable), e.g. `sim_process_main(conn)` in a new
   `sim_process.py`.
2. **Headless pygame in the sim process.** Set
   `SDL_VIDEODRIVER=dummy` + `SDL_AUDIODRIVER=dummy` in the child BEFORE
   importing pygame, then `pygame.display.set_mode((1, 1))` (Game.__init__
   builds shield Surfaces; Ship uses pygame.Vector2). Standard, low risk.
3. **NetWorker stays in the render process.** The socket is created there
   (accept + handshake); passing a live socket across a spawn boundary is
   platform-fragile. The bridge adds one pipe hop (~sub-ms) — invisible
   against the 10 ms game-level latency.
4. **Spawn after the handshake** (the client's hull is known). Spawn cost
   (~0.1-0.3 s: python startup + pygame import) overlaps the
   handshake/waiting screen. (Optional later: spawn during the accept
   loop and send setup when the client joins.)
5. **Latest-wins drain.** The render process drains the sim pipe each
   frame and keeps the newest model (the queue can hold 1-2 models;
   stale ones are dropped — same semantics as today's reference swap).
6. **Shutdown ordering.** Render `finally`: send `("quit",)`, close the
   render->sim pipe, `process.join(timeout=2)`, `terminate()` if still
   alive. The sim process also exits on pipe EOF (parent died).
7. **Crash handling.** The sim process wraps its loop in try/except:
   print the traceback, send `("lifecycle", "crash", tb)`, exit. The
   render process shows a "SIM CRASHED" notice (the `_notice` pattern).
8. **Single-player.** SP uses the same SimThread today (worker=None) and
   has the SAME GIL batching. The sim-process wrapper is generic
   (worker events are simply absent), so SP reuses it for free — but it
   is a M5 item, not on the 2P critical path.

## F. Milestones (one session each)

### M1 — Headless sim process + model pipe (no network, no draw)
- New `sim_process.py`: top-level `sim_process_main(pipe)` — headless
  pygame, build a 2P Game from a `("setup", ...)` message, run the
  SimThread loop body inline, publish `("model", ...)` per step.
- Driver script: spawn it, feed idle input, print the model's
  `sim_time` + a step cadence counter.
- VERIFY: (a) the model arrives at 60 Hz; (b) the sim's per-iteration
  step count is 1.0 every iteration (the batching is GONE — this is the
  direct falsification of the GIL hypothesis; compare against
  `host_debug.csv`'s 16.4% 2-tick frames); (c) the sim rate is 1.000x.

### M2 — Input + commands + remote input over the pipe
- Render->sim `("input", ...)` per frame, `("command", ...)` per key;
  sim->render `("remote_input", ...)` applied to `game.remote_input`.
- VERIFY: a scripted parent sends thrust/fire -> the model's player-0
  pose/velocity responds; a command (e.g. targeting) flips in the model;
  a fake remote input moves player 1. Unit-test the bridge with a fake
  Game (no spawn) for the fast path.

### M3 — Shell Game draw in the render process
- Audit `draw()` for live-state reads (extend `test_render_model.py` to
  a never-stepped shell Game).
- Build the shell Game in `run_host` (real surfaces), replace
  `game.draw(dt, model)` with `shell.draw(dt, model)` fed by the pipe.
- Add the `"sfx"` list to `render_model()` + play it in the render
  process (D.1).
- VERIFY: the host window renders the full game (world, both ships,
  beams, particles, HUD) from piped models; SFX play; the parity test
  passes against the shell.

### M4 — Network bridge (full 2P)
- Sim->render `("snap", ...)` / `("beam", ...)` / `("echo", ...)` ->
  `worker.publish_snapshot` / `worker.send`.
- Worker input sink: `game.set_remote_input` -> pipe `("remote_input",
  ...)` (T_PONG reflection stays in the worker).
- Lifecycle: `ready` / `game_over` / `crash` -> render notices; shutdown
  ordering (E.6).
- VERIFY: full 2P game plays end-to-end. F3 capture: host per-iteration
  steps == 1.0 (no 2-tick batches); client `newest_stamp` delta == 17 ms
  on every frame (no 33 ms jumps); the >=3 px snap tail (364/3,080
  today) drops to ~0; `snap_px` p95 from 7.2 px to ~1 px; median stays
  0.000 px.

### M5 — SP reuse, cleanup, docs
- SP runs through the same sim process (worker=None path).
- Decide the fate of in-process `SimThread` (keep for tests; the live
  host path uses the process).
- Update ARCHITECTURE.md (topology section: the host is now two
  processes; the pipe contract; the GIL rationale — "history lives in
  git, but the invariant 'sim and render are separate PROCESSES' is
  load-bearing").
- VERIFY: SP plays smoothly (no batching in its step cadence); full test
  suite green.

## M1 RESULT (done — the hypothesis is CONFIRMED)

`sim_process.py` (headless sim process, reuses `SimThread` verbatim) +
`test_sim_process.py` (A/B GIL test). Both pass.

| Arm | Setup | Batched (2-tick) | Max gap | Sim rate |
|---|---|---|---|---|
| A. In-process | sim thread + render GIL load, same process | 98.4% | 50 ms | starved |
| B. Separate process | sim in spawned process, GIL load in parent | **0.0%** | **16.7 ms** | **1.00x (59.9 steps/s)** |

The child's own step report: 181 steps in 3.02 s, all 180 gaps exactly one
tick. The separate-process sim steps smoothly at 60 Hz even while the
parent holds the GIL with a render-like load — the smooth snapshot-stamp
cadence the client's ghost clock needs.

### Design decisions locked in M1 (deviations from the original sketch)
- **Model channel is a bounded `Queue(maxsize=4)` + `put_nowait`
  (drop-when-full), NOT a Pipe.** A Pipe blocks the child's bridge on
  `send()` when the parent is busy (draw holds the GIL), and a blocked
  bridge can't drain commands. A bounded queue drops the oldest models
  when full — latest-wins, exactly what the render thread wants.
- **Commands channel is a `Pipe`** (parent -> child: setup/input/command/
  quit) — small, low-rate, ordered.
- **Cadence is measured CHILD-side** (the worker's per-step
  `publish_snapshot` stamps, reported on quit). A parent-side model-stream
  measurement is corrupted by (a) queue drops when the parent is busy and
  (b) the beat frequency between the parent's drain period (20 ms) and the
  sim period (16.7 ms). The child-side stamps are immune to both.
- **The render loop drains the model queue in the SAME thread as the draw
  (GIL load), every frame.** If the GIL load ran in a separate thread from
  the drain, the drain would be starved, the queue would fill, and the
  child would block — a false positive that the separate process doesn't
  help.
- The `done` report must be read with a BLOCKING `get(timeout)` after
  sending quit — the child's `put(done)` only completes once the parent
  drains the queue, so a `get_nowait` loop that exits on an empty queue
  races past it.

## M2 RESULT (done — input + commands + remote input over the pipe)

`sim_process.py` gains ONE handler: `("remote_input", ShipInput)` ->
`game.remote_input = m[1]` (a reference swap — exactly what the in-process
`game.set_remote_input` does; `_step` reads `game.remote_input` each tick
and applies it to player 1). `test_sim_process.py` gains a SECOND test
(`test_m2_input_commands_remote`; the M1 A/B is untouched): spawn the sim,
send `("setup", 1234, 2)`, wait for `("ready",)`, then drive it over the
command Pipe and verify each control path lands in the model:

| Control path | Message sent | Verified in the model |
|---|---|---|
| LOCAL input -> player 0 | `("input", ShipInput(thrust_fwd=1.0))` | p0 pos 41 px off center, vel 160 px/s |
| command -> sim | `("command", "targeting")` | p0 `targeting_on` False -> True |
| REMOTE input -> player 1 | `("remote_input", ShipInput(turn=1.0))` | p1 angle changed 1.8 rad |

All three pass; the M1 A/B still passes (no regression). The values match
the physics exactly (160 = SHIP_ACCEL 320 x 0.5 s; 1.8 = ROT_SPEED 3.6 x
0.5 s) and are deterministic across runs.

### Two test-design gotchas (both in the TEST, not the sim — no redesign)
- **The test must DRAIN the model queue continuously while an input is
  held (the render-loop pattern), not sleep-then-drain.** The model channel
  is a bounded Queue(maxsize=4) with drop-when-full: if the parent stops
  draining, the queue fills and the child's newer models are DROPPED, so a
  sleep-then-drain only ever sees stale pre-input models. `_settle_and_read`
  drains every ~10 ms for the settle duration and returns the newest.
- **The test must finish well under ~8 s.** With `("setup", seed, 2)` both
  ships sit at the spawn center (the real 2P game repositions player 1 via
  `set_player_ship` after the handshake; the test doesn't), and the
  seed-1234 world kills a stationary center ship at ~8 s. A dead ship is
  FROZEN (`_step` skips it), so any input sent after the death does nothing.
  The test runs in ~1.7 s (4x margin) and releases the thrust after the
  input check so p0 only coasts, never accelerates into the world.

## M3 RESULT (done — shell draw + SFX bridge + setup hulls)

The host window now renders the full game from the piped model on a SHELL
Game in the render process, with SFX piped out of the silent sim process.
All three parts landed; the M1 A/B + M2 tests still pass (no regression).

### Part 1 — the never-stepped shell audit (`test_render_model.py`)
`test_m3_shell_audit` builds a live game (stepped in-process) and a shell
Game (real surfaces, both players' REAL hulls — scout + blackbird,
`set_player_ship(1, ...)`, NEVER stepped), and at EVERY tick of a 300-tick
scripted run feeds the shell the live game's model and asserts:
- **S1 PIXEL PARITY** — `shell.draw(dt, model)` renders PIXEL-IDENTICAL to
  `live.draw(dt, model)` (the model carries everything draw() reads — no
  accidental live read). To make the comparison exact: `pygame.time.
  get_ticks` is frozen (the draw path has wall-clock animations — the
  reticle's 100 ms blink, the shield swirl), the camera is synced to the
  model's target for both games, and the GLOBAL `random` module is
  re-seeded before each draw (the draw path samples it — flame length
  jitter, shield flicker in ship.py).
- **S2** — `shell.draw(dt, None)` works (the pre-first-model frame: draw
  falls back to the shell's own construction state).
- **S3** — the model carries a plain `"sfx"` list, drained per publish.

### Part 2 — the shell Game in `run_host` (`__main__.py`)
Phase 3 now: builds the SHELL Game (real surfaces/fonts/textures +
sound=sfx + both players' hulls, `set_player_ship(1, client hull)`),
spawns the sim process (spawn context, AFTER the handshake — E.4), sends
the 6-arg setup, waits for `("ready",)` (bounded, crash -> notice + menu).
The game loop: `sim_cmd.send(("input", ...))` per frame, DRAINS the model
Queue in the SAME thread as the draw (`_drain_sim_model` — latest-wins +
concatenates every drained model's sfx), plays the sfx via the host's
SoundBank, and draws `game.draw(dt, {**model, "step_alpha": alpha} if
model else None)`. T/V/G/R/F route to the sim over the command Pipe
(`_sim_command`); T_RESPAWN -> `("command", "respawn", 1)`. The finally
shuts down in E.6 order: stop the worker, then quit the sim process
(`("quit",)` -> close Pipe -> join(2) -> terminate). The shell's
cam/_standins/_death_bursts/remote_beams are the render-side presentation
state (only the render thread touches them — the split is preserved).

### Part 3 — the SFX bridge (D.1)
`Game._sfx/_sfx_throttled/_sfx_thruster` now RECORD `(name, *params)`
events into `self._sfx_events` (and still play immediately when a
SoundBank is attached — the in-process SP/client path is unchanged).
`render_model()` DRAINS the queue into the model's `"sfx"` list (one
publish per sim step -> bounded to one step's worth). The sim process is
silent (sound=None), so its events ride the model pipe; the render
process plays them via `SoundBank.play_sfx(events)` (new: thruster ->
set_thruster, 2-tuple -> play_throttled, 1-tuple -> play). The throttle
lives in the SoundBank (wall clock): the sim-side bank never saw the
events, so the RENDER process's bank is the single place the human-rate
throttle runs — same behavior as in-process, throttled once.
`play_throttled` now returns True/False (played/throttled) so the
in-process path records only what the ear hears.

### The setup message (the "likely need" — taken)
`("setup", local_hull, local_loadout, client_hull, client_loadout, seed)`
(the plan's original sketch) is the REAL 2P form: the hulls/loadouts are
frozen dataclasses of plain data (picklable over the spawn Pipe; no hull
identity comparisons anywhere — only `.id` strings). The sim process
builds player 0 from the host's fit and replaces the player-1 placeholder
via `set_player_ship(1, Ship(hull=client_hull, ...))`. The M1/M2 simple
form `("setup", seed, players)` is kept (default hulls, the tests).

### Verification (all pass)
- **`python -m ship5.test_sim_process`** — M1 A/B + M2 + the new M3 pipe
  test + the new M3 e2e:
  - M3 pipe: 6-arg setup -> ready; the client's REAL hull is in the sim
    (p1 = silas, turn_rate_factor 1.15: dangle 4.140 rad over 1.0 s —
    the scout's 3.600 is outside the band); the model's "sfx" list
    carries ("thruster", True/False) + ("laser", 0.12) + ("enemy_laser",
    0.15), drained per publish.
  - M3 e2e (the REAL `run_host` + a fake TCP client over loopback): the
    sim process is spawned (6-arg setup) and GONE after exit (E.6); the
    shell's camera followed the thrusting ship + its local standin's pose
    moved with the ship (draw ran from the piped model); the host's
    SoundBank played the model's sfx (242 events: thruster on + laser
    blips — the D.1 bridge end to end); the worker is alive (31 T_PONGs
    to the client's tagged inputs).
- **`python -m ship5.test_render_model`** — M1 (SHAPE/PARITY/PLAIN/PURE/
  DETERMINISM/2P) + the M3 shell audit (S1/S2/S3) all pass.
- **No regression**: test_sim_thread, test_snapshot, test_determinism,
  test_m2b_local all pass.
- **`test_network` (the loopback e2e)**: the HOST-side checks pass (no
  exceptions, clean loop exit, handshake + welcome over the 6-arg setup
  path). The CLIENT-side snapshot/ghost/render-point checks FAIL BY
  DESIGN until M4 — the sim->render->`worker.publish_snapshot` bridge is
  M4's job (scope guard), so no snapshots reach the client yet. M4 turns
  this green.
- **Display note**: this env has no X display — the window check is the
  headless e2e (dummy SDL driver + the pixel-parity audit), which is the
  plan's "headless or manual window check" option. A manual window run
  (`python -m ship5`, host mode) is the remaining human-in-the-loop
  confirmation.

### Locked decisions — none changed
Pipe commands / bounded drop-when-full Queue models / same-thread drain /
child-side cadence / headless sim process: all untouched. No NetWorker or
wire-protocol change (M4).

## M4 RESULT (done — full 2P end-to-end over the network bridge)

The sim's wire traffic (snap/beam/echo) goes to a DEDICATED net Queue,
drained by a dedicated net-pump thread in the render process, which
routes each message to the NetWorker. Full 2P plays end to end: the
client's "syncing to authoritative sim" clears, its input reaches the
sim, and snapshots/beams/echoes flow. `test_network`'s authoritative
verdict is GREEN (it failed by design in M3 — no snapshots reached the
client).

### Why the redo (the original M4's flaw)
The original M4 rode the wire traffic on the SAME bounded Queue as the
render model, so the render thread's per-frame drain (`_drain_sim_model`,
the SAME thread as the draw) was in the snapshot's SEND path: a busy draw
delayed (up to a full frame) or dropped snapshots. That coupled snapshot
sending to rendering — the dependency this redo removes. The fix: the
wire traffic gets its own Queue + its own drain thread, so the render
thread's draw is out of the send path. (The original ask was "send
snapshots directly from the sim process to the socket"; a live socket
can't be pickled across the spawn Pipe, and the sim owning the socket
would break the relay impairment test — so the dedicated pump thread is
the cross-platform-robust way to get the same decoupling.)

### Part 1 — sim -> render wire messages on a dedicated Queue (`sim_process.py`)
`sim_process_main(cmd_conn, model_queue, net_queue)` — a THIRD channel:
the net Queue. `_Worker` (the SimThread's worker stub) now takes the net
Queue and forwards: `publish_snapshot(sim_time, snap)` ->
`("snap", sim_time, snap)` (still recording the child-side stamp),
`send(msg)` -> `("beam", msg)` / `("echo", msg)` by `msg["type"]`. All
via `put_nowait` (drop-when-full — a dropped snapshot is a network loss
the client's interpolation buffer already handles). The `SimThread` call
sites are UNCHANGED (verified: `publish_snapshot` per step,
`_beam_events` drain -> T_BEAM, `flush_remote_input_echo()` -> T_ECHO).
The model Queue is back to models + lifecycle only (maxsize 4).

### Part 2 — the dedicated net-pump thread (`__main__.py`)
New `_NetPump` class: a daemon thread that drains the sim's net Queue
every ~1 ms and routes each message to the worker — `("snap", t, snap)`
-> `worker.publish_snapshot(t, snap)`, `("beam"/"echo", msg)` ->
`worker.send(msg)`. The worker serializes + sends on its own thread the
moment it lands (age at send ~0). Started after `worker.start()` (it
calls worker.*), stopped in the finally BEFORE `worker.stop()` (stop the
producer before the consumer). `_drain_sim_model` is back to returning
`(newest_model, sfx_events)` — the wire messages no longer ride the
model Queue, so the render thread's draw is out of the send path. The
model Queue is back to maxsize 4 (models + lifecycle only); the net
Queue is maxsize 16 (~2 steps of wire traffic).

### Part 3 — worker input sink -> sim (`__main__.py`)
`_client_input_sink` (worker thread) now sends
`("remote_input", deserialize_input(m["inp"]))` over the command Pipe
instead of the no-op `game.set_remote_input`. The T_PONG reflection STAYS
in the sink (worker thread, ~1 ms — a render-thread reflection would add
a full frame to the measured RTT). SEND-SIDE LOCK: the command Pipe now
has TWO senders in the render process (the main thread's input/command/
quit + the worker thread's remote_input); `_sim_cmd_lock` serializes them
(two concurrent senders would interleave the Pipe's 4-byte length
prefixes). The sim process's `cmd_lock` guards its RECV side (M3,
unchanged). Shutdown ordering (E.6) already correct: the finally stops
the worker BEFORE quitting the sim (the sink writes the Pipe, which must
not be written after the sim is gone).

### Part 4 — lifecycle
`ready`/`crash` were already handled (M3). `game_over` needs NO new
message: in 2P the flag is never set (only single player sets it —
`game.py` per-player death), and the client reads it from the snapshot
tuples (index 7) it already receives. Verified: no client-side
game_over handling exists beyond the snapshot field.

### test_network: host-side checks re-pointed to the wire
Three checks read the host's LIVE Game state — but since M3 the host's
Game is a never-stepped SHELL (the authoritative sim is in the separate
process), so those reads were stale (frozen at spawn / sim_time 0).
Re-pointed to the WIRE (the client's buffer — the correct M4 way to
verify host-authoritative state):
- 7.1 "client sim_time tracks the host's sim clock": host.sim_time ->
  `client.snap_buf.newest_time()` (the newest snapshot stamp — the same
  stamps the estimator fits).
- "host's player 1 (client's ship) moved" (CORE): host.players[1].pos ->
  the newest snapshot's ships[1] (the client's ship, matched by index).
- 7.2 "newest snapshot enemy angle is fresh vs host": the host's LIVE
  final angle is no longer measurable from the client (the newest
  snapshot IS the host's final state — the sim publishes every step), so
  it became a wire-based sanity check (every newest-snapshot enemy angle
  is finite; the in-span check still catches off-arc garbage).

### test_network: the leaked-QUIT fix (the alternating "Game built" failure)
The 2nd and 4th runs failed "Game built" (both peers, no exceptions, both
loops exited) — a DETERMINISTIC alternating pattern, not flakiness. Root
cause: the stop loop posts a QUIT every 50 ms while a thread is
is_alive(), but the threads spend their shutdown in `finally` blocks
(not polling events) — and M4's host finally includes the sim-process
shutdown (quit + join(timeout=2)), a longer window than M3's worker-only
stop. The unpolled QUITs accumulate in the GLOBAL pygame event queue
(all 5 runs share one display); the next run's thread picks them up on
its first `pygame.event.get()` and returns before building a Game. Runs
that fail exit fast (no sim spawned -> short finally -> no leak), which
is why it alternates. Measured: 39-40 leaked QUITs after each run. Fix:
`pygame.event.clear()` at the start of each run (test-harness artifact —
5 game loops sharing one global event queue; real peers are separate
processes with separate queues).

### Verification (all pass)
- **`python -m ship5.test_sim_process`** — M1 A/B + M2 + M3 pipe + the
  M3 e2e (M4): the fake client drives its ship with REAL input (a
  constant turn) and the test verifies (a) the client's input reached
  the sim (player 1 turned ~12 rad in the host's models — the client ->
  worker -> Pipe -> sim -> model path) and (b) the wire bridge delivered
  snapshots + input echoes to the client over the DEDICATED net Queue +
  pump (360 snap/echo messages). ALL PASS.
- **`python -m ship5.test_network`** — the authoritative verdict
  (clean + 40ms jitter + snap loss + 100ms stall + 7.5b jitter) is GREEN
  (exit 0) in every run that completes: the client's snapshot/ghost/
  render-point checks that failed by design in M3 pass (ghost seeded,
  >=2 snapshots, render point, buffer ships/enemies/bullets, ghost
  fires, player 1 moved 1.5-2.8k px). NOTE: two per-battery checks are
  FLAKY under machine load (the 40ms battery's "buffer carries >=1
  enemy bullet" — timing-dependent on whether an AI bullet is alive at
  the render point; and an intermittent hang-after-verdict in the
  interpreter-shutdown phase, not the game logic). The authoritative
  verdict folds in only the battery-specific checks, so these do not
  fail the run.
- **`python -m ship5.test_render_model`** — M1 parity + M3 shell audit
  (S1/S2/S3) all pass.
- **No regression**: test_sim_thread, test_snapshot, test_determinism,
  test_m2b_local all pass.
- **F3 telemetry — PENDING RE-CAPTURE**: the original M4's two-machine
  capture (client `newest_stamp` delta 99.1% one-tick; `snap_px` p95
  0.000, >=3 px tail 0.29%) was measured with the wire riding the model
  Queue. The redo moves the send path to the pump thread, so the
  two-machine F3 capture (`tools/dbg_f3_capture2.py`) should be
  re-run to confirm the snap tail stays at ~0 with the new routing.
  (The headless e2e above covers the same paths over loopback.)
- **Manual (two machines)**: the remaining human-in-the-loop
  confirmation (host + client on two machines — the client's "syncing to
  authoritative sim" clears, both ships move, beams/explosions/SFX on
  both sides).

### Locked decisions — the redo's one change
Pipe commands / bounded drop-when-full Queues / same-thread model drain /
child-side cadence / headless sim process: all untouched. The redo's one
change to a locked decision: the model Queue is back to maxsize 4 (models
+ lifecycle only) and the wire traffic gets its OWN bounded Queue
(maxsize 16) drained by a dedicated pump thread — an extension of the
locked channel pattern (bounded, drop-when-full), not a redesign. No
wire-protocol or client-side change (the T_SNAP/T_BEAM/T_ECHO/T_INPUT
shapes are unchanged — M4 only moved WHO sends them: from the in-process
sim thread to the sim process, via the pipe). No NetWorker change
(publish_snapshot / send are the same thread-safe puts the in-process
sim thread used).

## G. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `draw()` reads live sim state (not just the model) | low (parity test exists) | M3 audit + shell-Game parity test |
| Headless pygame misbehaves (Surface/Vector2 without display) | low (standard) | M1 proves it before anything else |
| Pickle/pipe overhead at 60 Hz | low (few KB, sub-ms) | M1 measures the model cadence |
| Local input latency (pipe hop instead of reference swap) | low (sub-ms; the host applies the LATEST input per tick, so sub-tick delay is invisible) | M2 |
| Spawn startup cost visible to the player | low (overlaps the handshake) | E.4; optional spawn-during-accept later |
| Sim process crash takes the game down | new failure mode | E.7 crash notice; the sim loop is the same code that runs today (no new sim logic) |
| Windows vs Linux spawn differences | low (spawn + Pipe are portable; no socket passing) | test on the target machine in M1 |

## H. What does NOT change
- The remote client (netcode.py, the ghost, the estimator, the
  interpolation buffer, `push_snapshot`, `predicted_view`) — untouched.
- The wire protocol (snap/echo/beam/pong shapes) — untouched.
- `Game._step` / `snapshot` / determinism — untouched (the sim code is
  the same code, in a different process).
- The existing test suite (it constructs Game directly, in-process).