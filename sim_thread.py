"""The authoritative sim on its own thread.

The authoritative Game runs on a daemon thread that steps it at a
real-time 60 Hz clock (a monotonic accumulator, NOT tied to the render
frame rate). The render thread is a LOCAL CLIENT of that sim: it
publishes the latest local input + commands and reads the published
RenderModel. Single-player uses the same thread with worker=None (the
host loop minus the NetWorker), so there is exactly ONE sim
architecture in the game.

Threading model (the NetWorker hand-off patterns, everywhere):
  * All cross-thread hand-offs are single reference swaps (atomic under
    the GIL) or a queue the consumer drains for the newest item. No
    locks, no shared mutable state.
  * The sim thread OWNS the Game: it is the only thread that calls
    _step / reset / fire_scan / attribute writes on ships. The render
    thread never mutates the Game — it publishes input (a fresh
    ShipInput object per frame), commands (a queue), and reads
    `latest_model` (a reference swap the sim thread performs).
  * `remote_input` (the client's input, polled off the NetWorker on the
    render thread) is handed off by a direct reference swap
    (`game.remote_input = inp`). Safe: ShipInput is a plain dataclass
    that _step only READS — the world-cap `replace(p_inp, ...)` in
    _step builds a NEW object and never mutates the stored one.

Per iteration (a real-time accumulator, the same hiccup policy as
Game.update — MAX_STEPS_PER_FRAME + ACC_BACKLOG_CAP):
  1. drain the command queue (apply T/V/G/R/F to the sim; QUIT/ESC/F3
     stay on the render thread);
  2. acc = min(acc + elapsed, ACC_BACKLOG_CAP); run <= 5 _step(STEP, inp)
     with the LATEST published local input (pinned #5: the host applies
     the latest received input each tick — the local input gets the same
     treatment);
  3. PUBLISH:
     a. SNAPSHOT — after EACH step (inside the step loop):
        worker.publish_snapshot(game.sim_time, game.snapshot()). The
        stamp is the tick's END (sim_time advanced at the top of _step),
        so the stamps are a smooth 1-tick cadence even when the OS
        batches 2 ticks into one wake (a per-wake publish stamped the
        batch at its final sim_time — a 33 ms stamp jump, the ~8 px
        snap; per-step makes the batching invisible to the client).
        The worker serializes + sends it the moment it lands, so the
        snapshot's age at send is ~0 and the client's `now = sim_time +
        L` is exact.
     b. MODEL — only when >= 1 step ran, and PER WAKE (latest-wins):
        g.acc = acc (so render_model's step_alpha is the sim's real
        acc/STEP), then latest_model = game.render_model() (atomic
        reference swap; the render thread reads it and draws with it).
        Publishing only on a step keeps the model object STABLE between
        steps (the render clock detects a new model by identity and
        advances its alpha while the same model is re-read) and avoids
        the ~94% wasted render_model() calls (the thread iterates every
        ~1 ms but only steps 60x/sec). The model must stay per-wake —
        a per-step model would re-anchor the render clock every step
        and pin its alpha at 0 (the M4 bug);
  4. sleep until the next tick boundary (dynamic sleep — see _run).

The render thread's interpolation alpha is NOT this thread's acc/STEP —
the render thread computes its own alpha from its own clock + the
published models (see _HostRenderClock in __main__.py). The model still
CARRIES `step_alpha` (the sim's acc/STEP at publish time) so the parity
tests keep working unchanged.
"""
import queue
import threading
import time

from .game import STEP
from .intent import ShipInput
from .net import T_BEAM, T_ECHO, serialize_input


class SimThread:
    """A daemon thread that OWNS a Game and steps it at a real-time
    60 Hz clock, publishing the RenderModel for the render thread.

    Lifecycle:
      1. Built around a fully-constructed Game (handshake done, player
         ships set) + the host's NetWorker. The worker is None for
         single-player (SP runs this same thread with no network — the
         sim thread then publishes the model only) and in tests.
      2. `start()` launches the thread. The first published model
         appears within ~one iteration (~1-2 ms).
      3. The render thread calls `publish_input(ShipInput)` each frame
         (latest wins), `command(name)` for T/V/G/R/F, and reads
         `.latest_model` each frame.
      4. `stop()` sets the stop event and joins. Idempotent. Call it
         BEFORE worker.stop() (the sim thread feeds the worker).
    """

    # Unused by the run loop (it sleeps dynamically until the next tick
    # boundary — see _run). Kept for parity with NetWorker.WAIT.
    WAIT = 0.001

    # The hiccup policy is the sim's own (Game.update's guards): one
    # iteration may run at most this many fixed steps, and the backlog
    # is capped — a pathological stall (GC pause, OS suspend) drops its
    # excess instead of spiraling.
    MAX_STEPS_PER_FRAME = 5
    ACC_BACKLOG_CAP = 0.25

    def __init__(self, game, worker=None):
        self._game = game
        self._worker = worker
        # The latest local ShipInput, written by the render thread each
        # frame, read by the sim thread each step. A reference swap is
        # atomic under the GIL; the sim never mutates the object it
        # reads (ShipInput is only ever replaced, in place nowhere).
        self._latest_input = ShipInput()
        # The render thread's T/V/G/R/F commands (event -> sim command
        # routing). The sim thread drains this at the top of each
        # iteration, before stepping, so a command applies on the very
        # next step. A queue (not a single slot) so nothing is lost if
        # the render thread posts several between iterations.
        self._commands = queue.Queue()
        # The published RenderModel: a plain-data dict, swapped by
        # reference each iteration. None until the first publish. The
        # render thread reads this and calls game.draw(dt, model).
        self.latest_model = None
        # Cumulative fixed steps run by the sim thread (written by the
        # sim thread, read by the render thread's telemetry — a monotone
        # int, so the render thread computes the per-window delta without
        # a read/reset race). The host CSV logs the delta per telemetry
        # line: a stable 1-per-line (60 Hz) means the sim thread steps
        # exactly one tick per iteration (the dynamic-sleep design);
        # 0/2+ oscillation means it is batching, which makes the snapshot
        # stamps jump and is a candidate driver of the client's sawtooth
        # snap.
        self._steps_total = 0
        # Per-iteration cadence log (diagnostics): (inter_wake, body_time,
        # steps) tuples, one per loop iteration. `inter_wake` is the real
        # time between consecutive wakes (top of the loop); `body_time` is
        # the work done this iteration (wake -> just before the sleep);
        # `steps` is the number of fixed steps taken. Used to diagnose WHY
        # the sim batches: inter_wake = body_time + sleep + wake_latency, so
        # a ragged inter_wake with a small body_time means the OS scheduler
        # is waking the thread late (jitter); a large body_time means the
        # loop body is eating into the tick. None disables the log (the
        # default — it is only enabled for the cadence diagnostics).
        self._iter_log = None
        self._stop = threading.Event()
        self._thread = None
        self._started = False

    # -- render-thread API --------------------------------------------------
    def publish_input(self, inp):
        """Publish the latest local ShipInput (render thread, once per
        frame). The sim applies the LATEST published input at each step
        (pinned #5 — the same rule the host applies to the client's
        input)."""
        self._latest_input = inp

    def command(self, name, arg=None):
        """Enqueue one sim command (render thread). `name` is one of
        'targeting' (T), 'sensor' (V), 'scan' (G), 'reset' (R/F), or
        'respawn' (R while the local player is dead — `arg` is the
        player index). The sim thread applies it at the top of its next
        iteration, before stepping. QUIT/ESC and F3 never come here —
        they stay on the render thread (exit + render diagnostic)."""
        self._commands.put((name, arg))

    def start(self):
        """Launch the sim thread. Call once, after the Game is fully
        built (handshake done, player ships set)."""
        if self._started:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self):
        """Signal the sim thread to stop and join it. Idempotent — safe
        to call from a `finally` block more than once. Call BEFORE
        worker.stop(): the sim thread feeds the worker, and the worker
        must not be sending a snapshot the sim thread is about to
        mutate (a reset)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # -- sim thread ---------------------------------------------------------
    def _apply_command(self, cmd):
        """Apply one render-thread command to the sim (sim thread only).

        `cmd` is a (name, arg) pair (arg is None for all but 'respawn').
        Mirrors the KEYDOWN branches of Game.handle_events EXACTLY
        (same game_over gating). `self.ship` is players[0] (the host's
        own ship)."""
        name, arg = cmd
        g = self._game
        if name == "targeting" and not g.game_over:
            g.ship.targeting_on = not g.ship.targeting_on
        elif name == "reset" and g.game_over:
            g.reset()
        elif name == "sensor" and not g.game_over:
            g.ship.sensor_on = not g.ship.sensor_on
        elif name == "scan" and not g.game_over:
            g.ship.fire_scan()
        elif name == "reset" and g.test_mode:
            g.reset()
        elif name == "respawn":
            # Per-player death — respawn the local player's ship (a
            # no-op unless it is actually dead, so a stray R is
            # harmless). The world + the other player are untouched.
            g.respawn_player(arg)

    def _run(self):
        g = self._game
        acc = 0.0
        last = time.monotonic()
        prev_top = None   # previous iteration's top-of-loop time (cadence log)
        while not self._stop.is_set():
            # Cadence log: the top-of-loop time (the wake moment). The
            # inter-wake interval is measured between consecutive tops;
            # the body time is top -> just before the sleep.
            iter_t0 = time.monotonic()
            inter_wake = (iter_t0 - prev_top) if prev_top is not None else 0.0
            prev_top = iter_t0
            # 1. commands: drain everything posted since the last
            #    iteration (a few at most — the render thread posts at
            #    most one per key, per frame). Applied BEFORE stepping
            #    so the very next step sees them.
            while True:
                try:
                    self._apply_command(self._commands.get_nowait())
                except queue.Empty:
                    break
            # 2. the real-time 60 Hz accumulator (mirrors Game.update's
            #    fixed-step loop, with the same hiccup guards).
            now = time.monotonic()
            acc = min(acc + (now - last), self.ACC_BACKLOG_CAP)
            last = now
            inp = self._latest_input
            steps = 0
            for _ in range(self.MAX_STEPS_PER_FRAME):
                if acc < STEP:
                    break
                g._step(STEP, inp)
                acc -= STEP
                steps += 1
                # SNAPSHOT PUBLISH — after EACH step, not per wake. The
                # snapshot's stamp is g.sim_time, which _step advanced to
                # this tick's END at its top, so every publish is stamped
                # exactly STEP after the previous one — a smooth 1-tick
                # cadence the client's interpolation buffer can read.
                #
                # WHY per-step (not per-wake): the OS wakes this thread
                # ~1 tick late ~19% of the time (plan_13 telemetry), so a
                # batched wake steps 2 ticks in one go. Publishing once
                # per wake stamped that wake at the FINAL sim_time — the
                # stamp jumped 33 ms, the client's ghost reconcile
                # corrected a larger span, and that was the ~8 px snap.
                # Publishing after each step makes a batched wake emit 2
                # snapshots 16.7 ms apart: the batching is invisible to
                # the client's stamp cadence (the client's
                # HostTimeEstimator / SnapshotBuffer / reconcile all
                # handle two arrivals in one frame — the estimator
                # re-anchors, the buffer appends both, the reconcile
                # replays from the NEWER stamp).
                #
                # The host's NET process (plan_13) drains its net Queue
                # and sends EACH queued snap (no drain-to-newest), so both
                # snapshots reach the wire. The in-process NetWorker
                # (single-player / pre-plan_13) drains to the newest —
                # it would drop the intermediate one, but that path has
                # no client to serve (SP has no worker at all).
                #
                # Age at send is still ~0: the snapshot is taken the
                # moment the step finishes, so the client's
                # `now = sim_time + L` stays exact.
                if self._worker is not None:
                    self._worker.publish_snapshot(g.sim_time, g.snapshot())
            # The host CSV's per-window step count (see _steps_total in
            # __init__).
            self._steps_total += steps
            # 3. MODEL publish ONLY when the sim actually stepped — and
            #    still PER WAKE (one model per wake, latest-wins). The
            #    model is the host's own render feed: the render clock
            #    detects a new model by object identity and advances its
            #    alpha while the same model is re-read, so publishing a
            #    fresh model every ~1 ms iteration (even when nothing
            #    changed) would (a) re-anchor the render clock every
            #    frame and (b) waste ~94% of the render_model() calls.
            #    The model is latest-wins and the render clock needs it
            #    stable — it must NOT move to per-step like the snapshot
            #    (a per-step model would re-anchor the render clock on
            #    every step and pin its alpha at 0, the M4 bug).
            #
            #    Also write g.acc = acc before publishing: render_model()
            #    reads self.acc for step_alpha, but this thread uses its
            #    OWN local acc (it steps g._step directly, not
            #    g.update()). Without this, step_alpha is always 0 and the
            #    render clock's re-anchor (t = sim_time + step_alpha*STEP)
            #    lands on alpha=0 every frame.
            if steps:
                g.acc = acc
                self.latest_model = g.render_model()
                if self._worker is not None:
                    # Send any laser beams fired this wake as T_BEAM
                    # events (drained here, on the sim thread — the only
                    # thread that touches g._beam_events). The client
                    # draws each immediately on receipt; one-way LAN
                    # latency is imperceptible against the 0.15 s beam
                    # flash. Stays per-wake: a beam is a 0.15 s flash,
                    # not a cadence — one flush per wake covers every
                    # step's beams.
                    for (sx, sy, ex, ey, t, owner, lx, ly,
                         target) in g._beam_events:
                        self._worker.send({"type": T_BEAM, "sim_time": t,
                                           "sx": sx, "sy": sy,
                                           "ex": ex, "ey": ey,
                                           "owner": owner,
                                           "lx": lx, "ly": ly,
                                           "target": target})
                    g._beam_events.clear()
                    # Send the input the host ACTUALLY applied to player 1
                    # (the client's ship) on each tick stepped this
                    # iteration, as ONE T_ECHO (the T_BEAM pattern —
                    # per-tick event data that must not bloat the pruned
                    # snapshot). The client's rewind replay uses these
                    # instead of its own sent input (the in-flight input
                    # snapback fix). ~60 msg/s, a few KB/s — negligible.
                    # Stays per-wake: the echo list already carries one
                    # entry per stepped tick, so a batched wake's echo
                    # covers both ticks in one message.
                    echo = g.flush_remote_input_echo()
                    if echo:
                        self._worker.send({
                            "type": T_ECHO,
                            "entries": [[t, serialize_input(inp)]
                                        for (t, inp) in echo]})
            # 4. DYNAMIC SLEEP: sleep until the next tick boundary.
            #    After the step loop, `acc` is the remaining time (0 to
            #    STEP). The next tick is due in (STEP - acc) seconds.
            #    Sleeping for exactly that amount makes the thread wake
            #    up on the tick boundary, so it steps exactly 1 tick per
            #    wake-up (in steady state) instead of batching 1-7 ticks
            #    (a fixed 1 ms sleep + OS scheduler imprecision lets the
            #    accumulator build up 1-7 ticks of time between wake-ups,
            #    which batches the steps and makes the snapshot stamps
            #    jump).
            # Cadence log: record (inter_wake, body_time, steps) for this
            # iteration. body_time = top-of-loop -> just before the sleep
            # (the work done: command drain + steps + model publish).
            # inter_wake = the real time between consecutive wakes. The
            # decomposition: inter_wake = body_time + sleep_time +
            # wake_latency, where wake_latency is the OS scheduler's
            # overshoot (how long after the requested sleep the thread
            # actually resumed). A ragged inter_wake with a small
            # body_time => OS jitter; a large body_time => the loop body
            # is eating into the tick.
            if self._iter_log is not None:
                self._iter_log.append((inter_wake,
                                       time.monotonic() - iter_t0, steps))
            sleep_time = STEP - acc
            if sleep_time > 0:
                self._stop.wait(sleep_time)