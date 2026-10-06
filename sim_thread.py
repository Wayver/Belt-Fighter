"""Session 9.x M3: the authoritative sim on its own thread.

The authoritative Game moves off the render thread onto a daemon thread
that steps it at a real-time 60 Hz clock (a monotonic accumulator,
mirroring the NetWorker's real-time snapshot timer — NOT tied to the
render frame rate). The render thread becomes a LOCAL CLIENT of that
sim: it publishes the latest local input + commands, and reads the
published RenderModel. M3 did this for the 2P host; M5 (Session 9.x)
unified single-player onto the same thread — SP is now the host loop
minus the NetWorker (worker=None), so there is exactly ONE sim
architecture in the game.

Threading model (the NetWorker `_latest_snapshot` pattern, everywhere):
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
  1. drain the command queue (apply T/V/G/R/F to the sim — the M3
     event->sim-command routing; QUIT/ESC/F3 stay on the render thread);
  2. acc = min(acc + elapsed, ACC_BACKLOG_CAP); run <= 5 _step(STEP, inp)
     with the LATEST published local input (pinned #5: the host applies
     the latest received input each tick — the local input gets the same
     treatment);
  3. PUBLISH (only when >= 1 step ran): g.acc = acc (so render_model's
     step_alpha is the sim's real acc/STEP), then
     latest_model = game.render_model() (atomic reference swap; the
     render thread reads it and draws with it), and
     worker.set_latest_snapshot(game.sim_time, game.snapshot()) (moved
     from the render loop to the sim thread — the worker's 10 Hz timer
     now always has a fresh snapshot to send). Publishing only on a step
     keeps the model object STABLE between steps (the render clock
     detects a new model by identity and advances its alpha while the
     same model is re-read) and avoids ~94% wasted render_model() calls
     (the thread iterates every ~1 ms but only steps 60x/sec);
  4. stop_event.wait(WAIT) (~1 ms; bounds CPU + makes stop() prompt).

The render thread's interpolation alpha is NOT this thread's acc/STEP
anymore — the render thread computes its own alpha from its own clock +
the last two published models (see _render_alpha in __main__.py). The
model still CARRIES `step_alpha` (the sim's acc/STEP at publish time) so
the M1/M2b parity tests keep working unchanged.
"""
import queue
import threading
import time

from .game import STEP
from .intent import ShipInput
from .net import T_BEAM


class SimThread:
    """A daemon thread that OWNS a Game and steps it at a real-time
    60 Hz clock, publishing the RenderModel for the render thread.

    Lifecycle:
      1. Built around a fully-constructed Game (handshake done, player
         ships set) + the host's NetWorker. The worker is None for
         single-player (M5: SP runs this same thread with no network —
         the sim thread then publishes the model only) and in tests.
      2. `start()` launches the thread. The first published model
         appears within ~one iteration (~1-2 ms).
      3. The render thread calls `publish_input(ShipInput)` each frame
         (latest wins), `command(name)` for T/V/G/R/F, and reads
         `.latest_model` each frame.
      4. `stop()` sets the stop event and joins. Idempotent. Call it
         BEFORE worker.stop() (the sim thread feeds the worker).
    """

    # Sleep between iterations when idle. ~1 ms keeps the 60 Hz step
    # clock near-exact (a 16.67 ms step is checked every ~1 ms) and
    # makes stop() prompt, while bounding the thread's CPU — the same
    # trade-off as NetWorker.WAIT.
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
        self._stop = threading.Event()
        self._thread = None
        self._started = False

    # -- render-thread API --------------------------------------------------
    def publish_input(self, inp):
        """Publish the latest local ShipInput (render thread, once per
        frame). The sim applies the LATEST published input at each step
        (pinned #5 — the same rule the host applies to the client's
        input). Replaces the old in-loop `game.update(raw_dt, keys)`
        input sampling."""
        self._latest_input = inp

    def command(self, name, arg=None):
        """Enqueue one sim command (render thread). `name` is one of
        'targeting' (T), 'sensor' (V), 'scan' (G), 'reset' (R/F), or
        'respawn' (10.1, R while the local player is dead — `arg` is the
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
        (same game_over gating) — only the thread moved. `self.ship`
        is players[0] (the host's own ship), so the mapping is
        1:1 with the old in-loop code."""
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
            # 10.1: per-player death — respawn the local player's ship
            # (a no-op unless it is actually dead, so a stray R is
            # harmless). The world + the other player are untouched.
            g.respawn_player(arg)

    def _run(self):
        g = self._game
        acc = 0.0
        last = time.monotonic()
        while not self._stop.is_set():
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
            # 3. publish ONLY when the sim actually stepped. Publishing a
            #    fresh model every ~1 ms iteration (even when nothing
            #    changed) caused two problems: (a) the render clock
            #    detects a "new model" by object identity, so a fresh
            #    dict every ~1 ms made it re-anchor every render frame
            #    (it never advanced its alpha between publishes), and
            #    (b) ~94% of the render_model() calls were wasted CPU
            #    (the model was identical to the last step's). Publishing
            #    only on a step keeps the model object STABLE between
            #    steps (so the render clock can advance its alpha) and
            #    cuts the sim thread's CPU ~94%.
            #
            #    Also write g.acc = acc before publishing: render_model()
            #    reads self.acc for step_alpha, but this thread uses its
            #    OWN local acc (it steps g._step directly, not
            #    g.update()). Without this, step_alpha was always 0 and
            #    the render clock's re-anchor (t = sim_time +
            #    step_alpha*STEP) landed on alpha=0 every frame.
            if steps:
                g.acc = acc
                self.latest_model = g.render_model()
                if self._worker is not None:
                    self._worker.set_latest_snapshot(g.sim_time,
                                                     g.snapshot())
                    # 10.4: send any laser beams fired this step as T_BEAM
                    # events (drained here, on the sim thread — the only
                    # thread that touches g._beam_events). The client draws
                    # each immediately on receipt; one-way LAN latency is
                    # imperceptible against the 0.15 s beam flash.
                    for (sx, sy, ex, ey, t) in g._beam_events:
                        self._worker.send({"type": T_BEAM, "sim_time": t,
                                           "sx": sx, "sy": sy,
                                           "ex": ex, "ey": ey})
                    g._beam_events.clear()
            # 4. sleep ~1 ms (bounds CPU + makes stop() prompt).
            self._stop.wait(self.WAIT)