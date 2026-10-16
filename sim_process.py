"""The host's authoritative sim in its OWN process (plan_12, M1).

WHY A SEPARATE PROCESS
The host's render thread and the sim thread are separate THREADS in the
same process, so they share the GIL. The render thread holds the GIL for
its whole draw (~18-25 ms — longer than the 16.7 ms sim tick), starving
the sim thread; the sim then BATCHES 2 ticks when it gets the GIL back,
and the snapshot stamps come out ragged (17, 33, 17, 33 ms). The client's
ghost clock (anchored on the previous snapshot) can't anticipate a 33 ms
jump, so it sits ~7 ms behind the new stamp and the reconcile pulls the
clock+pose forward -> the residual ~8 px snap tail (plan_12 section A).

Moving the sim to a separate PROCESS removes the shared GIL: the render
process's draw can no longer starve the sim, so the sim steps exactly one
tick per iteration and the snapshot stamps are a smooth 17 ms cadence.

M1 SCOPE (this file)
Headless sim process + the model pipe, NO network, NO draw. It proves the
core hypothesis: with the sim in its own process, the step cadence is
smooth (one tick per publish) even while the PARENT process holds the GIL
with a render-like load. The A/B that demonstrates this is
test_sim_process.py.

M1 sends the full render model over the pipe (the real deliverable); the
parent reads model["sim_time"] to measure the cadence. M2 adds input +
commands + remote input; M3 adds the shell-Game draw + sfx; M4 adds the
network bridge: the wire traffic (snap/beam/echo) the SimThread hands to
its worker goes to a DEDICATED net Queue (separate from the model
Queue). plan_13 (Phase 1): that net Queue is drained by the host's
dedicated NET process (net_process.py) — NOT the render process — and the
client's input arrives on a DEDICATED remote_input Pipe (net -> sim)
instead of the command Pipe.

IPC (spawn context — see plan_12 section E + plan_13 section 2):
  parent -> child (Pipe, small + low-rate + ordered):
                   ("setup", local_hull, local_loadout, client_hull,
                    client_loadout, seed)   (M3: the real 2P fits; the
                                              plan's original sketch)
                   ("setup", seed, players)  (M1/M2: default hulls, tests)
                   ("input", ShipInput) |
                   ("command", name, arg) | ("quit",)
  net -> child (Pipe, small + low-rate + ordered — plan_13 Phase 1):
                   ("remote_input", ShipInput)   (the client's input; the
                                                  net process forwards it
                                                  the moment it lands off
                                                  the socket)
                   ("command", name, arg)        (e.g. ("command",
                                                  "respawn", 1) — the net
                                                  process forwards the
                                                  client's T_RESPAWN)
  child -> parent, TWO bounded Queues (put_nowait drop-when-full):
    model Queue (render-thread traffic):
                   ("model", render_model_dict)   (60 Hz, on a step)
                   ("ready",) | ("crash", traceback_text) |
                   ("done", steps, sim_time, wall, stamps)
    net Queue (wire traffic — the M4 network bridge):
                   ("snap", sim_time, snapshot)   (60 Hz, on a step)
                   ("beam", msg) | ("echo", msg)  (T_BEAM / T_ECHO wire
                        messages)
  Both channels are bounded Queues, NOT Pipes: a Pipe blocks the child's
  bridge on send() when the parent is busy (draw holds the GIL), and a
  blocked bridge can't drain commands. A bounded queue drops the oldest
  items when full — latest-wins, exactly what the render thread wants
  (it draws the newest model, never a stale one). A dropped snapshot is
  a network loss the client's interpolation buffer already handles
  (out-of-order/late arrivals are dropped there too), so drop-when-full
  is the right semantics for the net Queue as well.

  WHY TWO QUEUES: the wire messages (snap/beam/echo) ride a DEDICATED net
  Queue, NOT the model Queue the render thread drains — so the render
  thread's draw is OUT of the snapshot's send path. plan_13 (Phase 1):
  the net Queue is drained by the host's dedicated NET process
  (net_process.py), which serializes + sends each message the moment it
  lands (the render process never touches it). The model Queue is models
  + lifecycle only (maxsize 4).

The child is HEADLESS: sim_process_main sets SDL_VIDEODRIVER/AUDIODRIVER=
dummy BEFORE pygame is imported (inside the function, so the PARENT —
which imports this module for sim_process_main — keeps its real video
driver), and a 1x1 dummy display is created (Game.__init__ builds shield
Surfaces; Ship uses pygame.Vector2). The sim path (_step / snapshot /
render_model) never touches the display surfaces — only draw() does — so
a Game with None surfaces is safe here.
"""
import os

import threading
import time
import traceback


class _Worker:
    """Just enough of a NetWorker for the SimThread inside the sim
    process. The SimThread calls it exactly as it would the real
    NetWorker (the call sites are UNCHANGED — plan_12 M4):
      * publish_snapshot(sim_time, snap) — once per sim step (60 Hz);
      * send(msg) — T_BEAM / T_ECHO wire messages, per stepped iteration.

    plan_13 (Phase 1): instead of sending to a socket, these are FORWARDED
    over the DEDICATED net Queue as ("snap", sim_time, snap) / ("beam",
    msg) / ("echo", msg). The host's dedicated NET process (net_process.py)
    drains that Queue and serializes + sends each message the moment it
    lands — the in-process hand-off replaced by one pipe hop (~sub-ms).
    The net Queue is SEPARATE from the model Queue so the render thread's
    draw is out of the snapshot's send path (the net process, not the
    render thread, drains it). put_nowait + drop-when-full: a dropped
    snapshot is a network loss the client's interpolation buffer already
    handles (it drops out-of-order/late arrivals too), so this is the
    right semantics, and the bridge must never block (a blocked bridge
    can't drain commands).

    It ALSO records the sim_time of EVERY publish_snapshot (one per sim
    step) in `stamps` — the child-side snapshot-stamp cadence. This is
    the authoritative measure of "is the sim stepping smoothly in its own
    process": it is immune to pipe backpressure (the parent-side model
    stream can drop models when the parent is busy, which would corrupt a
    parent-side cadence measurement)."""

    def __init__(self, net_queue):
        self._q = net_queue
        self.stamps = []
        # Per-publish snap log (F3 telemetry, plan_13 follow-up): one row
        # per snapshot PUBLISH — (wall, mono, sim_time, seq, drops).
        # `wall`/`mono` are the publish time (the moment the sim stamped
        # the snapshot), `seq` is the 1-based publish count, `drops` is
        # the RUNNING count of snaps lost to a full net Queue. The mono
        # deltas are the authoritative "did the sim batch?" signal (a
        # descheduled sim process or a slow step shows up here, not in
        # sim_time — which is always a clean multiple of 16.7 ms); the
        # drops deltas are the "was the net process too slow to drain?"
        # signal. Written by the sim process (its own machine) — the
        # render process toggles it via ("command", "snap_log", path).
        self._snap_log_f = None
        self._snap_seq = 0
        self._snap_drops = 0

    def set_snap_log(self, path):
        """Open (path) or close (None) the per-publish snap log. Called
        from the bridge loop on ("command", "snap_log", path) — the
        render process's F3 edge. The file lives where the render
        process's CSVs land (the path is absolute, the repo root)."""
        if path is None:
            if self._snap_log_f is not None:
                self._snap_log_f.close()
                self._snap_log_f = None
            return
        if self._snap_log_f is None:
            self._snap_log_f = open(path, "w", newline="")
            self._snap_log_f.write("wall,mono,sim_time,seq,drops\n")

    def publish_snapshot(self, sim_time, snap):
        self.stamps.append(sim_time)   # child-side cadence measurement
        if self._snap_log_f is not None:
            self._snap_seq += 1
            self._snap_log_f.write("%.3f,%.3f,%.4f,%d,%d\n" % (
                time.time(), time.monotonic(), sim_time,
                self._snap_seq, self._snap_drops))
            if self._snap_seq % 60 == 0:   # ~1 s of rows: bound the buffer
                self._snap_log_f.flush()
        self._put(("snap", sim_time, snap))

    def send(self, msg):
        # T_BEAM / T_ECHO (the only message types the SimThread sends).
        kind = msg.get("type")
        if kind == "beam":
            self._put(("beam", msg))
        elif kind == "echo":
            self._put(("echo", msg))
        # Any other type is dropped (there is no network here; the
        # SimThread only ever sends beams + echoes).

    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except Exception:
            # Queue full (parent busy): drop it. A dropped snapshot is a
            # network loss the client's interpolation buffer already
            # handles; a dropped beam/echo is a missed 0.15 s flash / one
            # echoed input — both degrade gracefully (the client holds the
            # last echoed input, like a dropped echo today).
            if item[0] == "snap":
                # Count the drop (the snap log's `drops` column): a
                # rising drops count means the NET process was too slow
                # to drain the queue — the 33 ms client gaps are then
                # net-process backpressure, not sim batching.
                self._snap_drops += 1


def _build_game(seed, players, local_hull=None, local_loadout=None,
                client_hull=None, client_loadout=None):
    """Build a headless Game. The display surfaces are None: the sim path
    never reads them (verified — all 23 uses are in draw helpers), and
    draw() is never called in the sim process (M3 draws in the render
    process on a shell Game).

    `local_hull`/`local_loadout` go into player 0 (the host's ship). For a
    2P sim, `client_hull`/`client_loadout` replace the player-1 placeholder
    (the client's real fit, from the join handshake) — the authoritative
    sim needs the client's hull for correct physics (mass, weapons, shield)
    and the render process's shell needs it for the ship-body/shield render.
    None -> the default hull/loadout (the M1/M2 simple setup form)."""
    from .game import Game
    from .ship import Ship
    g = Game(None, None, None, None, None, None,
             hull=local_hull, loadout=local_loadout,
             seed=seed, players=players)
    if players == 2 and client_hull is not None:
        g.set_player_ship(1, Ship(hull=client_hull, loadout=client_loadout))
    return g


def sim_process_main(cmd_conn, model_queue, net_queue, remote_input_conn):
    """The spawn target (MUST be a top-level function: spawn re-imports
    this module by name to find it). Runs the authoritative sim in this
    process.

    Four channels (all from the spawn context):
      `cmd_conn`    a Pipe, parent -> child: ("setup", ...) (6-arg M3
                    hulls/loadouts/seed form, or the 3-arg M1/M2
                    seed/players form) | ("input", ShipInput) |
                    ("command", name, arg) | ("quit",). Small, low-rate,
                    ordered — a Pipe is fine. (plan_13: the client's
                    input NO LONGER rides this Pipe — it arrives on
                    `remote_input_conn`.)
      `model_queue` a bounded Queue, child -> parent: the render model
                    (60 Hz) + ("ready",) / ("crash", tb) / ("done", ...).
                    Drained by the RENDER thread (the same thread as the
                    draw — a locked decision).
      `net_queue`   a bounded Queue, child -> parent: the wire messages
                    (("snap", sim_time, snap) / ("beam", msg) /
                    ("echo", msg) — the SimThread's worker hand-off).
                    Drained by the host's dedicated NET process
                    (net_process.py) — NOT the render thread, so the
                    draw is out of the snapshot's send path.
                    Both child -> parent Queues are BOUNDED with
                    put_nowait: when the drain is busy a queue fills and
                    the OLDEST items are dropped — latest-wins, exactly
                    the semantics the render thread wants (it draws the
                    newest model); a dropped snapshot is a network loss
                    the client's interpolation buffer already handles. A
                    Pipe would BLOCK the child's bridge on send() when
                    the drain is busy, and a blocked bridge can't drain
                    commands — so both channels must be drop-on-full,
                    not back-pressure.
      `remote_input_conn` a Pipe, net -> child (plan_13 Phase 1): the
                    client's input (("remote_input", ShipInput)) +
                    forwarded commands (("command", name, arg), e.g.
                    ("command", "respawn", 1) — the net process forwards
                    the client's T_RESPAWN here). The bridge loop drains
                    it each iteration (a poll()), exactly like cmd_conn.
                    An EOF on it (the net process died) is treated like a
                    cmd_conn EOF (the render process died) — shut down
                    cleanly.

    Exits cleanly on ("quit",) or on a cmd_conn / remote_input_conn EOF
    (the render / net process died), or on a crash (after sending
    ("crash", traceback))."""
    # Headless: MUST be set before pygame is first imported in THIS
    # process. Set here (the child-only entry point), NOT at module
    # level: the render process imports this module (for sim_process_main)
    # and a module-level setdefault would force the DUMMY video driver on
    # the parent too — the host window would init headless (the game runs,
    # but there is no window). SDL reads the driver at pygame.init() time,
    # so setting it here (before the import below) is safe.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))   # dummy display for Surfaces/Vector2

    from .sim_thread import SimThread

    st = None
    worker = None
    try:
        # --- setup (blocking: the parent sends it right after spawn) ---
        msg = cmd_conn.recv()
        if msg[0] != "setup":
            raise RuntimeError("expected setup, got %r" % (msg,))
        if len(msg) == 6:
            # M3: ("setup", local_hull, local_loadout, client_hull,
            #      client_loadout, seed) — the host's + client's REAL fits
            #      (the plan's original sketch). The authoritative sim needs
            #      the client's hull for correct physics; the render
            #      process's shell needs it for the ship-body/shield render.
            #      players=2 (a real 2P game).
            _, lh, ll, ch, cl, seed = msg
            game = _build_game(seed, 2, lh, ll, ch, cl)
        else:
            # M1/M2: ("setup", seed, players) — default hulls (the tests).
            _, seed, players = msg
            game = _build_game(seed, players)
        worker = _Worker(net_queue)
        st = SimThread(game, worker)
        st.start()
        model_queue.put(("ready",))

        # --- the bridge loop: drain commands, publish the latest model ---
        # The SimThread (a thread in THIS process) steps the Game at 60 Hz
        # and sets st.latest_model on each step. This loop reads the
        # newest model and ships it to the parent. It does almost no CPU
        # work (a pipe recv-drain + one put_nowait), so it does not
        # starve the SimThread via the GIL — the contention we are
        # eliminating is between the SIM and the RENDER, which are now in
        # separate processes.
        last_model = None
        t_start = time.monotonic()
        # Guard the command Pipe's recv: the bridge loop is the only thread
        # in THIS process that touches the Pipe (the SimThread never does).
        # The lock is cheap insurance against a future second reader.
        # (plan_13: the SEND side has a single sender now — the render
        # thread; the worker's input sink is gone, the net process forwards
        # the client's input on the remote_input Pipe.)
        cmd_lock = threading.Lock()

        def _handle_cmd(m):
            """Apply one command-Pipe message. Returns True on ("quit",)
            (the caller returns after the "done" report)."""
            kind = m[0]
            if kind == "quit":
                st.stop()
                # M1 diagnostic: report the child's OWN step count +
                # sim clock + the per-step snapshot stamps so the
                # parent can verify the sim ran at 1x real time with a
                # smooth cadence (child-side — immune to queue
                # backpressure, which would corrupt a parent-side
                # model-stream measurement).
                model_queue.put(("done", st._steps_total, game.sim_time,
                                 time.monotonic() - t_start,
                                 worker.stamps))
                return True
            elif kind == "input":
                st.publish_input(m[1])
            elif kind == "command":
                name = m[1]
                arg = m[2] if len(m) > 2 else None
                if name == "snap_log":
                    # F3 telemetry: open/close the per-publish snap log
                    # (the render process's F3 edge sends the path, or
                    # None to close). NOT a sim command — it configures
                    # the worker's logging, so it is handled here, not
                    # routed to st.command.
                    worker.set_snap_log(arg)
                else:
                    st.command(name, arg)
            # ("remote_input", ...) no longer rides the command Pipe
            # (plan_13): the client's input arrives on the remote_input
            # Pipe (net -> sim) and is handled in the loop below.
            return False

        while True:
            # Drain everything the parent sent since last time (input /
            # command / quit). Latest input wins (the SimThread applies
            # the latest published input each step — pinned #5).
            while cmd_conn.poll():
                with cmd_lock:
                    try:
                        m = cmd_conn.recv()
                    except EOFError:
                        # The parent died (the Pipe is closed): shut down
                        # cleanly. No "done" report — there is no parent
                        # left to read it (the finally stops the SimThread).
                        return
                if _handle_cmd(m):
                    return
            # Drain the net process's remote_input Pipe (plan_13 Phase 1):
            # the client's input (("remote_input", ShipInput)) + forwarded
            # commands (("command", name, arg), e.g. ("command", "respawn",
            # 1) — the net process forwards the client's T_RESPAWN here).
            # remote_input is a reference swap — exactly what the
            # in-process game.set_remote_input does: _step reads
            # game.remote_input each tick and applies it to player 1.
            # ShipInput is a plain dataclass the sim only READS (the
            # world-cap replace() builds a new object), so the swap is safe
            # across the process boundary.
            while remote_input_conn.poll():
                try:
                    m = remote_input_conn.recv()
                except EOFError:
                    # The net process died (the Pipe is closed): shut down
                    # cleanly, like a cmd_conn EOF. No "done" report — the
                    # render process is still alive and notices the net
                    # process's exit on its own status Pipe.
                    return
                kind = m[0]
                if kind == "remote_input":
                    game.remote_input = m[1]
                elif kind == "command":
                    st.command(m[1], m[2] if len(m) > 2 else None)
            # Publish the newest model (only when it changed — the
            # SimThread publishes on a step, so the model object is
            # stable between steps). put_nowait + drop-when-full: the
            # render thread wants the NEWEST model, never a stale one,
            # and the bridge must never block (a blocked bridge can't
            # drain commands).
            model = st.latest_model
            if model is not None and model is not last_model:
                try:
                    model_queue.put_nowait(("model", model))
                    last_model = model
                except Exception:
                    # Queue full (parent busy): drop the NEWEST and keep
                    # last_model as-is — the parent will get the next
                    # model, which is newer still. (Dropping the newest
                    # is safe: the render thread only ever draws the
                    # newest it has.)
                    pass
            time.sleep(0.001)   # ~1 ms loop (the SimThread's WAIT pace)

    except BaseException:
        # A crash in the sim must not take the render process down
        # silently: report it, then exit (the parent shows a notice).
        try:
            model_queue.put(("crash", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.set_snap_log(None)   # flush + close the snap log
            except Exception:
                pass
        pygame.quit()


if __name__ == "__main__":
    # Direct run is not supported (spawn needs a picklable top-level
    # target + a Pipe from the parent). Use test_sim_process.py.
    raise SystemExit("sim_process_main is a spawn target; "
                     "run test_sim_process.py")