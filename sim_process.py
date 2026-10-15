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
network bridge (snap/beam/echo -> the render process's NetWorker).

IPC (spawn context — see plan_12 section E):
  parent -> child (Pipe, small + low-rate + ordered):
                   ("setup", seed, players) | ("input", ShipInput) |
                   ("command", name, arg) | ("remote_input", ShipInput) |
                   ("quit",)
  child -> parent (bounded Queue, maxsize=4, put_nowait drop-when-full):
                   ("model", render_model_dict)   (60 Hz, on a step)
                   ("ready",) | ("crash", traceback_text) |
                   ("done", steps, sim_time, wall, stamps)
  The model channel is a bounded Queue, NOT a Pipe: a Pipe blocks the
  child's bridge on send() when the parent is busy (draw holds the GIL),
  and a blocked bridge can't drain commands. A bounded queue drops the
  oldest models when full — latest-wins, exactly what the render thread
  wants (it draws the newest model, never a stale one).

The child is HEADLESS: SDL_VIDEODRIVER/AUDIODRIVER=dummy are set BEFORE
pygame is imported, and a 1x1 dummy display is created (Game.__init__
builds shield Surfaces; Ship uses pygame.Vector2). The sim path
(_step / snapshot / render_model) never touches the display surfaces —
only draw() does — so a Game with None surfaces is safe here.
"""
import os
# Headless: MUST be set before pygame is first imported in this process.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import traceback


class _Worker:
    """Just enough of a NetWorker for the SimThread inside the sim
    process. M1 has no network: the wire snapshot is M4's job (the render
    process's NetWorker sends it), and send() drops T_BEAM / T_ECHO (also
    M4). Kept as a class so M4 can forward these to the parent over the
    pipe without changing the SimThread call sites.

    M1 also records the sim_time of EVERY publish_snapshot (one per sim
    step) in `stamps` — the child-side snapshot-stamp cadence. This is
    the authoritative measure of "is the sim stepping smoothly in its own
    process": it is immune to pipe backpressure (the parent-side model
    stream can drop models when the parent is busy, which would corrupt a
    parent-side cadence measurement)."""

    def __init__(self):
        self.stamps = []

    def publish_snapshot(self, sim_time, snap):
        self.stamps.append(sim_time)   # M1 cadence measurement (M4: send)

    def send(self, msg):
        pass   # M4: forward ("beam"/"echo", msg) to the parent


def _build_game(seed, players):
    """Build a headless Game. The display surfaces are None: the sim path
    never reads them (verified — all 23 uses are in draw helpers), and
    draw() is never called in the sim process (M3 draws in the render
    process on a shell Game)."""
    import pygame
    from .game import Game
    return Game(None, None, None, None, None, None,
                seed=seed, players=players)


def sim_process_main(cmd_conn, model_queue):
    """The spawn target (MUST be a top-level function: spawn re-imports
    this module by name to find it). Runs the authoritative sim in this
    process.

    Two channels (both from the spawn context):
      `cmd_conn`    a Pipe, parent -> child: ("setup", seed, players) |
                    ("input", ShipInput) | ("command", name, arg) |
                    ("remote_input", ShipInput) | ("quit",). Small,
                    low-rate, ordered — a Pipe is fine.
      `model_queue` a Queue(maxsize=4), child -> parent: the render model
                    (60 Hz) + ("ready",) / ("crash", tb) / ("done", ...).
                    A BOUNDED queue with put_nowait: when the parent is
                    busy (draw holds the GIL) the queue fills and the
                    OLDEST models are dropped — latest-wins, exactly the
                    semantics the render thread wants (it draws the
                    newest model). A Pipe would BLOCK the child's bridge
                    on send() when the parent is busy, and a blocked
                    bridge can't drain commands — so the model channel
                    must be drop-on-full, not back-pressure.

    Exits cleanly on ("quit",) or on cmd_conn EOF (the parent died), or
    on a crash (after sending ("crash", traceback))."""
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))   # dummy display for Surfaces/Vector2

    from .sim_thread import SimThread

    st = None
    try:
        # --- setup (blocking: the parent sends it right after spawn) ---
        msg = cmd_conn.recv()
        if msg[0] != "setup":
            raise RuntimeError("expected setup, got %r" % (msg,))
        _, seed, players = msg
        game = _build_game(seed, players)
        worker = _Worker()
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
        while True:
            # Drain everything the parent sent since last time (input /
            # command / quit). Latest input wins (the SimThread applies
            # the latest published input each step — pinned #5).
            while cmd_conn.poll():
                m = cmd_conn.recv()
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
                    return
                elif kind == "input":
                    st.publish_input(m[1])
                elif kind == "command":
                    st.command(m[1], m[2] if len(m) > 2 else None)
                elif kind == "remote_input":
                    # The CLIENT's input (M4: the render process's NetWorker
                    # input sink sends it; M2: the test sends it directly).
                    # A reference swap — exactly what the in-process
                    # game.set_remote_input does: _step reads
                    # game.remote_input each tick and applies it to player 1.
                    # ShipInput is a plain dataclass the sim only READS
                    # (the world-cap replace() builds a new object), so the
                    # swap is safe across the process boundary.
                    game.remote_input = m[1]
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
        pygame.quit()


if __name__ == "__main__":
    # Direct run is not supported (spawn needs a picklable top-level
    # target + a Pipe from the parent). Use test_sim_process.py.
    raise SystemExit("sim_process_main is a spawn target; "
                     "run test_sim_process.py")