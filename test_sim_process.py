"""Session 12: the separate-process sim.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_sim_process

Two tests:
  M1 — A/B proof of the GIL fix (in-process batches, separate-process
       does not). See the arm_* functions below.
  M2 — input + commands + remote input over the command Pipe: the parent
       drives the spawned sim and verifies each control path (local input
       -> player 0, command -> targeting, remote input -> player 1) lands
       in the model. See test_m2_input_commands_remote.

WHY THIS TEST
The residual ~8 px snap tail is host-side: the host's render thread holds
the GIL for its whole draw (~18-25 ms — longer than the 16.7 ms sim tick),
starving the SAME-PROCESS sim thread, which then batches 2 ticks and the
snapshot stamps come out ragged (17, 33 ms). plan_12's fix moves the sim
to a separate PROCESS (no shared GIL). This test proves the mechanism:

  A. IN-PROCESS (reproduces the bug): the SimThread + a render-like GIL
     load in the SAME process. The GIL load starves the sim thread, so the
     published sim_time gaps show 2-tick (33 ms) batching.
  B. SEPARATE PROCESS (the fix): the sim in a spawned process; the SAME
     render-like GIL load runs in the PARENT. The sim's sim_time gaps are
     a smooth 1-tick (16.7 ms) cadence — the parent's GIL load can't
     starve a different process.

The metric is the gap between consecutive published sim_times (the
snapshot-stamp cadence the client's ghost clock sees). A smooth sim has
every gap ~= TICK (16.7 ms); a starved sim has gaps of 2*TICK (33 ms) or
more. The test asserts B is smooth and A is not (A batches).

Headless: SDL_VIDEODRIVER/AUDIODRIVER=dummy.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import sys
import time
import threading
import multiprocessing
import queue as _queue

from .config import WIDTH, HEIGHT
from .game import Game, STEP
from .intent import ShipInput
from .sim_thread import SimThread
from .sim_process import sim_process_main

TICK = STEP
RUN_SECONDS = 3.0          # how long each A/B arm runs
GIL_HOLD_MS = 20.0         # the render-like GIL hold per frame (~host draw)
FRAME_MS = 20.0            # the render frame period (~50 fps, host-like)
SWITCH_INTERVAL = 0.040    # force long GIL holds (the render doesn't yield)


def check(label, cond, extra=""):
    print(("PASS: " if cond else "FAIL: ") + label
          + (("  " + extra) if extra else ""))
    return bool(cond)


def _gil_hold(ms):
    """Hold the GIL for ~`ms` milliseconds with a pure-Python busy loop.

    A proxy for the render thread's draw: it is CPU-bound Python (holds
    the GIL) for a duration longer than one sim tick, which is exactly
    what starves the same-process sim thread. With a high switch interval
    (set in main) it does not yield to the sim thread during the hold."""
    t0 = time.monotonic()
    x = 0
    while (time.monotonic() - t0) * 1000.0 < ms:
        x += 1
    return x


def _render_load(stop_event, hold_ms, frame_ms):
    """A render-like thread: hold the GIL `hold_ms` per `frame_ms` frame.

    Runs until `stop_event` is set. This is the load that starves the
    in-process sim thread (arm A) and the load the parent applies while
    the sim is in a separate process (arm B)."""
    frame_s = frame_ms / 1000.0
    while not stop_event.is_set():
        t0 = time.monotonic()
        _gil_hold(hold_ms)
        elapsed = time.monotonic() - t0
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)   # the non-draw part of the frame


class _StampWorker:
    """A NetWorker stand-in that records the sim_time of every
    publish_snapshot (the snapshot-stamp cadence)."""

    def __init__(self):
        self.stamps = []

    def publish_snapshot(self, sim_time, snap):
        self.stamps.append(sim_time)

    def send(self, msg):
        pass


def _gap_stats(stamps):
    """Classify the gaps between consecutive sim_time stamps.

    Returns (n_gaps, n_1tick, n_2tick, n_more, max_gap). A gap is
    '1-tick' if it is in [0.5, 1.5)*TICK, '2-tick' if in [1.5, 2.5)*TICK,
    'more' if >= 2.5*TICK."""
    gaps = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    if not gaps:
        return 0, 0, 0, 0, 0.0
    n1 = sum(1 for g in gaps if 0.5 * TICK <= g < 1.5 * TICK)
    n2 = sum(1 for g in gaps if 1.5 * TICK <= g < 2.5 * TICK)
    nm = sum(1 for g in gaps if g >= 2.5 * TICK)
    return len(gaps), n1, n2, nm, max(gaps)


def _report(tag, stamps):
    n, n1, n2, nm, mx = _gap_stats(stamps)
    batched = n2 + nm
    pct = (100.0 * batched / n) if n else 0.0
    print("  [%s] %d gaps: 1-tick=%d  2-tick=%d  >2-tick=%d  "
          "batched=%d (%.1f%%)  max_gap=%.1f ms"
          % (tag, n, n1, n2, nm, batched, pct, mx * 1000.0))
    return n, batched, pct


def arm_in_process():
    """Arm A: the SimThread + the render GIL load in the SAME process.

    Returns (stamps, batched_pct). The GIL load starves the sim thread,
    so the stamps should show 2-tick batching."""
    g = Game(None, None, None, None, None, None, seed=1234, players=2)
    worker = _StampWorker()
    st = SimThread(g, worker)
    st.start()
    stop = threading.Event()
    loader = threading.Thread(target=_render_load,
                              args=(stop, GIL_HOLD_MS, FRAME_MS))
    loader.start()
    time.sleep(RUN_SECONDS)
    stop.set()
    loader.join()
    st.stop()
    return worker.stamps


def arm_separate_process():
    """Arm B: the sim in a spawned process; the render GIL load in the
    PARENT, in the SAME thread as the model drain (the real render loop:
    draw + drain every frame). Returns the child-reported per-step
    snapshot stamps (the child-side cadence — immune to pipe
    backpressure). The parent's GIL load cannot starve the sim (separate
    process / separate GIL), so the stamps should be a smooth 1-tick
    cadence.

    NOTE: the drain + draw MUST be the same thread. If the GIL load ran
    in a separate thread from the drain, the drain would be starved, the
    pipe would fill, and the CHILD would block on send() — which would
    starve the child's sim too (a false positive that the separate
    process doesn't help). The real render loop drains every frame, so
    the pipe never fills."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()       # parent -> child commands
    model_queue = ctx.Queue(maxsize=4)         # child -> parent models
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue))
    p.start()
    child_conn.close()   # the parent does not use the child end
    parent_conn.send(("setup", 1234, 2))
    # Wait for ready (the child builds the Game + starts the SimThread).
    while True:
        m = model_queue.get()
        if m[0] == "ready":
            break
        elif m[0] == "crash":
            print("  [separate-process] SIM CRASHED:\n" + m[1])
            parent_conn.close()
            p.join()
            return []
    # The render loop: each frame, DRAIN the model queue (the render
    # thread reads the newest model) then DRAW (the GIL load). Same
    # thread, every frame. The queue is bounded + drop-on-full, so a
    # busy frame just drops stale models — the child never blocks.
    frame_s = FRAME_MS / 1000.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < RUN_SECONDS:
        ft0 = time.monotonic()
        while True:          # drain (read the newest model)
            try:
                m = model_queue.get_nowait()
            except Exception:
                break
            if m[0] == "crash":
                print("  [separate-process] SIM CRASHED:\n" + m[1])
        _gil_hold(GIL_HOLD_MS)      # draw (the GIL load)
        elapsed = time.monotonic() - ft0
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)
    parent_conn.send(("quit",))
    # The child's done report: its OWN step count + sim clock + the
    # per-step snapshot stamps (the child-side cadence). BLOCKING get
    # with a timeout until "done" arrives: the child's put(done) can
    # only complete once the parent drains the queue, so a get_nowait
    # loop that exits on an empty queue would race past it.
    import queue as _queue
    done = None
    t_deadline = time.monotonic() + 3.0
    while time.monotonic() < t_deadline:
        try:
            m = model_queue.get(timeout=0.2)
        except _queue.Empty:
            if not p.is_alive():
                break
            continue
        if m[0] == "done":
            done = m
            break
    p.join(timeout=3.0)
    if p.is_alive():
        p.terminate()
        p.join()
    parent_conn.close()
    if done is not None:
        _, steps, simt, wall, stamps = done
        print("  [separate-process] CHILD step report: steps=%d sim_time=%.3f "
              "wall=%.3f -> sim/wall=%.2fx (%.1f steps/s)"
              % (steps, simt, wall, simt / wall, steps / wall))
        return stamps
    return []


# ---------------------------------------------------------------------------
# M2: input + commands + remote input over the command Pipe
# ---------------------------------------------------------------------------
#
# The M1 A/B test above proves the sim STEPS smoothly in its own process.
# M2 proves the CONTROL path: the parent can drive the sim over the command
# Pipe — local input (player 0), commands (targeting), and the client's
# remote input (player 1) — and each shows up in the model the parent reads
# from the model Queue. This is the exact contract M4's NetWorker bridge
# relies on; here the test sends the messages directly (no network).
#
#   ("input", ShipInput)         -> st.publish_input  -> player 0
#   ("command", name, arg)       -> st.command        -> e.g. targeting
#   ("remote_input", ShipInput)  -> game.remote_input -> player 1
#
# The model is read by DRAINING the model Queue to the newest (latest-wins,
# exactly what the render loop does). No GIL load is applied here — that was
# M1's A/B; M2 only checks that the messages take effect.

M2_SETTLE_S = 0.50          # how long each input/command is held (sim time)
M2_READY_TIMEOUT_S = 10.0   # spawn + pygame import + Game build budget


def _drain_newest(model_queue):
    """One non-blocking drain of the model Queue to the NEWEST model
    (latest-wins — the render loop's read pattern). Returns the newest model
    dict read this drain, or None if the queue was empty. A crash message
    aborts with a raised error. "ready" / "done" are not models; ignored."""
    newest = None
    while True:
        try:
            m = model_queue.get_nowait()
        except _queue.Empty:
            break
        if m[0] == "model":
            newest = m[1]
        elif m[0] == "crash":
            raise RuntimeError("SIM CRASHED:\n" + m[1])
    return newest


def _settle_and_read(model_queue, duration):
    """Hold the sim for `duration` seconds while DRAINING the model Queue
    continuously (every ~10 ms), then return the NEWEST model seen.

    This mirrors the render loop EXACTLY: it drains the queue every frame
    and keeps the newest. Draining continuously is load-bearing — the model
    channel is a BOUNDED queue (maxsize=4) with drop-when-full, so if the
    parent stopped draining (e.g. slept) the queue would fill and the child's
    newer models would be DROPPED, and the parent would only ever see stale
    pre-input models. The render loop never has this problem because it
    drains every frame; this test must do the same."""
    newest = None
    t_deadline = time.monotonic() + duration
    while time.monotonic() < t_deadline:
        m = _drain_newest(model_queue)
        if m is not None:
            newest = m
        time.sleep(0.01)
    # Final drain so we capture the very newest model before returning.
    m = _drain_newest(model_queue)
    if m is not None:
        newest = m
    return newest


def _ship_pos_dist(pos, center):
    return math.hypot(pos[0] - center[0], pos[1] - center[1])


def test_m2_input_commands_remote():
    """M2: spawn the sim, drive it over the command Pipe, and verify each
    control path lands in the model. Returns True when all checks pass."""
    ok = True
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    model_queue = ctx.Queue(maxsize=4)
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue))
    p.start()
    child_conn.close()
    parent_conn.send(("setup", 1234, 2))

    # Wait for ready (the child built the Game + started the SimThread).
    ready = False
    t_deadline = time.monotonic() + M2_READY_TIMEOUT_S
    while time.monotonic() < t_deadline:
        try:
            m = model_queue.get(timeout=0.2)
        except _queue.Empty:
            continue
        if m[0] == "ready":
            ready = True
            break
        elif m[0] == "crash":
            print("  [M2] SIM CRASHED:\n" + m[1])
            break
    ok &= check("M2: sim process became ready", ready)
    if not ready:
        parent_conn.send(("quit",))
        p.join(timeout=3.0)
        if p.is_alive():
            p.terminate()
            p.join()
        parent_conn.close()
        return ok

    # Baseline: with no input, both ships sit at the spawn center. Grab a
    # model to confirm the sim is stepping and record the center. Keep this
    # SHORT: the seed-1234 world kills a stationary center ship at ~8 s, so
    # the whole test must finish well before then (it takes ~2 s).
    center = (WIDTH / 2.0, HEIGHT / 2.0)
    base = _settle_and_read(model_queue, 0.2)
    ok &= check("M2: baseline model received", base is not None)
    if base is None:
        parent_conn.send(("quit",))
        p.join(timeout=3.0)
        if p.is_alive():
            p.terminate()
            p.join()
        parent_conn.close()
        return ok
    base_p0 = base["players"][0]
    base_p1 = base["players"][1]
    print("  [M2] baseline p0 pos=%s vel=%s angle=%.3f targeting=%s"
          % (tuple(round(v, 2) for v in base_p0["pos"]),
             tuple(round(v, 2) for v in base_p0["vel"]),
             base_p0["angle"], base_p0["targeting_on"]))
    print("  [M2] baseline p1 pos=%s angle=%.3f"
          % (tuple(round(v, 2) for v in base_p1["pos"]), base_p1["angle"]))

    # --- 1. LOCAL input drives player 0 -------------------------------
    # Hold forward thrust for M2_SETTLE_S while draining (the render-loop
    # pattern); the sim applies the LATEST published input each tick
    # (pinned #5), so p0 accelerates away from center.
    parent_conn.send(("input", ShipInput(thrust_fwd=1.0)))
    m = _settle_and_read(model_queue, M2_SETTLE_S)
    ok &= check("M2: model after local input", m is not None)
    if m is not None:
        p0 = m["players"][0]
        moved = _ship_pos_dist(p0["pos"], center)
        speed = math.hypot(p0["vel"][0], p0["vel"][1])
        # SHIP_ACCEL=320 px/s^2 for ~0.5 s -> ~160 px/s, ~40 px from center.
        ok &= check("M2: LOCAL input moved player 0 (pos off center)",
                    moved > 5.0, "dist=%.1f px" % moved)
        ok &= check("M2: LOCAL input gave player 0 velocity",
                    speed > 20.0, "speed=%.1f px/s" % speed)
        print("  [M2] p0 after thrust: pos=%s vel=%s (dist=%.1f px)"
              % (tuple(round(v, 2) for v in p0["pos"]),
                 tuple(round(v, 2) for v in p0["vel"]), moved))
    # Release the thrust (the player lets go of W): the input persists in
    # the sim until replaced, so without this p0 would keep accelerating to
    # max speed through the later phases and could fly into an asteroid.
    # Now it only coasts at ~160 px/s for the rest of the ~1 s test.
    parent_conn.send(("input", ShipInput()))

    # --- 2. A COMMAND is applied by the sim ---------------------------
    # "targeting" flips player 0's targeting_on (SimThread._apply_command
    # toggles g.ship.targeting_on). Baseline is False, so it must become
    # True.
    parent_conn.send(("command", "targeting"))
    m = _settle_and_read(model_queue, M2_SETTLE_S)
    ok &= check("M2: model after command", m is not None)
    if m is not None:
        p0 = m["players"][0]
        ok &= check("M2: COMMAND flipped player 0 targeting_on",
                    p0["targeting_on"] is True,
                    "targeting_on=%s" % p0["targeting_on"])
        print("  [M2] p0 after 'targeting': targeting_on=%s"
              % p0["targeting_on"])

    # --- 3. REMOTE input drives player 1 ------------------------------
    # The client's input path: ("remote_input", ShipInput) -> the sim sets
    # game.remote_input (a reference swap, exactly what the in-process
    # game.set_remote_input does); _step applies it to player 1. Hold a
    # turn for M2_SETTLE_S; p1's angle must change from its baseline.
    parent_conn.send(("remote_input", ShipInput(turn=1.0)))
    m = _settle_and_read(model_queue, M2_SETTLE_S)
    ok &= check("M2: model after remote input", m is not None)
    if m is not None:
        p1 = m["players"][1]
        dangle = abs(p1["angle"] - base_p1["angle"])
        # ROT_SPEED=3.6 rad/s * turn_rate_factor(1.0) for ~0.5 s -> ~1.8 rad.
        ok &= check("M2: REMOTE input turned player 1 (angle changed)",
                    dangle > 0.3, "dangle=%.3f rad" % dangle)
        print("  [M2] p1 after remote turn: angle=%.3f (d=%.3f rad)"
              % (p1["angle"], dangle))

    # --- shutdown ------------------------------------------------------
    parent_conn.send(("quit",))
    # Blocking get for the done report (the child's put(done) only
    # completes once the parent drains the queue — see M1's note).
    done = None
    t_deadline = time.monotonic() + 3.0
    while time.monotonic() < t_deadline:
        try:
            m = model_queue.get(timeout=0.2)
        except _queue.Empty:
            if not p.is_alive():
                break
            continue
        if m[0] == "done":
            done = m
            break
    p.join(timeout=3.0)
    if p.is_alive():
        p.terminate()
        p.join()
    parent_conn.close()
    if done is not None:
        _, steps, simt, wall, _stamps = done
        print("  [M2] CHILD step report: steps=%d sim_time=%.3f wall=%.3f"
              % (steps, simt, wall))
    return ok


def main():
    ok = True
    # Force long GIL holds so the render-like load actually starves the
    # in-process sim thread (the default 5 ms switch interval would let
    # the sim thread run between render slices and hide the batching).
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    try:
        print("M1: separate-process sim — A/B GIL test")
        print("  (GIL hold %.0f ms/frame, switch interval %.0f ms, "
              "run %.1f s each arm)\n" % (GIL_HOLD_MS, SWITCH_INTERVAL * 1000,
                                          RUN_SECONDS))

        print("A. IN-PROCESS (sim thread + render GIL load, same process):")
        stamps_a = arm_in_process()
        n_a, batched_a, pct_a = _report("in-process", stamps_a)

        print("\nB. SEPARATE PROCESS (sim in spawned process, GIL load in parent):")
        stamps_b = arm_separate_process()
        n_b, batched_b, pct_b = _report("separate", stamps_b)

        print()
        # The separate-process sim must actually be running (enough stamps
        # to measure a cadence) and at ~1x real time.
        ok &= check("separate-process: the sim ran and published a cadence",
                    n_b > 30, "n_gaps=%d" % n_b)
        if stamps_b:
            rate = (stamps_b[-1] - stamps_b[0]) / max(
                1e-9, (RUN_SECONDS))
            # sim_time span over the run ~= RUN_SECONDS at 1x real time.
            ok &= check("separate-process: sim clock runs at ~1x real time",
                        0.8 <= (stamps_b[-1] - stamps_b[0]) / RUN_SECONDS <= 1.2,
                        "sim span=%.2f s over %.1f s wall"
                        % (stamps_b[-1] - stamps_b[0], RUN_SECONDS))
        # THE FIX: the separate-process cadence is smooth (no 2-tick
        # batching) — the parent's GIL load did not starve the sim.
        ok &= check("separate-process: NO 2-tick batching (the fix)",
                    pct_b < 5.0, "batched=%.1f%%" % pct_b)
        # THE BUG: the in-process cadence DOES batch (the GIL load
        # starved the sim thread) — this is what the separate process
        # eliminates. If this fails, the GIL load is not strong enough to
        # reproduce the contention (tune GIL_HOLD_MS / SWITCH_INTERVAL).
        ok &= check("in-process: 2-tick batching reproduced (the bug)",
                    pct_a > 10.0, "batched=%.1f%%" % pct_a)
        # The A/B delta: the separate process must be clearly smoother.
        ok &= check("A/B: separate-process is smoother than in-process",
                    pct_b < pct_a, "separate=%.1f%% vs in-process=%.1f%%"
                    % (pct_b, pct_a))
    finally:
        sys.setswitchinterval(old_switch)

    print("\nM1:", "ALL PASS" if ok else "FAILURES")

    # M2: input + commands + remote input over the command Pipe. Runs
    # WITHOUT the GIL load (that was M1's A/B) — it only checks that the
    # control messages take effect in the model.
    print("\nM2: input + commands + remote input over the command Pipe")
    ok &= test_m2_input_commands_remote()

    print("\nSIM PROCESS (M1+M2):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()