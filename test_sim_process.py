"""Session 12: the separate-process sim.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_sim_process

Three tests:
  M1 — A/B proof of the GIL fix (in-process batches, separate-process
       does not). See the arm_* functions below.
  M2 — input + commands + remote input over the command Pipe: the parent
       drives the spawned sim and verifies each control path (local input
       -> player 0, command -> targeting, remote input -> player 1) lands
       in the model. See test_m2_input_commands_remote.
  M3 — the real 2P setup path: the 6-arg setup message (the host's +
       client's REAL hulls/loadouts — the plan's original sketch) and the
       model's "sfx" list (plan_12 D.1: presentation sound events since
       the last publish). See test_m3_hulls_and_sfx.
  M3 e2e — the REAL run_host with a fake client over loopback: the sim
       process is spawned (6-arg setup), the shell is drawn from the
       piped model, the host's SoundBank plays the model's sfx, and the
       sim process is shut down cleanly on exit. See test_m3_host_e2e.
  M4 — the network bridge (in the M3 e2e): the fake client drives its
       ship with REAL input (a constant turn) and the test verifies the
       client -> worker -> command Pipe -> sim -> model path (player 1
       turns in the host's models) AND the sim -> render -> worker ->
       client wire bridge (snapshots + input echoes reach the client).

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
import socket
import sys
import time
import threading
import multiprocessing
import queue as _queue
import traceback

import pygame   # after the SDL dummy-driver env (above); the M3 e2e's
                # _E2EKeys reads pygame.K_* at module scope

from .config import WIDTH, HEIGHT, ROT_SPEED
from .game import Game, STEP
from .hulls import PLAYER_HULLS, default_loadout
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
    net_queue = ctx.Queue(maxsize=16)          # child -> parent wire (M4)
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue, net_queue))
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
    # The wire traffic (M4) rides the DEDICATED net queue, drained by a
    # separate thread in the real run_host (_NetPump); here the render
    # loop drains it too (the test has no pump thread — draining keeps
    # the queue from filling and dropping, which is all arm B needs).
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
        while True:          # drain the wire queue (M4 net channel)
            try:
                net_queue.get_nowait()
            except Exception:
                break
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
    net_queue = ctx.Queue(maxsize=16)          # child -> parent wire (M4)
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue, net_queue))
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


# ---------------------------------------------------------------------------
# M3: the real 2P setup path + the model's "sfx" list (plan_12 D.1)
# ---------------------------------------------------------------------------
#
# M1/M2 use the simple ("setup", seed, players) form (default hulls). A
# REAL 2P game sends the 6-arg form — the plan's original sketch:
#
#   ("setup", local_hull, local_loadout, client_hull, client_loadout, seed)
#
# The authoritative sim needs the client's hull for correct physics
# (mass, weapons, shield) and the render process's shell needs it for the
# ship-body/shield render. This test proves the 6-arg path end to end:
# the hulls LOADOUTS are picklable over the spawn Pipe, the sim builds
# player 1 from the client's REAL hull (verified through the model — the
# silas's turn_rate_factor 1.15 makes p1 turn 15% faster than the
# scout's 1.0), and the model carries the drained "sfx" list.
#
# SFX: the sim process is silent (sound=None), so _sfx/_sfx_throttled/
# _sfx_thruster RECORD events into the per-Game queue instead of playing
# them; render_model() drains that queue into the model's "sfx" list.
# The checks:
#   * the model carries a plain "sfx" list, drained per publish (a second
#     render_model with no step in between is empty — the sim process
#     publishes exactly one model per step, so this is the per-publish
#     drain the render process relies on);
#   * thrusting fires ("thruster", True) then ("thruster", False) on
#     release (the engine loop is level-triggered per step);
#   * firing the gun fires ("laser", 0.12) — the throttled variant
#     (no SoundBank in the sim -> every shot records; the RENDER
#     process's bank applies the throttle when it plays the list).

M3_TURN_S = 1.0             # how long p1's turn is held (sim time)
M3_SETTLE_S = 0.50          # how long p0's thrust+fire is held (sim time)


def test_m3_hulls_and_sfx():
    """M3: the 6-arg setup (real hulls/loadouts) + the model's "sfx" list.
    Returns True when all checks pass."""
    ok = True
    # Host: the scout (turn_rate_factor 1.0). Client: the silas (1.15) —
    # a DIFFERENT turn rate, so the model's p1 turn delta proves the sim
    # built player 1 from the client's REAL hull, not the default.
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[3]   # scout, silas
    assert host_hull.turn_rate_factor == 1.0
    assert client_hull.turn_rate_factor == 1.15

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    model_queue = ctx.Queue(maxsize=4)
    net_queue = ctx.Queue(maxsize=16)          # child -> parent wire (M4)
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue, net_queue))
    p.start()
    child_conn.close()
    # The 6-arg setup: the host's + client's REAL hulls/loadouts + seed.
    # The hulls/loadouts are frozen dataclasses of plain data — picklable
    # over the spawn Pipe (the real run_host sends exactly this).
    parent_conn.send(("setup", host_hull, default_loadout(host_hull),
                      client_hull, default_loadout(client_hull), 1234))

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
            print("  [M3] SIM CRASHED:\n" + m[1])
            break
    ok &= check("M3: sim process became ready (6-arg setup)", ready)
    if not ready:
        parent_conn.send(("quit",))
        p.join(timeout=3.0)
        if p.is_alive():
            p.terminate()
            p.join()
        parent_conn.close()
        return ok

    # --- 1. the client's REAL hull is in the sim (through the model) ---
    # Baseline: both ships at the spawn center, idle.
    base = _settle_and_read(model_queue, 0.2)
    ok &= check("M3: baseline model received", base is not None)
    if base is None:
        parent_conn.send(("quit",))
        p.join(timeout=3.0)
        if p.is_alive():
            p.terminate()
            p.join()
        parent_conn.close()
        return ok
    base_p1_angle = base["players"][1]["angle"]

    # Turn p1 (the client's ship) for M3_TURN_S. The silas's
    # turn_rate_factor is 1.15 (the scout's is 1.0), so the same input
    # turns p1 15% faster: dangle ~= ROT_SPEED * 1.15 * t. If the sim had
    # built p1 from the DEFAULT hull (scout, factor 1.0), the delta would
    # be ~13% smaller — outside the tolerance band below.
    parent_conn.send(("remote_input", ShipInput(turn=1.0)))
    m = _settle_and_read(model_queue, M3_TURN_S)
    ok &= check("M3: model after remote turn", m is not None)
    if m is not None:
        dangle = abs(m["players"][1]["angle"] - base_p1_angle)
        want = ROT_SPEED * client_hull.turn_rate_factor * M3_TURN_S
        # The settle window is wall-clock; the sim steps at 60 Hz, so the
        # actual held time is within ~3 ticks of M3_TURN_S (input pickup
        # + drain quantization) -> dangle within ~0.2 rad of `want`.
        # The band around the silas prediction (4.14 rad) EXCLUDES the
        # scout prediction (3.6 rad): 0.54 rad apart, the band is 0.25.
        lo, hi = want - 0.25, want + 0.25
        ok &= check("M3: client's REAL hull is in the sim (silas turn "
                    "rate, not the default scout)",
                    lo < dangle <= hi,
                    "dangle=%.3f rad (silas ~%.3f, scout ~%.3f)"
                    % (dangle, want, ROT_SPEED * M3_TURN_S))
        print("  [M3] p1 after remote turn: dangle=%.3f rad "
              "(expected ~%.3f for the silas)" % (dangle, want))
    parent_conn.send(("remote_input", ShipInput()))

    # --- 2. the model carries the drained "sfx" list (D.1) ---
    # Thrust p0 for M3_SETTLE_S, draining continuously (the render-loop
    # pattern). The sim is silent (sound=None), so the engine loop's
    # on/off + the gun's blips arrive as the model's "sfx" list.
    sfx_seen = []
    t_deadline = time.monotonic() + M3_SETTLE_S
    parent_conn.send(("input", ShipInput(thrust_fwd=1.0, fire=True)))
    while time.monotonic() < t_deadline:
        m = _drain_newest(model_queue)
        if m is not None:
            sfx_seen.extend(m.get("sfx", ()))
        time.sleep(0.01)
    m = _drain_newest(model_queue)
    if m is not None:
        sfx_seen.extend(m.get("sfx", ()))
    # Release the input; the engine loop must come back off.
    parent_conn.send(("input", ShipInput()))
    off_seen = []
    t_deadline = time.monotonic() + 0.2
    while time.monotonic() < t_deadline:
        m = _drain_newest(model_queue)
        if m is not None:
            off_seen.extend(m.get("sfx", ()))
        time.sleep(0.01)
    m = _drain_newest(model_queue)
    if m is not None:
        off_seen.extend(m.get("sfx", ()))

    names = [ev[0] for ev in sfx_seen]
    ok &= check("M3: model carries a plain 'sfx' list with events",
                bool(sfx_seen), "events=%r" % (sfx_seen[:6],))
    ok &= check("M3: thrust fired ('thruster', True)",
                ("thruster", True) in sfx_seen,
                "names=%r" % sorted(set(names)))
    ok &= check("M3: firing the gun fired ('laser', 0.12) (throttled "
                "variant, recorded in the silent sim)",
                any(ev[0] == "laser" and len(ev) == 2 for ev in sfx_seen),
                "names=%r" % sorted(set(names)))
    ok &= check("M3: releasing thrust fired ('thruster', False)",
                ("thruster", False) in off_seen,
                "names=%r" % sorted(set(off_seen and [e[0] for e in off_seen] or [])))
    print("  [M3] sfx while thrusting+firing: %r" % (sfx_seen[:8],))
    print("  [M3] sfx after release: %r" % (off_seen[:4],))

    # --- 3. drained per publish: a second render_model with no step in
    #      between is empty. The sim process publishes exactly one model
    #      per step, so the model's "sfx" list holds at most one step's
    #      worth of events — the render process plays every drained model's
    #      list and never hears a sound twice.
    # (Verified in-process, where render_model is directly callable: the
    #  sim process's publish loop is one render_model per step.)
    g = Game(None, None, None, None, None, None, seed=1234, players=2)
    g._sfx("beam")
    first = g.render_model()["sfx"]
    second = g.render_model()["sfx"]
    # The constructor's reset records ("thruster", False) before the
    # "beam", so the first publish carries both; the point is the DRAIN:
    # the second publish (no step in between) is empty.
    ok &= check("M3: 'sfx' is drained per publish (second publish empty)",
                ("beam",) in first and second == [],
                "first=%r second=%r" % (first, second))

    # --- shutdown ------------------------------------------------------
    parent_conn.send(("quit",))
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
        print("  [M3] CHILD step report: steps=%d sim_time=%.3f wall=%.3f"
              % (steps, simt, wall))
    return ok


# ---------------------------------------------------------------------------
# M3 e2e: the REAL run_host with a fake client over loopback
# ---------------------------------------------------------------------------
#
# The pipe test above proves the sim process side (setup/hulls/sfx). This
# proves the RENDER process side end to end: the real run_host spawns the
# sim process (6-arg setup), drains the model queue in the render thread,
# draws the SHELL from the piped model, plays the model's sfx, and shuts
# the sim down cleanly on exit.
#
# The fake client is a REAL TCP connection: it sends the join handshake
# (its hull crosses the wire -> the 6-arg setup) and then T_INPUT messages
# carrying a REAL turn (M4: the client's input reaches the sim — the
# worker's sink forwards it as ("remote_input", ...) over the command
# Pipe, and the sim applies it to player 1). The host's OWN input is
# test-driven (pygame.key.get_pressed is patched to hold W+SPACE) so the
# sim actually does something (thrust + fire -> sfx events -> the host's
# SoundBank).
#
# Checks:
#   * the host loop runs + exits cleanly (no exception, no hang);
#   * the sim process was spawned and is GONE after exit (E.6 shutdown);
#   * the shell was drawn from the piped model (its camera moved with the
#     thrusting ship; its local standin's pose == the last model's p0
#     pose — the standin is synced from the model pack inside draw());
#   * the host's SoundBank played the model's sfx (thruster on + laser
#     blips — the D.1 bridge, end to end);
#   * the worker is alive: the fake client's tagged T_INPUT got a T_PONG;
#   * M4: the client's REAL input reached the sim (player 1's angle
#     changed in the host's models) and the wire bridge delivered
#     snapshots + input echoes to the client.

M3_E2E_RUN_S = 3.5          # how long the host runs before the QUIT


class _E2EKeys:
    """The host's held keys: W (thrust) + SPACE (fire) — the sim does
    something (sfx) and the ship moves (camera/standin checks)."""
    def __getitem__(self, k):
        return 1 if k in (pygame.K_w, pygame.K_SPACE) else 0


class _E2EMenu:
    """Just enough of a Menu for run_host (hull/loadout)."""
    def __init__(self, hull):
        self.hull = hull
        self.loadout = default_loadout(hull)


def test_m3_host_e2e():
    """M3 e2e: the real run_host + a fake client over loopback. Returns
    True when all checks pass."""
    from .net import (Connection, T_INPUT, T_PONG, serialize_hull,
                      serialize_loadout, serialize_input)

    ok = True
    if not pygame.get_init():
        pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    from .fog import make_light_texture
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    from .sound import SoundBank
    sfx = SoundBank()
    sfx.init()
    clock = pygame.time.Clock()

    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[3]   # scout, silas

    # --- capture the host's (shell) Game + the spawned sim process ---
    # Every constructed Game gets a draw() counter: _test_draw_from_model
    # counts the frames drawn FROM A PIPED MODEL (model is not None — the
    # pre-first-model frames pass None). The stable proof that the render
    # loop drained the model queue and drew the shell from it (the
    # camera/standin positions are timing-dependent — the ship coasts off
    # the map).
    created_games = []
    orig_init = Game.__init__
    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        created_games.append(self)
        orig_draw = self.draw
        self._test_draw_from_model = 0
        # Regression gate for the "berserk host" bug: after the first
        # model arrives, the host must NEVER draw None (a None draw falls
        # through to draw() building a model from the never-stepped SHELL
        # — frozen at spawn — so the host flickers between the live world
        # and the spawn world on every 0-model frame). Pre-first-model
        # None draws are expected (the shell's construction state).
        self._test_draw_none_after_model = 0
        def counting_draw(dt, model=None):
            if model is not None:
                self._test_draw_from_model += 1
            elif self._test_draw_from_model > 0:
                self._test_draw_none_after_model += 1
            return orig_draw(dt, model)
        self.draw = counting_draw
    Game.__init__ = patched_init

    created_procs = []
    orig_get_context = multiprocessing.get_context
    def tracking_get_context(name):
        ctx = orig_get_context(name)
        orig_process = ctx.Process
        def tracking_process(*a, **k):
            pr = orig_process(*a, **k)
            created_procs.append(pr)
            return pr
        ctx.Process = tracking_process
        return ctx
    multiprocessing.get_context = tracking_get_context

    # --- record the sfx the host's SoundBank plays (the D.1 bridge) ---
    sfx_played = []
    orig_play_sfx = sfx.play_sfx
    def tracking_play_sfx(events):
        sfx_played.extend(events)
        return orig_play_sfx(events)
    sfx.play_sfx = tracking_play_sfx

    # --- capture the host's models (M4: verify the client's input reaches
    #     the sim — player 1's angle must change as the fake client turns)
    #     + the wire messages the fake client receives (snap/beam/echo —
    #     the sim->render->worker->client bridge). ---
    from . import __main__ as M
    host_models = []
    orig_drain = M._drain_sim_model
    def tracking_drain(q):
        res = orig_drain(q)
        model = res[0]
        if model is not None:
            host_models.append(model)
        return res
    M._drain_sim_model = tracking_drain

    # --- the fake client (a real TCP connection) ---
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    pongs = []
    wire_seen = []          # wire messages the client received (M4 bridge)
    client_err = []
    client_stop = threading.Event()
    def fake_client():
        try:
            time.sleep(0.3)          # let the host bind + listen
            conn = Connection(socket.create_connection(
                ("127.0.0.1", port), timeout=5.0))
            from .net import do_handshake_client
            do_handshake_client(conn, serialize_hull(client_hull),
                                serialize_loadout(default_loadout(client_hull)))
            tag = 0
            # M4: drive the client's ship with REAL input (a constant turn)
            # instead of the M3 idle no-op. The host's worker sink sends it
            # to the sim process as ("remote_input", ...); the sim applies
            # it to player 1, so p1's angle must change (verified against
            # the host's models below).
            turn_inp = serialize_input(ShipInput(turn=1.0))
            while not client_stop.is_set():
                tag += 1
                conn.send({"type": T_INPUT, "inp": turn_inp, "tag": tag})
                conn.drain_send()
                for m in conn.poll():
                    t = m.get("type")
                    if t == T_PONG:
                        pongs.append(m.get("tag"))
                    elif t in ("snap", "beam", "echo"):
                        wire_seen.append(t)
                time.sleep(0.1)
            conn.close()
        except Exception as e:
            client_err.append(repr(e))
    th_client = threading.Thread(target=fake_client, daemon=True)

    # --- run the REAL run_host on a thread (the render thread) ---
    # The PACKAGE's __main__ (a relative import — `import __main__` would
    # give this test module, since we run via -m ship5.test_sim_process).
    from . import __main__ as M
    M.NET_PORT = port
    M._lan_ip = lambda: "127.0.0.1"
    orig_get_pressed = pygame.key.get_pressed
    pygame.key.get_pressed = lambda: _E2EKeys()
    host_err = []
    def host_side():
        try:
            M.run_host(screen, font, big_font, clock, sfx, _E2EMenu(host_hull),
                       None, light_tex, fog_surf, light_surf)
        except Exception:
            host_err.append(traceback.format_exc())
    th_host = threading.Thread(target=host_side, daemon=True)

    th_client.start()
    th_host.start()
    time.sleep(M3_E2E_RUN_S)
    # Stop the host (QUIT -> handle_events returns False -> run_host
    # returns -> the finally shuts the worker + sim process down).
    stop = time.time() + 10.0
    while time.time() < stop and th_host.is_alive():
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        time.sleep(0.05)
    th_host.join(timeout=5)
    client_stop.set()
    th_client.join(timeout=5)

    # Restore the patches.
    pygame.key.get_pressed = orig_get_pressed
    Game.__init__ = orig_init
    multiprocessing.get_context = orig_get_context
    sfx.play_sfx = orig_play_sfx
    M._drain_sim_model = orig_drain

    # --- verify -----------------------------------------------------------
    def check(label, cond, extra=""):
        nonlocal ok
        print(("PASS: " if cond else "FAIL: ") + label
              + (("  " + extra) if extra else ""))
        if not cond:
            ok = False

    check("M3e2e: no exception in the host loop", not host_err,
          repr(host_err))
    check("M3e2e: no exception in the fake client", not client_err,
          repr(client_err))
    check("M3e2e: the host loop exited (no hang)", not th_host.is_alive())

    shell = next((g for g in created_games if g.local_index == 0), None)
    check("M3e2e: the host built its (shell) Game", shell is not None)
    if shell is not None:
        # The stable proof that the render loop drained the model queue
        # and drew the shell from the piped model (patched_init counts
        # the draw() calls that received a non-None model).
        check("M3e2e: the shell was drawn from the piped model (draw "
              "received non-None models)",
              shell._test_draw_from_model > 30,
              "n=%d" % shell._test_draw_from_model)
        # Regression gate for the "berserk host" bug: after the first model
        # arrives, the host must never draw None (a None draw falls through
        # to draw() building a model from the never-stepped SHELL — frozen
        # at spawn — so the host flickers between the live world and the
        # spawn world on every 0-model frame). Pre-first-model None draws
        # are expected (the shell's construction state).
        check("M3e2e: the host never drew the frozen shell after the first "
              "model (no None draws post-model)",
              shell._test_draw_none_after_model == 0,
              "n=%d" % shell._test_draw_none_after_model)

    check("M3e2e: the sim process was spawned", len(created_procs) == 1,
          "n=%d" % len(created_procs))
    if created_procs:
        pr = created_procs[0]
        check("M3e2e: the sim process is GONE after exit (E.6 shutdown)",
              not pr.is_alive())

    if shell is not None:
        # The local standin was built + synced from the model pack inside
        # draw() (draw() builds it lazily on the first frame and syncs
        # its pose/presentation from the model pack every frame).
        standin = shell._standins.get(0)
        check("M3e2e: the shell's local standin was built + synced from "
              "the model (draw ran)", standin is not None)

    names = [ev[0] for ev in sfx_played]
    check("M3e2e: the host's SoundBank played the model's sfx (D.1)",
          bool(sfx_played), "n=%d names=%r" % (len(sfx_played),
                                               sorted(set(names))))
    check("M3e2e: the engine loop played ('thruster', True)",
          ("thruster", True) in sfx_played)
    check("M3e2e: the gun blips played ('laser', 0.12)",
          any(ev[0] == "laser" and len(ev) == 2 for ev in sfx_played))

    check("M3e2e: the worker is alive (the client's tagged input got a "
          "T_PONG)", len(pongs) >= 1, "pongs=%d" % len(pongs))

    # --- M4: the client's REAL input reaches the sim (player 1 responds) ---
    # The fake client sends a constant turn (ShipInput(turn=1.0)) every
    # 100 ms. The host's worker sink forwards it to the sim process as
    # ("remote_input", ...); the sim applies it to player 1 each tick, so
    # p1's angle must change across the run (verified against the host's
    # OWN models — the same data the host draws from). This is the M4
    # upgrade of the M3 e2e (whose fake client sent idle input as a no-op
    # placeholder): it proves the client -> worker -> Pipe -> sim -> model
    # path end to end, the way the M2 pipe test does in-process.
    if host_models:
        a0 = host_models[0]["players"][1]["angle"]
        a1 = host_models[-1]["players"][1]["angle"]
        # The ship's angle is UNBOUNDED (it accumulates forever — see
        # Ship.snapshot), so the TOTAL turn is the raw difference, NOT a
        # wrapped delta (a wrapped delta would lose the full rotations —
        # the client held turn=1.0 for the whole run, ~2 rotations).
        d = abs(a1 - a0)
        check("M4: the client's input reached the sim (player 1 turned "
              "in the host's models)", d > 0.5,
              "dangle=%.3f rad over %d models" % (d, len(host_models)))
        print("  [M4] p1 angle: %.3f -> %.3f (d=%.3f rad, %d models)"
              % (a0, a1, d, len(host_models)))
    else:
        check("M4: the client's input reached the sim (player 1 turned "
              "in the host's models)", False, "no host models captured")

    # --- M4: the wire bridge delivered snapshots (sim -> render -> worker
    #     -> client). The fake client records every snap/beam/echo it
    #     receives; snapshots must arrive (the client's ghost seeds from
    #     them) and echoes must flow (the host applied the client's input
    #     each tick and echoed it back — the input path, end to end). ---
    wire_kinds = set(wire_seen)
    check("M4: the client received snapshots over the wire (bridge)",
          "snap" in wire_kinds, "kinds=%r n=%d"
          % (sorted(wire_kinds), len(wire_seen)))
    check("M4: the client received input echoes over the wire",
          "echo" in wire_kinds, "kinds=%r" % sorted(wire_kinds))

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

    # M3: the real 2P setup path (6-arg hulls/loadouts) + the model's
    # "sfx" list (D.1).
    print("\nM3: 6-arg setup (real hulls/loadouts) + the model's 'sfx' list")
    ok &= test_m3_hulls_and_sfx()

    # M3 e2e: the REAL run_host with a fake client over loopback (spawn +
    # 6-arg setup + drain/draw the shell + play the sfx + clean shutdown),
    # now including the M4 network-bridge checks (the client's REAL input
    # reaches the sim; snapshots + echoes reach the client over the wire).
    print("\nM3 e2e: the real run_host + a fake client over loopback "
          "(+ M4 network bridge)")
    ok &= test_m3_host_e2e()

    print("\nSIM PROCESS (M1+M2+M3+M4):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()