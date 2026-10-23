"""The host's dedicated NET process (plan_13, Phase 1).

WHY A SEPARATE PROCESS
The host's snapshot SEND used to run in the render process (Option B: the
sim process piped the wire traffic to a net Queue that a _NetPump thread in
the render process drained, then a NetWorker thread sent). Those threads
share the render process's GIL with the draw, so a heavy draw frame could
starve the send for up to a GIL-switch interval and the snapshot send
cadence inherited that jitter. The sim's STEP rate was already fixed (the
sim runs in its own process), but the SEND was still coupled to the render
process's GIL.

Moving the socket + the send into a dedicated NET process removes the shared
GIL entirely: the net process has its own GIL, shared only with itself. The
render thread's draw can hold the render GIL for as long as it wants and the
net process keeps draining the queue and writing to the socket. The render
thread is COMPLETELY OUT of the send path.

SOCKET OWNERSHIP (the plan's locked decision)
A live socket can't be pickled across the spawn Pipe, so the net process
OWNS the socket end to end: it does the listen + accept + the join/welcome
handshake (the render process no longer touches the socket at all). The
render process learns the outcome by polling the status Pipe (non-blocking
`poll()` — the waiting screen's draw loop stays smooth):
  ("connected", client_hull_id, client_loadout_dict)  (after the handshake)
  ("disconnected",)                                    (the client went away)
  ("error", text)                                      (setup/handshake failed)
  ("crash", traceback_text)                            (the net process died)

IPC (spawn context — see plan_13 section 2):
  render -> net (the status Pipe, the render's end):
                   ("quit",)
  net -> render (the status Pipe, the net's end):
                   ("connected", client_hull_id, client_loadout_dict) |
                   ("disconnected",) | ("error", text) | ("crash", tb)
  sim -> net (a bounded Queue, put_nowait drop-when-full):
                   ("snap", sim_time, snapshot)   (60 Hz, on a step)
                   ("beam", msg) | ("echo", msg)  (T_BEAM / T_ECHO wire msgs)
  net -> sim (a Pipe, small + low-rate + ordered):
                   ("remote_input", ShipInput)    (the client's input)
                   ("command", name, arg)         (e.g. ("command","respawn",1))

The net process is HEADLESS: it sets SDL_VIDEODRIVER/AUDIODRIVER=dummy
BEFORE the net import (inside the function, so the PARENT — which imports
this module for net_process_main — keeps its real video driver). The net
import chain pulls in pygame (net -> intent -> pygame), so the dummy driver
is required; the net process never creates a display (it does no draw).
"""
import os

import time
import traceback

# The net import chain (net -> intent -> pygame) pulls in pygame, so the
# dummy drivers MUST be set before that import. They are set inside
# net_process_main (below), NOT here, so the PARENT process — which imports
# this module for net_process_main — keeps its real video driver. This
# mirrors sim_process.py exactly.

# SEND CADENCE (plan_20 — the residual-stutter fix). The net process paces
# its SENDS to a steady 60 Hz (SEND_TICK = 16.7 ms) with a deadline +
# busy-wait hybrid (see the game loop), INDEPENDENT of the OS timer
# resolution. WHY: the sim publishes at a clean 60 Hz, but the OLD loop
# paced itself with a bare `time.sleep(0.001)` — and on a coarse OS timer
# (Windows' default ~15.6 ms, or a coalesced virtualized timer on the
# Mac-hosted Linux host) that 1 ms sleep actually returns after ~15-17 ms,
# so the loop only iterated ~54x/sec and the client saw ragged 17-26 ms
# gaps (the residual stutter). The hybrid sleeps until SEND_BUSY_MARGIN
# before the deadline, then busy-waits the last slice, so the send lands on
# the deadline on ANY timer resolution (stdlib only — no timeBeginPeriod).
# 2 ms of busy-wait per 16.7 ms tick burns ~12% of one core — acceptable.
SEND_TICK = 1.0 / 60.0
SEND_BUSY_MARGIN = 0.002


def _raise_priority_best_effort():
    """Best-effort: raise this process's scheduling priority so the OS wakes
    it on time (the secondary lever — the deadline+busy-wait hybrid is the
    primary fix and works WITHOUT this).

    WHY it is best-effort and wrapped: raising priority needs permission
    (CAP_SYS_NICE on Linux, root on macOS, an elevated token on Windows).
    On a normal user session it fails — that is FINE, the hybrid still
    paces the sends. The elevation only helps the descheduling/jitter
    component (a render process burning a core stealing the net process's
    timeslice). Every branch is try/except with a silent fallback to normal
    priority, so the game runs on all three OSes whether or not the API is
    available or permitted.

    Returns a short human-readable tag (for the log), never raises."""
    import sys
    try:
        if sys.platform.startswith("win"):
            # Windows: bump the whole process to HIGH priority class.
            import ctypes
            # HIGH_PRIORITY_CLASS = 0x00000080 (winbase.h)
            ok = ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
            return "windows SetPriorityClass(HIGH)=%s" % bool(ok)
        # Linux / macOS: lower the nice value (higher priority). os.nice()
        # can only LOWER the value (raise priority) and needs permission;
        # -5 is a modest bump that rarely needs root.
        import os
        try:
            cur = os.getpriority(os.PRIO_PROCESS, 0)
        except Exception:
            cur = 0
        new = max(-5, cur - 5)
        try:
            os.nice(new)
            return "nice %d -> %d" % (cur, new)
        except (PermissionError, OSError) as e:
            return "nice denied (%s) — normal priority" % type(e).__name__
    except Exception as e:   # any unexpected failure: fall back silently
        return "priority elevation skipped (%s)" % type(e).__name__


def net_process_main(status_conn, net_queue, remote_input_conn,
                     host_hull, host_loadout, port):
    """The spawn target (MUST be a top-level function: spawn re-imports this
    module by name to find it). Runs the host's socket + snapshot send +
    client-input receive in this process.

    Channels (all from the spawn context):
      `status_conn`      a Pipe, render <-> net. The net process sends status
                         (connected/disconnected/error/crash) on it and
                         receives ("quit",) from the render process. The
                         render process polls it NON-BLOCKING (poll()) so the
                         waiting screen's draw loop stays smooth.
      `net_queue`        a bounded Queue, sim -> net: the wire messages
                         (snap/beam/echo) the SimThread hands to its worker.
                         Drained HERE (not the render process) and sent to the
                         socket — the render thread is out of the send path.
      `remote_input_conn` a Pipe, net -> sim: the client's input
                         (("remote_input", ShipInput)) + forwarded commands
                         (("command", name, arg), e.g. respawn). The sim
                         process's bridge loop drains it.
      `host_hull` / `host_loadout`  the host's real fit (HullType + dict),
                         pickled across the spawn boundary (frozen dataclass +
                         plain dict — both picklable). Serialized for the
                         welcome.
      `port`             the TCP port to listen on (config.NET_PORT).

    Exits cleanly on ("quit",) or on status_conn EOF (the render process
    died), or on a crash (after sending ("crash", traceback))."""
    # Headless: the net import chain pulls in pygame, so set the dummy
    # drivers BEFORE importing net (inside the function, so the parent keeps
    # its real video driver). Mirrors sim_process.py.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from .net import (Host, do_handshake_host, serialize_hull,
                      serialize_loadout, serialize_snapshot,
                      RockShapeTracker,
                      deserialize_input, T_SNAP, T_INPUT, T_RESPAWN, T_PONG)

    def _status(msg):
        """Send a status message to the render process. A failure means the
        render process is gone (the Pipe is closed) — there is no one left
        to report to, so return False and the caller exits."""
        try:
            status_conn.send(msg)
            return True
        except (OSError, EOFError, ConnectionError, BrokenPipeError):
            return False

    # Per-send snap log (F3 telemetry, plan_13 follow-up): one row per
    # snapshot SENT to the socket — (wall, mono, sim_time, seq). The wall
    # deltas are the "is the net process sending smoothly, or bursty?"
    # signal (a descheduled net process shows up here, not in sim_time).
    # Written by the net process (its own machine — the host's Mac). The
    # render process toggles it with ("snap_log", path) on the status Pipe
    # (the F3 edge).
    snap_log_f = None
    snap_seq = 0

    # SNAPSHOT LATEST-WINS (the staleness fix, commit 8c32708). The loop is
    # paced to 60 Hz (SEND_TICK — the deadline+busy-wait hybrid at the
    # bottom of the game loop), the SAME rate the sim publishes at, so
    # normally exactly one snap lands per iteration. But a hiccup (a slow
    # body, an OS deschedule) can still let a snap or two land between
    # iterations. A FIFO (oldest-first) would queue stale snaps behind
    # the newest — the client would render seconds-old state and the ship
    # would lag far behind the player's input.
    #
    # Instead, we use LATEST-WINS: drain the queue, keep only the NEWEST
    # snap, send it. The client always receives the most current state,
    # regardless of the loop rate. Dropped snaps are a network loss the
    # client's interpolation buffer already handles (it holds the last
    # frame). The rock-shape tracker handles dropped snaps gracefully
    # (it's a "have I seen this rock before?" check, not a delta-from-last
    # check, so a dropped snap just means the next snap sends full verts
    # for any new rocks — a one-time cost, not a correctness issue).
    #
    # LATEST-WINS and the cadence fix are ORTHOGONAL: latest-wins decides
    # WHICH snap to send (the newest — no staleness), the pacing decides
    # WHEN (a steady 16.7 ms — no ragged gaps). Keep both.
    #
    # `latest_snap`: the newest snap pulled from the queue (None until the
    # first). Replaced on each drain (latest-wins). Sent on each loop
    # iteration (the loop is paced to 60 Hz, so the client receives ~60
    # snaps/sec — one per frame, matching the sim's publish rate).
    latest_snap = None

    # Rock-shape tracker (plan_17): drives the verts delta-compression in
    # serialize_snapshot. One per connection; the net process is 2P-only
    # (accepts ONE client, no re-accept loop), so one tracker per process
    # lifetime is correct. Owned by this process (the only one that calls
    # serialize_snapshot).
    rock_tracker = RockShapeTracker()

    def _snap_log(path):
        nonlocal snap_log_f
        if path is None:
            if snap_log_f is not None:
                snap_log_f.close()
                snap_log_f = None
            return
        if snap_log_f is None:
            snap_log_f = open(path, "w", newline="")
            snap_log_f.write("wall,mono,sim_time,seq\n")

    # Per-iteration LOOP cadence log (plan_20, step 1 — measure first):
    # one row per game-loop iteration,
    #   (iter, wall, mono, sleep_requested, sleep_actual, work_time,
    #    total_period)
    # where work_time = top-of-loop -> just before the sleep (the
    # drain/send/receive body), sleep_actual = the real time the sleep
    # took, and total_period = the real time between consecutive
    # tops-of-loop (0 on the first row). The decomposition
    # total_period ~= work_time + sleep_actual tells us WHERE the ~18.5 ms
    # period comes from: a ragged sleep_actual with a tiny work_time is
    # sleep overshoot / OS descheduling; a large work_time is a slow loop
    # body. Toggled with ("loop_log", path) on the status Pipe (path None
    # closes it) — the same pattern as snap_log.
    loop_log_f = None
    loop_iter = 0

    def _loop_log(path):
        nonlocal loop_log_f
        if path is None:
            if loop_log_f is not None:
                loop_log_f.close()
                loop_log_f = None
            return
        if loop_log_f is None:
            loop_log_f = open(path, "w", newline="")
            loop_log_f.write("iter,wall,mono,sleep_requested,sleep_actual,"
                             "work_time,total_period\n")

    conn = None
    try:
        # --- 1. listen + accept + handshake (the socket is owned HERE) ---
        host = Host(port)
        # The render process's waiting screen (ESC/QUIT) sends ("quit",) on
        # the status Pipe; a status_conn EOF means the render process died.
        # Either way the net process exits (the render process is gone).
        while True:
            if status_conn.poll():
                m = status_conn.recv()
                if m[0] == "quit":
                    host.close()
                    return
                elif m[0] == "snap_log":
                    _snap_log(m[1])
                elif m[0] == "loop_log":
                    _loop_log(m[1])
            conn, _addr = host.accept_one(timeout=0.25)
            if conn is None:
                continue
            break
        host.close()   # 2P-only: stop listening once one client is accepted
        # The join/welcome handshake (blocking). The host's fit is serialized
        # for the welcome; the client's (hull id, loadout dict) comes back.
        ch, cl = do_handshake_host(conn, serialize_hull(host_hull),
                                   serialize_loadout(host_loadout))
        if ch is None:
            conn.close()
            _status(("error", "client vanished before joining"))
            return
        # Tell the render process the client is joined (it builds the shell +
        # spawns the sim from the client's real fit).
        if not _status(("connected", ch, cl)):
            conn.close()
            return

        # --- 2. the game loop (the socket is non-blocking now) ---
        conn.set_nonblocking()
        # Best-effort priority elevation (the SECONDARY lever — see
        # _raise_priority_best_effort). The deadline+busy-wait hybrid below
        # is the PRIMARY fix and works without this; the elevation only
        # helps the descheduling/jitter component (a render process burning
        # a core stealing the net process's timeslice). Wrapped: it falls
        # back to normal priority silently if the API is unavailable or
        # unpermitted, so the game runs on all three OSes either way.
        _raise_priority_best_effort()
        prev_top = None   # previous iteration's top-of-loop time (loop log)
        # The send deadline grid (the plan_20 cadence fix). The loop paces
        # itself to SEND_TICK (16.7 ms) with a deadline + busy-wait hybrid
        # (see the pacing block at the bottom of the loop), so the client
        # receives ~one snap per frame regardless of the OS timer
        # resolution. next_send is the deadline of the NEXT send; it is
        # advanced by SEND_TICK each iteration (resynced if we fall behind).
        next_send = time.monotonic() + SEND_TICK
        while True:
            # Top-of-loop time: the wake moment (loop_log total_period).
            iter_t0 = time.monotonic()
            # Quit / render-process-gone (checked first — snappy shutdown).
            if status_conn.poll():
                m = status_conn.recv()
                if m[0] == "quit":
                    break
                elif m[0] == "snap_log":
                    _snap_log(m[1])
                elif m[0] == "loop_log":
                    _loop_log(m[1])
            # SEND side: drain the sim's net Queue -> the socket.
            #   * beam / echo: sent the moment they land (low-rate events,
            #     no cadence — a beam is a 0.15 s flash, an echo is one
            #     input; neither is a cadence the client phase-locks to).
            #   * snap: LATEST-WINS. The loop is paced to 60 Hz (SEND_TICK, the
            #     deadline+busy-wait hybrid at the bottom of the loop) —
            #     the SAME rate the sim publishes at — so normally exactly
            #     one snap lands per iteration. But a hiccup (a slow body,
            #     an OS deschedule) can let a snap or two land between
            #     iterations. Sending them oldest-first (a FIFO) would
            #     queue stale snaps behind the newest — the client would
            #     render seconds-old state and the ship would lag far
            #     behind the input. Instead, each drained snap REPLACES
            #     the pending one (latest-wins) and the newest is sent
            #     this iteration. A dropped snap is a 1-tick loss the
            #     client's interpolation buffer already handles (it holds
            #     the last frame); the rock-shape tracker re-sends full
            #     verts for any rock it has not seen, so a drop is a
            #     one-time bandwidth cost, not a correctness issue.
            while True:
                try:
                    m = net_queue.get_nowait()
                except Exception:
                    break
                kind = m[0]
                if kind == "snap":
                    latest_snap = m   # latest-wins: newest replaces pending
                else:   # "beam" / "echo" — the msg dict is m[1]
                    conn.send(m[1])
            if latest_snap is not None:
                m = latest_snap
                latest_snap = None
                conn.send({"type": T_SNAP, "sim_time": m[1],
                           "snap": serialize_snapshot(m[2], rock_tracker)})
                if snap_log_f is not None:
                    snap_seq += 1
                    snap_log_f.write("%.3f,%.3f,%.4f,%d\n" % (
                        time.time(), time.monotonic(), m[1], snap_seq))
                    if snap_seq % 60 == 0:   # ~1 s of rows: bound buffer
                        snap_log_f.flush()
            conn.drain_send()
            # RECEIVE side: the client's input + commands. The client's input
            # is forwarded to the sim the moment it lands (~1 ms) — the same
            # hand-off the in-process worker input sink did, but in this
            # process (off the render thread's GIL). The T_PONG is reflected
            # HERE (~1 ms) so the measured RTT is honest (a render-thread
            # reflection would add a full frame).
            for m in conn.poll():
                t = m.get("type")
                if t == T_INPUT:
                    remote_input_conn.send(
                        ("remote_input", deserialize_input(m["inp"])))
                    if "tag" in m:
                        conn.send({"type": T_PONG, "tag": m["tag"]})
                elif t == T_RESPAWN:
                    # The client's dead player asks to respawn. The net
                    # process has no direct command channel to the sim (only
                    # the remote_input Pipe), so it forwards the command over
                    # that Pipe (the sim's bridge loop handles both
                    # ("remote_input", ...) and ("command", ...)).
                    remote_input_conn.send(("command", "respawn", 1))
            # Close detection: the client went away (FIN / RST).
            if conn.closed:
                _status(("disconnected",))
                break
            # SEND CADENCE (plan_20): pace the loop to SEND_TICK (16.7 ms)
            # with a deadline + busy-wait hybrid, INDEPENDENT of the OS
            # timer resolution. The body above (drain + serialize + send +
            # receive) runs NOW; then we wait until the next deadline:
            #   1. sleep until SEND_BUSY_MARGIN before the deadline — the
            #      OS can wake us early or late (a coarse timer wakes us
            #      on its grid, a fine timer wakes us on time);
            #   2. busy-wait the last SEND_BUSY_MARGIN — a tight spin that
            #      lands the NEXT iteration's body on the deadline within
            #      ~1 ms, even if the sleep returned early on a coarse
            #      timer (Windows' 15.6 ms grid, a coalesced VM timer).
            # WHY this and not a bare time.sleep: the OLD loop slept 1 ms
            # per iteration and let the OS timer set the cadence — on a
            # coarse timer that stretched to ~15-17 ms (the 53.9 Hz,
            # ragged 17-26 ms gaps). The hybrid owns the cadence: the
            # deadline grid is SEND_TICK, and the busy-wait absorbs the
            # timer's coarseness. Stdlib only (time.monotonic + a spin) —
            # no timeBeginPeriod / sched API required. The 2 ms spin per
            # 16.7 ms tick burns ~12% of one core.
            #
            # LATEST-WINS is UNCHANGED (the drain above still keeps only
            # the newest pending snap): it decides WHICH snap to send,
            # this pacing decides WHEN. If the body ever runs long (a
            # hiccup), next_send resyncs to now + SEND_TICK (below) so the
            # grid never accumulates backlog — and latest-wins means the
            # snap sent after the hiccup is still the newest (no
            # staleness, exactly as before).
            work_time = time.monotonic() - iter_t0
            sleep_requested = max(0.0, next_send - time.monotonic()
                                  - SEND_BUSY_MARGIN)
            sleep_t0 = time.monotonic()
            if sleep_requested > 0.0:
                time.sleep(sleep_requested)
            # Busy-wait the last slice (the spin that lands us on the
            # deadline). A no-op when the sleep already overshot.
            while time.monotonic() < next_send:
                pass
            sleep_actual = time.monotonic() - sleep_t0
            # Advance the deadline grid. If we fell behind (the body ran
            # long, or the OS starved us past the deadline), resync to
            # now + SEND_TICK instead of accumulating a backlog of missed
            # deadlines (which would burst-catch-up and re-rag the cadence).
            now = time.monotonic()
            next_send += SEND_TICK
            if next_send <= now:
                next_send = now + SEND_TICK
            # Loop cadence log (plan_20 step 1): work_time = top-of-loop
            # -> just before the wait (the drain/send/receive body);
            # sleep_requested = the hybrid's sleep ask (SEND_TICK -
            # SEND_BUSY_MARGIN in steady state); sleep_actual = the real
            # time the sleep+spin took; total_period = the real time
            # between consecutive tops (the SEND cadence — this is what
            # must be a steady 16.7 ms).
            if loop_log_f is not None:
                loop_iter += 1
                total_period = (iter_t0 - prev_top) if prev_top is not None else 0.0
                loop_log_f.write("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (
                    loop_iter, time.time(), iter_t0, sleep_requested,
                    sleep_actual, work_time, total_period))
                if loop_iter % 60 == 0:   # ~1 s of rows: bound the buffer
                    loop_log_f.flush()
            prev_top = iter_t0
    except BaseException:
        # A crash in the net process must not take the render process down
        # silently: report it, then exit (the render process shows a notice).
        try:
            status_conn.send(("crash", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if snap_log_f is not None:
            try:
                snap_log_f.close()
            except Exception:
                pass
        if loop_log_f is not None:
            try:
                loop_log_f.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Direct run is not supported (spawn needs a picklable top-level target +
    # a Pipe/Queue from the parent). Use test_net_process.py.
    raise SystemExit("net_process_main is a spawn target; "
                     "run test_net_process.py")