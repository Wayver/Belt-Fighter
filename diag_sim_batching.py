"""Diagnose WHY the separate-process sim batches (wall-time bursts).

Spawns the real sim process (sim_process_main), enables the SimThread's
per-iteration cadence log (inter_wake, body_time, steps), runs it for a
few seconds, and decomposes the batching:

  inter_wake = body_time + sleep_time + wake_latency

  * body_time  = the work done this iteration (command drain + steps +
                 model publish). If this is large (a non-trivial fraction
                 of the 16.7 ms tick), the loop body is eating into the
                 tick and the accumulator builds up -> batching.
  * wake_latency = inter_wake - body_time - sleep_time = how long AFTER
                 the requested sleep the OS actually resumed the thread.
                 If this is ragged (and body_time is small), the OS
                 scheduler is waking the thread late -> jitter -> batching.

Two arms:
  IDLE  — the parent does nothing (baseline OS jitter).
  BUSY  — the parent holds the GIL with a render-like load (CPU
          contention). Confirms whether the parent's load affects the
          separate-process sim's cadence (it should not, via the GIL, but
          CPU contention can still deschedule the sim thread).

Run from the repo root:  python -m ship5.diag_sim_batching
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys
import time
import threading
import multiprocessing

from .sim_process import sim_process_main

RUN_SECONDS = 6.0
GIL_HOLD_MS = 20.0
FRAME_MS = 20.0


def _gil_hold(ms):
    t0 = time.monotonic()
    x = 0
    while (time.monotonic() - t0) * 1000.0 < ms:
        x += 1
    return x


def _render_load(stop_event, hold_ms, frame_ms):
    frame_s = frame_ms / 1000.0
    while not stop_event.is_set():
        t0 = time.monotonic()
        _gil_hold(hold_ms)
        elapsed = time.monotonic() - t0
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)


def _run_arm(tag, busy):
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    model_queue = ctx.Queue(maxsize=4)
    net_queue = ctx.Queue(maxsize=16)
    ri_sim_end, ri_net_end = ctx.Pipe()
    p = ctx.Process(target=sim_process_main,
                    args=(child_conn, model_queue, net_queue, ri_sim_end))
    p.start()
    child_conn.close()
    parent_conn.send(("setup", 1234, 2))
    # Wait for ready.
    while True:
        m = model_queue.get()
        if m[0] == "ready":
            break
        elif m[0] == "crash":
            print("  [%s] SIM CRASHED:\n" % tag + m[1])
            parent_conn.close()
            p.join()
            return None
    # Enable the per-iteration cadence log.
    parent_conn.send(("command", "iter_log", True))

    stop = threading.Event()
    loader = None
    if busy:
        loader = threading.Thread(target=_render_load,
                                  args=(stop, GIL_HOLD_MS, FRAME_MS))
        loader.start()

    # Run for RUN_SECONDS, draining the queues (the render-loop pattern)
    # so the child never blocks on a full queue.
    t0 = time.monotonic()
    while time.monotonic() - t0 < RUN_SECONDS:
        while True:
            try:
                model_queue.get_nowait()
            except Exception:
                break
        while True:
            try:
                net_queue.get_nowait()
            except Exception:
                break
        if busy:
            time.sleep(0.001)   # the loader thread does the GIL hold
        else:
            time.sleep(0.01)

    if loader is not None:
        stop.set()
        loader.join()

    parent_conn.send(("quit",))
    done = None
    t_deadline = time.monotonic() + 3.0
    while time.monotonic() < t_deadline:
        try:
            m = model_queue.get(timeout=0.2)
        except Exception:
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
    ri_sim_end.close()
    ri_net_end.close()
    if done is None:
        print("  [%s] no done report" % tag)
        return None
    _, steps, simt, wall, stamps, iter_log = done
    print("  [%s] steps=%d sim_time=%.3f wall=%.3f -> sim/wall=%.3fx"
          % (tag, steps, simt, wall, simt / wall))
    return iter_log


def _analyze(tag, log):
    if not log:
        print("  [%s] no iter_log data" % tag)
        return
    # Skip the first iteration (inter_wake is 0 there).
    rows = log[1:]
    n = len(rows)
    inter = [r[0] for r in rows]
    body = [r[1] for r in rows]
    steps = [r[2] for r in rows]
    # wake_latency = inter_wake - body_time - sleep_time. sleep_time is
    # not recorded, but inter_wake - body_time = sleep_time + wake_latency.
    # In steady state (1 step), sleep_time ~= STEP - (inter_wake - body)
    # ... simpler: the sleep requested is STEP - acc, and acc after 1 step
    # is the leftover. The cleanest signal is: how much of inter_wake is
    # NOT body_time? That remainder = sleep + wake_latency. The wake
    # latency is the part of the remainder that exceeds the requested
    # sleep. We don't have the requested sleep here, so report the
    # remainder distribution and the body distribution separately.
    inter_ms = [x * 1000 for x in inter]
    body_ms = [x * 1000 for x in body]
    rem_ms = [(i - b) * 1000 for i, b in zip(inter, body)]

    def _stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return (min(xs), xs[n // 2], max(xs), sum(xs) / n)

    def _pct(xs, thr):
        return 100.0 * sum(1 for x in xs if x > thr) / len(xs)

    print("  [%s] %d iterations" % (tag, n))
    lo, med, hi, mean = _stats(inter_ms)
    print("    inter_wake ms: min=%.2f med=%.2f max=%.2f mean=%.2f"
          % (lo, med, hi, mean))
    lo, med, hi, mean = _stats(body_ms)
    print("    body_time  ms: min=%.2f med=%.2f max=%.2f mean=%.2f"
          % (lo, med, hi, mean))
    lo, med, hi, mean = _stats(rem_ms)
    print("    (sleep+lat) ms: min=%.2f med=%.2f max=%.2f mean=%.2f"
          % (lo, med, hi, mean))
    # Batching: iterations that took >= 2 steps.
    n2 = sum(1 for s in steps if s >= 2)
    n0 = sum(1 for s in steps if s == 0)
    n1 = sum(1 for s in steps if s == 1)
    print("    steps/iter: 0=%d 1=%d >=2=%d  (batched=%.1f%%)"
          % (n0, n1, n2, 100.0 * n2 / n))
    # How often does the body exceed half a tick (8.3 ms)?
    print("    body > 8.3ms (half tick): %.1f%%" % _pct(body_ms, 8.3))
    print("    body > 16.7ms (full tick): %.1f%%" % _pct(body_ms, 16.7))
    # The inter-wake raggedness: how often is it > 1.5 ticks (25 ms)?
    print("    inter_wake > 25ms (1.5 tick): %.1f%%" % _pct(inter_ms, 25.0))
    # The wake-latency proxy: inter_wake - body - (STEP). In steady state
    # the requested sleep is STEP - acc where acc is the leftover after
    # the steps. For a 1-step iteration, acc_after = (acc_before + inter)
    # - STEP, and sleep = STEP - acc_after. This is getting circular; the
    # cleanest single number is: the FRACTION of inter_wake that is body
    # time (the rest is sleep + latency).
    frac_body = 100.0 * sum(b for b in body_ms) / sum(i for i in inter_ms)
    print("    body is %.1f%% of inter_wake (rest = sleep + OS latency)"
          % frac_body)


def main():
    print("Sim batching diagnostic (separate-process sim)")
    print("  run %.1f s per arm, GIL hold %.0f ms/frame\n"
          % (RUN_SECONDS, GIL_HOLD_MS))
    print("IDLE arm (parent does nothing):")
    log_idle = _run_arm("idle", busy=False)
    _analyze("idle", log_idle)
    print("\nBUSY arm (parent holds the GIL, render-like load):")
    log_busy = _run_arm("busy", busy=True)
    _analyze("busy", log_busy)


if __name__ == "__main__":
    main()