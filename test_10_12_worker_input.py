"""10.12: the client's input is applied by the WORKER thread the moment it
lands (not polled off the render thread once per frame).

The 10.11 host polled T_INPUT on the RENDER thread (worker.poll() once per
frame). The host renders at ~42-47 FPS (21-24 ms/frame), so the input sat in
the worker's in_queue for up to a full frame before the sim thread could
apply it — a 0-24 ms (mean ~12 ms) poll-phase added to the wire latency. The
host applies the LATEST received input each tick (pinned #5), so that stale
input diverged from the client's prediction ghost (which used the input
immediately) and the reconcile snapped the ghost back (the in-flight-input
snap). The fix: a sink the WORKER thread invokes the moment a T_INPUT lands
(worker.set_input_sink), cutting the poll-phase to ~1 ms.

This test proves the wiring over a real loopback socket (no pygame game):
  1. SINK-ON-WORKER-THREAD — the sink is invoked on the WORKER thread, not
     the main thread (the whole point: the input no longer waits for the
     next render frame).
  2. INPUT-APPLIED — the sink receives the parsed input (the 'inp' field)
     intact.
  3. PONG-REFLECTED — the sink reflects the T_PONG (worker.send from the
     worker thread) and the client receives it with the same tag.
  4. INPUT-CONSUMED — the T_INPUT is NOT also queued in the host worker's
     in_queue (the sink returned True), so the main thread's poll() never
     sees it (no double-apply).
  5. NO-SINK-FALLBACK — with no sink registered, the T_INPUT is queued as
     before (the main thread polls it) — the pre-10.12 behavior is intact.

Run from the repo root (the directory that CONTAINS ship5/):
    python -m ship5.test_10_12_worker_input
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import threading
import time

from .net import (Host, connect, NetWorker, T_INPUT, T_PONG,
                  serialize_input)
from .intent import ShipInput


def _loopback():
    h = Host(0)
    p = h.sock.getsockname()[1]
    c = connect("127.0.0.1", p)
    hc, _addr = h.accept_one()
    return h, hc, c


def _wait_for(worker, want_type, timeout=2.0):
    """Poll `worker` until a message of `want_type` arrives (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for m in worker.poll():
            if m.get("type") == want_type:
                return m
        time.sleep(0.002)
    return None


def check(label, cond, extra=""):
    print(("PASS: " if cond else "FAIL: ") + label
          + (("  " + extra) if extra else ""))
    return bool(cond)


def main():
    main_ident = threading.get_ident()
    ok = True

    # --- 1-4. SINK path: applied on the worker thread, pong reflected,
    #     input consumed (not re-queued). ---
    host, hconn, client = _loopback()
    hw = NetWorker(hconn, is_host=True)
    cw = NetWorker(client, is_host=False)
    sink_calls = []

    def sink(m):
        # Record the calling thread + the parsed input. Reflect the pong
        # (worker.send is a thread-safe queue put, safe from the worker).
        sink_calls.append((threading.get_ident(), m.get("inp"),
                           m.get("tag")))
        if "tag" in m:
            hw.send({"type": T_PONG, "tag": m["tag"]})
        return True   # consumed: do NOT queue it into in_queue

    hw.set_input_sink(sink)
    hw.start()
    cw.start()
    try:
        inp = ShipInput(turn=1.0, thrust_fwd=1.0, fire=True)
        cw.send({"type": T_INPUT, "inp": serialize_input(inp), "tag": 7})
        # Wait for the pong (the sink reflects it on the worker thread).
        pong = _wait_for(cw, T_PONG)
        # Give the host worker a moment to drain its (now empty) in_queue.
        time.sleep(0.05)
        # The main thread's poll of the HOST worker must NOT see the
        # T_INPUT (it was consumed by the sink, not queued).
        leaked = [m for m in hw.poll() if m.get("type") == T_INPUT]

        ok &= check("sink: invoked exactly once", len(sink_calls) == 1,
                    "n=%d" % len(sink_calls))
        if sink_calls:
            ident, got_inp, got_tag = sink_calls[0]
            ok &= check("sink: invoked on the WORKER thread (not main)",
                        ident != main_ident,
                        "sink thread=%d main thread=%d"
                        % (ident, main_ident))
            ok &= check("sink: received the parsed input intact",
                        got_inp == serialize_input(inp),
                        "got=%r" % (got_inp,))
            ok &= check("sink: received the tag", got_tag == 7,
                        "tag=%r" % (got_tag,))
        ok &= check("pong: reflected + received by the client (same tag)",
                    pong is not None and pong.get("tag") == 7,
                    "pong=%r" % (pong,))
        ok &= check("input: consumed (NOT re-queued into in_queue)",
                    leaked == [], "leaked=%r" % (leaked,))
    finally:
        hw.stop()
        cw.stop()
        hconn.close()
        client.close()
        host.close()

    # --- 5. NO-SINK fallback: with no sink, the T_INPUT is queued as
    #     before (the main thread polls it) — pre-10.12 behavior intact. ---
    host, hconn, client = _loopback()
    hw = NetWorker(hconn, is_host=True)   # NO set_input_sink
    cw = NetWorker(client, is_host=False)
    hw.start()
    cw.start()
    try:
        cw.send({"type": T_INPUT, "inp": serialize_input(
            ShipInput(turn=-1.0)), "tag": 9})
        got = _wait_for(hw, T_INPUT)
        ok &= check("no-sink: T_INPUT is queued (main thread polls it)",
                    got is not None and got.get("tag") == 9,
                    "got=%r" % (got,))
    finally:
        hw.stop()
        cw.stop()
        hconn.close()
        client.close()
        host.close()

    print("\n10.12 WORKER-INPUT SINK:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()