"""The client's dedicated NET process (the client-side mirror of
net_process.py, plan_20 follow-up — the GIL fix).

WHY A SEPARATE PROCESS
The client's snapshot RECEIVE used to run in a NetWorker THREAD in the
render process. A thread shares the render process's GIL with the draw:
when a snapshot's bytes land on the socket, the worker thread's `recv()`
returns (it releases the GIL while blocked), but it then has to
RE-ACQUIRE the GIL to stamp the wire-arrival time and queue the message —
and the render thread is holding the GIL for the whole draw frame
(~16-25 ms). So a snap that arrived at time T got stamped at T + (one
frame), turning a 16 ms gap into a 32 ms gap. The client's
client_snap_debug.csv showed exactly this: a BIMODAL receive cadence
(76% at 10-18 ms, then a sharp 13% peak at 30-36 ms) that the host's
UNIMODAL send (55% at 10-18 ms, 31% at 18-22 ms, only 1.5% at 30-36 ms)
never produced. The extra ~666 32 ms gaps were manufactured by the GIL
re-acquisition, not the wire.

Moving the socket + the JSON parse into a dedicated NET process removes
the shared GIL entirely: the net process has its own GIL, shared only
with itself. The render thread's draw can hold the render GIL for as long
as it wants and the net process keeps draining the socket and stamping
the wire-arrival time on time. The render thread is COMPLETELY OUT of the
receive path — it only sends the local input (a Pipe put) and drains the
net process's message Pipe (snaps/beam/echo/pong).

SOCKET OWNERSHIP (mirrors the host's net process)
A live socket can't be pickled across the spawn Pipe, so the net process
OWNS the socket end to end: it does the connect + the join/welcome
handshake (the render process no longer touches the socket at all). The
render process learns the outcome by polling the message Pipe
(non-blocking `poll()` — the "connecting" screen's draw loop stays
smooth):
  ("connected", host_hull_id, host_loadout_dict)  (after the handshake)
  ("snap", sim_time, snap, wire_mono, wire_wall)  (a received snapshot)
  ("beam", msg) | ("echo", msg) | ("pong", tag)   (the other wire msgs)
  ("disconnected",)                                (the host went away)
  ("error", text)                                  (connect/handshake failed)
  ("crash", traceback_text)                        (the net process died)

IPC (spawn context — see net_process.py):
  render -> net (the cmd Pipe):
                   ("input", inp_dict, tag)   (the local input, 60 Hz)
                   ("respawn",)               (the local player asks to respawn)
                   ("quit",)
  net -> render (the msg Pipe):
                   ("connected", host_hull_id, host_loadout_dict) |
                   ("snap", sim_time, snap, wire_mono, wire_wall) |
                   ("beam", msg) | ("echo", msg) | ("pong", tag) |
                   ("disconnected",) | ("error", text) | ("crash", tb)

The net process is HEADLESS: it sets SDL_VIDEODRIVER/AUDIODRIVER=dummy
BEFORE the net import (inside the function, so the PARENT — which imports
this module for client_net_process_main — keeps its real video driver).
The net import chain pulls in pygame (net -> intent -> pygame), so the
dummy driver is required; the net process never creates a display (it
does no draw).
"""
import os

import time
import traceback


def _raise_priority_best_effort():
    """Best-effort: raise this process's scheduling priority so the OS wakes
    it on time (the secondary lever — the separate GIL is the primary fix).

    WHY it is best-effort and wrapped: raising priority needs permission
    (CAP_SYS_NICE on Linux, root on macOS, an elevated token on Windows).
    On a normal user session it fails — that is FINE, the separate GIL
    still removes the render-thread contention. The elevation only helps
    the OS-descheduling component (a render process burning a core
    stealing the net process's timeslice). Every branch is try/except
    with a silent fallback to normal priority, so the game runs on all
    three OSes whether or not the API is available or permitted.

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


def client_net_process_main(cmd_conn, msg_conn, ip, port,
                            client_hull, client_loadout):
    """The spawn target (MUST be a top-level function: spawn re-imports this
    module by name to find it). Runs the client's socket + the input send +
    the snapshot receive in this process.

    Channels (all from the spawn context):
      `cmd_conn`       a Pipe, render -> net. The render process sends the
                       local input (("input", inp_dict, tag), 60 Hz),
                       respawn (("respawn",)), and ("quit",). Drained HERE
                       (non-blocking poll) and the input/respawn are sent
                       to the socket the moment they land.
      `msg_conn`       a Pipe, net -> render. The net process sends the
                       handshake outcome (("connected", host_hull_id,
                       host_loadout_dict)), the received wire messages
                       (("snap", sim_time, snap, wire_mono, wire_wall),
                       ("beam", msg), ("echo", msg), ("pong", tag)), and
                       the disconnect/crash status. The render process
                       polls it NON-BLOCKING (poll()) so the "connecting"
                       screen's draw loop stays smooth.
      `ip` / `port`    the host's address (the render process's menu choice).
      `client_hull` / `client_loadout`  the client's real fit (HullType +
                       dict), pickled across the spawn boundary (frozen
                       dataclass + plain dict — both picklable). Serialized
                       for the join.

    Exits cleanly on ("quit",) or on cmd_conn EOF (the render process
    died), or on a crash (after sending ("crash", traceback))."""
    # Headless: the net import chain pulls in pygame, so set the dummy
    # drivers BEFORE importing net (inside the function, so the parent keeps
    # its real video driver). Mirrors net_process.py.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from .net import (connect, do_handshake_client, serialize_hull,
                      serialize_loadout, T_SNAP, T_INPUT, T_RESPAWN,
                      T_BEAM, T_ECHO, T_PONG)

    def _status(msg):
        """Send a message to the render process. A failure means the render
        process is gone (the Pipe is closed) — there is no one left to
        report to, so return False and the caller exits."""
        try:
            msg_conn.send(msg)
            return True
        except (OSError, EOFError, ConnectionError, BrokenPipeError):
            return False

    conn = None
    try:
        # --- 1. connect + handshake (the socket is owned HERE) ---
        # The connect is BLOCKING (connection-setup phase, before the game
        # loop). A refusal raises socket.error; the handshake returns
        # (None, None) on timeout. Either way the render process learns of
        # the failure via ("error", text) and bails to the menu.
        try:
            conn = connect(ip, port, timeout=10.0)
        except OSError as e:
            _status(("error", "no host at %s:%d (%s)"
                     % (ip, port, type(e).__name__)))
            return
        wh, wl = do_handshake_client(conn, serialize_hull(client_hull),
                                     serialize_loadout(client_loadout))
        if wh is None:
            conn.close()
            _status(("error", "the host went away or took too long"))
            return
        # Tell the render process the join succeeded (it builds the Game
        # from the host's real fit + its own).
        if not _status(("connected", wh, wl)):
            conn.close()
            return

        # --- 2. the game loop (the socket is non-blocking now) ---
        conn.set_nonblocking()
        # Best-effort priority elevation (the SECONDARY lever — see
        # _raise_priority_best_effort). The separate GIL is the PRIMARY fix
        # and works without this; the elevation only helps the
        # OS-descheduling component (a render process burning a core
        # stealing the net process's timeslice). Wrapped: it falls back to
        # normal priority silently if the API is unavailable or unpermitted.
        _raise_priority_best_effort()
        while True:
            # cmd side: drain the render's input/respawn/quit (non-blocking).
            # The input is sent to the socket the moment it lands (~1 ms) —
            # the same hand-off the in-process worker input sink did, but in
            # this process (off the render thread's GIL). The tag is
            # forwarded verbatim (the render process generated it + recorded
            # its send time for the RTT measurement).
            while cmd_conn.poll():
                m = cmd_conn.recv()
                if m[0] == "quit":
                    conn.close()
                    return
                elif m[0] == "input":
                    # m = ("input", inp_dict, tag)
                    conn.send({"type": T_INPUT, "inp": m[1], "tag": m[2]})
                elif m[0] == "respawn":
                    conn.send({"type": T_RESPAWN})
            conn.drain_send()
            # RECEIVE side: drain the socket, parse JSON HERE (on this
            # process's own GIL — NOT the render thread's), stamp the
            # wire-arrival time, and forward to the render process.
            for m in conn.poll():
                t = m.get("type")
                if t == T_SNAP:
                    # WIRE-ARRIVAL STAMP: the moment the snapshot's bytes
                    # were parsed off the socket, on THIS process's own GIL
                    # (uncontended by the render thread's draw). This is the
                    # true wire cadence — the client's client_snap_debug.csv
                    # uses it as its "receive" time. (The old NetWorker
                    # thread stamped this AFTER re-acquiring the render
                    # GIL, which the render thread held for a whole draw
                    # frame — the bimodal 16/32 ms artifact.)
                    _status(("snap", m["sim_time"], m["snap"],
                             time.monotonic(), time.time()))
                elif t == T_BEAM:
                    _status(("beam", m))
                elif t == T_ECHO:
                    _status(("echo", m))
                elif t == T_PONG:
                    _status(("pong", m.get("tag")))
            # Close detection: the host went away (FIN / RST).
            if conn.closed:
                _status(("disconnected",))
                break
            # Idle sleep: bounds this process's CPU when nothing is in
            # flight. ~1 ms keeps the input/respawn forward + the snap
            # forward latency to ~1 ms (the snaps arrive at 60 Hz, so a
            # snap is forwarded within ~1 ms of landing).
            time.sleep(0.001)
    except BaseException:
        # A crash in the net process must not take the render process down
        # silently: report it, then exit (the render process shows a notice).
        try:
            msg_conn.send(("crash", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Direct run is not supported (spawn needs a picklable top-level target +
    # a Pipe from the parent). Use test_client_net_process.py.
    raise SystemExit("client_net_process_main is a spawn target; "
                     "run test_client_net_process.py")