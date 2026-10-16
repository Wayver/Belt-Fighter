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

WAIT = 0.001   # ~1 ms loop pace (like NetWorker.WAIT)


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
        while True:
            # Quit / render-process-gone (checked first — snappy shutdown).
            if status_conn.poll():
                m = status_conn.recv()
                if m[0] == "quit":
                    break
            # SEND side: drain the sim's net Queue -> the socket. The sim
            # publishes a snap (+ beam/echo) per step; the net process sends
            # each the moment it lands, so the snapshot's age at send is ~0
            # (the client's `now = sim_time + L` is exact). The queue is
            # bounded + drop-when-full (the sim's put_nowait), so a backlog
            # never blocks the sim — a dropped snapshot is a network loss the
            # client's interpolation buffer already handles.
            while True:
                try:
                    m = net_queue.get_nowait()
                except Exception:
                    break
                kind = m[0]
                if kind == "snap":
                    conn.send({"type": T_SNAP, "sim_time": m[1],
                               "snap": serialize_snapshot(m[2])})
                else:   # "beam" / "echo" — the msg dict is m[1]
                    conn.send(m[1])
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
            time.sleep(WAIT)
    except BaseException:
        # A crash in the net process must not take the render process down
        # silently: report it, then exit (the render process shows a notice).
        try:
            status_conn.send(("crash", traceback.format_exc()))
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
    # a Pipe/Queue from the parent). Use test_net_process.py.
    raise SystemExit("net_process_main is a spawn target; "
                     "run test_net_process.py")