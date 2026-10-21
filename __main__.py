"""Entry point: run with  python -m ship5

Three modes (chosen on the menu's mode screen):
  single — the classic solo run. Unified onto the SimThread — the same
           architecture as the host (the sim on its own real-time 60 Hz
           thread, the render thread a local client of the published
           RenderModel; no NetWorker).
  host   — wait for ONE client, run the authoritative 2-ship sim, apply
           the client's input, broadcast a snapshot every
           SNAPSHOT_INTERVAL ticks. ESC/QUIT or a client disconnect
           returns to the menu (pinned decision #8: no reconnect in v1).
  join   — connect to a host, run the client sim (predict the local
           ship, interpolate the remote entities), send input every
           frame, push each snapshot, render via predicted_view.
"""
import gc
import multiprocessing
import os
import queue as _queue
import socket
import sys
import time

import pygame

from .config import WIDTH, HEIGHT, FPS, NET_PORT, BG, SNAPSHOT_INTERVAL, TICK
from .fog import make_light_texture
from .game import Game, STEP
from .menu import Menu
from .net import (connect, do_handshake_client,
                  serialize_hull, serialize_loadout,
                  deserialize_hull, deserialize_loadout,
                  serialize_snapshot, deserialize_snapshot,
                  serialize_input, deserialize_input,
                  NetWorker,
                  T_INPUT, T_SNAP, T_RESPAWN, T_BEAM, T_ECHO, T_PONG)
from .netcode import (PredictedShip, HostTimeEstimator, LatencyTracker,
                      RenderPoint, BEAM_TTL)
from .ship import Ship
from .sim_thread import SimThread
from .sim_process import sim_process_main
from .net_process import net_process_main
from .sound import SoundBank
from .intent import ShipInput
from . import missile_telemetry
from . import bullet_telemetry
from . import bullet_time_telemetry
from . import bullet_handoff_telemetry
from . import hit_fx_telemetry


class _HostRenderClock:
    """The host render thread's own interpolation clock.

    With the sim on the SimThread, the render thread no longer owns the
    sim's accumulator — so it computes its OWN interpolation alpha
    instead of reading the model's `step_alpha` (the sim's acc/STEP at
    publish time). The model still CARRIES step_alpha (the parity tests
    read it), and it is also what this clock re-anchors on.

    The model carries the local ship's prev+curr pose (the window
    [T - STEP, T], where T = model.sim_time), so no 2-deep model ring is
    needed — the latest model alone spans the interpolation window. The
    clock keeps an estimate of the sim's CURRENT time (sim_time + the
    in-progress step's fraction):

      * on each NEW published model: re-anchor on the data —
        t = model.sim_time + model.step_alpha * STEP (the sim's current
        time at publish, ~1-2 ms old). Re-anchoring (instead of
        free-running) is what keeps the estimate honest: the sim's
        hiccup policy DROPS backlog time (ACC_BACKLOG_CAP), so a
        free-running 1x-real-time estimate would drift ahead of the sim
        clock after any stall and stay clamped at alpha=1 forever.
      * between publishes (the same model re-read): advance t by the
        frame's real dt (1x real time — the host's sim rate).

    The alpha is then (t - T) / STEP, clamped to [0, 1] (on a hiccup
    render the LATEST simulated state, never extrapolate past the
    current pose). In steady state t - T equals the sim's acc at publish
    (plus ~1 ms of publish latency), so the alpha matches the old
    sim-acc/STEP behavior exactly — the rendered pose is sim_now - STEP,
    one step behind the sim's current time.
    """

    def __init__(self):
        self._curr = None    # the newest published model
        self._t = None       # the render-thread estimate of the sim's current time

    def advance(self, dt, model):
        """One render frame of real time `dt` (the CLAMPED dt — a hiccup
        frame must not drag the estimate forward faster than 1x).
        `model` is the latest published model (an atomic reference read
        of SimThread.latest_model; None before the first publish).
        Returns the interpolation alpha for this frame."""
        if model is None:
            return 0.0
        if model is not self._curr:
            # A new model was published: re-anchor on its stamp + the
            # sim's in-progress fraction (the sim's current time at
            # publish). Do NOT also add dt — the anchor is fresh.
            self._curr = model
            self._t = model["sim_time"] + model["step_alpha"] * STEP
        else:
            # Same model as last frame: advance the estimate at 1x real
            # time (the host's sim rate) until the next publish.
            self._t += dt
        T = self._curr["sim_time"]
        alpha = (self._t - T) / STEP
        return max(0.0, min(1.0, alpha))


def _drain_sim_model(sim_model_q):
    """Drain the sim process's model Queue (the render thread, the SAME
    thread as the draw — a locked decision: a separate drain thread would
    be starved by the draw's GIL hold, the queue would fill, and the sim
    process would block).

    Returns (newest_model, sfx_events):
      * newest_model — the NEWEST model drained this call (latest-wins,
        for drawing). None if no model was drained this frame.
      * sfx_events — the CONCATENATED "sfx" list from EVERY model drained
        this call, in arrival order. The render process plays ALL of them
        (not just the newest model's) — a stale model's sfx would
        otherwise be lost when only the newest is kept for drawing
        (plan_12 D.1).

    The wire messages (snap/beam/echo) do NOT ride this Queue: they go to
    the sim process's DEDICATED net Queue, drained by the dedicated NET
    process (plan_13) — the render thread's draw is out of the
    snapshot's send path.

    Raises RuntimeError on a ("crash", tb) message (the sim process died —
    the caller shows a notice and returns to the menu). "ready" / "done"
    are lifecycle messages, not models; ignored here."""
    newest = None
    sfx_events = []
    while True:
        try:
            m = sim_model_q.get_nowait()
        except _queue.Empty:
            break
        kind = m[0]
        if kind == "model":
            newest = m[1]
            sfx_events.extend(m[1].get("sfx", ()))
        elif kind == "crash":
            raise RuntimeError("SIM CRASHED:\n" + m[1])
    return newest, sfx_events


def _lan_ip():
    """Best-effort LAN IP to show the host (the client types it in).

    A UDP 'connect' sends no packet — it just makes the kernel pick the
    interface it WOULD use, so the local address is the LAN IP. Falls
    back to 127.0.0.1 (loopback testing) on any failure.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


def _notice(screen, font, big_font, clock, lines, min_s=2.5):
    """Full-screen notice (disconnect / error). Dismissed by any key or
    after `min_s` seconds. Returns False only when the window was closed."""
    t0 = pygame.time.get_ticks()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return True
        screen.fill(BG)
        y = HEIGHT // 2 - 60
        for i, line in enumerate(lines):
            f = big_font if i == 0 else font
            c = (200, 210, 225) if i == 0 else (110, 120, 140)
            surf = f.render(line, True, c)
            screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
            y += 50 if i == 0 else 32
        pygame.display.flip()
        if pygame.time.get_ticks() - t0 >= min_s * 1000:
            return True
        clock.tick(FPS)


def _draw_waiting(screen, font, big_font, ip, port):
    """The host's 'waiting for a player' screen (drawn while accepting)."""
    screen.fill(BG)
    title = big_font.render("WAITING FOR A PLAYER", True, (200, 210, 225))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2,
                        HEIGHT // 2 - 120))
    line1 = font.render("they type:  %s:%d" % (ip, port), True,
                        (120, 200, 255))
    screen.blit(line1, (WIDTH // 2 - line1.get_width() // 2,
                        HEIGHT // 2 - 40))
    hint = font.render("ESC back to menu", True, (110, 120, 140))
    screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2,
                       HEIGHT // 2 + 20))


def run_host(screen, font, big_font, clock, sfx, menu, seed,
             light_tex, fog_surf, light_surf):
    """Host a 2P game.

    Phases (plan_13, Phase 1): (0) spawn the dedicated NET process EARLY
    (it owns the socket: listen + accept + the join/welcome handshake —
    the render process never touches the socket); (1) draw the waiting
    screen while polling the net process's status Pipe for the client's
    join; (2) [the handshake runs in the net process — the render learns
    the client's hull/loadout from the ("connected", ...) status message];
    (3) spawn the authoritative sim in its OWN process (sim_process.py —
    plan_12 M1/M2) with the 4-arg form (the sim's net Queue is drained by
    the net process; the client's input arrives on the net -> sim
    remote_input Pipe) and build the SHELL Game in this (render) process;
    (4) the game loop: publish local input + commands to the sim, drain
    the sim's model queue (SAME thread as the draw — a locked decision),
    play the model's sfx, draw the shell from the piped model, and poll
    the net status Pipe for a disconnect. Returns to the caller (the
    menu) on ESC/QUIT or a client disconnect (pinned decision #8).

    The render thread is COMPLETELY OUT of the send path: the socket, the
    snapshot send, and the client's input all live in the net process
    (its own GIL). The render thread only sends local input/commands to
    the sim, drains the model Queue, plays sfx, draws, and polls the net
    status Pipe.

    M3 (plan_12): the authoritative Game no longer lives in this process.
    The render thread is a LOCAL CLIENT of the sim process — the same
    relationship the remote client has over the network, but over a pipe.
    `game` here is a SHELL: built with the REAL surfaces/fonts/textures +
    sound + both players' hulls, then NEVER stepped. draw(dt, model) runs
    on it from the piped model; its cam/_standins/_death_bursts/
    remote_beams are the render-side presentation state (today they live
    on the shared Game and are only touched by the render thread — the
    shell preserves that split). The shell's sound=sfx is inert (it is
    never stepped, so it never fires SFX) — the host's SFX play from the
    model's "sfx" list (plan_12 D.1: the silent sim process records the
    events; this process plays them via the same SoundBank).
    """
    ip = _lan_ip()
    # plan_13: the socket lives in the dedicated NET process (net_process.py),
    # NOT the render process. Bind a temp socket to reserve the port + read
    # the bound port back (shown on the waiting screen), then spawn the net
    # process EARLY (it owns the listen socket + accept + handshake). The
    # render process never touches the socket — it polls the status Pipe.
    try:
        _port_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _port_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _port_sock.bind(("0.0.0.0", NET_PORT))
        port = _port_sock.getsockname()[1]
        _port_sock.close()
    except OSError:
        _notice(screen, font, big_font, clock,
                ["COULD NOT LISTEN", "port %d is unavailable" % NET_PORT])
        return
    game = None   # the SHELL Game (M3: never stepped — draw-only). Defined
                  # here so the finally can close its debug log even if the
                  # handshake fails before phase 3 builds it.
    # The dedicated NET process (plan_13 Phase 1): owns the socket (listen +
    # accept + handshake), drains the sim's net Queue -> the socket, and
    # forwards the client's input -> the sim's remote_input Pipe. Defined
    # here so the finally can shut it down on every exit path.
    net_proc = None
    net_status = None   # the render's end of the status Pipe (render <-> net)
    # The authoritative sim process (M3, plan_12): a spawned child running
    # the real Game + SimThread loop (sim_process.py). Defined here so the
    # finally can shut it down on every exit path.
    sim_proc = None
    sim_cmd = None      # the parent end of the command Pipe (render -> sim)
    sim_model_q = None  # the bounded model Queue (sim -> render)
    sim_net_q = None    # the bounded net Queue (sim -> net, wire msgs)
    try:
        # --- 0. spawn the NET process EARLY (it owns the listen socket) ---
        # The net process binds 0.0.0.0:port, listens, accepts ONE client,
        # and runs the join/welcome handshake — all in its OWN process (its
        # own GIL, off the render thread). The render process learns the
        # outcome by polling the status Pipe (non-blocking poll()).
        _net_ctx = multiprocessing.get_context("spawn")
        net_status, _net_status_child = _net_ctx.Pipe()   # render <-> net
        # The sim -> net wire channel: the net process drains this (NOT the
        # render thread), so the draw is out of the snapshot's send path.
        # maxsize=16 holds ~2 steps of wire traffic; a dropped snapshot is a
        # network loss the client's interpolation buffer already handles.
        sim_net_q = _net_ctx.Queue(maxsize=16)            # sim -> net (wire)
        # The net -> sim channel: the client's input (("remote_input",
        # ShipInput)) + forwarded commands (("command", name, arg), e.g.
        # respawn). The sim process's bridge loop drains it.
        _ri_sim_end, _ri_net_end = _net_ctx.Pipe()        # net -> sim
        net_proc = _net_ctx.Process(target=net_process_main,
                                    args=(_net_status_child, sim_net_q,
                                          _ri_net_end, menu.hull,
                                          menu.loadout, port))
        net_proc.start()
        _net_status_child.close()   # the render uses net_status both ways
        _ri_net_end.close()         # the render never sends to the sim's ri

        # --- 1. wait for a client (waiting screen + poll the status Pipe) ---
        # The net process owns the accept; the render process only polls the
        # status Pipe (non-blocking) + pumps events (ESC/QUIT -> quit the net
        # process + return). No socket I/O here (the render is out of the
        # connection path).
        ch = cl = None
        while ch is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            if net_status.poll():
                m = net_status.recv()
                if m[0] == "connected":
                    ch, cl = m[1], m[2]     # the client joined (the net
                    break                   # process did the handshake)
                elif m[0] == "error":
                    _notice(screen, font, big_font, clock,
                            ["JOIN FAILED", m[1]])
                    return
                elif m[0] == "crash":
                    _notice(screen, font, big_font, clock,
                            ["NET PROCESS CRASHED",
                             m[1].splitlines()[-1] if m[1] else ""])
                    return
            elif not net_proc.is_alive():
                # The net process died without a status (a crash before it
                # could report): treat it as a net failure.
                _notice(screen, font, big_font, clock,
                        ["NET PROCESS DIED", "the net process exited"])
                return
            _draw_waiting(screen, font, big_font, ip, port)
            pygame.display.flip()
            clock.tick(FPS)

        # --- 3. spawn the sim process + build the SHELL Game -----------------
        # The authoritative sim runs in its OWN process (plan_12 M1/M2):
        # the render thread holds the GIL for its whole draw (~18-25 ms —
        # longer than the 16.7 ms sim tick), so a same-process sim thread
        # is starved and batches 2 ticks (ragged 17/33 ms snapshot stamps
        # -> the client's ~8 px snap tail). A separate process has no
        # shared GIL: the sim steps exactly one tick per iteration.
        #
        # Spawn AFTER the handshake (E.4): the client's hull is known (it
        # arrived on the status Pipe from the net process), so the setup
        # message carries BOTH players' real fits. The spawn cost
        # (~0.1-0.3 s: python startup + pygame import) overlaps the
        # waiting screen / handshake.
        chull = deserialize_hull(ch)
        clout = deserialize_loadout(chull, cl)
        # The SHELL Game (plan_12 D.2): built with the REAL
        # surfaces/fonts/textures + sound + both players' hulls, then
        # NEVER stepped. draw(dt, model) runs on it from the piped model;
        # its cam/_standins/_death_bursts/remote_beams are the render-side
        # presentation state (only the render thread touches them — the
        # shell preserves that split). sound=sfx is inert (the shell is
        # never stepped, so it never fires SFX) — the host's SFX play from
        # the model's "sfx" list (D.1) via the same SoundBank.
        game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                    hull=menu.hull, loadout=menu.loadout,
                    seed=seed, sound=sfx, players=2)
        game.set_player_ship(1, Ship(hull=chull, loadout=clout))
        # Spawn the authoritative sim (the spawn context re-imports
        # sim_process by name — sim_process_main is a top-level function).
        # The sim's net Queue (sim_net_q) + the net -> sim remote_input
        # Pipe were created in phase 0 (with the net process) — the net
        # process drains sim_net_q and the sim drains the remote_input
        # Pipe. The render process passes the SIM's end of the
        # remote_input Pipe (_ri_sim_end) to the sim process.
        _sim_ctx = multiprocessing.get_context("spawn")
        sim_cmd, _sim_child_conn = _sim_ctx.Pipe()   # render -> sim
        # The sim -> render MODEL channel: the render model (60 Hz) +
        # lifecycle, drained by the render thread (the same thread as the
        # draw — a locked decision). maxsize=4 holds ~1 step of models —
        # the wire traffic rides the DEDICATED net Queue (sim_net_q,
        # phase 0), drained by the NET process, so the original sizing
        # stands.
        sim_model_q = _sim_ctx.Queue(maxsize=4)      # sim -> render (models)
        sim_proc = _sim_ctx.Process(target=sim_process_main,
                                    args=(_sim_child_conn, sim_model_q,
                                          sim_net_q, _ri_sim_end))
        sim_proc.start()
        _sim_child_conn.close()   # the render process never sends to it
        _ri_sim_end.close()       # the render never sends to the sim's ri
        # The 6-arg setup (the plan's original sketch): the host's +
        # client's REAL hulls/loadouts + the seed. The sim process builds
        # its authoritative Game from these (the client's hull matters for
        # physics — mass, weapons, shield).
        sim_cmd.send(("setup", menu.hull, menu.loadout, chull, clout, seed))
        # Wait for the sim's ("ready",) (it built the Game + started its
        # SimThread). Bounded: a crash shows a notice + returns to the
        # menu (the finally shuts the process down).
        _sim_ready = False
        _sim_deadline = time.monotonic() + 10.0
        while time.monotonic() < _sim_deadline:
            try:
                _m = sim_model_q.get(timeout=0.2)
            except _queue.Empty:
                if not sim_proc.is_alive():
                    break
                continue
            if _m[0] == "ready":
                _sim_ready = True
                break
            elif _m[0] == "crash":
                _notice(screen, font, big_font, clock,
                        ["SIM CRASHED", "the sim process died at startup"])
                return
        if not _sim_ready:
            _notice(screen, font, big_font, clock,
                    ["SIM UNAVAILABLE", "the sim process did not start"])
            return
        # plan_13: the socket I/O + snapshot send + the client's input all run in
        # the dedicated NET process (net_process.py) — NOT a NetWorker
        # thread in the render process. The net process drains the sim's
        # net Queue (sim_net_q) and sends each message the moment it lands
        # (the snapshot's age at send is ~0, so the client's
        # `now = sim_time + L` is exact), and forwards the client's input
        # to the sim's remote_input Pipe the moment it lands (~1 ms) — the
        # same hand-off the in-process worker input sink did, but off the
        # render thread's GIL. The T_PONG is reflected in the net process
        # (~1 ms) so the measured RTT is honest. The render thread never
        # touches the socket, the net Queue, or the snapshot send.
        #
        # The command Pipe now has a SINGLE sender (the render thread —
        # the worker's input sink is gone), so _sim_cmd_send needs no
        # lock (the plan's recommendation: remove it).
        def _sim_cmd_send(msg):
            sim_cmd.send(msg)

        # --- 4. the game loop (the sim is the spawned process) ---
        # The sim runs in its OWN process (phase 3): the render thread is
        # a LOCAL CLIENT of it — it publishes the latest local input +
        # T/V/G/R/F commands over the command Pipe, DRAINS the model Queue
        # (latest-wins, _drain_sim_model) in the SAME thread as the draw,
        # plays the model's sfx, and draws the SHELL from the newest model.
        # The wire traffic (snap/beam/echo) is NOT the render thread's job
        # (plan_13): the dedicated NET process drains the sim's net Queue
        # and sends it to the socket, so the draw is out of the snapshot's
        # send path. The shell Game is NEVER stepped: the render thread
        # never mutates sim state (it only reads the piped model + its own
        # presentation state: cam/_standins/_death_bursts/remote_beams).
        # The render thread's own interpolation clock (the sim's acc/STEP
        # is no longer the render thread's — see the class docstring).
        render_clock = _HostRenderClock()
        # The newest model the render thread has (PERSISTENT — mirrors the
        # in-process path's sim_thread.latest_model). The drain returns the
        # newest model drained THIS frame, or None on a frame where the sim
        # published nothing new (the render thread wakes less often than the
        # sim's 60 Hz whenever the draw is slower than 60 FPS). Drawing `None`
        # would fall through to draw() building a model from the never-stepped
        # SHELL (frozen at spawn) — the host would flicker between the live
        # world and the spawn world on every 0-model frame (the "berserk"
        # glitch). So we keep the last model and always draw the freshest
        # world we have.
        last_model = None
        # Host-side frame-rate telemetry (F3-toggled). The send moved to the
        # NET process (plan_13), so the telemetry runs on its OWN
        # real-time timer in the main loop (it measures the host's FRAME
        # rate — a main-thread concern — and the wall-clock `t` is the
        # join key to the client's CSV, not the send event).
        # last_telem_time anchors it (0.0 -> the first line is immediate).
        last_telem_time = 0.0
        TELEMETRY_PERIOD = SNAPSHOT_INTERVAL * STEP   # 0.0167 s (60 Hz)
        # The F3 edge state for the per-snapshot cadence logs (the sim +
        # net processes' CSVs — see the F3-edge block in the loop).
        snap_log_on = False
        # Host-side frame-rate telemetry (F3-toggled, mirrors the client's
        # columns). The client's CSV can't tell a WIRE stall (host sent
        # smoothly, packets queued + released in a burst) from a HOST frame
        # stall (a host hiccup stalls the sim AND the snapshot sending).
        # We log the host's raw frame time on each telemetry tick so the
        # next re-test can be correlated: if the client's snapshot gap
        # lines up with a host hiccup (dt_max > 50 ms / low fps) at the
        # same wall-clock time, it's a HOST stall; if the host was clean
        # (60 fps, no hiccups) at that moment, it's a WIRE stall.
        game._dbg_frames = 0
        game._dbg_dt_sum = 0.0
        game._dbg_dt_max = 0.0
        game._dbg_hiccups = 0
        game._dbg_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "host_debug.csv")
        game._dbg_log_f = None
        # Per-frame sim-clock diagnostic (F3-toggled, additive). Logs raw_dt
        # (real frame time — what the sim is fed), draw_dt (the clamped
        # dt that drives draw()), sim_before/sim_after (the sim clock),
        # advance (sim time gained this frame), and acc_after
        # (accumulator remainder). Used to verify the sim-clock rate is
        # ~1.0 and advance tracks raw_dt (modulo STEP quantization + the
        # backlog cap).
        game._dbg_sim_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "host_sim_debug.csv")
        game._dbg_sim_log_f = None
        # T/V/G/R/F -> the sim process (the command Pipe). The sim process
        # applies the command at the top of its next iteration (the pipe
        # analog of the in-process sim_thread.command queue). The render
        # thread is the command Pipe's only sender (the worker's input
        # sink is gone — plan_13), so _sim_cmd_send needs no lock.
        def _sim_command(name, arg=None):
            _sim_cmd_send(("command", name, arg))
        # Models drained since the last telemetry line (the host CSV's
        # `steps` column — the snapshot cadence as seen by the render
        # thread; the child-side per-step stamps are the authoritative
        # cadence measure, M1).
        _models_window = 0
        # GC tuning (PERFORMANCE.md rec C): freeze the long-lived baseline
        # (surfaces, hulls, the shell Game, fonts, textures) so the cyclic
        # GC only scans the small set of per-frame temporaries. Trims the
        # draw max/p99 tail (measured 5.63->5.11 ms max, 4.44->4.26 p99).
        # Reference counting is untouched; objects allocated after this are
        # still collected normally.
        gc.freeze()
        while True:
            # Capture the RAW frame time before the clamp (the clamp hides
            # hiccups from the sim, but a raw frame > 50 ms is exactly
            # the frame that stalls the sim + snapshot sending).
            raw_dt = clock.tick(FPS) / 1000.0
            dt = min(raw_dt, 0.05)
            if game.debug_net:
                game._dbg_frames += 1
                game._dbg_dt_sum += raw_dt
                if raw_dt > game._dbg_dt_max:
                    game._dbg_dt_max = raw_dt
                if raw_dt > 0.05:
                    game._dbg_hiccups += 1
            # The T/V/G/R/F keys are ROUTED to the sim process (command_sink ->
            # the command Pipe) instead of mutating live state — the sim
            # process applies them at the top of its next iteration.
            # QUIT/ESC (exit) + F3 (the render diagnostic) stay handled
            # here. handle_events still reads game.game_over/test_mode
            # for the same gating (a bool read on the shell — always
            # False/False for a 2P host, matching the sim's gating).
            if not game.handle_events(_sim_command):
                return
            # F3 edge: toggle the per-snapshot cadence logs (plan_13
            # follow-up). The SIM + NET processes each write their own
            # CSV (sim_snap_debug.csv / net_snap_debug.csv — both on the
            # host's machine, where those processes run); the render
            # process only tells them the path (or None to close). The
            # paths are absolute (the repo root — the same dir as the
            # render's own CSVs). The client writes client_snap_debug.csv
            # on its own machine (run_client's F3 edge). Correlating the
            # three by sim_time localizes a 33 ms client gap to the sim
            # (batching), the net process (bursty send / queue drops), or
            # the wire + client receive.
            if game.debug_net != snap_log_on:
                snap_log_on = game.debug_net
                _sim_cmd_send(("command", "snap_log",
                               os.path.join(os.path.dirname(
                                   os.path.abspath(__file__)),
                                   "sim_snap_debug.csv")
                               if snap_log_on else None))
                try:
                    net_status.send(("snap_log",
                                     os.path.join(os.path.dirname(
                                         os.path.abspath(__file__)),
                                         "net_snap_debug.csv")
                                     if snap_log_on else None))
                except Exception:
                    pass
            keys = pygame.key.get_pressed()
            # Poll the NET process's status Pipe (non-blocking). The
            # client's input + the T_PONG + the T_RESPAWN are all handled
            # in the NET process (it forwards the input to the sim's
            # remote_input Pipe and reflects the T_PONG the moment they
            # land — the render thread never touches the socket). The
            # render thread only learns of a DISCONNECT (or a net-process
            # crash) here.
            if net_status.poll():
                m = net_status.recv()
                if m[0] == "disconnected":
                    _notice(screen, font, big_font, clock,
                            ["DISCONNECTED", "the other player left"])
                    return
                elif m[0] == "crash":
                    _notice(screen, font, big_font, clock,
                            ["NET PROCESS CRASHED",
                             m[1].splitlines()[-1] if m[1] else ""])
                    return
            elif not net_proc.is_alive():
                # The net process died without a status (a crash before it
                # could report): treat it as a disconnect.
                _notice(screen, font, big_font, clock,
                        ["DISCONNECTED", "the net process exited"])
                return
            # The sim runs in its OWN process (phase 3). The render thread's job
            # each frame: publish the latest local input (the command
            # Pipe), DRAIN the model Queue (latest-wins, in THIS thread —
            # the same thread as the draw, a locked decision), play the
            # drained sfx, and draw the shell from the newest model. `dt`
            # (clamped) drives draw() + the render clock — presentation
            # only. The wire traffic (snap/beam/echo) is the NET
            # process's job (plan_13) — NOT this thread's.
            _sim_cmd_send(("input", ShipInput.from_keys(keys)))
            # Drain the model Queue (latest-wins) + collect the sfx. A
            # crash message raises (the sim process died) — caught below
            # (notice + return to the menu; the finally shuts it down).
            try:
                model, sfx_events = _drain_sim_model(sim_model_q)
            except RuntimeError as _sim_err:
                _notice(screen, font, big_font, clock,
                        ["SIM CRASHED", str(_sim_err).splitlines()[0]])
                return
            _models_window += 1 if model is not None else 0
            if model is not None:
                last_model = model
            # Play the drained sfx (plan_12 D.1): the sim process is
            # silent (sound=None), so its presentation sounds arrive as
            # the model's "sfx" list and play HERE, via the host's own
            # SoundBank (the same bank the in-process path used).
            if sfx_events:
                sfx.play_sfx(sfx_events)
            # Compute the render thread's own interpolation alpha from
            # the newest model we have (see _HostRenderClock). last_model
            # (not the per-frame drain) so a 0-model frame still advances
            # the clock off the freshest world, not the frozen shell.
            alpha = render_clock.advance(dt, last_model)
            # The per-frame sim-clock diagnostic (F3-toggled, additive)
            # reads the published model + the render thread's alpha. One
            # line per frame: t (wall clock), raw_dt (real frame time),
            # draw_dt (the clamped dt that drives draw() — presentation
            # only), sim_time (the published model's sim clock), alpha
            # (the render thread's interpolation alpha).
            if game.debug_net:
                f = game._dbg_sim_log_f
                if f is None:
                    f = open(game._dbg_sim_log_path, "w", newline="")
                    f.write("t,raw_dt,draw_dt,sim_time,alpha\n")
                    game._dbg_sim_log_f = f
                f.write("%.3f,%.4f,%.4f,%.4f,%.4f\n" % (
                    time.time(), raw_dt, dt,
                    last_model["sim_time"] if last_model else 0.0, alpha))
            # Host-side frame-rate telemetry (F3-toggled), on its OWN real-time
            # timer (the send moved to the NET process, so it no longer
            # rides the send). One line per 100 ms, mirroring the client's
            # columns: t (wall clock), sim_time (the host's sim clock),
            # fps (1/mean raw frame time since the last line), dt_max
            # (longest raw frame), hic (frames past the 50 ms clamp), n
            # (frames in the window). The wall-clock `t` is the join key:
            # the client's CSV has the same `t` (both machines' wall
            # clocks are roughly synced on the same LAN), so a client
            # snapshot gap at t=X can be checked against the host's
            # fps/dt_max/hic at t=X to tell a wire stall from a host stall.
            now_t = pygame.time.get_ticks() / 1000.0
            if now_t - last_telem_time >= TELEMETRY_PERIOD:
                last_telem_time = now_t
                if game.debug_net:
                    f = game._dbg_log_f
                    if f is None:
                        f = open(game._dbg_log_path, "w", newline="")
                        f.write("t,sim_time,fps,dt_max,hic,n,steps\n")
                        game._dbg_log_f = f
                    n = game._dbg_frames
                    fps = (1.0 / (game._dbg_dt_sum / n)) if n > 0 else 0.0
                    dt_max = game._dbg_dt_max
                    hic = game._dbg_hiccups
                    game._dbg_frames = 0
                    game._dbg_dt_sum = 0.0
                    game._dbg_dt_max = 0.0
                    game._dbg_hiccups = 0
                    # Models drained this window (the snapshot cadence as
                    # seen by the render thread — the child-side per-step
                    # stamps are the authoritative cadence measure, M1).
                    steps = _models_window
                    _models_window = 0
                    f.write("%.3f,%.4f,%.1f,%.4f,%d,%d,%d\n" % (
                        time.time(),
                        last_model["sim_time"] if last_model else 0.0,
                        fps, dt_max, hic, n, steps))
                    f.flush()
            # Draw the newest model we have (draw(dt, model) — the model is built by
            # the sim process, not here), with the render thread's OWN
            # interpolation alpha (the sim's acc/STEP is no longer the
            # render thread's — see the _HostRenderClock docstring). The
            # model still carries the sim's step_alpha (the parity tests
            # read it); draw() uses the render thread's instead.
            #
            # last_model (PERSISTENT), not the per-frame drain: on a frame
            # where the drain returned None (the sim published nothing new
            # since the last frame — common when the draw is slower than
            # 60 FPS), we still draw the freshest world we have. Drawing
            # None would fall through to draw() building a model from the
            # never-stepped SHELL (frozen at spawn) — the host would
            # flicker between the live world and the spawn world on every
            # 0-model frame (the "berserk" glitch). Mirrors the in-process
            # path's persistent sim_thread.latest_model.
            game.draw(dt, {**last_model, "step_alpha": alpha} if last_model else None)
            pygame.display.flip()
    finally:
        if getattr(game, "_dbg_log_f", None) is not None:
            game._dbg_log_f.close()
            game._dbg_log_f = None
        # Close the per-frame sim-clock diagnostic log.
        if getattr(game, "_dbg_sim_log_f", None) is not None:
            game._dbg_sim_log_f.close()
            game._dbg_sim_log_f = None
        # plan_13 shutdown ordering: stop the NET process BEFORE the sim process.
        # The net process writes to the sim's remote_input Pipe (net -> sim);
        # the Pipe must not be written after the sim is gone. Stopping the
        # net process first (it stops writing to the sim's Pipe) then the sim
        # process mirrors the old "stop the worker before the sim" rule.
        # Both are no-ops when not yet spawned (the waiting screen returned
        # early on ESC/QUIT / a net error), so this is safe on every exit
        # path.
        if net_proc is not None:
            try:
                net_status.send(("quit",))
            except Exception:
                pass
            net_status.close()
            net_proc.join(timeout=2.0)
            if net_proc.is_alive():
                net_proc.terminate()
                net_proc.join()
        # Shut down the sim process (E.6): send ("quit",), close the
        # render->sim Pipe, join(timeout=2), terminate() if still alive.
        # The sim process ALSO exits on Pipe EOF (the parent died), so a
        # missed quit still can't leak it.
        if sim_proc is not None:
            try:
                sim_cmd.send(("quit",))
            except Exception:
                pass
            sim_cmd.close()
            sim_proc.join(timeout=2.0)
            if sim_proc.is_alive():
                sim_proc.terminate()
                sim_proc.join()


def _draw_joining(screen, font, big_font, ip, port):
    """The client's 'connecting' screen (drawn while connect/handshake run)."""
    screen.fill(BG)
    title = big_font.render("CONNECTING", True, (200, 210, 225))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2,
                        HEIGHT // 2 - 120))
    line1 = font.render("to  %s:%d" % (ip, port), True, (120, 200, 255))
    screen.blit(line1, (WIDTH // 2 - line1.get_width() // 2,
                        HEIGHT // 2 - 40))
    hint = font.render("one moment...", True, (110, 120, 140))
    screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2,
                       HEIGHT // 2 + 20))


def _input_str(inp):
    """A compact string of the held input for the net debug log (7.8):
    e.g. 'W', 'W+Q', 'W+Q+SPACE'. Empty string = no input."""
    parts = []
    if inp.turn < 0:
        parts.append("Q")
    if inp.turn > 0:
        parts.append("E")
    if inp.thrust_fwd:
        parts.append("W")
    if inp.thrust_rev:
        parts.append("S")
    if inp.thrust_left:
        parts.append("A")
    if inp.thrust_right:
        parts.append("D")
    if inp.fire:
        parts.append("SPACE")
    if inp.laser_fire:
        parts.append("R")
    if inp.missile_fire:
        parts.append("2")
    if inp.stop:
        parts.append("B")
    return "+".join(parts)


def _draw_net_debug(screen, font, game, inp):
    """The net debug overlay (client only, F3-toggled). Small
    text top-right (the HUD occupies the top-left). Shows the live
    feel-layer state: adaptive delay, latency jitter, buffer depth, newest
    snapshot age, render point, host-time estimate, and the LAST snap size
    (the ghost's displacement across the most recent reconcile — the key
    number: ~0 = dead-reckoning is exact, large = prediction diverged)."""
    rp = game.render_point.now()
    newest = game.snap_buf.newest_time()
    est = game.host_time.now(pygame.time.get_ticks() / 1000.0)
    jit = game.latency.jitter_ema   # None until the first sample
    lines = [
        "NET DEBUG (F3 off)",
        "delay   %.3f s" % game.latency.delay,
        "jitter  %s" % ("%.3f s" % jit if jit is not None else "-"),
        "buf     %d snaps" % len(game.snap_buf),
        "newest  %s" % ("%.3f s" % newest if newest is not None else "-"),
        "render  %s" % ("%.3f s" % rp if rp is not None else "-"),
        "host est%s" % (" %.3f s" % est if est is not None else " -"),
        "SNAP    %.2f px" % game.last_snap_px,
        "input   %s" % (_input_str(inp) or "-"),
    ]
    y = 8
    for ln in lines:
        s = font.render(ln, True, (120, 220, 120))
        screen.blit(s, (WIDTH - s.get_width() - 8, y))
        y += s.get_height() + 2


def _log_net_debug(game, inp):
    """Append one CSV line per reconcile (10 Hz) to
    net_debug.csv (client only, F3-toggled). The overlay shows the LATEST
    snap size; the log captures the DISTRIBUTION over the session + the
    input that was active, so the snap-size vs input-change correlation
    (in-flight input vs clock error) can be analyzed offline."""
    f = game._dbg_log_f
    if f is None:
        f = open(game._dbg_log_path, "w", newline="")
        f.write("t,snap_px,render_snap_px,input,delay,jitter_ema,"
                "buf_depth,newest_stamp,render_t,host_time_est,fps,dt_max,"
                "hic,n,replay_ticks,replay_span,rtt_ow,ghost_sim_time,"
                "ghost_last_step_time,ghost_acc,phase_ticks,advance_steps,"
                "frame_dt,host_rate\n")
        game._dbg_log_f = f
    rp = game.render_point.now()
    newest = game.snap_buf.newest_time()
    est = game.host_time.now(pygame.time.get_ticks() / 1000.0)
    jit = game.latency.jitter_ema   # None until the first sample
    # Frame-rate stats over the frames since the
    # last line, then reset the accumulators. fps = 1/mean(raw_dt);
    # dt_max = longest raw frame; hic = frames past the 50 ms clamp;
    # n = frames in the window.
    n = game._dbg_frames
    fps = (1.0 / (game._dbg_dt_sum / n)) if n > 0 else 0.0
    dt_max = game._dbg_dt_max
    hic = game._dbg_hiccups
    game._dbg_frames = 0
    game._dbg_dt_sum = 0.0
    game._dbg_dt_max = 0.0
    game._dbg_hiccups = 0
    f.write("%.3f,%.3f,%.3f,%s,%.4f,%s,%d,%s,%s,%s,%.1f,%.4f,%d,%d,%d,%.4f,"
        "%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f\n" % (
        time.time(), game.last_snap_px, game.last_render_snap_px,
        _input_str(inp),
        game.latency.delay,
        ("%.4f" % jit) if jit is not None else "",
        len(game.snap_buf),
        ("%.4f" % newest) if newest is not None else "",
        ("%.4f" % rp) if rp is not None else "",
        ("%.4f" % est) if est is not None else "",
        fps, dt_max, hic, n,
        game.last_replay_ticks, game.last_replay_span,
        game._rtt_ow,
        game.last_ghost_sim_time,
        game.last_ghost_last_step_time,
        game.last_ghost_acc,
        game.last_ghost_phase_ticks,
        game.last_advance_steps,
        game.last_frame_dt,
        game.host_time.rate))
    f.flush()


def _make_ghost_clock_writer(path):
    """11.x diagnostics: open ghost_clock.csv and return (file, sink).

    The `sink` is set as `PredictedShip.diag_sink`; the ghost calls it once
    per `advance()` (kind='ADV') and once per `reconcile_rewind()`
    (kind='REC'). One row per call, so the CSV captures the per-frame clock
    trace that net_debug.csv (logged once per reconcile) cannot: the
    estimator's UNCAPPED vs capped `now`, the computed ghost clock, the
    pre/post `_sim_time`/`_last_step_time`, and whether a step fired. This
    is what resolves the "why is advance_steps 0 so much" question — it
    shows, for each zero-step interval, exactly where the clock was and
    whether the MAX_LEAD cap bound.

    Columns are the UNION of the ADV and REC fields (empty when a field
    does not apply to that record kind).
    """
    f = open(path, "w", newline="")
    f.write("kind,real_time,pre_sim,pre_last,pre_acc,ht_raw,ht_cap,rtt_ow,"
            "clock,steps,post_last,post_acc,dead,snap_time,now,n,post_sim\n")

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, int):
            return str(v)
        return "%.4f" % v

    def sink(kind, rec):
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            kind,
            _fmt(rec.get('real_time')), _fmt(rec.get('pre_sim')),
            _fmt(rec.get('pre_last')), _fmt(rec.get('pre_acc')),
            _fmt(rec.get('ht_raw')), _fmt(rec.get('ht_cap')),
            _fmt(rec.get('rtt_ow')), _fmt(rec.get('clock')),
            _fmt(rec.get('steps')), _fmt(rec.get('post_last')),
            _fmt(rec.get('post_acc')), _fmt(rec.get('dead')),
            _fmt(rec.get('snap_time')), _fmt(rec.get('now')),
            _fmt(rec.get('n')), _fmt(rec.get('post_sim'))))
        f.flush()
    return f, sink


def run_client(screen, font, big_font, clock, sfx, menu, seed,
               light_tex, fog_surf, light_surf):
    """Join a 2P game.

    Phases: (1) a 'connecting' screen while the BLOCKING connect +
    join/welcome handshake run (the socket is still blocking here); (2) build
    the 2-ship sim — player 0 is the HOST's ship (from the welcome), player 1
    is the client's own ship (the menu choice), local_index=1; (3) the
    client game loop: send the local input every frame, push each received
    snapshot into the interpolation buffer + prediction ghost, and render via
    predicted_view (local ship = the ghost, remote entities = the buffer).
    Returns to the caller (the menu) on ESC/QUIT or a disconnect (pinned
    decision #8: no reconnect in v1).
    """
    ip, port = menu.host_ip, menu.host_port
    _draw_joining(screen, font, big_font, ip, port)
    pygame.display.flip()
    try:
        conn = connect(ip, port)
    except OSError:
        _notice(screen, font, big_font, clock,
                ["COULD NOT CONNECT", "no host at %s:%d" % (ip, port)])
        return

    # The join/welcome handshake (blocking; the socket is still blocking).
    # Returns the host's (hull, loadout) -> player 0, or (None, None) when the
    # host goes away / the join times out.
    wh, wl = do_handshake_client(conn, serialize_hull(menu.hull),
                                 serialize_loadout(menu.loadout))
    if wh is None:
        conn.close()
        _notice(screen, font, big_font, clock,
                ["COULD NOT JOIN", "the host went away or took too long"])
        return

    # Build the 2-ship sim: player 0 = the host's ship (from the welcome),
    # player 1 = the client's own ship (the menu choice). local_index=1: the
    # prediction ghost tracks the LOCAL ship (player 1). The constructor puts
    # `hull` in player 0, so pass the HOST's hull there and set player 1 to
    # the client's own ship explicitly (the constructor's player-1 slot is a
    # default-hull placeholder).
    whull = deserialize_hull(wh)
    wlout = deserialize_loadout(whull, wl)
    game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=whull, loadout=wlout,
                seed=seed, sound=sfx, players=2, local_index=1)
    game.set_player_ship(1, Ship(hull=menu.hull, loadout=menu.loadout))
    # The prediction ghost must predict the LOCAL ship (player 1 = the
    # client's own hull/loadout), not the default-hull placeholder Game
    # builds. Rebuild it with the client's fit (same seam the host uses for
    # its ship, but the ghost is a private presentation object).
    #
    # `local_index` MUST be passed. The default is 0 (the
    # host), but the client is player 1 — and the ghost stamps its own
    # missiles with (local_index, seq) as the dedup key. With the default
    # 0 the ghost's ids were (0, seq) while the host's authoritative copy
    # of the SAME missile was (1, seq): the id-dedup never matched, so the
    # buffer's copy was never suppressed (a second missile) AND the
    # handback id-check never matched, so the ghost's copy was never
    # culled (it flew through the target for the full MISSILE_LIFE). The
    # "two missiles from one fire" bug. Pass game.local_index (1) so the
    # ghost's ids match the host's.
    game.ghost = PredictedShip(hull=menu.hull, loadout=menu.loadout,
                               local_index=game.local_index)
    # The client's sensor state is CLIENT-
    # AUTHORITATIVE. The client's V/G/T keys mutate the ghost directly
    # (via the command sink below), and reconcile preserves them across
    # apply_snapshot (the host never saw the client's V/G/T, so its
    # snapshot carries the host's own sensor state — restoring it would
    # flicker the client's sensor off every ~100 ms). The host's ship
    # stays authoritative for everything else (pose, weapons, power,
    # shield). The host's ghost keeps the default (host-authoritative).
    game.ghost.client_sensor_authoritative = True
    # The client's estimate of the host's sim clock. The client never runs
    # the sim, so it has no sim clock of its own — the estimator derives
    # one from the snapshots: each is stamped with the host's sim time at
    # send and observed at a known local time, so every arrival is a
    # sample of (host_time - local_time). The estimate is NOT the render
    # clock — the render point is anchored on the buffer's newest
    # ARRIVED snapshot (below). The estimator still runs because its
    # per-arrival jitter sample is what the adaptive delay consumes.
    game.host_time = HostTimeEstimator()
    # The adaptive interpolation delay: a pure function of the arrival
    # pattern — delay = clamp(BASE + k * EMA(jitter), MIN, MAX), moved at
    # most 1/60 s per frame. Fed by the estimator's per-arrival jitter
    # sample, ticked once per frame.
    game.latency = LatencyTracker()
    # RTT-echo state — the client tags each T_INPUT with a monotonic
    # `tag` + records its LOCAL send time; the host immediately replies
    # T_PONG with the same tag (a pure reflection, no sim work). On
    # receipt the client computes RTT = now - sent_time[tag], one-way
    # L = RTT / 2 (EMA-smoothed, clamped), and anchors the prediction
    # ghost's clock at snap_time + L (the ghost must sit at the
    # real-time present, not at the snapshot's time L in the past). A
    # dropped pong just means no sample that round — the EMA holds the
    # last value.
    game._rtt_tag = 0
    game._rtt_sent = {}
    game._rtt_ow = 0.02   # one-way latency estimate (s); 20 ms LAN default
    # The render point: newest ARRIVED snapshot stamp minus the adaptive
    # delay, chased at a bounded per-frame rate so a late packet holds
    # the point on the last window instead of the point outrunning the
    # data and clamp-stuttering.
    game.render_point = RenderPoint(game.latency)
    # Net debug overlay + CSV log (client only). F3 toggles game.debug_net
    # (via handle_events); while on, the loop draws the overlay and
    # appends one CSV line per reconcile to net_debug.csv. `last_logged`
    # tracks the last snap_count written so each reconcile is logged
    # exactly once (reconciles are 10 Hz, far slower than the 60 Hz frame
    # rate).
    game._dbg_last_logged = 0
    game._dbg_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "net_debug.csv")
    game._dbg_log_f = None
    # 11.x diagnostics: the per-frame ghost-clock trace (F3-toggled, like
    # net_debug.csv). Opened when debug_net turns on (below); closed on
    # every exit path (the finally-style block after the loop).
    game._ghost_clock_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ghost_clock.csv")
    game._ghost_clock_f = None
    game._ghost_clock_sink = None
    # Per-receive snap log (F3 telemetry, plan_13 follow-up): one row per
    # snapshot RECEIVED off the wire — (wall, mono, sim_time, seq). The
    # wall deltas are the "is the client receive smooth?" signal (the
    # client's NetWorker thread + the wire, as seen from the client).
    # Written by the client process (its own machine — the 4090). Correlate
    # with the host's sim_snap_debug.csv + net_snap_debug.csv by sim_time
    # to localize a 33 ms gap: sim batching (sim CSV), net-process burst
    # (net CSV), or wire + client receive (this CSV + the cross-machine
    # wall-clock latency).
    game._snap_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "client_snap_debug.csv")
    game._snap_log_f = None
    game._snap_seq = 0
    # Per-frame frame-rate telemetry. The net_debug CSV is written once per
    # reconcile (10 Hz), so it can't see individual frames — the
    # render-point drift could be a low frame rate OR occasional hiccup
    # frames (dt clamped to 50 ms), and the 10 Hz log can't tell them
    # apart. We accumulate the RAW (pre-clamp) frame time every frame and
    # emit the rolling stats on each reconcile line:
    #   fps      = 1 / mean(raw_dt) over the frames since the last line
    #   dt_max   = the longest raw frame in that window (a hiccup)
    #   hic      = count of frames whose raw dt exceeded the 50 ms clamp
    #              (those frames advanced the render point < 1x real time)
    #   n        = number of frames in the window
    # raw_dt is the UNCLAMPED clock.tick() time; dt (clamped) is what the
    # sim/render consume.
    game._dbg_frames = 0
    game._dbg_dt_sum = 0.0
    game._dbg_dt_max = 0.0
    game._dbg_hiccups = 0
    # The socket I/O + JSON run on a NetWorker thread (the "Network Queue"
    # pattern). The worker owns the Connection: it drains the OS socket
    # buffer the instant a packet lands (no bunching) and parses the JSON
    # off the render thread. The main thread talks to it only through
    # worker.send() / worker.poll() / worker.closed — it never touches
    # conn.* once the worker starts (the ownership rule that avoids the
    # "Python Pygame Trap" shared-state corruption). start() flips the
    # socket to non-blocking (the handshake used a 0.05 s recv timeout; it
    # must be cleared before the loop) and launches the thread.
    worker = NetWorker(conn, is_host=False)
    worker.start()

    # The client's command sink. handle_events routes T/V/G/R/F through it
    # (a sink is provided), so it must handle all five — not just the
    # sensor keys. T/V/G mutate the GHOST (the client's local ship)
    # directly: the client's sensor state is client-authoritative
    # (game.ghost.client_sensor_authoritative), so these take effect
    # immediately and survive reconciles (the host never saw them, so its
    # snapshot can't flicker them off). R and F keep the client's prior
    # behavior: R sets _respawn_requested (the loop drains it into a
    # T_RESPAWN — the host respawns player 1) and F/reset calls
    # game.reset() (the client's local presentation reset).
    def _client_command(name, *args):
        if name == "targeting" and not game.game_over:
            game.ghost.ship.targeting_on = not game.ghost.ship.targeting_on
        elif name == "sensor" and not game.game_over:
            game.ghost.ship.sensor_on = not game.ghost.ship.sensor_on
        elif name == "scan" and not game.game_over:
            game.ghost.ship.fire_scan()
        elif name == "respawn":
            game._respawn_requested = True
        elif name == "reset":
            game.reset()

    # GC tuning (PERFORMANCE.md rec C): freeze the long-lived baseline
    # (surfaces, hulls, the Game, ghost, buffer, fonts, textures) so the
    # cyclic GC only scans the small set of per-frame temporaries. Trims
    # the draw max/p99 tail. Reference counting is untouched; objects
    # allocated after this are still collected normally.
    gc.freeze()
    while True:
        # Capture the RAW frame time before the
        # clamp — the clamp hides hiccups from the sim, but the render
        # point only advances by the clamped dt, so a raw frame longer than
        # 50 ms is exactly the frame that lets the point fall behind.
        raw_dt = clock.tick(FPS) / 1000.0
        dt = min(raw_dt, 0.05)
        if game.debug_net:
            game._dbg_frames += 1
            game._dbg_dt_sum += raw_dt
            if raw_dt > game._dbg_dt_max:
                game._dbg_dt_max = raw_dt
            if raw_dt > 0.05:
                game._dbg_hiccups += 1
        # Missile telemetry: rides the same F3 toggle as the net
        # debug overlay (client only — the ghost never runs on the host).
        missile_telemetry.set_enabled(game.debug_net)
        # Bullet telemetry (Phase 0, plan_14): the fire-drift + double-draw
        # measurement, same F3 toggle, client only.
        bullet_telemetry.set_enabled(game.debug_net)
        # Bullet TIME-DOMAIN telemetry (ship-vs-bullet clock gap + muzzle
        # trailing), same F3 toggle, client only.
        bullet_time_telemetry.set_enabled(game.debug_net)
        # Bullet HANDOFF-OFFSET telemetry (ghost vs buffer drawn position at
        # the handoff frame, decomposed into parallel/perpendicular), same
        # F3 toggle, client only.
        bullet_handoff_telemetry.set_enabled(game.debug_net)
        # Bullet-impact (option A) trigger telemetry: the closest-approach
        # geometry to ships vs enemies + the spark decision, same F3 toggle,
        # client only.
        hit_fx_telemetry.set_enabled(game.debug_net)
        # 11.x diagnostics: open/close ghost_clock.csv on the F3 edge and
        # set/clear the ghost's diag_sink. Opening on the rising edge
        # (not every frame) keeps the file fresh per capture; clearing the
        # sink on the falling edge stops the per-frame write when F3 is off.
        if game.debug_net and game._ghost_clock_f is None:
            game._ghost_clock_f, game._ghost_clock_sink = (
                _make_ghost_clock_writer(game._ghost_clock_path))
            PredictedShip.diag_sink = game._ghost_clock_sink
        elif not game.debug_net and game._ghost_clock_f is not None:
            game._ghost_clock_f.close()
            game._ghost_clock_f = None
            PredictedShip.diag_sink = None
        # Per-receive snap log (F3 edge, like the ghost_clock log above).
        if game.debug_net and game._snap_log_f is None:
            game._snap_log_f = open(game._snap_log_path, "w", newline="")
            game._snap_log_f.write("wall,mono,sim_time,seq\n")
        elif not game.debug_net and game._snap_log_f is not None:
            game._snap_log_f.close()
            game._snap_log_f = None
        now = pygame.time.get_ticks() / 1000.0
        # Route T/V/G/R/F through the client command
        # sink (T/V/G -> the ghost; R/F -> the client's existing behavior).
        # Without a sink these keys mutated the STALE host ship (self.ship),
        # so the client's V/G/T never reached the ghost.
        if not game.handle_events(_client_command):
            break
        keys = pygame.key.get_pressed()
        # Sample the local input ONCE per frame (the intent edge) and use
        # that same object to send, to buffer, and (below) to render — so
        # the three can never disagree about what the player did this
        # frame.
        inp = ShipInput.from_keys(keys)
        # Send the local input every frame (pinned #5: the host applies the
        # LATEST received input each tick). Queued to the worker (it
        # flushes to the socket off the render thread). Tag it with a
        # monotonic id + record the LOCAL send time — the host reflects
        # the tag back in a T_PONG, and the RTT is the one-way-latency
        # measurement that anchors the ghost's clock.
        game._rtt_tag += 1
        game._rtt_sent[game._rtt_tag] = time.time()
        if len(game._rtt_sent) > 500:   # bound the map (a lost pong)
            del game._rtt_sent[min(game._rtt_sent)]
        worker.send({"type": T_INPUT, "inp": serialize_input(inp),
                     "tag": game._rtt_tag})
        # Per-player death — R while the local ship is dead set
        # game._respawn_requested in handle_events; send the respawn
        # request to the host (it respawns player 1 — us — leaving the
        # host's ship + world untouched).
        if game._respawn_requested:
            game._respawn_requested = False
            worker.send({"type": T_RESPAWN})
        # Record the input in the ghost's rewind buffer, stamped with the
        # client's estimate of the host's sim clock at this moment (the
        # HostTimeEstimator — the same clock the snapshot stamps are in).
        # When a snapshot arrives, reconcile_rewind replays the inputs the
        # host applied since it, selecting them by this stamp (the host
        # applies the LATEST received input each tick — pinned #5 — so the
        # replay must use the same selection). Before the first snapshot
        # the estimate is None and there is nothing to rewind against, so
        # skip the record.
        est_now = game.host_time.now(now)
        if est_now is not None:
            game.ghost.record_input(est_now, inp)
        # Advance the ghost to the CURRENT frame using the estimator's
        # CURRENT anchor. Called once before the poll (so no-snapshot
        # frames still advance the ghost and the render sees a
        # current-frame pose) and AGAIN after each snapshot's record() in
        # the poll loop — record() moves the anchor to that snapshot, so
        # the re-advance steps the ghost's PHYSICS POSE up to that
        # snapshot's time before the reconcile. Without the in-poll
        # re-advance, a double-arrival frame's second reconcile would
        # overwrite the first reconcile's pose (auth at stamp1) with
        # auth at stamp2 — a full tick of motion -> the 8.67 px snap,
        # even with the clock correct. For a single arrival the in-poll
        # re-advance is a no-op (the pre-poll advance already caught up).
        # Mirrors predicted_view's advance (same dt/inp/enemies/clock);
        # predicted_view's own advance later in the frame is a no-op on
        # the step grid (the branch-2 clock is a pure function of
        # real_time, and steps are boundary-based).
        def _advance_ghost():
            if not game.ghost.seeded:
                return
            _rp = game.render_point.now()
            if _rp is None:
                return
            _pos = game.snap_buf.positions_at(_rp)
            if _pos is None:
                return
            game.ghost.advance(
                dt, inp, enemies=game._ghost_enemy_proxies(_pos),
                real_time=now, host_time=game.host_time,
                rtt_ow=game._rtt_ow,
                rocks=game._ghost_rock_proxies(_pos))
        _advance_ghost()
        # Poll the host: each snapshot feeds the host-time estimator (its
        # wire stamp vs the local arrival time — the estimator's
        # per-arrival jitter sample is the adaptive delay's input) and
        # then the interpolation buffer + prediction ghost (seed on the
        # first, dead-reckoning rewind on the rest). The rewind replays
        # from the snapshot's stamp up to the ghost's own clock
        # (ghost.sim_time — advanced to this frame above). worker.poll()
        # drains the messages the worker thread already parsed off the
        # render thread (the JSON decode no longer happens here).
        for m in worker.poll():
            if m.get("type") == T_SNAP:
                if game._snap_log_f is not None:
                    game._snap_seq += 1
                    game._snap_log_f.write("%.3f,%.3f,%.4f,%d\n" % (
                        time.time(), time.monotonic(), m["sim_time"],
                        game._snap_seq))
                    if game._snap_seq % 60 == 0:   # ~1 s: bound the buffer
                        game._snap_log_f.flush()
                if game.host_time.record(now, m["sim_time"]):
                    game.latency.update(game.host_time.last_jitter)
                # Re-advance the ghost to the FRESH anchor (this snapshot's
                # stamp) BEFORE the reconcile: record() above moved the
                # anchor, so this steps the ghost's PHYSICS POSE up to this
                # snapshot's time. In a double-arrival frame the pre-poll
                # advance only caught up to the FIRST stamp; without this
                # the second reconcile would overwrite the first
                # reconcile's pose (auth at stamp1) with auth at stamp2 —
                # a full tick of motion -> the 8.67 px snap, even with the
                # clock correct. For a single arrival this is a no-op (the
                # pre-poll advance already caught up to this stamp).
                _advance_ghost()
                # Reconcile at the ghost's clock anchored on THIS snapshot's
                # FRESH anchor — not the once-per-frame game.ghost.sim_time
                # (advanced before the poll loop, anchored on the PREVIOUS
                # frame's newest snapshot). record() above just moved the
                # anchor to this snapshot, and its local_time is exactly
                # `now`, so host_time.now(now) == m["sim_time"] and the
                # reconcile span is rtt_ow (positive, sub-tick) for EVERY
                # snapshot in the burst. The old code passed the frozen
                # pre-poll clock: in a double-arrival frame the second
                # snapshot's stamp (16.7 ms ahead) landed AHEAD of that
                # frozen clock -> negative span -> no replay -> the clock
                # jumped forward to the stamp -> the ~8.67 px snap. Using
                # the fresh anchor keeps the span positive on every
                # reconcile, so the clock never jumps.
                game.push_snapshot(m["sim_time"],
                                   deserialize_snapshot(m["snap"]),
                                   now=game.host_time.now(now) + game._rtt_ow,
                                   real_time=now,
                                   rate=game.host_time.rate)
            elif m.get("type") == T_BEAM:
                # A laser beam the host fired — an EVENT (a 0.15 s flash, too short
                # to ride the snapshot's INTERP_DELAY window). Push it
                # into game.remote_beams as [local_start, end, age, ttl,
                # owner] (hull-local muzzle + world endpoint);
                # predicted_view draws it immediately (before the ships),
                # re-anchoring the origin to the firing ship's
                # interpolated pose, and ages it each frame
                # (_step_remote_beams). One-way LAN latency (~10-20 ms)
                # is imperceptible against the 0.15 s flash.
                # SKIP beams from OUR OWN ship (owner == local_index) —
                # the ghost already draws those, and drawing the host's
                # copy too (at the slightly different authoritative pose,
                # ~10-20 ms later) produced a double-draw "criss-cross".
                # Same dedup idea as the ghost-missile id skip.
                if m.get("owner") == game.local_index:
                    continue
                # Store the HULL-LOCAL muzzle (lx, ly) + the firing owner,
                # not the frozen world-space start. predicted_view
                # re-anchors the origin to the firing ship's INTERPOLATED
                # pose each frame (mirrors the host's _draw_world_beam) —
                # the world-space start is where the muzzle WAS at fire
                # time, so drawing it directly left the origin behind in
                # empty space as the ship moved during the 0.15 s flash.
                # Entry shape: [local_start, end, age, ttl, owner].
                game.remote_beams.append([
                    (m.get("lx", 0.0), m.get("ly", 0.0)),
                    pygame.Vector2(m["ex"], m["ey"]),
                    0.0, BEAM_TTL, m.get("owner", 0)])
            elif m.get("type") == T_ECHO:
                # The input the host ACTUALLY applied to OUR ship (player 1) on each
                # tick — one (sim_time, input) pair per entry. Record
                # each in the ghost's echo buffer (in arrival order —
                # in-order TCP keeps it sorted by sim_time).
                # reconcile_rewind prefers these over the client's own
                # sent input for the replay: the host applied the latest
                # input it had RECEIVED (up to one-way latency stale), so
                # on an input change the client's own input diverges from
                # what the host did — the high-speed snapback. A dropped
                # echo degrades gracefully (the replay holds the last
                # echoed input — what the host itself did).
                for (st, d) in m.get("entries", []):
                    game.ghost.record_echo(st, deserialize_input(d))
            elif m.get("type") == T_PONG:
                # The host reflected our T_INPUT tag back. The RTT is now - the
                # LOCAL time we sent that tag; one-way L = RTT / 2.
                # EMA-smooth + clamp (a hiccup can't spike the anchor). A
                # missing tag (the map was bounded) just skips the sample.
                t0 = game._rtt_sent.pop(m.get("tag"), None)
                if t0 is not None:
                    rtt = max(0.0, time.time() - t0)
                    ow = rtt / 2.0
                    game._rtt_ow = 0.1 * ow + 0.9 * game._rtt_ow
                    game._rtt_ow = max(0.002, min(0.25, game._rtt_ow))
        # The worker owns the socket now — read worker.closed (the worker
        # mirrors conn.closed) and never touch conn.* here. The worker
        # also does the drain_send() (off the render thread).
        if worker.closed:
            _notice(screen, font, big_font, clock,
                    ["DISCONNECTED", "the host left"])
            break
        # Render point: the newest ARRIVED snapshot's stamp minus the
        # adaptive delay, chased at a bounded per-frame rate — the buffer
        # is the anchor, not the host-time estimate (a model of the host
        # clock that disagrees with the data under jitter/loss). The
        # tracker's per-frame smoothing step runs here too (one frame
        # moves the delay at most 1/60 s). None until the first snapshot
        # arrives — draw a waiting state meanwhile (predicted_view would
        # do the same, but this keeps the waiting frame free of a ghost
        # step).
        game.latency.tick(dt)
        rp = game.render_point.advance(dt, game.snap_buf.newest_time())
        if rp is None:
            _draw_waiting_client(screen, font, big_font)
        else:
            # Render: local ship from the prediction ghost (advanced to
            # this frame before the poll loop; predicted_view's advance
            # is a no-op on the step grid), remote entities from the
            # interpolation buffer at the render point. host_time keeps
            # self.sim_time = the host's clock as carried by the wire,
            # for the ghost's reconcile bookkeeping and diagnostics.
            game.predicted_view(dt, keys, host_time=game.host_time.now(now),
                              real_time=now)
        # Net debug overlay + CSV log (F3-toggled, client only).
        # The overlay shows the live feel-layer state; the log appends one
        # line per reconcile (10 Hz) so the snap-size distribution + its
        # correlation with input changes can be analyzed offline.
        if game.debug_net:
            _draw_net_debug(screen, font, game, inp)
            if game.snap_count > game._dbg_last_logged:
                _log_net_debug(game, inp)
                game._dbg_last_logged = game.snap_count
        pygame.display.flip()
    if game._dbg_log_f is not None:
        game._dbg_log_f.close()
        game._dbg_log_f = None
    # 11.x diagnostics: close the ghost-clock CSV + clear the sink on every
    # exit path (ESC/QUIT, disconnect).
    if getattr(game, "_ghost_clock_f", None) is not None:
        game._ghost_clock_f.close()
        game._ghost_clock_f = None
    PredictedShip.diag_sink = None
    # Close the per-receive snap log on every exit path (like the
    # ghost_clock log above).
    if getattr(game, "_snap_log_f", None) is not None:
        game._snap_log_f.close()
        game._snap_log_f = None
    # Flush + close the missile telemetry CSV on every exit path.
    missile_telemetry.close()
    # Flush + close the bullet telemetry CSV (Phase 0, plan_14).
    bullet_telemetry.close()
    # Flush + close the bullet time-domain telemetry CSV.
    bullet_time_telemetry.close()
    # Flush + close the bullet handoff-offset telemetry CSV.
    bullet_handoff_telemetry.close()
    # Flush + close the bullet-impact (option A) trigger telemetry CSV.
    hit_fx_telemetry.close()
    # Stop the worker (join its thread) BEFORE closing the
    # connection — the worker owns the socket, and closing a socket a worker
    # still owns crashes that worker with EBADF (the 8.1 gotcha). stop() is
    # idempotent, so this is safe on every exit path (ESC/QUIT, disconnect).
    worker.stop()
    conn.close()


def _draw_waiting_client(screen, font, big_font):
    """The client's 'waiting for snapshots' state (before the buffer holds a
    window to interpolate). Drawn over the cleared screen while predicted_view
    returns None."""
    screen.fill(BG)
    title = big_font.render("WAITING FOR THE HOST", True, (200, 210, 225))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2,
                        HEIGHT // 2 - 40))
    hint = font.render("syncing to the authoritative sim...", True,
                       (110, 120, 140))
    screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2,
                       HEIGHT // 2 + 20))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Belt Fighter")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)

    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    seed = None
    if '--seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('--seed') + 1])

    sfx = SoundBank()
    sfx.init()

    while True:
        menu = Menu(font, big_font)
        while not menu.done:
            clock.tick(FPS)
            if not menu.handle_events():
                pygame.quit(); return
            menu.draw(screen)
            pygame.display.flip()

        if menu.mode == 'host':
            run_host(screen, font, big_font, clock, sfx, menu, seed,
                     light_tex, fog_surf, light_surf)
            continue          # back to the menu (pinned #8: no reconnect)

        if menu.mode == 'join':
            run_client(screen, font, big_font, clock, sfx, menu, seed,
                       light_tex, fog_surf, light_surf)
            continue          # back to the menu (pinned #8: no reconnect)

        # --- single player (unified onto the SimThread) ---
        # The classic solo run now uses the SAME architecture as the host:
        # the sim runs on a SimThread (a real-time 60 Hz clock — the 8.5
        # hiccup policy, its own monotonic accumulator, NOT tied to the
        # frame rate) and the render thread is a LOCAL CLIENT of that sim:
        # it publishes the latest local input + T/V/G/R/F commands, reads
        # the published RenderModel, and draws it with the render thread's
        # OWN interpolation alpha (_HostRenderClock — the same class the
        # host loop uses; "Host" is a misnomer now, it is just "the render
        # thread's clock for a sim it does not own"). worker=None: SP has
        # no NetWorker, so the sim thread publishes the model only.
        #
        # The sim itself is UNCHANGED (Game._step) — only its thread moved,
        # exactly what M3 proved for the host: test_determinism (which
        # steps the Game directly, not through this loop) stays
        # bit-identical. The old in-loop `game.update(raw_dt, keys)` +
        # `game.draw(dt)` path is gone for SP; the render thread never
        # mutates sim state anymore (it publishes input, a fresh ShipInput
        # per frame, and commands, a queue — the sim thread applies them).
        game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                    hull=menu.hull, loadout=menu.loadout,
                    test_mode=('--test' in sys.argv), seed=seed, sound=sfx)
        sim_thread = SimThread(game)
        sim_thread.start()
        render_clock = _HostRenderClock()
        # GC tuning (PERFORMANCE.md rec C): freeze the long-lived baseline
        # (surfaces, hulls, the Game, fonts, textures) so the cyclic GC
        # only scans the small set of per-frame temporaries. Trims the
        # draw max/p99 tail. Reference counting is untouched; objects
        # allocated after this are still collected normally.
        gc.freeze()
        running = True
        try:
            while running:
                raw_dt = clock.tick(FPS) / 1000.0
                dt = min(raw_dt, 0.05)
                # T/V/G/R/F are ROUTED to the sim thread (command_sink)
                # instead of mutating live state — the sim thread applies
                # them at the top of its next iteration (<= 1-2 ms).
                # QUIT/ESC (exit) + F3 (the render diagnostic) stay
                # handled here.
                if not game.handle_events(sim_thread.command):
                    running = False
                keys = pygame.key.get_pressed()
                # Publish the latest local input (reference swap; the sim
                # applies the LATEST published input at each step — the
                # same rule the host applies to the client's input).
                sim_thread.publish_input(ShipInput.from_keys(keys))
                # Read the PUBLISHED model (an atomic reference read — the
                # render thread never reads live sim state, which would
                # race the sim thread's _step) and compute the render
                # thread's own interpolation alpha (see _HostRenderClock).
                model = sim_thread.latest_model
                alpha = render_clock.advance(dt, model)
                # Draw the PUBLISHED model with the render thread's alpha
                # (the model still carries the sim's step_alpha for the
                # M1/M2b parity tests; draw() uses the render thread's).
                game.draw(dt, {**model, "step_alpha": alpha} if model else None)
                pygame.display.flip()
        finally:
            # Stop the sim thread (join it) before the menu loop resumes —
            # it owns the Game, and the next mode's Game must not be built
            # while a live sim thread is still stepping the old one.
            # stop() is idempotent.
            sim_thread.stop()
        break

    pygame.quit()


if __name__ == "__main__":
    main()