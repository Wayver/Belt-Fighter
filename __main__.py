"""Entry point: run with  python -m ship5

Three modes (chosen on the menu's mode screen, Session 6.4):
  single — the classic solo run. Session 9.x M5: unified onto the
           SimThread — the same architecture as the host (the sim on its
           own real-time 60 Hz thread, the render thread a local client
           of the published RenderModel; no NetWorker).
  host   — Session 6.5: wait for ONE client, run the authoritative 2-ship
           sim, apply the client's input, broadcast a snapshot every
           SNAPSHOT_INTERVAL ticks. ESC/QUIT or a client disconnect
           returns to the menu (pinned decision #8: no reconnect in v1).
  join   — Session 6.6: connect to a host, run the client sim (predict the
           local ship, interpolate the remote entities), send input every
           frame, push each snapshot, render via predicted_view.
"""
import os
import socket
import sys
import time

import pygame

from .config import WIDTH, HEIGHT, FPS, NET_PORT, BG, SNAPSHOT_INTERVAL
from .fog import make_light_texture
from .game import Game, STEP
from .menu import Menu
from .net import (Host, connect, do_handshake_host, do_handshake_client,
                  serialize_hull, serialize_loadout,
                  deserialize_hull, deserialize_loadout,
                  serialize_snapshot, deserialize_snapshot,
                  serialize_input, deserialize_input,
                  NetWorker,
                  T_INPUT, T_SNAP, T_RESPAWN, T_BEAM, T_ECHO)
from .netcode import (PredictedShip, HostTimeEstimator, LatencyTracker,
                      RenderPoint, BEAM_TTL)
from .ship import Ship
from .sim_thread import SimThread
from .sound import SoundBank
from .intent import ShipInput
from . import missile_telemetry


class _HostRenderClock:
    """Session 9.x M3: the host render thread's own interpolation clock.

    With the sim on the SimThread, the render thread no longer owns the
    sim's accumulator — so it computes its OWN interpolation alpha
    instead of reading the model's `step_alpha` (the sim's acc/STEP at
    publish time). The model still CARRIES step_alpha (the M1/M2b parity
    tests read it), and it is also what this clock re-anchors on.

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

    The alpha is then (t - T) / STEP, clamped to [0, 1] (the 70460de
    rule: on a hiccup render the LATEST simulated state, never
    extrapolate past the current pose). In steady state t - T equals the
    sim's acc at publish (plus ~1 ms of publish latency), so the alpha
    matches the old sim-acc/STEP behavior exactly — the rendered pose is
    sim_now - STEP, one step behind the sim's current time, as before.
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
    """Host a 2P game (Session 6.5).

    Phases: (1) draw the waiting screen while a timed accept() waits for
    ONE client; (2) the blocking join/welcome handshake; (3) build the
    2-ship sim — player 0 is the host's ship, player 1 is built from the
    client's (validated) hull/loadout; (4) the authoritative game loop:
    poll the client's input, step the sim, broadcast a snapshot every
    SNAPSHOT_INTERVAL ticks, draw. Returns to the caller (the menu) on
    ESC/QUIT or a client disconnect (pinned decision #8).
    """
    ip = _lan_ip()
    try:
        host = Host(NET_PORT)
    except OSError:
        _notice(screen, font, big_font, clock,
                ["COULD NOT LISTEN", "port %d is unavailable" % NET_PORT])
        return
    port = host.sock.getsockname()[1]
    conn = None
    game = None   # defined here so the finally can close its debug log even
                  # if the handshake fails before phase 3 builds the Game.
    worker = None  # Session 8.3: the host's NetWorker (created in phase 3);
                   # defined here so the finally can stop() it even if the
                   # handshake fails before phase 3 runs.
    sim_thread = None  # Session 9.x M3: the host's SimThread (created in
                       # phase 4); defined here so the finally can stop() it
                       # even if the handshake fails before phase 4 runs.
    try:
        # --- 1. wait for a client (waiting screen + timed accept) ---
        while conn is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            conn, _addr = host.accept_one(timeout=0.25)
            if conn is None:
                _draw_waiting(screen, font, big_font, ip, port)
                pygame.display.flip()
            clock.tick(FPS)

        # --- 2. handshake: the client's join -> our welcome ---
        ch, cl = do_handshake_host(conn, serialize_hull(menu.hull),
                                   serialize_loadout(menu.loadout))
        if ch is None:
            conn.close()
            return                      # vanished before joining: back to menu

        # --- 3. build the 2-ship sim (player 0 = us, player 1 = client) ---
        chull = deserialize_hull(ch)
        clout = deserialize_loadout(chull, cl)
        game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                    hull=menu.hull, loadout=menu.loadout,
                    seed=seed, sound=sfx, players=2)
        game.set_player_ship(1, Ship(hull=chull, loadout=clout))
        # Session 8.3: the socket I/O + the 10 Hz snapshot send move to a
        # NetWorker thread (the "Network Queue" pattern). The worker owns the
        # Connection: it drains the client's input off the render thread and
        # runs the REAL-TIME snapshot timer (checked every ~1 ms, NOT once
        # per frame — the 7.10c in-loop timer was frame-quantized and fired
        # at 112-150 ms at 24 FPS). The main thread publishes a fresh
        # snapshot each frame via set_latest_snapshot; the worker's timer
        # grabs the latest and sends it every 100 ms of REAL time. start()
        # flips the socket to non-blocking + launches the thread.
        worker = NetWorker(conn, is_host=True)
        worker.start()

        # --- 4. the authoritative game loop ---
        # Session 9.x M3: the sim moves OFF the render thread onto the
        # SimThread — a daemon thread that steps the Game at a real-time
        # 60 Hz clock (a monotonic accumulator, NOT tied to the frame
        # rate) and publishes the plain-data RenderModel via an atomic
        # reference swap (the NetWorker _latest_snapshot pattern). The
        # render thread becomes a LOCAL CLIENT of that sim: it publishes
        # the latest local input + T/V/G/R/F commands, and reads
        # sim_thread.latest_model to draw. The snapshot hand-off to the
        # worker (set_latest_snapshot) moves to the sim thread too — the
        # worker's 10 Hz real-time timer (unchanged) now always has a
        # fresh snapshot to send, and the snapshot is still stamped with
        # game.sim_time (the host's sim clock, 1x real time regardless of
        # frame rate), so the client's interpolation window is unchanged.
        # `game` is now OWNED by the sim thread: the render thread never
        # calls game.update() or mutates sim state — it publishes input
        # (a fresh ShipInput per frame), commands (a queue), and reads
        # latest_model (a reference swap).
        sim_thread = SimThread(game, worker)
        sim_thread.start()
        # Session 9.x M3: the render thread's own interpolation clock
        # (the sim's acc/STEP is no longer the render thread's — see the
        # class docstring).
        render_clock = _HostRenderClock()
        # Session 7.10b: host-side frame-rate telemetry (F3-toggled). The
        # send moved to the worker (8.3), so the telemetry runs on its OWN
        # 10 Hz real-time timer in the main loop (it measures the host's
        # FRAME rate — a main-thread concern — and the wall-clock `t` is
        # the join key to the client's CSV, not the send event).
        # Session 9.x M3: this is now the headline gate — with the sim
        # cost OFF the render thread, the host's render FPS should rise
        # (the 8.5 caveat's outstanding item). last_telem_time anchors it
        # (0.0 -> the first line is immediate).
        last_telem_time = 0.0
        TELEMETRY_PERIOD = SNAPSHOT_INTERVAL * STEP   # 0.033 s (30 Hz, 10.8)
        # Session 7.10b: host-side frame-rate telemetry (F3-toggled, mirrors
        # the client's 7.10a columns). The client's CSV can't tell a WIRE
        # stall (host sent smoothly, packets queued + released in a burst)
        # from a HOST frame stall (the host's sim is tied to its frame loop,
        # so a host hiccup stalls the sim AND the snapshot sending). We log
        # the host's raw frame time on each snapshot send (30 Hz, 10.8) so the
        # next re-test can be correlated: if the client's snapshot gap
        # (newest jumping 1.3-1.6 s) lines up with a host hiccup (dt_max
        # > 50 ms / low fps) at the same wall-clock time, it's a HOST stall
        # (-> decouple the sim from the frame loop); if the host was clean
        # (60 fps, no hiccups) at that moment, it's a WIRE stall (-> the
        # catch-up is the right fix, nothing to do on the host).
        game._dbg_frames = 0
        game._dbg_dt_sum = 0.0
        game._dbg_dt_max = 0.0
        game._dbg_hiccups = 0
        game._dbg_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "host_debug.csv")
        game._dbg_log_f = None
        # Session 8.5 Step 1: per-frame sim-clock diagnostic (F3-toggled,
        # additive). Logs raw_dt (real frame time — what the sim is fed),
        # draw_dt (the clamped dt that drives draw()), sim_before/sim_after
        # (the sim clock), advance (sim time gained this frame), and
        # acc_after (accumulator remainder). Step 1 (pre-fix) used it to
        # prove the dt clamp was losing time on hiccup frames; post-Step-2
        # it verifies the fix: sim-clock rate ~1.0, advance tracking
        # raw_dt (modulo STEP quantization + the backlog cap).
        game._dbg_sim_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "host_sim_debug.csv")
        game._dbg_sim_log_f = None
        while True:
            # Session 7.10b: capture the RAW frame time before the clamp
            # (the clamp hides hiccups from the sim, but a raw frame > 50 ms
            # is exactly the frame that stalls the sim + snapshot sending).
            raw_dt = clock.tick(FPS) / 1000.0
            dt = min(raw_dt, 0.05)
            if game.debug_net:
                game._dbg_frames += 1
                game._dbg_dt_sum += raw_dt
                if raw_dt > game._dbg_dt_max:
                    game._dbg_dt_max = raw_dt
                if raw_dt > 0.05:
                    game._dbg_hiccups += 1
            # Session 9.x M3: the T/V/G/R/F keys are ROUTED to the sim thread
            # (command_sink) instead of mutating live state — the sim
            # thread applies them at the top of its next iteration.
            # QUIT/ESC (exit) + F3 (the render diagnostic) stay handled
            # here. handle_events still reads game.game_over/test_mode
            # for the same gating (a bool read — atomic, no mutation).
            if not game.handle_events(sim_thread.command):
                return
            keys = pygame.key.get_pressed()
            # Poll the client: apply its latest input (pinned #5: the host
            # applies the LATEST received input each tick). Session 8.3:
            # worker.poll() drains the input the worker thread already
            # parsed off the render thread (the JSON decode no longer
            # happens here). Session 9.x M3: the hand-off to the sim
            # thread is a direct reference swap (game.remote_input = inp)
            # — atomic under the GIL, and _step only READS the input
            # (the world-cap replace() builds a new object), so the swap
            # is race-free.
            for m in worker.poll():
                if m.get("type") == T_INPUT:
                    game.set_remote_input(deserialize_input(m["inp"]))
                elif m.get("type") == T_RESPAWN:
                    # 10.1: the client's dead player asks to respawn.
                    # The client is player 1; the sim thread applies it
                    # (a no-op unless that ship is actually dead).
                    sim_thread.command("respawn", 1)
            # Session 8.3: the worker owns the socket now — read
            # worker.closed (the worker mirrors conn.closed) and never
            # touch conn.* here. The worker also does the drain_send()
            # (off the render thread), so the in-loop conn.drain_send() is
            # gone.
            if worker.closed:
                _notice(screen, font, big_font, clock,
                        ["DISCONNECTED", "the other player left"])
                return
            # Session 9.x M3: the sim no longer runs here. The SimThread steps it
            # at a real-time 60 Hz clock (its own monotonic accumulator —
            # the 8.5 hiccup policy, MAX_STEPS_PER_FRAME + ACC_BACKLOG_CAP,
            # now lives in sim_thread.py) and publishes the RenderModel +
            # the wire snapshot. The render thread's job each frame:
            # publish the latest local input, read the published model,
            # draw. `dt` (clamped) drives draw() + the render clock —
            # presentation only.
            sim_thread.publish_input(ShipInput.from_keys(keys))
            # Session 9.x M3: read the PUBLISHED model (an atomic reference read —
            # the render thread never reads live sim state, which would
            # race the sim thread's _step) and compute the render
            # thread's own interpolation alpha (see _HostRenderClock).
            model = sim_thread.latest_model
            alpha = render_clock.advance(dt, model)
            # Session 9.x M3: the per-frame sim-clock diagnostic (F3-
            # toggled, additive) now reads the published model + the
            # render thread's alpha instead of the live game.sim_time /
            # game.acc. One line per frame: t (wall clock), raw_dt (real
            # frame time), draw_dt (the clamped dt that drives draw() —
            # presentation only), sim_time (the published model's sim
            # clock), alpha (the render thread's interpolation alpha).
            if game.debug_net:
                f = game._dbg_sim_log_f
                if f is None:
                    f = open(game._dbg_sim_log_path, "w", newline="")
                    f.write("t,raw_dt,draw_dt,sim_time,alpha\n")
                    game._dbg_sim_log_f = f
                f.write("%.3f,%.4f,%.4f,%.4f,%.4f\n" % (
                    time.time(), raw_dt, dt,
                    model["sim_time"] if model else 0.0, alpha))
            # Session 7.10b: host-side frame-rate telemetry (F3-toggled), on
            # its OWN 10 Hz real-time timer (the send moved to the worker, so
            # it no longer rides the send). One line per 100 ms, mirroring
            # the client's 7.10a columns: t (wall clock), sim_time (the
            # host's sim clock), fps (1/mean raw frame time since the last
            # line), dt_max (longest raw frame), hic (frames past the 50 ms
            # clamp), n (frames in the window). The wall-clock `t` is the
            # join key: the client's CSV has the same `t` (both machines'
            # wall clocks are roughly synced on the same LAN), so a client
            # snapshot gap at t=X can be checked against the host's
            # fps/dt_max/hic at t=X to tell a wire stall from a host stall.
            now_t = pygame.time.get_ticks() / 1000.0
            if now_t - last_telem_time >= TELEMETRY_PERIOD:
                last_telem_time = now_t
                if game.debug_net:
                    f = game._dbg_log_f
                    if f is None:
                        f = open(game._dbg_log_path, "w", newline="")
                        f.write("t,sim_time,fps,dt_max,hic,n\n")
                        game._dbg_log_f = f
                    n = game._dbg_frames
                    fps = (1.0 / (game._dbg_dt_sum / n)) if n > 0 else 0.0
                    dt_max = game._dbg_dt_max
                    hic = game._dbg_hiccups
                    game._dbg_frames = 0
                    game._dbg_dt_sum = 0.0
                    game._dbg_dt_max = 0.0
                    game._dbg_hiccups = 0
                    f.write("%.3f,%.4f,%.1f,%.4f,%d,%d\n" % (
                        time.time(),
                        model["sim_time"] if model else 0.0,
                        fps, dt_max, hic, n))
                    f.flush()
            # Session 9.x M3: draw the PUBLISHED model (the M3 seam:
            # draw(dt, model) — the model is built by the sim thread, not
            # here), with the render thread's OWN interpolation alpha
            # (the sim's acc/STEP is no longer the render thread's — see
            # the _HostRenderClock docstring). The model still carries
            # the sim's step_alpha (the M1/M2b parity tests read it);
            # draw() uses the render thread's instead.
            game.draw(dt, {**model, "step_alpha": alpha} if model else None)
            pygame.display.flip()
    finally:
        if getattr(game, "_dbg_log_f", None) is not None:
            game._dbg_log_f.close()
            game._dbg_log_f = None
        # Session 8.5 Step 1: close the per-frame sim-clock diagnostic log.
        if getattr(game, "_dbg_sim_log_f", None) is not None:
            game._dbg_sim_log_f.close()
            game._dbg_sim_log_f = None
        # Session 9.x M3: stop the sim thread (join it) BEFORE stopping the
        # worker — the sim thread feeds the worker (set_latest_snapshot),
        # and the worker must not be sending a snapshot the sim thread is
        # about to mutate (a reset). stop() is idempotent.
        if sim_thread is not None:
            sim_thread.stop()
        # Session 8.3: stop the worker (join its thread) BEFORE closing the
        # connection — the worker owns the socket, and closing a socket a
        # worker still owns crashes that worker with EBADF (the 8.1 gotcha).
        # stop() is idempotent + a no-op when the worker was never started
        # (handshake failed before phase 3), so this is safe on every exit
        # path (ESC/QUIT, disconnect, port-in-use).
        if worker is not None:
            worker.stop()
        if conn is not None:
            conn.close()
        host.close()


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
    """Session 7.8: the net debug overlay (client only, F3-toggled). Small
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
    """Session 7.8: append one CSV line per reconcile (10 Hz) to
    net_debug.csv (client only, F3-toggled). The overlay shows the LATEST
    snap size; the log captures the DISTRIBUTION over the session + the
    input that was active, so the snap-size vs input-change correlation
    (in-flight input vs clock error) can be analyzed offline."""
    f = game._dbg_log_f
    if f is None:
        f = open(game._dbg_log_path, "w", newline="")
        f.write("t,snap_px,input,delay,jitter_ema,buf_depth,newest_stamp,"
                "render_t,host_time_est,fps,dt_max,hic,n,"
                "replay_ticks,replay_span\n")
        game._dbg_log_f = f
    rp = game.render_point.now()
    newest = game.snap_buf.newest_time()
    est = game.host_time.now(pygame.time.get_ticks() / 1000.0)
    jit = game.latency.jitter_ema   # None until the first sample
    # Session 7.10 (Chunk 1): frame-rate stats over the frames since the
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
    f.write("%.3f,%.3f,%s,%.4f,%s,%d,%s,%s,%s,%.1f,%.4f,%d,%d,%d,%.4f\n" % (
        time.time(), game.last_snap_px, _input_str(inp),
        game.latency.delay,
        ("%.4f" % jit) if jit is not None else "",
        len(game.snap_buf),
        ("%.4f" % newest) if newest is not None else "",
        ("%.4f" % rp) if rp is not None else "",
        ("%.4f" % est) if est is not None else "",
        fps, dt_max, hic, n,
        game.last_replay_ticks, game.last_replay_span))
    f.flush()


def run_client(screen, font, big_font, clock, sfx, menu, seed,
               light_tex, fog_surf, light_surf):
    """Join a 2P game (Session 6.6).

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
    # 10.3b bug fix: `local_index` MUST be passed. The default is 0 (the
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
    # 10.6 V/G/T (option A): the client's sensor state is CLIENT-
    # AUTHORITATIVE. The client's V/G/T keys mutate the ghost directly
    # (via the command sink below), and reconcile preserves them across
    # apply_snapshot (the host never saw the client's V/G/T, so its
    # snapshot carries the host's own sensor state — restoring it would
    # flicker the client's sensor off every ~100 ms). The host's ship
    # stays authoritative for everything else (pose, weapons, power,
    # shield). The host's ghost keeps the default (host-authoritative).
    game.ghost.client_sensor_authoritative = True
    # The client's estimate of the host's sim clock (Session 7.1). The
    # client never runs the sim, so it has no sim clock of its own — the
    # estimator derives one from the snapshots: each is stamped with the
    # host's sim time at send and observed at a known local time, so every
    # arrival is a sample of (host_time - local_time). This replaced the
    # 6.6 wall-clock `sim_time += dt` (two independent clocks drifting).
    # Session 7.5b: the estimate is no longer the RENDER clock — the
    # render point is anchored on the buffer's newest ARRIVED snapshot
    # (below). The estimator still runs because its per-arrival jitter
    # sample is what the adaptive delay consumes.
    game.host_time = HostTimeEstimator()
    # The adaptive interpolation delay (Session 7.5a): a pure function of
    # the arrival pattern — delay = clamp(BASE + k * EMA(jitter), MIN,
    # MAX), moved at most 1/60 s per frame. Fed by the estimator's
    # per-arrival jitter sample, ticked once per frame.
    game.latency = LatencyTracker()
    # The render point (Session 7.5b): newest ARRIVED snapshot stamp
    # minus the adaptive delay, chased at a bounded per-frame rate so a
    # late packet holds the point on the last window instead of the
    # point outrunning the data and clamp-stuttering. This replaces the
    # 7.1 host-time-estimate render clock for the render point.
    game.render_point = RenderPoint(game.latency)
    # Session 7.8: net debug overlay + CSV log (client only). F3 toggles
    # game.debug_net (via handle_events); while on, the loop draws the
    # overlay and appends one CSV line per reconcile to net_debug.csv.
    # `last_logged` tracks the last snap_count written so each reconcile is
    # logged exactly once (reconciles are 10 Hz, far slower than the 60 Hz
    # frame rate).
    game._dbg_last_logged = 0
    game._dbg_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "net_debug.csv")
    game._dbg_log_f = None
    # Session 7.10 (Chunk 1): per-frame frame-rate telemetry. The 7.8 CSV is
    # written once per reconcile (10 Hz), so it can't see individual frames —
    # the render-point drift could be a low frame rate OR occasional hiccup
    # frames (dt clamped to 50 ms), and the 10 Hz log can't tell them apart.
    # We accumulate the RAW (pre-clamp) frame time every frame and emit the
    # rolling stats on each reconcile line:
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
    # Session 8.2: the socket I/O + JSON move to a NetWorker thread (the
    # "Network Queue" pattern). The worker owns the Connection: it drains the
    # OS socket buffer the instant a packet lands (no bunching) and parses
    # the JSON off the render thread. The main thread talks to it only
    # through worker.send() / worker.poll() / worker.closed — it never
    # touches conn.* once the worker starts (the ownership rule that avoids
    # the "Python Pygame Trap" shared-state corruption). start() flips the
    # socket to non-blocking (the handshake used a 0.05 s recv timeout; it
    # must be cleared before the loop) and launches the thread.
    worker = NetWorker(conn, is_host=False)
    worker.start()

    # 10.6 V/G/T (option A): the client's command sink. handle_events
    # routes T/V/G/R/F through it (a sink is provided), so it must handle
    # all five — not just the sensor keys. T/V/G mutate the GHOST (the
    # client's local ship) directly: the client's sensor state is
    # client-authoritative (game.ghost.client_sensor_authoritative), so
    # these take effect immediately and survive reconciles (the host never
    # saw them, so its snapshot can't flicker them off). R and F keep the
    # client's pre-10.6 behavior: R sets _respawn_requested (the loop
    # drains it into a T_RESPAWN — the host respawns player 1) and F/reset
    # calls game.reset() (the client's local presentation reset).
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

    while True:
        # Session 7.10 (Chunk 1): capture the RAW frame time before the
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
        # 10.3b missile telemetry: rides the same F3 toggle as the net
        # debug overlay (client only — the ghost never runs on the host).
        missile_telemetry.set_enabled(game.debug_net)
        now = pygame.time.get_ticks() / 1000.0
        # 10.6 V/G/T (option A): route T/V/G/R/F through the client command
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
        # LATEST received input each tick). Session 8.2: queued to the
        # worker (it flushes to the socket off the render thread).
        worker.send({"type": T_INPUT, "inp": serialize_input(inp)})
        # 10.1: per-player death — R while the local ship is dead set
        # game._respawn_requested in handle_events; send the respawn
        # request to the host (it respawns player 1 — us — leaving the
        # host's ship + world untouched).
        if game._respawn_requested:
            game._respawn_requested = False
            worker.send({"type": T_RESPAWN})
        # Session 7.6: record the input in the ghost's rewind buffer,
        # stamped with the client's estimate of the host's sim clock at
        # this moment (the 7.1 HostTimeEstimator — the same clock the
        # snapshot stamps are in). When a snapshot arrives,
        # reconcile_rewind replays the inputs the host applied since it,
        # selecting them by this stamp (the host applies the LATEST
        # received input each tick — pinned #5 — so the replay must use
        # the same selection). Before the first snapshot the estimate is
        # None and there is nothing to rewind against, so skip the record.
        est_now = game.host_time.now(now)
        if est_now is not None:
            game.ghost.record_input(est_now, inp)
        # Poll the host: each snapshot feeds the host-time estimator (its
        # wire stamp vs the local arrival time — the estimator's
        # per-arrival jitter sample is the adaptive delay's input,
        # Session 7.5a) and then the interpolation buffer + prediction
        # ghost (seed on the first, dead-reckoning rewind on the rest —
        # Session 7.6). `now` is the client's current estimate of the
        # host's sim clock: the rewind replays from the snapshot's stamp
        # up to it. Session 8.2: worker.poll() drains the messages the
        # worker thread already parsed off the render thread (the JSON
        # decode no longer happens here).
        for m in worker.poll():
            if m.get("type") == T_SNAP:
                if game.host_time.record(now, m["sim_time"]):
                    game.latency.update(game.host_time.last_jitter)
                # 10.11: pass the ghost's CURRENT physics sim time as `now`
                # (the replay target) instead of the host-time estimate. The
                # estimate is anchored on the newest snapshot and equals
                # `snap_time` exactly, which made the replay 0 ticks and left
                # the ghost's real-time advance un-replayed (the sawtooth).
                # The ghost's sim time is AHEAD of the snapshot's stamp by
                # the amount of time the ghost has advanced since that
                # snapshot was taken, so the replay rebuilds the prediction
                # forward and the snap is ~0.
                game.push_snapshot(m["sim_time"],
                                   deserialize_snapshot(m["snap"]),
                                   now=game.ghost.sim_time)
            elif m.get("type") == T_BEAM:
                # 10.4: a laser beam the host fired — an EVENT (a 0.15 s
                # flash, too short to ride the snapshot's INTERP_DELAY
                # window). Push it into game.remote_beams as
                # [local_start, end, age, ttl, owner] (hull-local muzzle
                # + world endpoint); predicted_view draws it immediately
                # (before the ships), re-anchoring the origin to the
                # firing ship's interpolated pose, and ages it each frame
                # (_step_remote_beams). One-way LAN latency (~10-20 ms) is
                # imperceptible against the 0.15 s flash.
                # 10.4 fix: SKIP beams from OUR OWN ship (owner ==
                # local_index) — the ghost already draws those (10.3a),
                # and drawing the host's copy too (at the slightly
                # different authoritative pose, ~10-20 ms later) produced
                # a double-draw "criss-cross". Same dedup idea as the
                # 10.3b ghost-missile id skip.
                if m.get("owner") == game.local_index:
                    continue
                # 10.4b (detach fix): store the HULL-LOCAL muzzle (lx, ly)
                # + the firing owner, not the frozen world-space start.
                # predicted_view re-anchors the origin to the firing ship's
                # INTERPOLATED pose each frame (mirrors the host's
                # _draw_world_beam / the original 1p fix) — the world-space
                # start is where the muzzle WAS at fire time, so drawing it
                # directly left the origin behind in empty space as the ship
                # moved during the 0.15 s flash. Entry shape:
                # [local_start, end, age, ttl, owner].
                game.remote_beams.append([
                    (m.get("lx", 0.0), m.get("ly", 0.0)),
                    pygame.Vector2(m["ex"], m["ey"]),
                    0.0, BEAM_TTL, m.get("owner", 0)])
            elif m.get("type") == T_ECHO:
                # 10.10: the input the host ACTUALLY applied to OUR ship
                # (player 1) on each tick — one (sim_time, input) pair per
                # entry. Record each in the ghost's echo buffer (in
                # arrival order — in-order TCP keeps it sorted by
                # sim_time). reconcile_rewind prefers these over the
                # client's own sent input for the replay: the host applied
                # the latest input it had RECEIVED (up to one-way latency
                # stale), so on an input change the client's own input
                # diverges from what the host did — the high-speed
                # snapback. A dropped echo degrades gracefully (the replay
                # holds the last echoed input — what the host itself did).
                for (st, d) in m.get("entries", []):
                    game.ghost.record_echo(st, deserialize_input(d))
        # Session 8.2: the worker owns the socket now — read worker.closed (the
        # worker mirrors conn.closed) and never touch conn.* here. The
        # worker also does the drain_send() (off the render thread), so the
        # in-loop conn.drain_send() is gone.
        if worker.closed:
            _notice(screen, font, big_font, clock,
                    ["DISCONNECTED", "the host left"])
            break
        # Render point (Session 7.5b): the newest ARRIVED snapshot's
        # stamp minus the adaptive delay, chased at a bounded per-frame
        # rate — the buffer is the anchor, not the 7.1 host-time
        # estimate (a model of the host clock that disagrees with the
        # data under jitter/loss). The tracker's per-frame smoothing
        # step runs here too (one frame moves the delay at most 1/60 s).
        # None until the first snapshot arrives — draw a waiting state
        # meanwhile (predicted_view would do the same, but this keeps
        # the waiting frame free of a ghost step).
        game.latency.tick(dt)
        rp = game.render_point.advance(dt, game.snap_buf.newest_time())
        if rp is None:
            _draw_waiting_client(screen, font, big_font)
        else:
            # Render: local ship from the prediction ghost (advanced at
            # the sim's fixed rate inside predicted_view, Session 7.1),
            # remote entities from the interpolation buffer at the
            # render point. host_time keeps self.sim_time = the host's
            # clock as carried by the wire (7.1), for the ghost's
            # reconcile bookkeeping and diagnostics.
            game.predicted_view(dt, keys, host_time=game.host_time.now(now))
        # Session 7.8: net debug overlay + CSV log (F3-toggled, client only).
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
    # 10.3b: flush + close the missile telemetry CSV on every exit path.
    missile_telemetry.close()
    # Session 8.2: stop the worker (join its thread) BEFORE closing the
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

        # --- single player (Session 9.x M5: unified onto the SimThread) ---
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