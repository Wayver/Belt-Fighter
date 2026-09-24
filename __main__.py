"""Entry point: run with  python -m ship5

Three modes (chosen on the menu's mode screen, Session 6.4):
  single — the classic solo run (unchanged).
  host   — Session 6.5: wait for ONE client, run the authoritative 2-ship
           sim, apply the client's input, broadcast a snapshot every
           SNAPSHOT_INTERVAL ticks. ESC/QUIT or a client disconnect
           returns to the menu (pinned decision #8: no reconnect in v1).
  join   — Session 6.6: connect to a host, run the client sim (predict the
           local ship, interpolate the remote entities), send input every
           frame, push each snapshot, render via predicted_view.
"""
import socket
import sys

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
                  T_INPUT, T_SNAP)
from .netcode import (PredictedShip, HostTimeEstimator, LatencyTracker,
                      RenderPoint)
from .ship import Ship
from .sound import SoundBank
from .intent import ShipInput


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
        conn.set_nonblocking()

        # --- 4. the authoritative game loop ---
        last_sent = -1
        while True:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)
            if not game.handle_events():
                return
            keys = pygame.key.get_pressed()
            # Poll the client: apply its latest input (pinned #5: the host
            # applies the LATEST received input each tick).
            for m in conn.poll():
                if m.get("type") == T_INPUT:
                    game.set_remote_input(deserialize_input(m["inp"]))
            if conn.closed:
                conn.close()
                _notice(screen, font, big_font, clock,
                        ["DISCONNECTED", "the other player left"])
                return
            conn.drain_send()
            game.update(dt, keys)
            # Broadcast a snapshot every SNAPSHOT_INTERVAL sim ticks.
            # round() guards against float drift in sim_time/STEP.
            tick = round(game.sim_time / STEP)
            if tick != last_sent and tick % SNAPSHOT_INTERVAL == 0:
                conn.send({"type": T_SNAP, "sim_time": game.sim_time,
                           "snap": serialize_snapshot(game.snapshot())})
                last_sent = tick
            game.draw(dt)
            pygame.display.flip()
    finally:
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
    game.ghost = PredictedShip(hull=menu.hull, loadout=menu.loadout)
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
    # Flip the socket to non-blocking for the game loop (the handshake used a
    # 0.05 s recv timeout; it must be cleared before the loop).
    conn.set_nonblocking()

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        now = pygame.time.get_ticks() / 1000.0
        if not game.handle_events():
            break
        keys = pygame.key.get_pressed()
        # Sample the local input ONCE per frame (the intent edge) and use
        # that same object to send, to buffer, and (below) to render — so
        # the three can never disagree about what the player did this
        # frame.
        inp = ShipInput.from_keys(keys)
        # Send the local input every frame (pinned #5: the host applies the
        # LATEST received input each tick).
        conn.send({"type": T_INPUT, "inp": serialize_input(inp)})
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
        # up to it.
        for m in conn.poll():
            if m.get("type") == T_SNAP:
                if game.host_time.record(now, m["sim_time"]):
                    game.latency.update(game.host_time.last_jitter)
                game.push_snapshot(m["sim_time"],
                                   deserialize_snapshot(m["snap"]),
                                   now=game.host_time.now(now))
        if conn.closed:
            conn.close()
            _notice(screen, font, big_font, clock,
                    ["DISCONNECTED", "the host left"])
            break
        conn.drain_send()
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
        pygame.display.flip()
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

        # --- single player (the classic path, unchanged) ---
        game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                    hull=menu.hull, loadout=menu.loadout,
                    test_mode=('--test' in sys.argv), seed=seed, sound=sfx)
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)
            running = game.handle_events()
            keys = pygame.key.get_pressed()
            game.update(dt, keys)
            game.draw(dt)
            pygame.display.flip()
        break

    pygame.quit()


if __name__ == "__main__":
    main()