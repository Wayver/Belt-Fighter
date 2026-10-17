"""Phase 5 (plan_14): fresh 2P bullet-drift capture, HEADLESS.

The Phase-5 checklist says: "Re-run tools/analyze_bullet_drift.py on a fresh
capture (expect seq gap 0, true-extra 0)." The live 2P playtest is the
ultimate capture (F3 on), but this script drives the REAL client loop path
headlessly so the capture is reproducible and does not need a human:

  * a 2P host Game steps at 60 Hz, player 1 (the client) holds fire;
  * the host's snapshot is delivered to the client L ticks late (one-way
    wire latency);
  * the client runs the REAL client path: host_time.record + latency.update
    on each arrival, push_snapshot (seed + reconcile/rewind), and
    predicted_view (ghost advance + the id-dedup + the DEDUP telemetry row).

The bullet_telemetry hooks (FIRE in PredictedShip.step, RECONCILE in
push_snapshot, HANDBACK in reconcile, DEDUP in predicted_view) are the same
ones the live client uses, so the CSV is identical in shape to a live F3
capture. After this runs, `python tools/analyze_bullet_drift.py` reports the
seq gap (expect 0) and true-extra (expect 0).

Run from /mnt:  python -m ship5.capture_drift
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, STEP
from .intent import ShipInput
from .netcode import (PredictedShip, HostTimeEstimator, LatencyTracker,
                      RenderPoint)
from . import bullet_telemetry

N_TICKS = 600          # 10 s of sustained fire
L = 3                  # one-way wire latency, in ticks
LOCAL_INDEX = 1        # the client is player 1
FIRE = ShipInput(fire=True)
IDLE = ShipInput()


class Keys:
    def __getitem__(self, k):
        return 0


IDLE_KEYS = Keys()


def make_res():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return (screen, font, big_font, light_tex, fog_surf, light_surf)


def make_host(res):
    g = Game(*res, seed=1234, players=2, local_index=0)
    return g


def make_client(res, host):
    g = Game(*res, seed=1234, players=2, local_index=LOCAL_INDEX)
    p1 = host.players[LOCAL_INDEX]
    g.ghost = PredictedShip(hull=p1.hull, loadout=p1.components,
                            local_index=LOCAL_INDEX)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    g._rtt_ow = L * STEP   # the capture's fixed one-way latency
    return g


def main():
    pygame.init()
    res = make_res()
    host = make_host(res)
    client = make_client(res, host)
    ghost = client.ghost

    # Seed the ghost at the host's initial state (t=0).
    s0 = host.snapshot()
    client.push_snapshot(0.0, s0, now=0.0)

    # Enable the REAL telemetry (the same hooks the live F3 client uses).
    bullet_telemetry.set_enabled(True)

    snaps = [s0]
    last_echo_idx = 0
    n_reconcile = 0
    n_dedup = 0

    for t in range(1, N_TICKS + 1):
        # --- HOST tick: player 0 idle, player 1 (the client) holds fire.
        host.remote_input = FIRE
        host._step(STEP, IDLE)
        snaps.append(host.snapshot())   # snaps[t] at t*STEP

        # --- Forward the host's new input-echo entries (T_ECHO wire).
        for (st, inp) in host._remote_input_echo[last_echo_idx:]:
            ghost.record_echo(st, inp)
        last_echo_idx = len(host._remote_input_echo)

        # --- SNAPSHOT ARRIVAL: the snapshot taken L ticks ago arrives now.
        if t - L >= 0:
            s_idx = t - L
            snap = snaps[s_idx]
            s_time = s_idx * STEP
            now = t * STEP
            if client.host_time.record(now, s_time):
                client.latency.update(client.host_time.last_jitter)
            client.push_snapshot(s_time, snap,
                                 now=client.host_time.now(now)
                                 + client._rtt_ow,
                                 real_time=now,
                                 rate=client.host_time.rate)
            n_reconcile += 1

        # --- RENDER: the real client draw path (ghost advance + id-dedup
        #     + the DEDUP telemetry row).
        client.latency.tick(STEP)
        rp = client.render_point.advance(STEP,
                                         client.snap_buf.newest_time())
        if rp is not None:
            client.predicted_view(STEP, IDLE_KEYS,
                                  host_time=client.host_time.now(t * STEP),
                                  real_time=t * STEP)
            n_dedup += 1

    bullet_telemetry.set_enabled(False)
    bullet_telemetry.close()

    # --- CLEAN double-draw signal (uncontaminated by the L-tick
    #     prediction-ahead): the ghost's fired bullet ids must be a subset
    #     of the host's fired bullet ids. Both start at seq 0 and bump by
    #     1 per shot, so if the final seqs are equal the id sets are
    #     identical (no ghost-only id, no missing shot). This is the same
    #     check as test_14_2 check 3, but in the real client loop. ---
    host_seq = host.players[LOCAL_INDEX].bullet_seq
    ghost_seq = ghost.ship.bullet_seq
    ghost_only = [s for s in range(ghost_seq) if s >= host_seq]
    print("capture complete: %d ticks, %d reconciles, %d render frames"
          % (N_TICKS, n_reconcile, n_dedup))
    print("CLEAN SIGNAL: host bullet_seq=%d, ghost bullet_seq=%d"
          % (host_seq, ghost_seq))
    if ghost_seq == host_seq:
        print("  NO GHOST-ONLY ID — the ghost fired exactly the same shots "
              "as the host (id sets identical). The double-draw is gone.")
    else:
        print("  GHOST-ONLY IDS — %d ghost shots the host never fired: %r"
              % (len(ghost_only), ghost_only[:8]))
    print("note: the analyzer's SEQ GAP / TRUE-EXTRA metrics are "
          "contaminated by the L-tick prediction-ahead in this synthetic "
          "capture (the ghost is legitimately L ticks ahead of the stale "
          "snapshot stamp). The clean signal above is the authoritative "
          "double-draw check; the live F3 capture is needed for the "
          "analyzer's 'gap 0, true-extra 0' reading.")
    pygame.quit()


if __name__ == "__main__":
    main()