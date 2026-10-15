"""F3 telemetry capture, host in a SEPARATE PROCESS (real topology).

The host (run_host + its own spawned sim) runs in a subprocess with its own
GIL — matching the real 2-machine case where the host's render process is not
starved by a client loop. The client (run_client) runs in THIS process (the
parent), so its render loop runs at full 60 FPS and its net_debug.csv is the
clean measure of the snapshot cadence the client's ghost clock sees over the
wire.

This isolates the M4 metric that matters: the client's newest_stamp delta
(== 17 ms every frame, no 33 ms jumps) + the snap_px distribution.

M4 REDO note: the client's newest_stamp delta is now the measure of the
SNAPSHOT SEND cadence end to end (sim process -> dedicated net Queue ->
_NetPump thread -> NetWorker -> socket). The host_debug.csv `steps` column
is ALSO captured, but after the redo it is the render thread's MODEL-DRAIN
cadence (presentation only — the wire no longer rides the model Queue), so
it is reported for context, not as the send-path measure.

Run:  python -m ship5.tools.dbg_f3_capture2
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import csv
import statistics
import subprocess
import sys
import threading
import time

import pygame

from ship5.config import WIDTH, HEIGHT
from ship5.fog import make_light_texture
from ship5.game import Game
from ship5.hulls import PLAYER_HULLS, default_loadout
from ship5.sound import SoundBank
from ship5 import __main__ as M

RUN_S = 12.0
_ORIG_GAME_INIT = Game.__init__


class _Keys:
    def __getitem__(self, k):
        return 1 if k == pygame.K_w else 0


def _free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pctl(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    sfx = SoundBank()
    sfx.init()
    clock = pygame.time.Clock()

    port = _free_port()
    M._lan_ip = lambda: "127.0.0.1"
    pygame.key.get_pressed = lambda: _Keys()

    # --- host in a subprocess (its own GIL + its own spawned sim) ---
    host_proc = subprocess.Popen(
        [sys.executable, "-m", "ship5.tools.f3_host_runner", str(port)],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    time.sleep(1.5)   # let the host bind + listen

    created = []
    def patched_init(self, *a, **k):
        _ORIG_GAME_INIT(self, *a, **k)
        created.append(self)
        self.debug_net = True
    Game.__init__ = patched_init

    class _Menu:
        def __init__(self, hull, ip, port):
            self.hull = hull
            self.loadout = default_loadout(hull)
            self.mode = "host"
            self.host_ip = ip
            self.host_port = port

    errors = []
    def client_side():
        try:
            M.run_client(screen, font, big_font, clock, sfx,
                         _Menu(PLAYER_HULLS[1], "127.0.0.1", port), None,
                         light_tex, fog_surf, light_surf)
        except Exception:
            import traceback
            errors.append("client: " + traceback.format_exc())

    th_client = threading.Thread(target=client_side, daemon=True)
    th_client.start()
    time.sleep(RUN_S)
    stop = time.time() + 10.0
    while time.time() < stop and th_client.is_alive():
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        time.sleep(0.05)
    th_client.join(timeout=8)
    Game.__init__ = _ORIG_GAME_INIT

    # Stop the host subprocess (it has no QUIT source from us; SIGTERM).
    host_proc.terminate()
    try:
        host_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        host_proc.kill()
        host_proc.wait()

    print("errors: %r" % (errors,))
    print("client games built: %d" % len(created))

    ship5_dir = os.path.dirname(os.path.abspath(M.__file__))

    # --- host_debug.csv (host subprocess): the `steps` column ---
    # M4 REDO: `steps` is the render thread's MODEL-DRAIN cadence
    # (presentation only — the wire traffic rides the dedicated net Queue +
    # _NetPump thread, NOT the model Queue). It is reported for context
    # (a render-thread hiccup still shows up here), but the SNAPSHOT SEND
    # cadence is measured client-side (the newest_stamp delta below).
    steps = []
    with open(os.path.join(ship5_dir, "host_debug.csv")) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["steps"]))
    n = len(steps)
    mean_steps = sum(steps) / n if n else 0.0
    n2 = sum(1 for s in steps if s >= 2)
    print("\n=== HOST (host_debug.csv, separate process) ===")
    print("(steps = render-thread MODEL-DRAIN cadence, presentation only —")
    print(" the snapshot SEND cadence is the client's newest_stamp delta below)")
    print("windows=%d  mean steps/window=%.3f  >=2-tick windows=%d (%.1f%%)"
          % (n, mean_steps, n2, 100.0 * n2 / n if n else 0.0))
    dist = {}
    for s in steps:
        dist[s] = dist.get(s, 0) + 1
    print("steps distribution: %r" % {k: dist[k] for k in sorted(dist)})

    # --- net_debug.csv (client, this process): stamp deltas + snap_px ---
    rows = []
    with open(os.path.join(ship5_dir, "net_debug.csv")) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    stamps = []
    for r in rows:
        try:
            stamps.append(float(r["newest_stamp"]))
        except ValueError:
            pass
    deltas = [1000.0 * (b - a) for a, b in zip(stamps, stamps[1:])
              if b > a]
    snaps = [float(r["snap_px"]) for r in rows if r["snap_px"]]
    d17 = sum(1 for d in deltas if 10.0 <= d <= 24.0)
    d33 = sum(1 for d in deltas if 24.0 < d <= 40.0)
    dbig = sum(1 for d in deltas if d > 40.0)
    tail = sum(1 for s in snaps if s >= 3.0)
    print("\n=== CLIENT (net_debug.csv, separate-process host) ===")
    print("reconcile lines=%d  stamp deltas=%d" % (len(rows), len(deltas)))
    if deltas:
        print("newest_stamp delta: mean=%.2f ms  1-tick(10-24ms)=%d (%.1f%%)  "
              "2-tick(24-40ms)=%d (%.1f%%)  >40ms=%d"
              % (statistics.mean(deltas), d17, 100.0 * d17 / len(deltas),
                 d33, 100.0 * d33 / len(deltas), dbig))
        print("  max delta=%.2f ms" % max(deltas))
    if snaps:
        print("snap_px: n=%d median=%.3f  p95=%.3f  max=%.3f  "
              ">=3px tail=%d (%.2f%%)"
              % (len(snaps), statistics.median(snaps), _pctl(snaps, 95),
                 max(snaps), tail, 100.0 * tail / len(snaps)))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()