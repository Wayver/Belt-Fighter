"""Session 9.x M3: the SimThread (the authoritative sim on its own thread).

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_sim_thread

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

The SimThread moves the host's Game off the render thread onto a daemon
thread with a real-time 60 Hz accumulator clock. This test drives it the
way the host loop will: a fake render thread publishes input + commands,
the sim thread steps the Game and publishes the RenderModel, and a fake
worker records the snapshots the sim thread hands it.

The checks:
  1. SMOKE    — the thread runs, publishes a plain-data model, and the
                sim clock advances at ~1x real time (not tied to any
                render frame rate — there is no render loop here at all).
  1b. PUBLISH-ON-STEP — the thread publishes a NEW model only on a step
                (the model object is stable between steps), so a render
                thread reading at its own pace sees the same model across
                reads and _HostRenderClock's alpha VARIES (not pinned at
                0). Regression for the M4 alpha=0 / wasted-CPU bug.
  1c. SP-ON-SIMTHREAD (M5) — the single-player loop pattern: the same
                thread with worker=None (no NetWorker), driven the way
                the SP render loop does (publish_input + read
                latest_model + draw the published model with the render
                clock's alpha). The production SP path is the host loop
                minus the network, so it must run end-to-end.
  2. INPUT    — the LATEST published local input drives player 0 (the
                ship moves), while the remote input (a direct reference
                swap on game.remote_input, the render-thread hand-off)
                drives player 1. The two inputs are independent.
  3. COMMANDS — T (targeting) and R (reset on game_over) are applied by
                the sim thread at the top of its next iteration, with the
                same game_over gating as Game.handle_events.
  4. STOP     — stop() joins the thread promptly and is idempotent.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import time

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .intent import ShipInput
from .sim_thread import SimThread


class FakeWorker:
    """Just enough of a NetWorker for the SimThread: records the
    (sim_time, snapshot) the sim thread publishes via
    set_latest_snapshot. The real worker's 10 Hz timer is out of scope
    here (it is covered by the net self-test + e2e)."""

    def __init__(self):
        self.snapshots = []

    def set_latest_snapshot(self, sim_time, snap):
        self.snapshots.append((sim_time, snap))


def make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed, players=1):
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                seed=seed, players=players)


def wait_for(cond, timeout=5.0, msg="condition"):
    """Poll `cond` until true or `timeout` seconds pass. Returns cond()."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def check(label, cond, extra=""):
    print(("PASS: " if cond else "FAIL: ") + label
          + (("  " + extra) if extra else ""))
    return bool(cond)


def _model(sim_time, step_alpha):
    """A minimal published model for the render-clock unit checks (only
    the two fields _HostRenderClock reads)."""
    return {"sim_time": sim_time, "step_alpha": step_alpha}


def check_render_clock():
    """The render thread's own interpolation alpha (the M3 #6 change):
    derived from the render clock + the published model's stamp, NOT the
    sim's acc. The model carries the prev+curr pose (the window
    [T - STEP, T]), so the alpha = (t - T) / STEP, clamped to [0, 1]."""
    from .__main__ import _HostRenderClock
    ok = True
    # Steady state: publishes arrive every STEP with the sim's acc
    # fraction; the alpha must track the sim's current-time fraction
    # (what the old sim-acc/STEP gave the render thread).
    rc = _HostRenderClock()
    worst = 0.0
    for i in range(1, 121):
        T = i * STEP
        acc_frac = 0.3
        m = _model(T, acc_frac)
        a = rc.advance(STEP, m)
        worst = max(worst, abs(a - acc_frac))
    ok &= check("render clock: steady-state alpha tracks the sim's "
                "current-time fraction", worst < 1e-9,
                "worst |a - acc/STEP| = %.2e" % worst)
    # Hiccup: the estimate must never extrapolate past the current pose
    # (alpha clamped to 1), and a render hiccup (dt clamped to 50 ms)
    # must not drag it forward faster than 1x.
    rc = _HostRenderClock()
    m = _model(STEP, 0.0)
    rc.advance(STEP, m)
    a = rc.advance(0.05, m)   # SAME model object (no new publish), hiccup
    ok &= check("render clock: hiccup clamps alpha to 1 (no extrapolate)",
                a == 1.0, "alpha=%.3f" % a)
    # Backlog drop: the sim's hiccup policy DROPS time (ACC_BACKLOG_CAP),
    # so the sim clock can JUMP BACKWARD relative to a free-running
    # estimate. The re-anchor on each new model must correct it — the
    # alpha must recover to the sim's fraction on the next publish.
    rc = _HostRenderClock()
    m = _model(STEP, 0.0)
    rc.advance(STEP, m)
    rc.advance(0.05, m)                       # same model: estimate ~0.05 ahead
    a = rc.advance(STEP, _model(2 * STEP, 0.4))   # new model (fresh stamp)
    ok &= check("render clock: re-anchors on a new model after a backlog "
                "drop (no permanent clamp)", abs(a - 0.4) < 1e-9,
                "alpha=%.3f (want 0.4)" % a)
    # No model yet: alpha 0 (draw the latest state, no interpolation).
    ok &= check("render clock: no model -> alpha 0",
                _HostRenderClock().advance(STEP, None) == 0.0)
    return ok


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ok = True

    # --- 0. RENDER CLOCK: the render thread's own alpha (M3 #6) --------
    ok &= check_render_clock()

    # --- 1. SMOKE: the thread steps the sim at ~1x real time ------------
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234)
    worker = FakeWorker()
    st = SimThread(g, worker)
    st.start()
    ok &= check("smoke: a model is published",
                wait_for(lambda: st.latest_model is not None),
                "sim_time=%.3f" % (st.latest_model["sim_time"]
                                   if st.latest_model else -1))
    time.sleep(0.4)
    st.stop()
    m = st.latest_model
    ok &= check("smoke: the model is a dict with the sim clock",
                isinstance(m, dict) and m["sim_time"] > 0.1,
                "sim_time=%.3f" % (m["sim_time"] if m else -1))
    # The sim clock tracks REAL time (the thread has no render loop to be
    # tied to): after ~0.4 s of wall time it should be in [0.2, 0.6] —
    # well inside that if the accumulator is running at 1x, far outside
    # if it were tied to a frame rate or stalled.
    ok &= check("smoke: sim clock advances at ~1x real time",
                0.2 <= m["sim_time"] <= 0.6,
                "sim_time=%.3f after ~0.4 s wall" % m["sim_time"])
    ok &= check("smoke: step_alpha is the sim's acc/STEP fraction",
                0.0 <= m["step_alpha"] < 1.0,
                "step_alpha=%.3f" % m["step_alpha"])
    ok &= check("smoke: the worker received snapshots from the sim thread",
                len(worker.snapshots) > 5,
                "n=%d" % len(worker.snapshots))
    ok &= check("smoke: the last snapshot is stamped with the sim clock",
                abs(worker.snapshots[-1][0] - m["sim_time"]) < 2 * STEP)

    # --- 1b. PUBLISH-ON-STEP: the REAL host-loop pattern (regression for
    #     the M4 alpha=0 / wasted-CPU bug). The thread iterates every
    #     ~1 ms but only STEPS 60x/sec; it must publish a NEW model only
    #     on a step, so the model object is STABLE between steps. A render
    #     thread reading at its own pace (here ~22 FPS, 45 ms — the M4
    #     host's measured rate) must then see the SAME model object across
    #     reads (so _HostRenderClock advances its alpha between publishes)
    #     and the alpha must VARY, not sit at 0. Before the fix the thread
    #     published a fresh dict every ~1 ms iteration, so the render
    #     clock (which detects a new model by object identity) re-anchored
    #     every frame and alpha was 0.0000 for 100% of frames — exactly
    #     what the M4 host_sim_debug.csv showed. ---
    from .__main__ import _HostRenderClock
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234)
    worker = FakeWorker()
    st = SimThread(g, worker)
    st.start()
    wait_for(lambda: st.latest_model is not None)
    rc = _HostRenderClock()
    alphas = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        m = st.latest_model
        if m is None:
            time.sleep(0.005)
            continue
        alphas.append(rc.advance(0.045, m))   # a ~22 FPS render frame
        time.sleep(0.045)
    st.stop()
    # The worker receives set_latest_snapshot ONLY on a step. ~1 s of sim
    # time = ~60 steps, so ~60 snapshots. Before the fix the thread
    # published (and snapshotted) every ~1 ms iteration -> ~1000 in 1 s.
    # This is the direct "publish-on-step" signal.
    ok &= check("publish-on-step: the worker got ~one snapshot per step "
                "(~60/s, not ~1000/s)",
                40 <= len(worker.snapshots) <= 80,
                "n=%d in ~1 s (sim_time=%.2f)"
                % (len(worker.snapshots), g.sim_time))
    # The render clock's alpha must NOT be pinned at exactly 0. Pre-fix,
    # g.acc was never written by the sim thread (it steps g._step directly,
    # not g.update()), so step_alpha was always 0.0 and the render clock
    # (re-anchoring on every fresh ~1 ms model) gave alpha=0.0000 for 100%
    # of frames — exactly the M4 host_sim_debug.csv. Post-fix, the render
    # (at ~22 FPS here) sees a fresh model each frame and re-anchors to the
    # sim's step_alpha at publish, which is the sub-step overshoot (varies
    # in ~[0, 0.1]) — so the alphas vary and reach > 0.01. (They do NOT
    # span [0,1): the sim thread feeds its own acc at ~1 ms granularity, so
    # the post-step remainder is small; the smoothness comes from t
    # advancing between publishes at a higher render FPS.)
    ok &= check("publish-on-step: render alpha is not pinned at 0 (varies)",
                len(alphas) > 5 and max(alphas) > 0.01
                and len(set(round(a, 4) for a in alphas)) > 2,
                "alpha: n=%d min=%.4f max=%.4f distinct=%d"
                % (len(alphas), min(alphas) if alphas else -1,
                   max(alphas) if alphas else -1,
                   len(set(round(a, 4) for a in alphas))))

    # --- 1c. SP-ON-SIMTHREAD (M5): the production single-player loop ---
    # SP is the host loop minus the NetWorker: SimThread(game) with
    # worker=None, the render thread publishes input + reads the
    # published model + draws it with the render clock's alpha. This
    # drives that exact pattern (worker=None + a real draw() of the
    # published model each frame) so the SP path is covered end-to-end,
    # not just the host's worker-fed variant.
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234)
    st = SimThread(g)   # worker=None: the SP shape
    st.start()
    wait_for(lambda: st.latest_model is not None)
    rc = _HostRenderClock()
    frames = 0
    alphas = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.6:
        m = st.latest_model
        if m is None:
            time.sleep(0.005)
            continue
        a = rc.advance(0.016, m)
        # The SP render loop's draw: the published model + the render
        # thread's own alpha (the same {**model, "step_alpha": alpha}
        # seam the host loop uses).
        g.draw(0.016, {**m, "step_alpha": a})
        frames += 1
        alphas.append(a)
        time.sleep(0.005)
    st.stop()
    ok &= check("sp-on-simthread: worker=None thread runs + the render "
                "thread draws the published model",
                frames > 10 and g.sim_time > 0.2,
                "frames=%d sim_time=%.3f" % (frames, g.sim_time))
    ok &= check("sp-on-simthread: the render clock's alpha varies "
                "(interpolation is live, not pinned)",
                len(alphas) > 10 and max(alphas) > 0.01
                and len(set(round(a, 4) for a in alphas)) > 2,
                "alpha: n=%d min=%.4f max=%.4f distinct=%d"
                % (len(alphas), min(alphas) if alphas else -1,
                   max(alphas) if alphas else -1,
                   len(set(round(a, 4) for a in alphas))))

    # --- 2. INPUT: local (published) + remote (reference swap) ----------
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234, players=2)
    st = SimThread(g, FakeWorker())
    st.start()
    wait_for(lambda: st.latest_model is not None)
    time.sleep(0.1)   # let a few idle steps settle (spawn protection on)
    # The fake render thread: publish the local input (player 0 thrusts
    # forward) and hand off the remote input (player 1 turns) the way the
    # host loop will — a direct reference swap on game.remote_input.
    st.publish_input(ShipInput(thrust_fwd=1.0))
    g.remote_input = ShipInput(turn=1.0)
    time.sleep(0.5)
    st.stop()
    p0, p1 = g.players
    ok &= check("input: the published local input drives player 0",
                p0.pos.distance_to(pygame.Vector2(WIDTH / 2, HEIGHT / 2)) > 5,
                "p0 moved %.1f px from center"
                % p0.pos.distance_to(pygame.Vector2(WIDTH / 2, HEIGHT / 2)))
    ok &= check("input: the remote input drives player 1 (turn)",
                abs(p1.angle - (-math.pi / 2)) > 0.05,
                "p1 angle=%.3f (start -1.571)" % p1.angle)

    # --- 3. COMMANDS: T (targeting) + R (reset on game_over) ------------
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234)
    st = SimThread(g, FakeWorker())
    st.start()
    wait_for(lambda: st.latest_model is not None)
    st.command("targeting")
    ok &= check("commands: T toggles targeting (applied by the sim thread)",
                wait_for(lambda: g.ship.targeting_on),
                "targeting_on=%r" % g.ship.targeting_on)
    st.command("targeting")
    ok &= check("commands: T toggles back",
                wait_for(lambda: not g.ship.targeting_on))
    # R on game_over: force the game-over state (the command's gate), then
    # the reset must restore the ship to center + clear game_over.
    g.game_over = True
    g.players[0].pos = pygame.Vector2(1.0, 1.0)
    st.command("reset")
    ok &= check("commands: R resets on game_over",
                wait_for(lambda: not g.game_over
                         and g.ship.pos.distance_to(
                             pygame.Vector2(WIDTH / 2, HEIGHT / 2)) < 1.0),
                "game_over=%r pos=%s" % (g.game_over, g.ship.pos))
    st.stop()

    # --- 4. STOP: joins promptly + idempotent ----------------------------
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  seed=1234)
    st = SimThread(g, FakeWorker())
    st.start()
    wait_for(lambda: st.latest_model is not None)
    t0 = time.monotonic()
    st.stop()
    join_ms = (time.monotonic() - t0) * 1000.0
    ok &= check("stop: joins promptly (< 50 ms)", join_ms < 50.0,
                "%.1f ms" % join_ms)
    st.stop()   # idempotent: a second stop() is a no-op
    ok &= check("stop: idempotent (second stop() is a no-op)", True)

    pygame.quit()
    print("\nSIM THREAD (M3/M5):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()