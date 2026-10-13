"""Diagnostic: is the residual snap in ship.pos or render_pose?

Runs the same harness as test_10_11c_phase_lock but measures BOTH the
physics pose (ship.pos) and the rendered pose (render_pose) across each
reconcile. The player sees render_pose, so that's the one that matters.

Run from the repo root:
    python -m ship5.tools.diag_phase
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import pygame

from ..config import TICK
from ..hulls import SILAS_HULL, default_loadout
from ..ship import Ship
from ..intent import ShipInput
from ..netcode import PredictedShip, HostTimeEstimator

L = 0.020
SNAP_INTERVAL = 2 * TICK
N_INTERVALS = 30


def main():
    pygame.init()
    thrust = ShipInput(thrust_fwd=1.0)

    host = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    host.vel = pygame.Vector2(0.0, 300.0)

    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    g.seed(host.snapshot(), snap_time=0.0, real_time=0.0)

    host_sim = 0.0
    real_time = 0.0
    snaps_pos = []
    snaps_render = []

    for i in range(N_INTERVALS):
        steps = g.advance(SNAP_INTERVAL, thrust,
                          real_time=real_time + SNAP_INTERVAL)
        for _ in range(2):
            host.update(TICK, thrust)
        host_sim += SNAP_INTERVAL
        real_time += SNAP_INTERVAL

        snap_time = host_sim
        now = snap_time + L
        # Capture BOTH poses before the reconcile.
        bx, by = g.ship.pos.x, g.ship.pos.y
        rpos, _ = g.render_pose()
        rx, ry = rpos.x, rpos.y
        snap = host.snapshot()
        g.record_input(snap_time, thrust)
        g.reconcile_rewind(snap, snap_time, now,
                           real_time=real_time, rate=1.0)
        # Capture BOTH poses after the reconcile.
        snaps_pos.append(math.hypot(g.ship.pos.x - bx, g.ship.pos.y - by))
        rpos2, _ = g.render_pose()
        snaps_render.append(math.hypot(rpos2.x - rx, rpos2.y - ry))

    def report(name, snaps):
        mx = max(snaps)
        mn = sum(snaps) / len(snaps)
        big = sum(1 for s in snaps if s > 1.0)
        worst = snaps.index(mx)
        print(f"{name}: max {mx:.3f}px (interval {worst}), mean {mn:.3f}px, "
              f"{big}/{N_INTERVALS} > 1px")
        # Show the per-interval values around the worst one.
        lo = max(0, worst - 2)
        hi = min(N_INTERVALS, worst + 3)
        print(f"  around worst: " +
              " ".join(f"{snaps[k]:.2f}" for k in range(lo, hi)))

    report("ship.pos   ", snaps_pos)
    report("render_pose", snaps_render)

    # Also: is the physics pose one tick ahead of the clock after reconcile?
    # (The 10.16 hack's signature.)
    print(f"\nAfter last reconcile: _sim_time={g._sim_time:.4f}, "
          f"_last_step_time={g._last_step_time:.4f}, "
          f"pose_time(=last_step)={g._last_step_time:.4f}")
    print(f"  _sim_time - _last_step_time = "
          f"{g._sim_time - g._last_step_time:.4f} (should be in [0, TICK))")

    pygame.quit()


def main_branch2():
    """Drive the LIVE-client clock path (branch 2: host_time.now + rtt_ow).

    Mirrors the real client (__main__.py): 60 Hz snapshots (SNAPSHOT_INTERVAL
    = 1 tick), one-way latency L, HostTimeEstimator re-anchored on every
    arrival, ghost clock = host_time.now(real_time) + rtt_ow, reconcile
    replay target now = snap_time + rtt_ow.

    Measures, across each reconcile:
      - ship.pos snap        (what the net-debug CSV's snap_px reports)
      - render_pose snap     (what the player actually sees)
      - phase offset         (snap_time - _last_step_time) in ticks BEFORE
      - _acc before          (should be in [0, TICK) per the contract)
      - pose lead after      (pose time - _sim_time) in ticks — the 10.16
                             pre-step signature
      - prediction accuracy  (ghost pose vs the host's pose at the SAME time)
    """
    L = 0.020
    N = 30
    FRAME = 1.0 / 120.0   # client render frames, finer than the sim

    pygame.init()
    thrust = ShipInput(thrust_fwd=1.0)

    host = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    host.vel = pygame.Vector2(0.0, 300.0)

    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    ht = HostTimeEstimator()
    rtt_ow = L

    # `host` is the snapshot source (in lockstep: at tick k after update).
    # `ref` is the prediction-accuracy reference — it runs 2 ticks AHEAD of
    # the snapshot (the ghost's pose after the 10.16 pre-step sits at
    # snap_time + 2*TICK, so it must be compared against the host's pose at
    # that same time).
    ref = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    ref.vel = pygame.Vector2(0.0, 300.0)
    ref_ticks = 0

    # Snapshot 0: taken at host time 0, arrives at local time L.
    g.seed(host.snapshot(), snap_time=0.0, real_time=L)
    ht.record(L, 0.0)
    t_local = L

    snaps_pos, snaps_render = [], []
    phase_before, acc_before, pose_lead_after = [], [], []
    pred_err = []

    for k in range(1, N):
        # Host advances one tick -> it is now at tick k.
        host.update(TICK, thrust)
        snap = host.snapshot()
        t_arrive = k * TICK + L
        # Client frames up to the arrival (branch-2 advance).
        while t_local < t_arrive - 1e-9:
            t_next = min(t_local + FRAME, t_arrive)
            g.advance(t_next - t_local, thrust, real_time=t_next,
                      host_time=ht, rtt_ow=rtt_ow)
            t_local = t_next
        snap_time = k * TICK
        # Record the input at the estimated host time (as __main__.py does).
        est_now = ht.now(t_local)
        if est_now is not None:
            g.record_input(est_now, thrust)
        # Arrival: feed the estimator, then reconcile (now = snap + rtt_ow,
        # exactly as __main__.py passes it).
        ht.record(t_local, snap_time)
        now = snap_time + rtt_ow
        phase_before.append((snap_time - g._last_step_time) / TICK)
        acc_before.append(g._acc)
        bx, by = g.ship.pos.x, g.ship.pos.y
        rp, _ = g.render_pose()
        g.reconcile_rewind(snap, snap_time, now,
                           real_time=t_local, rate=ht.rate)
        snaps_pos.append(math.hypot(g.ship.pos.x - bx, g.ship.pos.y - by))
        rp2, _ = g.render_pose()
        snaps_render.append(math.hypot(rp2.x - rp.x, rp2.y - rp.y))
        # The 10.16 pre-step puts the pose at snap_time + (n+1)*TICK.
        n = g._last_replay_ticks
        pose_time = snap_time + (n + 1) * TICK
        pose_lead_after.append((pose_time - g._sim_time) / TICK)
        # Prediction accuracy: run the reference to pose_time and compare.
        while ref_ticks < int(round(pose_time / TICK)):
            ref.update(TICK, thrust)
            ref_ticks += 1
        pred_err.append(math.hypot(g.ship.pos.x - ref.pos.x,
                                   g.ship.pos.y - ref.pos.y))

    def report(name, vals, unit="px"):
        mx = max(vals)
        mn = sum(vals) / len(vals)
        big = sum(1 for v in vals[1:] if v > 1.0)
        print(f"{name}: max {mx:.3f}{unit} (incl. first), "
              f"steady-state mean {mn:.3f}{unit}, "
              f"{big}/{len(vals) - 1} steady > 1{unit}")
        print("  steady: " + " ".join(f"{v:.2f}" for v in vals[1:8])
              + " ...")

    report("ship.pos snap   ", snaps_pos)
    report("render_pose snap", snaps_render)
    report("phase offset (pre-reconcile, ticks)", phase_before, " ticks")
    report("_acc (pre-reconcile, ticks)", acc_before, " ticks")
    report("pose lead over _sim_time (post-reconcile, ticks)",
           pose_lead_after, " ticks")
    report("prediction error (ghost vs host @ same time)", pred_err)

    print(f"\nOne tick of motion = {520.0 * TICK:.3f}px (MAX_SPEED*TICK)")
    pygame.quit()


def main_branch2_echo():
    """Branch 2 + echo buffer: measure the in-flight-input divergence.

    Simulates the real-client timing with the echo buffer (T_ECHO). The
    host applies the client's input after one-way latency L; the ghost
    predicts with the CURRENT local input (ahead by L). At each reconcile,
    the replay uses the echo buffer (what the host actually applied).

    Measures, across each reconcile:
      - snap size (ship.pos displacement)
      - input divergence: the ghost's predicted input (current local) vs.
        the replayed input (echo) — the in-flight-input gap
      - which buffer the replay used (echo or own-input fallback)
      - the echo's staleness (how many ticks behind the replay window)

    The hypothesis: the snap size correlates with the input divergence
    (the ghost predicted with I_new, but the replay used the old input).
    """
    L = 0.020
    N = 40
    FRAME = 1.0 / 120.0

    pygame.init()
    # Input script: thrust on/off to create in-flight-input transitions.
    # Change input every 5 ticks to create many transitions.
    def input_at(tick):
        # Alternate thrust on/off every 5 ticks.
        return ShipInput(thrust_fwd=1.0) if (tick // 5) % 2 == 0 else ShipInput()

    host = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    host.vel = pygame.Vector2(0.0, 300.0)

    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    ht = HostTimeEstimator()
    rtt_ow = L

    # The host's ACTUAL applied input (with one-way latency L). The host
    # receives the client's input at T + L and applies it from the first
    # tick after T + L.
    host_applied_input = ShipInput()  # the input the host is currently applying
    host_input_change_tick = 0  # the tick at which the host switches input

    # Snapshot 0.
    g.seed(host.snapshot(), snap_time=0.0, real_time=L)
    ht.record(L, 0.0)
    t_local = L

    snaps_pos = []
    input_divergence = []  # 1.0 if the ghost's predicted input != the replayed input
    buffer_used = []  # 'echo' or 'own'
    echo_staleness = []  # how many ticks the latest echo is behind the replay start

    for k in range(1, N):
        # Host advances one tick. The host applies the input it received
        # L ago. If the client changed input at tick k - L_ticks, the host
        # switches now.
        tick_k = k
        # The client's input at tick k - L_ticks (L ticks ago).
        l_ticks = int(round(L / TICK))
        client_input_ago = input_at(tick_k - l_ticks)
        # The host applies the client's input from L ticks ago.
        host_applied_input = client_input_ago
        host.update(TICK, host_applied_input)
        snap = host.snapshot()

        # Record the echo (the host's actual applied input, tagged with the
        # tick's start).
        echo_time = tick_k * TICK
        # The echo arrives at the client at echo_time + TICK + L. We'll
        # record it in the ghost's echo buffer when it "arrives."

        t_arrive = tick_k * TICK + L
        # Client frames up to the arrival (branch-2 advance). The ghost
        # predicts with the CURRENT local input (input_at(tick_k)).
        while t_local < t_arrive - 1e-9:
            t_next = min(t_local + FRAME, t_arrive)
            current_input = input_at(int(round(t_next / TICK)))
            g.advance(t_next - t_local, current_input, real_time=t_next,
                      host_time=ht, rtt_ow=rtt_ow)
            t_local = t_next

        # Record the client's own input (for the fallback buffer).
        est_now = ht.now(t_local)
        if est_now is not None:
            g.record_input(est_now, input_at(tick_k))

        # Arrival: feed the estimator, then reconcile. The echo for this
        # iteration is processed AFTER the snapshot (TCP ordering: the host
        # sends the snapshot first, then the echo), so at reconcile time the
        # latest echo in the buffer is from the PREVIOUS iteration.
        ht.record(t_local, tick_k * TICK)
        snap_time = tick_k * TICK
        now = snap_time + rtt_ow

        # Capture the ghost's predicted input (the current local input).
        predicted_input = input_at(tick_k)

        bx, by = g.ship.pos.x, g.ship.pos.y
        g.reconcile_rewind(snap, snap_time, now,
                           real_time=t_local, rate=ht.rate)
        snaps_pos.append(math.hypot(g.ship.pos.x - bx, g.ship.pos.y - by))

        # NOW record the echo (it arrives after the snapshot in the same
        # poll, so it's available for the NEXT reconcile, not this one).
        g.record_echo(echo_time, host_applied_input)

        # Which buffer did the replay use?
        buf_used = 'echo' if g._echo_buffer else 'own'
        buffer_used.append(buf_used)

        # The echo's staleness: how many ticks is the latest echo behind
        # the replay start (snap_time)?
        if g._echo_buffer:
            latest_echo_time = g._echo_buffer[-1][0]
            staleness = (snap_time - latest_echo_time) / TICK
        else:
            staleness = float('inf')
        echo_staleness.append(staleness)

        # The input divergence: did the ghost predict with a different
        # input than the replay used? (Approximate: compare the predicted
        # input to the latest echo's input.)
        if g._echo_buffer:
            replayed_input = g._echo_buffer[-1][1]
            diverged = (predicted_input.thrust_fwd != replayed_input.thrust_fwd)
        else:
            diverged = False
        input_divergence.append(1.0 if diverged else 0.0)

    # Report.
    print(f"rows={N - 1}")
    print(f"buffer used: echo={buffer_used.count('echo')}, "
          f"own={buffer_used.count('own')}")
    print(f"echo staleness (ticks behind replay start): "
          f"mean={sum(echo_staleness) / len(echo_staleness):.2f}, "
          f"max={max(echo_staleness):.2f}")
    print(f"input divergence (predicted != replayed): "
          f"{sum(input_divergence)}/{len(input_divergence)} rows")
    print()
    # Correlate snap size with input divergence.
    diverged_snaps = [s for s, d in zip(snaps_pos, input_divergence) if d]
    nondiv_snaps = [s for s, d in zip(snaps_pos, input_divergence) if not d]
    if diverged_snaps:
        print(f"snaps when input diverged: mean={sum(diverged_snaps) / len(diverged_snaps):.3f}px, "
              f"max={max(diverged_snaps):.3f}px, n={len(diverged_snaps)}")
    if nondiv_snaps:
        print(f"snaps when input matched:   mean={sum(nondiv_snaps) / len(nondiv_snaps):.3f}px, "
              f"max={max(nondiv_snaps):.3f}px, n={len(nondiv_snaps)}")
    print()
    print("per-row (row, snap_px, diverged, buffer, staleness):")
    for i in range(len(snaps_pos)):
        print(f"  {i:3d} snap={snaps_pos[i]:7.3f} div={input_divergence[i]:.0f} "
              f"buf={buffer_used[i]:4s} staleness={echo_staleness[i]:.2f}")

    pygame.quit()


def main_branch2_jitter():
    """Branch 2 + frame-timing jitter + network jitter + host sim rate jitter.

    Models the real-client timing with:
      - Client frame dt varying (the frame rate is not exactly 120 Hz).
      - Network delay varying (the snapshot's arrival time has jitter).
      - Host sim rate varying (the host's sim thread is slightly slower
        than real time, so the clock's rate is slightly < 1.0).
      - L = 0.0575 s (3.45 ticks), matching the real 2P CSV (delay ~0.115 s).

    Measures the snap size across each reconcile. The hypothesis: the
    0/8.667 sawtooth appears when the host's sim rate is slightly < 1.0,
    which causes the step during the interval to fire later, and sometimes
    after the next arrival (snap=8.667px).
    """
    import random
    random.seed(42)

    L = 0.0575  # one-way latency (s), matching the real 2P CSV
    N = 60
    FRAME_BASE = 1.0 / 120.0  # base frame dt
    FRAME_JITTER = 0.002  # frame dt jitter (s)
    NET_JITTER = 0.004  # network delay jitter (s)
    HOST_RATE_JITTER = 0.02  # host sim rate jitter (the host's sim is
                             # slightly slower than real time)

    pygame.init()
    thrust = ShipInput(thrust_fwd=1.0)

    host = Ship(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    host.vel = pygame.Vector2(0.0, 300.0)

    g = PredictedShip(hull=SILAS_HULL, loadout=default_loadout(SILAS_HULL))
    ht = HostTimeEstimator()
    rtt_ow = L

    # Snapshot 0.
    g.seed(host.snapshot(), snap_time=0.0, real_time=L)
    ht.record(L, 0.0)
    t_local = L

    snaps_pos = []
    n_replay = []

    for k in range(1, N):
        # Host advances one tick.
        host.update(TICK, thrust)
        snap = host.snapshot()

        # The host's sim rate is slightly < 1.0 (the host's sim thread is
        # slower than real time). The snapshot for tick k is generated at
        # host sim time k * TICK, which corresponds to local time
        # k * TICK / rate_h (later than k * TICK if rate_h < 1.0).
        rate_h = 1.0 + random.uniform(-HOST_RATE_JITTER, 0.0)
        net_delay = L + random.uniform(-NET_JITTER, NET_JITTER)
        t_arrive = k * TICK / rate_h + net_delay

        # Client frames up to the arrival (branch-2 advance). The frame dt
        # has jitter.
        while t_local < t_arrive - 1e-9:
            frame_dt = FRAME_BASE + random.uniform(-FRAME_JITTER, FRAME_JITTER)
            t_next = min(t_local + frame_dt, t_arrive)
            g.advance(t_next - t_local, thrust, real_time=t_next,
                      host_time=ht, rtt_ow=rtt_ow)
            t_local = t_next

        # Record the client's own input.
        est_now = ht.now(t_local)
        if est_now is not None:
            g.record_input(est_now, thrust)

        # Arrival: feed the estimator, then reconcile.
        ht.record(t_local, k * TICK)
        snap_time = k * TICK
        now = snap_time + rtt_ow

        bx, by = g.ship.pos.x, g.ship.pos.y
        g.reconcile_rewind(snap, snap_time, now,
                           real_time=t_local, rate=ht.rate)
        snaps_pos.append(math.hypot(g.ship.pos.x - bx, g.ship.pos.y - by))
        n_replay.append(g._last_replay_ticks)

    # Report.
    print(f"rows={N - 1}, L={L:.4f}s ({L / TICK:.2f} ticks)")
    print(f"frame jitter=±{FRAME_JITTER * 1000:.1f}ms, net jitter=±{NET_JITTER * 1000:.1f}ms")
    print()
    big = [(i, s) for i, s in enumerate(snaps_pos) if s > 1.0]
    print(f"snaps > 1px: {len(big)}/{len(snaps_pos)}")
    if big:
        print(f"max snap={max(snaps_pos):.3f}px, mean={sum(snaps_pos) / len(snaps_pos):.3f}px")
        print("big snaps (row, snap_px, n_replay):")
        for i, s in big[:20]:
            print(f"  {i:3d} snap={s:7.3f} n_replay={n_replay[i]}")
    else:
        print(f"max snap={max(snaps_pos):.3f}px, mean={sum(snaps_pos) / len(snaps_pos):.3f}px")
        print("  (no big snaps — the sawtooth did NOT appear)")
    print()
    print("replay length distribution:",
          {n: n_replay.count(n) for n in sorted(set(n_replay))})

    pygame.quit()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "branch2":
        main_branch2()
    elif len(sys.argv) > 1 and sys.argv[1] == "branch2-echo":
        main_branch2_echo()
    elif len(sys.argv) > 1 and sys.argv[1] == "branch2-jitter":
        main_branch2_jitter()
    else:
        main()