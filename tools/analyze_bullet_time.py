"""Analyze bullet_time_debug.csv (the ship-vs-bullet time-domain capture).

Reads the long-format CSV written by bullet_time_telemetry (F3-gated,
client-only) and reconstructs, per bullet, the timeline of its DRAWN
position relative to the ship's DRAWN position:

  * the FREEZE phase (age < D): the bullet is pinned at the muzzle, so
    its position relative to the ship = the ship's displacement since the
    shot fired (growing at the ship's speed). This is the "firing from
    empty space" trailing.
  * the FLIGHT phase (age >= D): the bullet flies at full speed; the
    relative position shrinks at (bullet_speed - ship_speed) along the
    barrel direction.

Usage:
  cd /mnt && python -m ship5.tools.analyze_bullet_time [csv_path] [max_bullets]

Defaults: csv next to the package (bullet_time_debug.csv), first 3 bullets.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(HERE), "bullet_time_debug.csv")


def _f(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def group_by_bullet(rows):
    """bid -> list of BUL rows (ordered by t). Also returns SHIP rows."""
    bullets = {}
    ships = []
    for r in rows:
        if r["event"] == "BUL":
            bullets.setdefault(r["bid"], []).append(r)
        elif r["event"] == "SHIP":
            ships.append(r)
    for bid in bullets:
        bullets[bid].sort(key=lambda r: r["t"])
    return bullets, ships


def summarize(rows):
    """One summary line per bullet.

    The 'empty space' gap is the distance the bullet is BEHIND the ship
    during the FREEZE phase (age < D): the bullet is pinned at the muzzle
    while the ship moves away, so |brel| during freeze = the ship's
    displacement since the shot fired. We report the max |brel| over the
    freeze-phase rows only (not the whole flight — the bullet later flies
    AHEAD of the ship, which would dominate a plain max)."""
    out = []
    for bid, rs in rows.items():
        if not rs:
            continue
        freeze = [r for r in rs if _f(r["bage"]) < _f(r["D"])]
        pool = freeze if freeze else rs
        best = max(pool, key=lambda r: _f(r["bdship"]))
        out.append((bid, best, len(freeze)))
    out.sort(key=lambda p: p[1]["t"])
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    max_b = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if not os.path.exists(path):
        print("No %s — run a 2P session with F3 (debug_net) on the CLIENT,"
              % os.path.basename(path))
        print("move laterally, stop, fire, then ESC. Re-run this analyzer.")
        return
    rows = load(path)
    if not rows:
        print("CSV is empty — was F3 on while firing?")
        return
    bullets, ships = group_by_bullet(rows)
    print("=== bullet_time_debug.csv: %d rows, %d bullets, %d ship-only rows"
          % (len(rows), len(bullets), len(ships)))
    if not bullets:
        print("No BUL rows — no bullets were in flight while F3 was on.")
        return

    # Global clock context (the gap D + delay + rtt_ow, sampled over time).
    Ds = [_f(r["D"]) for r in rows if r["event"] == "BUL"]
    delays = [_f(r["delay"]) for r in rows if r["event"] == "BUL"]
    rtt = [_f(r["rtt_ow"]) for r in rows if r["event"] == "BUL"]
    if Ds:
        print("D (freeze window)   : min %.3f  max %.3f  s" % (min(Ds), max(Ds)))
        print("adaptive delay      : min %.3f  max %.3f  s" % (min(delays), max(delays)))
        print("rtt_ow              : min %.3f  max %.3f  s" % (min(rtt), max(rtt)))
        print()

    # Per-bullet summary (max trailing during the freeze phase).
    print("=== per-bullet summary (max trailing while FROZEN = the 'empty space' gap) ===")
    for bid, best, n_freeze in summarize(bullets):
        sv = (_f(best["ship_vx"]) ** 2 + _f(best["ship_vy"]) ** 2) ** 0.5
        print("  %s  max_trailing=%6.1f px  at t=%s  ship_speed=%5.1f px/s  "
              "age=%.3f  D=%.3f  freeze_frames=%d  rel=(%+.1f, %+.1f)"
              % (bid, _f(best["bdship"]), best["t"], sv,
                 _f(best["bage"]), _f(best["D"]), n_freeze,
                 _f(best["brelx"]), _f(best["brely"])))
    print()

    # Per-bullet timeline (first max_b bullets).
    for bid in list(bullets)[:max_b]:
        rs = bullets[bid]
        print("=== bullet %s  (%d frames, age %.3f -> %.3f) ==="
              % (bid, len(rs), _f(rs[0]["bage"]), _f(rs[-1]["bage"])))
        print("  t        age    freeze  rel_x    rel_y   dist   ship_vx  ship_vy  "
              "ship_pos")
        t0 = rs[0]["t"]
        for r in rs:
            age = _f(r["bage"])
            D = _f(r["D"])
            freeze = "FROZEN" if age < D else "flight"
            print("  %+.3f  %5.3f   %-6s  %+7.1f  %+7.1f  %6.1f  %+7.1f  %+7.1f  "
                  "(%7.1f, %7.1f)"
                  % (_f(r["t"]) - _f(t0), age, freeze,
                     _f(r["brelx"]), _f(r["brely"]), _f(r["bdship"]),
                     _f(r["ship_vx"]), _f(r["ship_vy"]),
                     _f(r["ship_x"]), _f(r["ship_y"])))
        print()

    print("Reading the summary:")
    print("  * max_trailing ~ ship_speed * D  => the freeze-window trailing")
    print("    (the 'firing from empty space' distance). If it is ~one hull")
    print("    length, the diagnosis is confirmed.")
    print("  * In the timeline, rel should GROW during FROZEN (the ship")
    print("    leaves the muzzle) and SHRINK during flight (the bullet")
    print("    outruns the ship).")
    print("  * If rel is ~0 the whole flight, the bullet is NOT trailing")
    print("    (the bug is elsewhere — e.g. a velocity/direction issue).")


if __name__ == "__main__":
    main()