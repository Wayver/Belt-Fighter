"""Per-frame ship-vs-bullet TIME-DOMAIN telemetry, CLIENT-SIDE ONLY.

Diagnoses the "bullets fire from empty space / appear frozen then fly off"
symptom. The root hypothesis: the local ship is drawn at the ghost's
PREDICTED clock (_sim_time ~= host clock + rtt_ow, i.e. "now"), while the
ghost's own bullets are drawn at the RENDER POINT (rp = newest snapshot -
adaptive delay, i.e. "now - delay"). The gap D = _last_step_time - rp
(~delay + rtt_ow, 0.1-0.37 s) means a bullet younger than D is drawn pinned
at the muzzle (b.pos - b.vel*age), so it trails the ship by v_ship * D while
the ship moves. This module makes that gap and the trailing measurable
instead of eyeballed.

One CSV row per (frame, bullet) — LONG format — plus a ship-only row when no
bullet is live (so the ship's trajectory is captured even between bursts).
The ship state (drawn pos + vel + clocks) is repeated on every row so the
analyzer can correlate a bullet's drawn position against the ship's drawn
position on the SAME frame.

Key columns:
  sim_t    the ghost's predicted clock (the ship's time domain)
  rp       the render point (the bullets'/enemies' time domain)
  D        sim_t-ish gap = ghost._last_step_time - rp (the freeze window)
  delay    the adaptive interpolation delay
  rtt_ow   the one-way latency estimate
  ship_*   the ship's DRAWN pose (gpos) + vel
  bid      the bullet's (player_index, seq) id, "p:s"
  bx,by    the bullet's DRAWN position (what the player sees)
  bvx,bvy  the bullet's velocity (ship_vel + barrel_vel after the muzzle fix)
  bage     the bullet's age (BULLET_LIFE - life)
  brelx, brely  bullet drawn pos RELATIVE to the ship drawn pos (the trailing
                vector — behind the ship = opposite the ship's velocity)
  bdship   |brel| — the trailing magnitude (px)

Enable: the client loop calls set_enabled(game.debug_net) each frame (F3
toggles debug_net). While enabled, one row is appended per (frame, bullet)
and flushed immediately (so the file can be grabbed live). The analyzer is
tools/analyze_bullet_time.py.
"""
import os
import time

_enabled = False
_f = None
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "bullet_time_debug.csv")

_COLS = (
    "t", "event", "sim_t", "rp", "D", "delay", "rtt_ow",
    "ship_x", "ship_y", "ship_vx", "ship_vy",
    "bid", "bx", "by", "bvx", "bvy", "bage",
    "brelx", "brely", "bdship",
)
_HEADER = ",".join(_COLS)


def set_enabled(on):
    """Turn logging on/off. Opening the file (on enable) truncates it and
    writes the header; closing it (on disable) flushes + releases the
    handle. Matches the bullet_telemetry / missile_telemetry pattern."""
    global _enabled, _f
    _enabled = bool(on)
    if _enabled and _f is None:
        _f = open(_path, "w", newline="")
        _f.write(_HEADER + "\n")
        _f.flush()
    elif not _enabled and _f is not None:
        _f.close()
        _f = None


def close():
    """Flush + close on shutdown (idempotent)."""
    global _f
    if _f is not None:
        _f.close()
        _f = None


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    return "%.3f" % v


def _fmt_id(bid):
    if bid is None:
        return ""
    try:
        return "%d:%d" % (bid[0], bid[1])
    except (TypeError, IndexError):
        return str(bid)


def log_frame(game, gpos, rp, D, bullets):
    """Log one frame's ship state + each ghost bullet's DRAWN state.

    `game` is the Game (read for latency.delay, _rtt_ow, ghost._sim_time,
    ghost.ship.vel). `gpos` is the ship's DRAWN pose (the smoothed render
    pose). `rp` is the render point. `D` is the freeze-window gap
    (ghost._last_step_time - rp). `bullets` is a list of
    (id, bx, by, bvx, bvy, age) — the DRAWN position (bx,by) is what the
    player sees (b.pos - b.vel*t_off), not the raw b.pos.

    Writes one row per bullet (event=BUL); if there are no bullets, writes a
    single ship-only row (event=SHIP) so the ship's trajectory is still
    captured between bursts. No-op when disabled.
    """
    if not _enabled or _f is None:
        return
    g = game.ghost
    sim_t = g._sim_time
    delay = getattr(game.latency, "delay", 0.0)
    rtt_ow = getattr(game, "_rtt_ow", 0.0)
    svx, svy = g.ship.vel.x, g.ship.vel.y
    wall = "%.3f" % time.time()
    ship = (wall, sim_t, rp, D, delay, rtt_ow,
            gpos.x, gpos.y, svx, svy)

    if not bullets:
        row = {c: "" for c in _COLS}
        row["t"], row["event"] = wall, "SHIP"
        for c, v in zip(("sim_t", "rp", "D", "delay", "rtt_ow",
                         "ship_x", "ship_y", "ship_vx", "ship_vy"), ship[1:]):
            row[c] = _fmt(v)
        _f.write(",".join(row[c] for c in _COLS) + "\n")
        _f.flush()
        return

    for (bid, bx, by, bvx, bvy, age) in bullets:
        brelx = bx - gpos.x
        brely = by - gpos.y
        bdship = (brelx * brelx + brely * brely) ** 0.5
        row = {c: "" for c in _COLS}
        row["t"], row["event"] = wall, "BUL"
        for c, v in zip(("sim_t", "rp", "D", "delay", "rtt_ow",
                         "ship_x", "ship_y", "ship_vx", "ship_vy"), ship[1:]):
            row[c] = _fmt(v)
        row["bid"] = _fmt_id(bid)
        row["bx"], row["by"] = _fmt(bx), _fmt(by)
        row["bvx"], row["bvy"] = _fmt(bvx), _fmt(bvy)
        row["bage"] = _fmt(age)
        row["brelx"], row["brely"], row["bdship"] = (
            _fmt(brelx), _fmt(brely), _fmt(bdship))
        _f.write(",".join(row[c] for c in _COLS) + "\n")
    _f.flush()