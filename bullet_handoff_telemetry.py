"""Bullet HANDOFF-OFFSET telemetry, CLIENT-SIDE ONLY.

Diagnoses the "handoff is not seamless" symptom: when a ghost (locally
fired) bullet is handed off to the interpolation buffer, the buffer's copy
is drawn PULLED FORWARD by V*delay so that `buf_pos + V*delay == ghost_pos`
(a no-op visually). If that equality does NOT hold, the bullet JUMPS at the
handoff frame — the user sees it as the bullet "shifting magically".

This module logs, at each handoff frame, BOTH drawn positions and the
resulting offset, decomposed into:
  * off_par  — the component ALONG the bullet's velocity (signed, px).
               This is the TIME-DOMAIN mismatch: the ghost draws at its
               predicted clock (host_time + rtt_ow) while the buffer draws
               at the newest snapshot stamp. ghost_present != newest -> a
               parallel offset of ~V*(ghost_present - newest).
  * off_perp — the component PERPENDICULAR to the velocity (magnitude, px).
               This is the PREDICTION-ERROR / MUZZLE mismatch: the ghost's
               bullet line (muzzle + velocity at the fire tick) differs from
               the host's actual bullet line, so the two are parallel lines
               offset laterally. A significant off_perp means the ghost's
               ship pose (pos/angle) at the fire tick drifted from the
               host's actual pose.

One CSV row per handoff (a handful per burst), flushed immediately. Enable
with set_enabled(game.debug_net) each frame (F3 toggles debug_net), matching
the bullet_telemetry / bullet_time_telemetry pattern.
"""
import math
import os
import time

_enabled = False
_f = None
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "bullet_handoff_debug.csv")

_COLS = (
    "t", "bid",
    "ghost_x", "ghost_y",
    "buf_x", "buf_y",
    "off_x", "off_y", "off_mag", "off_par", "off_perp",
    "bvx", "bvy", "delay",
    "ship_x", "ship_y", "ship_ang", "ship_vx", "ship_vy",
)
_HEADER = ",".join(_COLS)


def set_enabled(on):
    """Turn logging on/off. Opening the file (on enable) truncates it and
    writes the header; closing it (on disable) flushes + releases the
    handle. Matches the bullet_telemetry / bullet_time_telemetry pattern."""
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


def log_handoff(bid, ghost_x, ghost_y, buf_x, buf_y, bvx, bvy, delay,
                ship=None):
    """Log one handoff: the ghost's drawn position, the buffer's drawn
    position (ALREADY pulled forward by V*delay), and the offset between
    them, decomposed into parallel (along V) and perpendicular components.

    `ship` is an optional (x, y, ang, vx, vy) tuple of the ghost's ship pose
    at the handoff frame (for correlating the offset with ship motion).
    """
    if not _enabled or _f is None:
        return
    off_x = ghost_x - buf_x
    off_y = ghost_y - buf_y
    off_mag = math.hypot(off_x, off_y)
    speed = math.hypot(bvx, bvy)
    if speed > 1e-6:
        # Unit vector along the bullet's velocity.
        ux, uy = bvx / speed, bvy / speed
        # Signed parallel component (px): + = ghost ahead of buffer.
        off_par = off_x * ux + off_y * uy
        # Perpendicular component (magnitude, px): the lateral offset.
        # off_perp_vec = off - off_par * u; |off_perp_vec| = sqrt(mag^2 - par^2).
        off_perp = max(0.0, off_mag * off_mag - off_par * off_par) ** 0.5
    else:
        off_par = 0.0
        off_perp = off_mag
    row = {c: "" for c in _COLS}
    row["t"] = "%.3f" % time.time()
    row["bid"] = _fmt_id(bid)
    row["ghost_x"], row["ghost_y"] = _fmt(ghost_x), _fmt(ghost_y)
    row["buf_x"], row["buf_y"] = _fmt(buf_x), _fmt(buf_y)
    row["off_x"], row["off_y"] = _fmt(off_x), _fmt(off_y)
    row["off_mag"] = _fmt(off_mag)
    row["off_par"] = _fmt(off_par)
    row["off_perp"] = _fmt(off_perp)
    row["bvx"], row["bvy"] = _fmt(bvx), _fmt(bvy)
    row["delay"] = _fmt(delay)
    if ship is not None:
        row["ship_x"], row["ship_y"], row["ship_ang"] = (
            _fmt(ship[0]), _fmt(ship[1]), _fmt(ship[2]))
        row["ship_vx"], row["ship_vy"] = _fmt(ship[3]), _fmt(ship[4])
    _f.write(",".join(row[c] for c in _COLS) + "\n")
    _f.flush()