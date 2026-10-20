"""Client-side bullet-impact (option A) trigger telemetry.

Diagnoses "no particle effects on bullet collisions when the client player
hits enemies with bullets." The option-A trigger (game.predicted_view)
spawns a shield_burst spark + culls the bullet when a PRESENT-time local
bullet's per-frame segment comes within a remote target's drawn collision
radius. This module makes the trigger's INPUTS and DECISIONS visible
instead of guessed at, so a single playtest shows WHY no spark fires.

The prime suspect it is built to confirm/refute: the trigger builds its
target list from `pos['ships']` ONLY (the other PLAYER, skipping the local
index) and does NOT test `pos['enemies']` (AI enemies). If the client is
shooting AI enemies (the normal mode), the target list is empty and the
spark can never fire. The FRAME row records, per frame, the closest any
local bullet came to (a) a remote player ship and (b) an AI enemy — so if
`n_ships==0` / `min_d_ship` is large while `min_d_enemy` is small (and
`enemy_hit==1`), the missing-enemy-target bug is confirmed.

Two row types (long format, one flat schema):
  FRAME  one row per frame that has >=1 local bullet in flight (ghost or
         buffer) or fired a hit this frame. Carries the trigger's inputs
         (counts) + the closest-approach geometry to ships vs enemies.
  HIT    one row per spark actually spawned (the trigger fired). Carries
         the bullet id, which source (ghost/buffer), the target, the impact
         point, and the distance at impact (should be <= the target radius).

Key columns (FRAME):
  n_ghost     ghost local bullets tested (present-time, b.pos)
  n_buf       buffer local bullets tested (pulled to the present)
  n_ships     remote PLAYER ships in the target list (alive, non-local)
  n_enemies   AI enemies in the buffer (NOT tested by the trigger)
  min_d_ship  min segment->center distance to any remote ship ("" if none)
  min_r_ship  that ship's collision radius
  ship_hit    1 if min_d_ship <= min_r_ship (the trigger WOULD fire)
  min_d_enemy min segment->center distance to any AI enemy ("" if none)
  min_r_enemy that enemy's collision radius
  enemy_hit   1 if min_d_enemy <= min_r_enemy (a hit the trigger MISSED)
  cull_n      live entries in the persistent cull set
  delay       the adaptive interpolation delay (context)
  D           the pull-forward gap _last_step_time - rp (context)

Enable: the client loop calls set_enabled(game.debug_net) each frame (F3
toggles debug_net). While enabled, rows are appended + flushed immediately
(so the file can be grabbed live). The file is hit_fx_debug.csv next to
this module.
"""
import os
import time

_enabled = False
_f = None
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "hit_fx_debug.csv")

_COLS = (
    "t", "event",
    # FRAME
    "n_ghost", "n_buf", "n_ships", "n_enemies",
    "min_d_ship", "min_r_ship", "ship_hit",
    "min_d_enemy", "min_r_enemy", "enemy_hit",
    "cull_n", "delay", "D",
    # FRAME (15.3: drawn-position distance — the segment/draw mismatch
    # signal). min_ddrawn = min distance from a bullet's DRAWN position
    # (the trigger segment's endpoint) to an enemy center; min_ddrawn_src
    # is that bullet's source (ghost/buf). If min_ddrawn is small (the
    # drawn bullet is visually on the enemy) while min_d_enemy (the swept
    # segment) is large, the segment does not cover the draw.
    "min_ddrawn", "min_ddrawn_src",
    # HIT
    "bid", "src", "tgt", "hx", "hy", "d",
    # HIT (shield decision inputs — 10.13b: "shield up but no shield FX")
    "es_charge", "has_comp", "s_up", "oval_a", "oval_b", "hull_r",
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
    return "%.2f" % v


def _fmt_id(bid):
    if bid is None:
        return ""
    try:
        return "%d:%d" % (bid[0], bid[1])
    except (TypeError, IndexError):
        return str(bid)


def seg_point_dist(x0, y0, x1, y1, cx, cy):
    """Minimum distance from the segment (p0 -> p1) to the point (cx, cy).

    Pure math (no pygame) — the same closest-point-on-segment projection the
    trigger's `_seg_circle_hit` uses, but returning the distance so the
    telemetry can measure "how close did the bullet get" without firing. A
    hit occurs when this distance <= the target's collision radius."""
    dx = x1 - x0
    dy = y1 - y0
    fx = x0 - cx
    fy = y0 - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return (fx * fx + fy * fy) ** 0.5
    t = -(fx * dx + fy * dy) / a
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    hx = x0 + t * dx - cx
    hy = y0 + t * dy - cy
    return (hx * hx + hy * hy) ** 0.5


def log_frame(n_ghost, n_buf, n_ships, n_enemies,
              min_ship, min_enemy, cull_n, delay, D,
              mindrawn=None):
    """Log one frame's trigger inputs + closest-approach geometry.

    `min_ship` / `min_enemy` are (dist, radius, label) tuples for the
    closest target of each kind, or None when there is no target of that
    kind. `label` is the ship index or the enemy tag (context only — not
    written, the counts + distances are the diagnostic). `mindrawn` is a
    (dist, src, label) tuple: the min distance from a bullet's DRAWN
    position (the trigger segment's endpoint) to an enemy center — the
    segment/draw mismatch signal (15.3). No-op when disabled.
    """
    if not _enabled or _f is None:
        return
    # Only log frames that have something to diagnose: a local bullet in
    # flight, or a target present. (A frame with neither is pure noise.)
    if (n_ghost + n_buf) == 0 and (n_ships + n_enemies) == 0:
        return
    ms_d, ms_r, _ = (min_ship if min_ship is not None
                     else (None, None, None))
    me_d, me_r, _ = (min_enemy if min_enemy is not None
                     else (None, None, None))
    md_d, md_src, _ = (mindrawn if mindrawn is not None
                       else (None, None, None))
    row = {c: "" for c in _COLS}
    row["t"] = "%.3f" % time.time()
    row["event"] = "FRAME"
    row["n_ghost"] = str(n_ghost)
    row["n_buf"] = str(n_buf)
    row["n_ships"] = str(n_ships)
    row["n_enemies"] = str(n_enemies)
    row["min_d_ship"] = _fmt(ms_d)
    row["min_r_ship"] = _fmt(ms_r)
    row["ship_hit"] = ("" if ms_d is None or ms_r is None
                       else str(1 if ms_d <= ms_r else 0))
    row["min_d_enemy"] = _fmt(me_d)
    row["min_r_enemy"] = _fmt(me_r)
    row["enemy_hit"] = ("" if me_d is None or me_r is None
                        else str(1 if me_d <= me_r else 0))
    row["cull_n"] = str(cull_n)
    row["delay"] = _fmt(delay)
    row["D"] = _fmt(D)
    row["min_ddrawn"] = _fmt(md_d)
    row["min_ddrawn_src"] = ("" if md_src is None else str(md_src))
    _f.write(",".join(row[c] for c in _COLS) + "\n")
    _f.flush()


def log_hit(bid, src, tgt, hx, hy, d,
            es_charge=None, has_comp=None, s_up=None,
            oval_a=None, oval_b=None, hull_r=None):
    """Log one spark the trigger actually spawned. `src` is 'ghost' or
    'buf' (which bullet source fired it); `tgt` is the target label (ship
    index or enemy tag); (hx, hy) is the impact point; `d` is the distance
    at impact (should be <= the target radius).

    The shield-decision inputs (10.13b) let a playtest show WHY a hit took
    the hull path instead of the shield path when the shield is up:
      es_charge  the buffer's enemy shield_charge (the wire value)
      has_comp   1 if the stand-in's ship has a shield component
      s_up       the computed shield-up flag (has_comp and es_charge>0)
      oval_a/b   the stand-in's shield oval semi-axes (the circle approx
                 uses max(a,b))
      hull_r     the stand-in's hull collision radius (the shield-down test)
    Ship hits (the other player) leave these blank. No-op when disabled."""
    if not _enabled or _f is None:
        return
    row = {c: "" for c in _COLS}
    row["t"] = "%.3f" % time.time()
    row["event"] = "HIT"
    row["bid"] = _fmt_id(bid)
    row["src"] = src
    row["tgt"] = tgt
    row["hx"] = _fmt(hx)
    row["hy"] = _fmt(hy)
    row["d"] = _fmt(d)
    row["es_charge"] = _fmt(es_charge)
    row["has_comp"] = ("" if has_comp is None else str(int(has_comp)))
    row["s_up"] = ("" if s_up is None else str(int(s_up)))
    row["oval_a"] = _fmt(oval_a)
    row["oval_b"] = _fmt(oval_b)
    row["hull_r"] = _fmt(hull_r)
    _f.write(",".join(row[c] for c in _COLS) + "\n")
    _f.flush()