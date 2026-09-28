"""10.3b missile telemetry: capture the missile-id lifecycle on the CLIENT
so the double-draw (two missiles from one fire) can be diagnosed offline.

Why this exists
---------------
The client's prediction ghost fires its OWN missiles (GhostMissiles) and the
host's snapshot carries a BUFFER copy of the same missile. They are supposed
to be deduped by a shared id — (player_index, missile_seq) — so only ONE is
drawn. When the bug appears (two missiles from one fire: one hits + explodes,
the other flies through), the id match is failing somewhere in the chain:

    ghost fire  ->  reconcile (seq resync + rewind replay)  ->  handback
    (FIRE)        (RECONCILE)                                 (HANDBACK)
                                                                        |
                                                            predicted_view (DEDUP)

This module logs those four events so the failure point is VISIBLE in the CSV
instead of guessed at. It is CLIENT-SIDE ONLY: the host never runs the ghost,
so only the joining peer writes the file.

Enable
------
The client loop calls `set_enabled(game.debug_net)` each frame (F3 toggles
`debug_net`). While enabled, one CSV row is appended per event and flushed
immediately (so the file can be grabbed live). The file is `missile_debug.csv`
next to this module.

Reading the CSV
---------------
`t` is the wall clock (the join key — all rows are on the same client machine,
so `t` orders the whole lifecycle). `sim_t` is the host sim clock where known
(RECONCILE = the snapshot's stamp; DEDUP = the client's host-clock estimate)
for cross-machine correlation; empty where the ghost has no clock (FIRE,
HANDBACK). Ids are serialized as `player:seq` lists joined by `|` (e.g.
`1:0|1:1`). The `event` column discriminates the row's populated fields.
"""
import os
import time

_enabled = False
_f = None
_last_dedup = None
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "missile_debug.csv")

# One flat schema; each event populates its own columns, the rest stay empty.
_COLS = (
    "t", "event", "sim_t",
    # FIRE (ghost created a GhostMissile)
    "f_mid", "f_seq_before", "f_seq_after", "f_guard", "f_local_ids",
    # RECONCILE (snapshot applied + rewind replay ran)
    "r_snap_t", "r_snap_seq", "r_seq_before", "r_seq_after",
    "r_buf_ids", "r_local_before", "r_local_after",
    # HANDBACK (ghost missiles handed back to the buffer)
    "h_host_seq", "h_buf_ids", "h_before", "h_after", "h_culled", "h_jump",
    # DEDUP (a frame's predicted_view draw decision)
    "d_ghost_ids", "d_buf_drawn", "d_buf_skipped",
)
_HEADER = ",".join(_COLS)


def set_enabled(on):
    """Turn logging on/off. Opening the file (on enable) truncates it and
    writes the header; closing it (on disable) flushes + releases the handle."""
    global _enabled, _f, _last_dedup
    _enabled = bool(on)
    if _enabled and _f is None:
        _f = open(_path, "w", newline="")
        _f.write(_HEADER + "\n")
        _f.flush()
        _last_dedup = None   # a fresh session logs its first frame
    elif not _enabled and _f is not None:
        _f.close()
        _f = None


def close():
    """Flush + close on shutdown (idempotent)."""
    global _f
    if _f is not None:
        _f.close()
        _f = None


def fmt_ids(ids):
    """Serialize an iterable of (player, seq) ids as 'p:s|p:s' (compact)."""
    return "|".join("%d:%d" % (i[0], i[1]) for i in ids)


def log(event, sim_t=None, **fields):
    """Append one row. No-op when disabled. `fields` are named columns; any
    column not supplied is left empty. Values are stringified (ids should be
    pre-formatted via `fmt_ids`)."""
    if not _enabled or _f is None:
        return
    row = {c: "" for c in _COLS}
    row["t"] = "%.3f" % time.time()
    row["event"] = event
    if sim_t is not None:
        row["sim_t"] = "%.4f" % sim_t
    for k, v in fields.items():
        if k in row and v is not None:
            row[k] = v if isinstance(v, str) else str(v)
    _f.write(",".join(row[c] for c in _COLS) + "\n")
    _f.flush()


_last_dedup = None


def log_dedup(sim_t, ghost_ids, buf_drawn, buf_skipped):
    """DEDUP with change-detection: the render loop calls this every frame
    (60 Hz) while any missile is in flight, but only the TRANSITIONS matter
    (a ghost id appears, a buffer copy starts drawing, a skip begins). Log a
    row only when the (ghost, drawn, skipped) triple changes — this collapses
    a sustained flight into a handful of rows instead of a 60 Hz stream."""
    global _last_dedup
    key = (fmt_ids(ghost_ids), fmt_ids(buf_drawn), fmt_ids(buf_skipped))
    if key == _last_dedup:
        return
    _last_dedup = key
    log("DEDUP", sim_t=sim_t, d_ghost_ids=key[0], d_buf_drawn=key[1],
        d_buf_skipped=key[2])