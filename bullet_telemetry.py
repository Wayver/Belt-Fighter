"""Bullet-id lifecycle + fire-drift telemetry, CLIENT-SIDE ONLY (the host
never runs the ghost, so only the joining peer writes the file).

Phase 0 of the bullet-cap foundation fix (plan_14.md). The hypothesis to
confirm: the host gates fire on a WORLD bullet count (`len(self.bullets) >=
MAX_BULLETS`), but the ghost gates on its LOCAL-DRAW count
(`len(self.local_bullets) < MAX_BULLETS`) and bumps its gun cooldown /
`bullet_seq` even when its cap binds. So when the host's cap binds (host does
NOT fire), the ghost's fire decision diverges -> the ghost's `bullet_seq` and
gun cooldown drift from the host's -> the ghost fires a shot the host never
fired -> its id `(local_index, seq)` matches no buffer bullet -> the id-dedup
fails -> BOTH the ghost's and the host's bullet are drawn (the "stream from
empty space").

This module logs the bullet lifecycle to `bullet_debug.csv` (next to this
module) so the drift is visible instead of guessed at:

    ghost fire (FIRE) -> reconcile (RECONCILE) -> handback (HANDBACK)
    -> predicted_view (DEDUP)

The RECONCILE row is the key one: it records the ghost's `bullet_seq` and gun
cooldown BEFORE/AFTER the reconcile next to the host's authoritative
`bullet_seq` (ship snapshot field 24) and gun cooldown (snapshot weapons
tuple). A growing (ghost_seq - host_seq) gap is the drift; a DEDUP row where a
bullet id is in BOTH `d_ghost_bids` and `d_buf_bdrawn` is a double-draw.

Enable: the client loop calls `set_enabled(game.debug_net)` each frame (F3
toggles `debug_net`). While enabled, one CSV row is appended per event and
flushed immediately (so the file can be grabbed live).

Reading the CSV: `t` is the wall clock (the join key — all rows are on the
same client machine, so `t` orders the whole lifecycle). `sim_t` is the host
sim clock where known (RECONCILE = the snapshot's stamp; DEDUP = the client's
host-clock estimate) for cross-machine correlation; empty where the ghost has
no clock (FIRE, HANDBACK). Ids are serialized as `player:seq` lists joined by
`|` (e.g. `1:0|1:1`). The `event` column discriminates the row's populated
fields.
"""
import os
import time

_enabled = False
_f = None
_last_dedup = None
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "bullet_debug.csv")

# One flat schema; each event populates its own columns, the rest stay empty.
_COLS = (
    "t", "event", "sim_t",
    # FIRE (ghost created a Bullet)
    "f_bid", "f_seq_before", "f_seq_after", "f_guard", "f_local_ids",
    "f_cap_before",
    # RECONCILE (snapshot applied + rewind replay ran)
    "r_snap_t", "r_snap_bseq", "r_gseq_before", "r_gseq_after",
    "r_glocal_ids", "r_buf_bids", "r_gcd", "r_host_cd",
    # HANDBACK (ghost bullets marked pending)
    "h_host_bseq", "h_buf_bids", "h_before", "h_after", "h_culled",
    # DEDUP (a frame's predicted_view draw decision, bullet-only)
    "d_ghost_bids", "d_buf_bdrawn", "d_buf_bskipped",
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


def log_dedup(sim_t, ghost_bids, buf_bdrawn, buf_bskipped):
    """DEDUP with change-detection: the render loop calls this every frame
    (60 Hz) while any bullet is in flight, but only the TRANSITIONS matter
    (a ghost id appears, a buffer copy starts drawing, a skip begins). Log a
    row only when the (ghost, drawn, skipped) triple changes — this collapses
    a sustained flight into a handful of rows instead of a 60 Hz stream.

    A DOUBLE-DRAW is a bullet id present in BOTH `ghost_bids` (the ghost is
    drawing it) and `buf_bdrawn` (the buffer is ALSO drawing it) — the id-dedup
    failed. The analyzer flags those."""
    global _last_dedup
    key = (fmt_ids(ghost_bids), fmt_ids(buf_bdrawn), fmt_ids(buf_bskipped))
    if key == _last_dedup:
        return
    _last_dedup = key
    log("DEDUP", sim_t=sim_t, d_ghost_bids=key[0], d_buf_bdrawn=key[1],
        d_buf_bskipped=key[2])