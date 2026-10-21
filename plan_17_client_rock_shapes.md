# Plan 17 — Client-Side Asteroid Polygons (2P: draw real rocks, not circles)

**Status:** Session 1 (wire + buffer) DONE + VERIFIED. Session 2 (render + tests) pending.
**Scope:** the CLIENT's rendering of asteroids in 2P. The authoritative sim,
the ghost, the reconcile, and the host's render path are untouched. This is
wire + interpolation-buffer + presentation: it changes WHAT the client
receives per rock and HOW it draws it.

**Session fit:** two focused sessions (wire+buffer, then render+tests), each
comfortably inside a 64k context window.

---

## 1. The problem

In 2P the host draws each asteroid as its real filled+stroked polygon
(`Asteroid.draw` / `_draw_world_asteroid`, game.py:232 — 10-vertex rock shape,
rotated by `angle`). The client draws a **hardcoded 20 px circle**
(`_draw_remote_rock`, game.py:3866, called from `predicted_view` game.py:2630
and `remote_view` game.py:2283) — no shape, no size, no spin.

Root cause is deliberate: the wire PRUNES each rock to `[id, x, y]`
(`serialize_snapshot`, net.py:242 — ~545 B/rock -> ~23 B/rock), and
`interp_positions` (netcode.py:838) therefore only outputs `(x, y)` per rock.
The client has no `Asteroid` objects (the buffer is a pure presentation
store), so it has nothing to draw a polygon from.

## 2. Design

| Field | Wire | Interpolation | Notes |
|---|---|---|---|
| `id` | sent | — | stable identity (Asteroid._next_id, never reused in a run) |
| `x, y` | sent (1-dec, as today) | lerp | unchanged |
| `angle` | sent (3-dec) | `lerp_angle` | same wrapped-delta rule as ships/enemies; spin <= 2 rad/s makes even "take curr" fine, but lerp is free and consistent |
| `verts` | sent **only for rocks new to the stream** | — | shape is immutable for a rock's lifetime (split children get fresh ids) |
| `vel, spin, size` | NOT sent | — | nothing on the client reads them (size is derivable from verts if ever needed) |

**New wire entry:** `[id, x, y, angle, verts_or_None]` (replaces `[id, x, y]`).

### 2.1 Shape delta-compression (the key decision)

Verts are ~411 B/rock. Sending them every snapshot at 60 Hz would add
~25 KB/s. Instead:

- **Host** keeps a per-connection `RockShapeTracker` (a `set` of rock ids
  whose verts have already been sent). `serialize_snapshot(snap, tracker)`
  emits full `verts` for a rock whose id is new to the tracker, `None`
  otherwise, and records the id. The tracker is owned by the send worker
  (both send paths: net.py in-process worker ~L860 and net_process.py
  separate-process worker ~L191) and cleared on a full game reset (ids
  restart at 1 on a new run; the host then re-sends full verts and the
  client's registry entries are simply overwritten — no explicit invalidation).
- **Client** keeps `self.rock_shapes = {}` (id -> verts) on `Game`
  (init in `__init__`, cleared in `reset()`). It is populated in
  `push_snapshot` — the single seam where a received snapshot enters the
  client: for each rock in `snap[5]`, if `verts is not None` store it.
  The interpolation buffer stays a pure plain-snapshot store; the registry
  is presentation state owned by the client Game.

Bandwidth: +~411 B per NEW rock only (spawns are capped at 4/tick, splits
rare) — amortized ~1-2 KB/s. Steady state adds one angle float per rock
(~+0.5 KB/snapshot at ~85 rocks).

### 2.2 Interpolation buffer (netcode.py)

- `interp_positions`: asteroid entries become `(id, x, y, angle)` — x/y
  lerp exactly as today, angle via `lerp_angle(prev, curr, a)`. Membership
  by rock id and output order (curr's order) unchanged.
- `interp_positions` consumes the **wire format** (that is what production
  feeds it — `deserialize_snapshot` is a passthrough of the pruned form).
  Tests that currently push raw full sim snapshots must push
  `serialize_snapshot(snap)` instead (see 4.3 — this also fixes a test
  fidelity gap: they currently bypass the wire pruning entirely).
- Docstrings updated (netcode.py:697, :838, :925).

### 2.3 Client render (game.py)

- `_draw_remote_rock(screen, cam, x, y, angle, verts)`:
  - `verts` present -> call the EXISTING `_draw_world_asteroid(screen, cam,
    pos, angle, verts)` (game.py:232) — it takes plain data and mirrors
    `Asteroid.draw` exactly, so host/client parity is by construction.
  - `verts` missing (registry miss — should be impossible) -> fall back to
    the current 20 px circle (graceful degradation, never a crash).
- Both call sites (`predicted_view` ~L2630, `remote_view` ~L2283) unpack the
  4-tuple, look up `self.rock_shapes.get(id)`, and pass it through.
- Viewport culling unchanged: `_in_view(x, y, bounds)` is center-based and
  `CULL_MARGIN` (80) exceeds the max rock radius (50), same as the host.

### 2.4 What does NOT change

- Host sim + host render (model path, `_draw_world_asteroid` consumers).
- Ghost / reconcile / dedup / handoff: the ghost's laser targets come from
  `pos['enemies']` proxies only — beams at rocks are host-side and reach the
  client as T_BEAM events with a frozen world endpoint. Asteroid entries are
  not read by the ghost at all.
- `Asteroid.snapshot()` (the sim-side 9-tuple) and `apply_snapshot` — the
  wire format is a serialization concern only.
- Single-player and the test range (host render path).

## 3. Work breakdown

### Session 1 — wire + buffer + render (DONE + VERIFIED)

**Implementation note (deviation from the original draft):** instead of
rewriting ~20 test files to serialize snapshots before pushing them,
`interp_positions` and `push_snapshot` were made **tolerant of both rock
formats** (sim 9-tuple and wire 5-tuple, detected by `len` — the two forms
differ only in which fields the wire keeps). This is well-justified: the
buffer is a presentation store fed either the authoritative sim snapshot
(loopback/tests) or the pruned wire snapshot (the network). The
`test_viewport_culling` change (original item 6) is therefore NOT needed —
its raw-snapshot path still works and now exercises the registry.

1. **net.py**: `RockShapeTracker` (per-connection `set` of sent rock ids);
   `serialize_snapshot(snap, tracker=None)` emits `[id, x, y, angle,
   verts_or_None]` (x/y 1-dec, angle 3-dec, verts full-precision on first
   sight of an id, None after). `NetWorker` owns a tracker; the module
   self-test now checks the new shape, first-sight verts, and delta
   compression (repeated id -> None, fresh id -> full verts).
2. **net_process.py**: owns a `RockShapeTracker` (2P-only, one client per
   process) and passes it to `serialize_snapshot`.
3. **netcode.py**: `interp_positions` asteroid entries -> `(id, x, y,
   angle)`; x/y lerp'd, angle via `lerp_angle` (wrapped delta, same rule as
   ships/enemies). Reads the rock in either form (`_rock_angle`: `[6]` sim /
   `[3]` wire, by `len`). Docstrings updated.
4. **game.py**: `self.rock_shapes = {}` (id -> verts) in `__init__` +
   `reset()`; populated in `push_snapshot` (verts at `a[-1]`, works for both
   forms). `_draw_remote_rock(screen, x, y, angle, verts)` now draws the
   real polygon via `_draw_world_asteroid` (the host's own plain-data
   helper — parity by construction), falling back to the old 20 px circle on
   a registry miss. Both call sites (`predicted_view`, `remote_view`) unpack
   the 4-tuple and look up the registry.
5. **test_interpolation.py**: the independent `id_oracle` now emits
   `(id, x, y, angle)` per rock (x/y via `mix`, angle via
   `mix_enemy_angle`, both reading either form) — the alpha battery +
   turnover stress validate the new entries.
6. **test_net_process.py**: `_fake_snapshot` now provides SIM 9-tuple rocks
   (the net process serializes them); the pruned-wire assertion checks the
   5-tuple + first-sight verts.

**Verification:** net self-test, test_interpolation, test_snapshot,
test_render_model, test_viewport_culling, test_net_process, test_sim_thread,
test_sim_process, and the 10.x/14.x/15.x client-path tests all pass.
Pre-existing failures (verified by stashing the changes): test_10_3a
(beam-origin re-anchor), test_10_9 (ghost lerp/advance), test_network
(`__main__.Host` import) — all unrelated to this work.

### Session 2 — tests + docs (pending)

7. **New test** (`test_17_rock_shapes.py`):
   - wire: first snapshot carries verts, second carries None; ids never
     re-sent; reset re-sends.
   - interp: angle lerps across the window (incl. a wrap case); x/y
     behavior identical to today.
   - pixel parity: one static rock (vel=0, spin=0), locked camera at the
     same world point — host frame vs client frame pixel diff == 0
     (same method as test_viewport_culling's pixel-diff harness).
8. **Docs**: ARCHITECTURE.md wire line (L86: `[id, x, y]` -> new shape),
   PERFORMANCE.md bandwidth note.
9. **Manual 2P check**: client sees spinning polygons matching the host;
   split children pop in with correct shapes; F3 snap-size log shows no
   steady-state growth.

## 4. Deferred follow-up (separate session, agreed 2026-07)

**Ghost rock collision.** The client's prediction ghost does no asteroid
collision (today or after this plan) — a predicted clip-through is
corrected by the next reconcile (~20-40 ms LAN). Real polygons make that
pre-existing artifact more visible. If it bothers in playtesting: add
rock collision to the ghost's step (buffer rocks, same swept-hull-vs-
circle test as the sim; the ghost may only REACT to a hit, never resolve
a split — splits stay host-authoritative). Explicitly out of scope here.

## 5. Risks / edge cases

- **Angle wrap** at the lerp: handled by `lerp_angle` (same as ships).
- **Registry growth**: dead rocks' verts linger (ids never reused). ~160 B
  per dead rock — negligible, but optionally prune ids absent from the
  latest snapshot on each push (one set rebuild).
- **Format split (sim 9-tuple vs wire 5-tuple)**: `interp_positions` must
  only ever see the wire form. Any test pushing raw `Game.snapshot()` into
  `push_snapshot` breaks silently (index 3 is vel.x, not angle) — the
  Session-2 test updates exist to catch exactly this.
- **RockShapeTracker lifetime**: must be per-connection (a reconnecting
  client starts fresh) and cleared on game reset. Both live in the worker
  that owns the connection — keep it there, not in `Game`.
- **Backward compatibility**: LAN, both peers run the same codebase — no
  version negotiation needed.
- **No new message type**: shapes ride the existing snapshot stream, so
  ordering/reliability come for free (TCP) and a client can resolve every
  rock from the snapshot stream alone.