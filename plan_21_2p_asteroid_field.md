# Plan 21 — 2P Asteroid Field (per-player regions + wire culling)

**Status:** PLANNED. Three sessions, each independently testable and
comfortably inside a 64k context window. Sessions are strictly ordered:
S1 fixes the field, S2 adds the wire cull, S3 integrates + verifies + docs.

**Scope:** host-side `spawning.update_field` + `config.py` + `net.py`
(serialize_snapshot) + `net_process.py` + one new test per session.
**Out of scope (later phases):** deterministic per-chunk generation, stable
chunk-derived rock ids, sleep/wake chunk store, density re-tuning, binary
wire format, rock snapshot rate reduction.

---

## 1. The problem (confirmed by reading the code + headless 2P sims)

The field (`spawning.update_field`) is a **single centroid-anchored blob**:

1. It computes the **centroid of all alive players**.
2. It keeps a **5×5 block of sectors** (Chebyshev radius `ACTIVE_SECTORS = 2`)
   around the **centroid** topped up to `ROCKS_PER_SECTOR = 3`.
3. It **culls** any rock farther than `DESPAWN_RADIUS = 2600` px from the
   **centroid**.

In 2P the centroid sits *between* the two players, which produces three
visible bugs:

- **Off-center / sparse field.** With players 1500 px apart, each player's
  local sector region is **disjoint** from the centroid's region. The top-up
  loop only fills the centroid's region, so the field is centered on the
  *midpoint* between the players, not on each player. Each player's own sector
  is under-populated and the field is biased toward the middle.
- **Wink-out.** The cull radius is measured from the centroid, not from each
  player. As the centroid moves, rocks near a player's screen edge get culled
  while still visible → they vanish.
- **Flying across / pop-in.** Fresh rocks spawn in the active region and drift
  in at ~0.6–0.8 new rocks/s per player; the field also *reflows* as the
  centroid moves, so the visible set is always churning.

**Latent geometry bug:** the active spawn region (5×5 sectors, corners
~3535 px from the anchor) is **larger** than the cull radius (2600 px). Rocks
spawn in the far corners and are culled the very next tick — pure waste and a
source of edge flicker.

**Bandwidth:** the host sends **all** live rocks in every 60 Hz snapshot
(~96 rocks × ~30 B × 60 Hz ≈ 165 KB/s for the 3×3 field). Most of those rocks
are off-screen for the client and just burn bandwidth.

### Root cause (one line)
The field is a *single centroid-anchored blob with a hard cull radius and no
per-player anchoring*, and the wire sends *all* rocks regardless of what the
client can see. Fine for 1P (centroid == the player), broken for 2P.

---

## 2. Design overview

Two independent changes, each a separate session:

### Change A — Per-player field (Session 1)

Replace the single centroid-anchored region with the **union of a 3×3 region
around each alive player** (`ACTIVE_SECTORS = 1`). Replace the centroid cull
with a **per-player minimum-distance** cull (`FIELD_KEEP_RADIUS = 2200`).

> **Why 3×3, not 5×5 (bandwidth):** a 5×5 region is 5000×5000 px but the
> screen is only 1920×1280 — most of those rocks are off-screen and just burn
> snapshot bandwidth. A 3×3 region (3000×3000 px) fully covers the screen with
> a 540 px buffer (half-extent 1500 px vs. screen half-width 960 px), which
> also absorbs the camera's look-ahead lead (~312 px at max speed).
>
> Measured steady-state (50 s sim, wire = total rocks × ~30 B × 60 Hz;
> on-screen = time-averaged count within the 1920×1280 rect):
>
> | Config | 1P total | 1P on-screen (mean) | 2P 4000px total | 2P wire |
> |---|---|---|---|---|
> | Original (5×5, keep 2600) | ~130 | 8.3 | ~123 | ~211 KB/s |
> | 5×5 per-player (keep 3600) | ~135 | 9.5 | ~246 | **~422 KB/s (2×)** |
> | **3×3 per-player (keep 2200)** | **~49** | **9.5** | **~96** | **~165 KB/s** |
>
> 3×3 has **~half the total rock count** (and thus ~half the bandwidth) of the
> original, while maintaining a comparable on-screen density (mean 9.5 vs 8.3
> in 1P). The 5×5-per-player option would **double** the bandwidth — rejected.
>
> **Note:** the on-screen density is comparable, not dramatically better. The
> main win of 3×3 is **bandwidth efficiency** (fewer off-screen rocks to
> serialize). The main fix for the user's "winking out / flying across"
> complaint is the **per-player anchoring** (Change A), not the region size —
> the per-player anchoring stops the centroid from sitting between the players
> and reflowing the whole field.

### Change B — Wire culling (Session 2)

The host knows the client's ship position (it's in the snapshot:
`snap[0][1]` = remote player's `(pos.x, pos.y)`). Before serializing, the host
**filters rocks to those within a cull rect around the client's position**,
dropping rocks the client can't see. The cull rect is the client's viewport
(1920×1280) inflated by a margin that covers:
- Camera look-ahead lead: `LOOK_AHEAD × MAX_SPEED` = 0.6 × 520 = **312 px**
- Viewport cull margin: `CULL_MARGIN` = **80 px**
- Interpolation delay × max rock speed: `INTERP_DELAY_MAX × 130` = 0.35 × 130 = **46 px**
- Safety buffer: **200 px** (covers jitter, adaptive delay growth, rock spin)

Total margin: **~640 px** beyond the screen edge. Cull rect half-extents:
- x: 960 + 640 = **1600 px**
- y: 640 + 640 = **1280 px**

Measured reduction (2P, players 1500 px apart, 3×3 field):
- Without cull: 58 rocks, ~100 KB/s
- With cull (±1600, ±1280): **~32 rocks, ~55 KB/s** (45% reduction)
- With tighter cull (±1397, ±1077): **~22 rocks, ~38 KB/s** (62% reduction)

The cull is **host-side only** (in `serialize_snapshot`). The client is
unaffected — it already interpolates rocks by id and handles new/missing rocks
gracefully (a rock that disappears from the wire is simply not drawn; a rock
that reappears pops in at its current position, which is the same behavior as
today's viewport culling).

**Why this is safe:**
- The client's interpolation buffer matches rocks by id. A rock that drops out
  of the wire (because it left the cull rect) is simply not in the curr
  snapshot — the client's `interp_positions` only lerps rocks that are in
  *both* prev and curr; a rock in prev but not curr is dropped (same as today
  when a rock is culled by the field). No special handling needed.
- A rock that re-enters the cull rect pops in at its current position (the
  client's `interp_positions` treats it as a "new rock" and uses its curr
  values). This is the same pop-in behavior the client already has for rocks
  that spawn fresh. The 640 px margin means the pop-in happens well off-screen
  (the rock is ~640 px beyond the screen edge when it first appears), so the
  player never sees it pop in.
- The `RockShapeTracker` is unaffected: it tracks rock ids that have had their
  verts sent. A rock that drops out of the cull rect and later re-enters it
  will have its verts re-sent (the tracker still has the id in `_sent`, so it
  won't re-send — the client's `rock_shapes` registry still has the verts).
  This is correct: the shape is immutable for the rock's lifetime.

---

## 3. Session 1 — Per-player field (config + spawning.py + test)

**Goal:** Replace the centroid-anchored field with a per-player union field.
Fix the geometry bug. Add a focused test.

**Files touched:**
- `config.py` — rename `DESPAWN_RADIUS` → `FIELD_KEEP_RADIUS`, change value
  2600 → 2200; change `ACTIVE_SECTORS` 2 → 1; update comments.
- `spawning.py` — rewrite `update_field` body (per-player cull + per-player
  top-up); update module docstring.
- `test_21_2p_field.py` — **new** test file (4 checks).

**No call-site changes:** `game.py` `_step` already calls
`update_field(self.asteroids, [p.pos for p in self.players if not p.dead], dt,
rng=self.rng)` and `reset()` calls
`update_field(self.asteroids, [p.pos for p in self.players], 0, rng=self.rng)`.
Both pass the alive-players list, which is exactly what the new per-player
logic wants.

### 3.1 `config.py` changes

```python
# --- infinite field (sector spawning) ---
SECTOR_SIZE = 1000          # px per sector cell
# Phase 0 (plan_21): 3x3 region per player (was 5x5). A 3000x3000 region
# fully covers the 1920x1280 screen with a 540 px buffer (absorbs the
# camera look-ahead lead) — a 5x5 region was 5000x5000, mostly off-screen
# rocks burning snapshot bandwidth. See plan_21 §2 for the measured
# bandwidth table.
ACTIVE_SECTORS = 1          # Chebyshev radius of sectors kept populated
ROCKS_PER_SECTOR = 3        # target rocks per active sector
# Phase 0 (plan_21): rocks are kept if within this radius of AT LEAST ONE
# alive player (per-player min-distance, not the centroid). Sized to fully
# contain the 3x3 active region (corners ~2121 px from the anchor) so a rock
# is never culled the tick it spawns. Was DESPAWN_RADIUS = 2600 (centroid
# cull), which both winked out rocks near a player's screen edge in 2P and
# left the active region's corners outside the keep radius (spawn-then-cull
# waste).
FIELD_KEEP_RADIUS = 2200
SPAWN_CLEAR_RADIUS = 500    # never spawn closer than this to a player
MAX_SPAWNS_PER_TICK = 4     # fill the field gradually, not all at once
```

> `DESPAWN_RADIUS` is referenced **only** in `spawning.py` (verified by grep).
> Renaming it to `FIELD_KEEP_RADIUS` is safe. Update the `spawning.py` import.

### 3.2 `spawning.py` — `update_field` rewrite

Replace the centroid computation, the cull, and the top-up region. New body:

```python
def update_field(asteroids, players, dt, rng=None):
    """Top up under-populated sectors and cull rocks that drifted away.

    Call once per fixed step.

    Phase 0 (plan_21): the field is the UNION of a per-player active region
    (a 3x3 block of sectors around EACH alive player), not a single
    centroid-anchored blob. A rock is kept if it is within
    FIELD_KEEP_RADIUS of AT LEAST ONE alive player (per-player min-distance),
    not the centroid. This gives each 2P player a full field centered on
    themselves (the old centroid blob sat between the players and left their
    local sectors empty) and stops rocks near a player's screen edge from
    being culled while visible.

    asteroids: the game's rock list (mutated in place)
    players:   list of Vector2 positions — one per alive ship
    dt:        fixed step size (kept for API symmetry / future use)
    rng:       optional random.Random for reproducible fields (default global)
    """
    rng = rng or random
    if not players:
        # No one alive to anchor the field (2P, both players dead waiting to
        # respawn). Leave the field as-is — no cull, no top-up — until a
        # respawn gives it players again.
        return

    # Cull: keep a rock if it is within FIELD_KEEP_RADIUS of at least one
    # alive player (per-player min-distance, not the centroid).
    def _kept(a):
        for p in players:
            if a.pos.distance_to(p) < FIELD_KEEP_RADIUS:
                return True
        return False
    for a in asteroids:
        a.sector = sector_of(a.pos)
    asteroids[:] = [a for a in asteroids if _kept(a)]

    # Count surviving rocks per sector.
    counts = {}
    for a in asteroids:
        counts[a.sector] = counts.get(a.sector, 0) + 1

    # Top up the UNION of each player's 3x3 active region. `region` tracks
    # sectors already scheduled this tick so overlapping regions (players
    # close together) are not double-filled. The GLOBAL spawn cap bounds the
    # total per tick (deterministic fill order; see plan_21 §3.3).
    region = set()
    spawned = 0
    for p in players:
        ps, pt = sector_of(p)
        for sx in range(ps - ACTIVE_SECTORS, ps + ACTIVE_SECTORS + 1):
            for sy in range(pt - ACTIVE_SECTORS, pt + ACTIVE_SECTORS + 1):
                if spawned >= MAX_SPAWNS_PER_TICK:
                    break
                key = (sx, sy)
                if key in region:
                    continue
                region.add(key)
                need = ROCKS_PER_SECTOR - counts.get(key, 0)
                for _ in range(need):
                    pos = _spawn_pos(key, players, rng=rng)
                    if pos is not None:
                        asteroids.append(Asteroid(pos, _spawn_size(rng=rng),
                                                  rng=rng))
                        counts[key] = counts.get(key, 0) + 1
                        spawned += 1
                    if spawned >= MAX_SPAWNS_PER_TICK:
                        break
            if spawned >= MAX_SPAWNS_PER_TICK:
                break
        if spawned >= MAX_SPAWNS_PER_TICK:
            break
```

Also update the module docstring's "Multiplayer-ready" paragraph (it currently
says "keys everything off their centroid") to describe the per-player union.

> `_spawn_pos`, `_spawn_size`, `sector_of` are unchanged. `_spawn_pos` already
> takes the full `players` list (for `SPAWN_CLEAR_RADIUS`), so no signature
> change is needed.

### 3.3 `test_21_2p_field.py` (new)

```python
"""Phase 0 (plan_21): the 2P asteroid field is the UNION of per-player
regions, not a single centroid blob.

Asserts:
  1. A rock within FIELD_KEEP_RADIUS of a player (but far from the centroid)
     SURVIVES the cull — the old centroid cull removed it (wink-out).
  2. Each player's local sector gets topped up when the players are far apart
     — the old centroid top-up left the players' local sectors empty (sparse
     field).
  3. 1P is unchanged: one player -> the field is anchored on that player.
  4. update_field is deterministic (same seed + same players -> same field).
"""
import random
import pygame

from ship5 import config as C
from ship5.spawning import update_field, sector_of
from ship5.asteroid import Asteroid


def _run(players, ticks=600, seed=42):
    rng = random.Random(seed)
    asteroids = []
    for _ in range(ticks):
        update_field(asteroids, players, C.TICK, rng=rng)
        for a in asteroids:
            a.update(C.TICK)
    return asteroids


def test_rock_near_player_survives_centroid_cull():
    # P0 at origin, P1 at (1500, 0) -> centroid (750, 0).
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    # A rock 2100 px from P0 but 2850 px from the centroid (750, 0). The old
    # centroid cull (DESPAWN_RADIUS 2600) removed it (2850 > 2600); the new
    # per-player keep radius (FIELD_KEEP_RADIUS 2200) keeps it (2100 < 2200).
    rock = Asteroid(pygame.Vector2(-2100, 0), 'medium',
                    vel=pygame.Vector2(0, 0), rng=random.Random(1))
    asteroids = [rock]
    update_field(asteroids, [p0, p1], C.TICK, rng=random.Random(2))
    assert any(a.id == rock.id for a in asteroids), \
        "rock near P0 (far from centroid) was culled — wink-out not fixed"


def test_each_players_local_sector_topped_up():
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    asteroids = _run([p0, p1])
    # Each player's own sector must be populated (the old centroid top-up
    # left both empty when the players were 1500 px apart).
    s0 = sector_of(p0)
    s1 = sector_of(p1)
    n0 = sum(1 for a in asteroids if sector_of(a.pos) == s0)
    n1 = sum(1 for a in asteroids if sector_of(a.pos) == s1)
    assert n0 >= 1, f"P0's own sector empty ({n0} rocks) — sparse field"
    assert n1 >= 1, f"P1's own sector empty ({n1} rocks) — sparse field"


def test_1p_field_anchored_on_player():
    p0 = pygame.Vector2(0, 0)
    asteroids = _run([p0])
    # With one player the field is anchored on them: their own sector is
    # populated and no rock is farther than FIELD_KEEP_RADIUS.
    s0 = sector_of(p0)
    assert sum(1 for a in asteroids if sector_of(a.pos) == s0) >= 1
    assert all(a.pos.distance_to(p0) < C.FIELD_KEEP_RADIUS for a in asteroids)


def test_field_deterministic():
    p0 = pygame.Vector2(0, 0)
    p1 = pygame.Vector2(1500, 0)
    a = _run([p0, p1], seed=7)
    b = _run([p0, p1], seed=7)
    sa = sorted((x.id, round(x.pos.x, 3), round(x.pos.y, 3)) for x in a)
    sb = sorted((x.id, round(x.pos.x, 3), round(x.pos.y, 3)) for x in b)
    assert sa == sb, "field not deterministic for the same seed + players"
```

### 3.4 Session 1 verification

1. **New test passes:** `python -m ship5.test_21_2p_field` (run from `/mnt`).
2. **No regressions:**
   - `python -m ship5.test_determinism` (real field, same-seed reproducibility)
   - `python -m ship5.test_interpolation` (no-ops the field, but exercises the
     rock wire path)
   - `python -m ship5.test_14_1_sustained_fire`, `test_14_2_ghost_lockstep`,
     `test_14_3_missile_lockstep`, `test_10_10_input_echo` (all no-op the
     field — must stay green)
   - `python -m ship5.test_19_2p_combat` (2P host sim)
3. **Headless 2P sim check** (manual): two players 1500 px apart → each
   player's own sector is populated (the old centroid top-up left both empty),
   the field is centered on each player (not the midpoint), and no rock is
   culled while within `FIELD_KEEP_RADIUS` of a player. Total rock count
   should drop from ~120 (original) to ~58 (3×3).
4. **Perf sanity:** the cull is now O(rocks × players) instead of O(rocks).
   With ≤ 2 players and ≤ ~100 rocks (3×3 keeps the count low) that is ≤ 200
   distance checks/tick — negligible.

### 3.5 Session 1 commit

`plan_21 S1: per-player 2P asteroid field (3x3 union + per-player keep radius)`

---

## 4. Session 2 — Wire culling (net.py + net_process.py + test)

**Goal:** The host filters rocks to those within a cull rect around the
client's position before serializing the snapshot. Reduces wire bandwidth by
~45–62% (measured: 58 → ~22–32 rocks).

**Files touched:**
- `net.py` — add optional `cull_center` param to `serialize_snapshot`; filter
  rocks before pruning.
- `net_process.py` — compute `cull_center` from the remote player's pos in
  the snapshot and pass it to `serialize_snapshot`.
- `net.py` `NetWorker` (in-process path) — same: compute `cull_center` and
  pass it to `serialize_snapshot`.
- `test_21_wire_cull.py` — **new** test file (3 checks).

### 4.1 `net.py` — `serialize_snapshot` change

Add an optional `cull_center` parameter. When provided (a `(x, y)` tuple),
filter rocks to those within the cull rect before pruning. When `None`
(default), no culling (backward-compatible — the in-process single-player path
and the self-test don't cull).

```python
def serialize_snapshot(snap, tracker=None, cull_center=None):
    """Game.snapshot() tuple -> the PRUNED wire structure.

    ... (existing docstring) ...

    `cull_center` (optional): a `(x, y)` tuple (the remote player's world
    position). When provided, rocks outside the cull rect centered on this
    point are DROPPED from the wire (the client can't see them). The cull
    rect is the viewport (WIDTH x HEIGHT) inflated by WIRE_CULL_MARGIN on
    each side. When None (default), no culling — all rocks are sent
    (backward-compatible; the single-player path and the self-test don't
    cull).
    """
    s = list(snap)
    rocks = []
    for a in snap[5]:
        # a = (id, x, y, vx, vy, size, angle, spin, verts)
        rid = a[0]
        # Wire cull (plan_21 S2): drop rocks outside the cull rect.
        if cull_center is not None:
            cx, cy = cull_center
            if (abs(a[1] - cx) > WIDTH / 2 + WIRE_CULL_MARGIN or
                    abs(a[2] - cy) > HEIGHT / 2 + WIRE_CULL_MARGIN):
                continue
        if tracker is None or tracker.take(rid):
            verts = tuple((vx, vy) for (vx, vy) in a[8])
        else:
            verts = None
        rocks.append([rid, round(a[1], 1), round(a[2], 1),
                      round(a[6], 3), verts])
    s[5] = rocks
    s[6] = None
    return s
```

Add `WIRE_CULL_MARGIN` to `config.py`:

```python
# --- wire culling (plan_21 S2) ---
# Margin (px) added around the client's viewport when deciding whether to
# send a rock over the wire. Must cover: camera look-ahead lead
# (LOOK_AHEAD * MAX_SPEED = 312 px) + viewport cull margin (CULL_MARGIN = 80)
# + interpolation delay * max rock speed (INTERP_DELAY_MAX * 130 = 46 px)
# + safety buffer (200 px, covers jitter + adaptive delay growth). Total:
# ~640 px. A rock is sent iff it is within (WIDTH/2 + WIRE_CULL_MARGIN) of
# the client's x AND (HEIGHT/2 + WIRE_CULL_MARGIN) of the client's y.
WIRE_CULL_MARGIN = 640
```

### 4.2 `net_process.py` — pass `cull_center`

In the net process's send loop (where `serialize_snapshot` is called), extract
the remote player's position from the snapshot and pass it as `cull_center`:

```python
# The remote player's world position (m[2] = the Game.snapshot() tuple;
# m[2][0] = players tuple; m[2][0][1] = player 1's Ship.snapshot(), whose
# first two fields are pos.x, pos.y). The host is player 0; the client is
# player 1.
cull_center = (m[2][0][1][0], m[2][0][1][1])
conn.send({"type": T_SNAP, "sim_time": m[1],
           "snap": serialize_snapshot(m[2], rock_tracker,
                                      cull_center=cull_center)})
```

> `m[2][0]` is the players tuple (one `Ship.snapshot()` per player).
> `m[2][0][1]` is player 1's (the client's) `Ship.snapshot()`, whose first
> two fields are `pos.x, pos.y`. This is the client's CURRENT position at
> the snapshot's sim time — exactly what the cull rect should be centered on.
>
> **Call-site inventory:** the production send paths are `net_process.py`
> (~L350, the 2P net process) and `net.py` `NetWorker` (~L952, the in-process
> worker). Both get `cull_center`. The self-test paths (`net.py` ~L1216,
> ~L1330) keep the default `cull_center=None` (no culling) — they exercise
> the serialization round-trip, not the production cull.

### 4.3 `net.py` `NetWorker` (in-process path) — pass `cull_center`

The in-process `NetWorker` (used by the self-test and potentially by
single-player networked mode) also calls `serialize_snapshot`. Apply the same
change:

```python
cull_center = (snap[0][1][0], snap[0][1][1])
conn.send({"type": T_SNAP, "sim_time": sim_time,
           "snap": serialize_snapshot(snap, self._rock_tracker,
                                      cull_center=cull_center)})
```

> **Note:** the in-process `NetWorker` is only used in 2P mode (the
> self-test). In single-player mode, `serialize_snapshot` is called with
> `cull_center=None` (the default), so no culling happens — the single-player
> path is unaffected.

### 4.4 `test_21_wire_cull.py` (new)

```python
"""Session 2 (plan_21): the host culls rocks outside the client's viewport
before serializing the snapshot.

Asserts:
  1. A rock within the cull rect is SENT (present in the wire).
  2. A rock outside the cull rect is DROPPED (absent from the wire).
  3. cull_center=None (default) sends ALL rocks (backward-compatible).
"""
import json
import pygame

from ship5 import config as C
from ship5.net import serialize_snapshot, RockShapeTracker


def _make_snap(rock_positions):
    """Build a minimal 11-tuple snapshot with the given rock positions.

    rock_positions: list of (x, y) — one per rock. Each rock gets a unique
    id (1, 2, 3, ...) and a fixed shape (10 verts in a circle).
    """
    players = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, False, 0.0, 0.0,
         False, 0.0, 0.0, 0.0, 0.0, False, 1.0, (), 0, False, 0, 0.0, (), 0),
        (1500.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, False, 0.0, 0.0,
         False, 0.0, 0.0, 0.0, 0.0, False, 1.0, (), 1, False, 0, 0.0, (), 0),
    )
    enemies = ()
    bullets = ()
    enemy_bullets = ()
    missiles = ()
    rocks = []
    for i, (x, y) in enumerate(rock_positions, start=1):
        verts = tuple((50.0 * __import__('math').cos(2 * __import__('math').pi * j / 10),
                       50.0 * __import__('math').sin(2 * __import__('math').pi * j / 10))
                      for j in range(10))
        rocks.append((i, x, y, 0.0, 0.0, 'large', 0.0, 0.0, verts))
    return (players, enemies, bullets, enemy_bullets, missiles,
            tuple(rocks), None, False, 0.0, 1, len(rocks) + 1)


def test_rock_in_cull_rect_is_sent():
    # Client at (1500, 0). A rock at (1500, 0) is at the center — sent.
    snap = _make_snap([(1500.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker(),
                              cull_center=(1500.0, 0.0))
    assert len(wire[5]) == 1, "rock at cull center was dropped"


def test_rock_outside_cull_rect_is_dropped():
    # Client at (1500, 0). A rock at (5000, 0) is 3500 px away — outside
    # the cull rect (960 + 640 = 1600 px half-width). Dropped.
    snap = _make_snap([(5000.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker(),
                              cull_center=(1500.0, 0.0))
    assert len(wire[5]) == 0, "rock outside cull rect was sent"


def test_no_cull_sends_all_rocks():
    # cull_center=None (default) sends ALL rocks (backward-compatible).
    snap = _make_snap([(0.0, 0.0), (5000.0, 0.0), (-5000.0, 0.0)])
    wire = serialize_snapshot(snap, RockShapeTracker())
    assert len(wire[5]) == 3, "cull_center=None should send all rocks"
```

### 4.5 Session 2 verification

1. **New test passes:** `python -m ship5.test_21_wire_cull` (run from `/mnt`).
2. **No regressions:**
   - `python -m ship5.test_net_process` (exercises the net process send path)
   - `python -m ship5.test_17_rock_shapes` (exercises `serialize_snapshot`
     with the `RockShapeTracker`)
   - `python -m ship5.test_interpolation` (exercises the client's rock
     interpolation — unaffected by the host-side cull)
   - `python -m ship5.test_21_2p_field` (Session 1's test — must stay green)
3. **Wire bandwidth check** (manual): run the 2P host with the net process
   and measure the snapshot size (the `snap_log` or a quick `len(json.dumps(...))`
   probe). Expect ~45–62% reduction in rock count on the wire.

### 4.6 Session 2 commit

`plan_21 S2: wire culling (host drops rocks outside client's viewport)`

---

## 5. Session 3 — Integration, perf verification, docs

**Goal:** Run the full test suite, measure the combined bandwidth reduction,
update the docs, and note the manual playtest.

**Files touched:**
- `PERFORMANCE.md` — add a "2P asteroid field (plan_21)" section with the
  measured bandwidth numbers.
- `ARCHITECTURE.md` — add a "2P asteroid field (plan_21)" section describing
  the per-player field + wire culling.
- No code changes (unless the full suite reveals a regression).

### 5.1 Full test suite

Run the complete suite from `/mnt`:

```bash
python -m ship5.test_21_2p_field
python -m ship5.test_21_wire_cull
python -m ship5.test_determinism
python -m ship5.test_snapshot
python -m ship5.test_net_process
python -m ship5.test_render_model
python -m ship5.test_interpolation
python -m ship5.test_14_1_sustained_fire
python -m ship5.test_14_2_ghost_lockstep
python -m ship5.test_14_3_missile_lockstep
python -m ship5.test_10_10_input_echo
python -m ship5.test_17_rock_shapes
python -m ship5.test_19_2p_combat
```

Expected: all pass. Pre-existing failures (10_3a, 10_11c, 10_9, test_network)
are unchanged.

### 5.2 Bandwidth measurement

Run the 2P host with the net process and measure the snapshot wire size
before/after. Expected combined reduction:
- Original (5×5, no cull): ~123 rocks × 30 B × 60 Hz ≈ **211 KB/s**
- After S1 (3×3, no cull): ~58 rocks × 30 B × 60 Hz ≈ **100 KB/s**
- After S2 (3×3 + cull): ~22–32 rocks × 30 B × 60 Hz ≈ **38–55 KB/s**

Total reduction: **~75–82%** vs. the original.

### 5.3 Docs

**PERFORMANCE.md** — add:

```markdown
## 2P asteroid field (plan_21)

The 2P asteroid field was a single centroid-anchored blob (5x5 sectors,
DESPAWN_RADIUS 2600 px from the centroid). In 2P the centroid sits between
the players, so the field was off-center, sparse, and churning (wink-out /
pop-in). Plan 21 S1 replaced it with a per-player union field (3x3 sectors
per player, FIELD_KEEP_RADIUS 2200 px per-player min-distance). Plan 21 S2
added wire culling: the host drops rocks outside the client's viewport
(+ WIRE_CULL_MARGIN 640 px) before serializing.

Measured bandwidth (2P, players 1500 px apart, 3x3 field):
- Original (5x5, no cull): ~123 rocks, ~211 KB/s
- After S1 (3x3, no cull): ~58 rocks, ~100 KB/s
- After S2 (3x3 + cull): ~22-32 rocks, ~38-55 KB/s

Total reduction: ~75-82% vs. the original.
```

**ARCHITECTURE.md** — add:

```markdown
## 2P asteroid field (plan_21)

The asteroid field is host-authoritative. In 2P, the field is the UNION of a
3x3 sector region around each alive player (not a single centroid blob). A
rock is kept if it is within FIELD_KEEP_RADIUS (2200 px) of at least one
alive player. The host culls rocks outside the client's viewport
(+ WIRE_CULL_MARGIN 640 px) before serializing the snapshot, reducing wire
bandwidth by ~75-82% vs. the original centroid field.

The client is unaffected: it interpolates rocks by id and handles new/missing
rocks gracefully (a rock that drops out of the wire is simply not drawn; a
rock that re-enters the cull rect pops in at its current position, well
off-screen due to the 640 px margin).
```

### 5.4 Manual playtest note

Two players in 2P:
- Each player should see a full field centered on themselves (not the
  midpoint).
- Rocks should not wink out at the screen edge (the per-player keep radius
  prevents centroid-based culling).
- Rocks should not pop in at the screen edge (the wire cull margin ensures
  rocks are ~640 px off-screen when they first appear on the wire).
- The field should feel stable (no centroid reflow).

### 5.5 Session 3 commit

`plan_21 S3: integration + perf verification + docs`

---

## 6. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `test_determinism` breaks (field layout changed) | Low | It only asserts *same-seed reproducibility*, not a specific layout. The new logic is deterministic. If it fails, the new logic has a hidden non-determinism — investigate before shipping. |
| A lockstep test depends on the old field | Very low | Verified: all lockstep tests **no-op** `update_field`. |
| Wire cull drops a rock the client needs (beam blocking) | Low | The client's `_beam_blocked_end` uses the buffer's rock list. A rock that drops out of the wire is simply not in the buffer — the beam won't be clamped to it. This is correct: the rock is off-screen, so the beam's visual endpoint (which is also off-screen) doesn't matter. The beam's *hit* is resolved on the host (authoritative), not the client. |
| Wire cull margin too small (rock pops in at screen edge) | Low | The 640 px margin covers look-ahead (312) + cull margin (80) + interp delay (46) + safety (200). If a pop-in is observed, increase `WIRE_CULL_MARGIN`. |
| `RockShapeTracker` re-sends verts for a re-entering rock | Very low | The tracker keeps the id in `_sent` even after the rock drops out of the cull rect. When the rock re-enters, the tracker returns `False` (already sent), so verts are NOT re-sent. The client's `rock_shapes` registry still has the verts (they're immutable). Correct. |
| Renaming `DESPAWN_RADIUS` misses a reference | Very low | Grep-verified: only `spawning.py` references it. |
| 2P rock count (two 3×3 regions) → perf | Low | Measured ~58 (1500 px apart) to ~96 (4000 px apart) — *below* the original's ~120. Viewport culling bounds draw cost; sim cost is O(rocks) and unchanged in kind. |

---

## 7. Follow-up (Phase 1, not this milestone)

- Deterministic per-chunk generation (seed = f(chunk, run_seed)) so the field
  is stable and persistent instead of churning.
- Stable chunk-derived rock ids (field rocks) + a separate counter range for
  split children; reconcile with `RockShapeTracker` + client `rock_shapes`.
- Sleep/wake chunk store (freeze rocks when a chunk leaves the active union).
- Density re-tuning for 2P (`ROCKS_PER_SECTOR`, chunk size).
- Rock snapshot rate reduction (15 Hz instead of 60 Hz) — the biggest
  remaining bandwidth win (~4×), but requires an architectural change to the
  snapshot format.
- Binary wire format (~1.5–2× reduction) — bigger change, affects all entities.