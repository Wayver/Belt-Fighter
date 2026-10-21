# Plan 19 — 2P Player-vs-Player Combat (team-based targeting, damage, lock-on, hit FX)

**Status:** Session 2 DONE (commit 075c045). Next: Session 3 (client-side
lock-on + wire). Session 2 landed: _collisions player-bullet-vs-other-
player + missile-vs-other-player passes (the projectile's id
(player_index, seq) selects the OTHER player's slot — a projectile starts
at the muzzle, inside the shooter's shield, so it can only reach the
other player; same shield-oval / swept-hull criteria as the enemy path;
_handle_ship_hit / m.dmg loop / _kill_ship on hit); _resolve_beam extends
the enemy loop to the other player's ship (distance test, vis_end on
their shield oval, beam.damage register_hit calls) and the beam entry's
LAST field is now the target_id (enemy ship id / "p<i>" / None for a
rock) — render_model serializes it (no live reference in the model);
_PlayerHomingTarget adapter (lets Missile.update's target.hp>0 +
target.lead_point work unchanged for a player target) — a player missile
target is wrapped at launch, Missile.snapshot encodes it as "p<i>"
(enemy -> ship id int, else None), Game.apply_snapshot re-wires "p<i>"
to a fresh adapter over self.players[i]; _kill_ship extracted from
_handle_ship_hit. Single-player bit-identical (all new paths gated on
len(players)==2; the beam target_id is a render-model-only field, the
snapshot stays the 11-tuple). Full suite green (4 pre-existing failures
unchanged: 10_3a, 10_11c, 10_9, test_network).

NOTE for Session 3: the beam target_id is currently a RENDER-MODEL-only
field (host draw path). The T_BEAM WIRE message does NOT yet carry it —
Session 3 adds `target` to the T_BEAM message (net.py / sim_thread.py
send + __main__.py receive into remote_beams) so the client can know
"this beam hit me" (Session 4 hit FX) and anchor the endpoint. The
missile "p<i>" encoding is already on the wire (Missile.snapshot), so
Session 3's ghost work is to re-resolve the "p<i>" target by id in
step_local_missiles (the by_id map must include the _GhostPlayerProxy).
**Scope:** let the two human players actually fight each other in 2P. Today the
game has **no "who can shoot whom" concept** — it is hardcoded to "players
shoot AI enemies." This plan introduces a **team** field on the ship, a single
team-based targeting predicate, host-side damage between the two player ships,
client-side lock-on (the prediction ghost sees the remote player), and
client-side hit FX (shield flash + particle burst) when the other player's
weapons impact the local ship.

**Session fit:** four focused sessions, each comfortably inside a 64k context
window — the same shape as plan_17/plan_18. Sessions are strictly ordered:
each builds on the previous and is independently testable.

---

## 1. The problem

In 2P the host-authoritative sim steps both player ships, but the weapons
system only knows about AI enemies:

- **Targeting** — `Game._pick_laser_target` / `_pick_missile_target`
  (game.py ~L1782 / ~L1797) loop **only** `self.enemies`. The other player's
  ship is never a candidate, so neither player can lock on to the other.
- **Damage** — `Game._collisions` (game.py ~L1997) has `player bullet vs
  enemy`, `missile vs enemy`, `enemy bullet vs player`, and `player vs enemy
  (ram)` — but **no `player bullet vs player`** path. A player's weapons
  physically cannot touch the other player's ship.
- **Beam** — `Game._resolve_beam` (game.py ~L1839) hitscans only
  `self.enemies` (after the rock check). A player's laser cannot hit the
  other player.
- **Client lock-on** — the prediction ghost's targeting
  (`netcode.py` `PredictedShip._pick_laser_target` / `_pick_missile_target`,
  ~L1920 / ~L1940) picks from **enemy proxies only** (`_ghost_enemy_proxies`).
  The ghost never sees the remote player's ship, so even if the host allowed
  it, the client's predicted lock/charge would desync.
- **Client hit FX** — the existing `_local_incoming_hit_fx` (game.py ~L3932)
  fires a shield_burst when an **enemy** bullet hits the local ship (detected
  by a jump in the ghost's authoritative `shield_dump`). It does not cover
  the other player's bullets, missiles, or beams.

The user's insight: give each ship a **team**, and let the game derive viable
targets from team membership. That single abstraction replaces the hardcoded
"loop over enemies" and is what makes the feature future-proof (3P, team
modes, alliances, a friendly-fire toggle all become one-line changes).

## 2. Design

### 2.1 Team model (the core abstraction)

**`team` is a property of the *ship*, not the player.** Add `Ship.team: int`.

- Player 0's ship → `TEAM_0` (== 0, == player index).
- Player 1's ship → `TEAM_1` (== 1, == player index).
- AI enemies → `TEAM_NEUTRAL` (a constant, `-1`). An enemy is not a team
  member; it is simply "not the attacker's team," so it is always a valid
  target (preserving today's behavior).

**One targeting predicate** replaces the hardcoded "loop over enemies":

```python
def can_target(attacker, target):
    """True if `attacker` may target `target`: alive and off-team."""
    return target.team != attacker.team and not getattr(target, "dead", False)
```

(Enemies have no `.dead`; `getattr(..., False)` keeps them always-targetable.
Player ships have `.dead` — a frozen corpse is not a target, matching the
existing "a dead ship is frozen and no longer a collision or enemy target"
rule in `_step`.)

**No wire change for `team` itself.** The client already knows its own index
(`self.local_index`) and the remote index, so it derives the same assignment
locally (team == player_index for player ships, neutral for enemies). The
ship snapshot stays at its current 25 fields — we do not touch a load-bearing
wire format for this.

**Team assignment site:** `Game` sets `ship.team` when it owns the ship:
- host: in `reset()` / `set_player_ship()` for the two player ships (by
  index); AI enemies get `TEAM_NEUTRAL` at construction (or in the spawn
  path). Single-player: the one ship is `TEAM_0`, enemies neutral — the
  predicate reduces to today's "enemies only" (the other player does not
  exist), so single-player is bit-identical.

### 2.2 Unified "targetable" surface

The host's targeting/damage/beam code reads a small set of attributes off a
target. To let the **same** code handle an AI enemy *and* the other player's
ship, both must expose:

| Attribute | AIEnemy | Player Ship |
|---|---|---|
| `.pos`, `.vel`, `.angle` | `e.ship.*` (or `e.pos`) | `p.*` |
| `.team` | `TEAM_NEUTRAL` | player index |
| `.dead` | (absent → treat as False) | `p.dead` |
| `.collision_radius` | `e.collision_radius` | `p.collision_radius` |
| `.shield_on`, `.shield_contains`, `.shield_impact_point` | `e.ship.*` | `p.*` |
| `.register_hit(world_pos)` | `e.register_hit` (hp) | `p.register_hit` (shield/hull) |
| `.lead_point(shooter_pos, speed)` | `e.lead_point` | **new** `Ship.lead_point` |
| `.hp` (missile homing gate) | `e.hp` | **adapter** (see 2.5) |

`Ship` already has `.pos/.vel/.angle/.team/.dead/.collision_radius/
.shield_on/.shield_contains/.shield_impact_point/.register_hit`. The one
missing piece is **`Ship.lead_point(shooter_pos, bullet_speed)`** — the
constant-velocity intercept math `AIEnemy.lead_point` uses (a plain method on
`Ship`, reading `self.pos`/`self.vel`; no accel term — a player ship's
accel is not synced to the client, and constant-velocity lead is the right
order of accuracy, the same as the ghost's enemy proxies). This is needed for
both host-side missile homing on a player and client-side ghost homing.

### 2.3 Host-side targeting (Phase 1)

`Game._pick_laser_target` / `_pick_missile_target` change from "loop
`self.enemies`" to "loop the **targetable candidates**":

```python
def _targetable_candidates(self, ship):
    """Alive, off-team ships: the AI enemies + the other player's ship."""
    out = [e for e in self.enemies if can_target(ship, e)]
    for i, p in enumerate(self.players):
        if i != <ship's own index> and can_target(ship, p):
            out.append(p)
    return out
```

The pickers keep their existing nearest-in-range math, just over this list.
The laser/missile wedge test (`_in_wedge` / `_laser_in_wedge`) reads only
`t.pos`, so it works unchanged for a player target.

**Result:** both players can now **lock on** to each other (laser charge +
missile lock build). No damage yet.

### 2.4 Host-side damage (Phase 2)

Extend the three weapon paths in `_collisions` / `_resolve_beam` to also test
the **other player's ship**, using the **same** hit criteria as the enemy path
(shield UP → point test against the shield oval; shield DOWN → swept hull
polygon). On a hit, call the existing `_handle_ship_hit` (shield absorb →
hull → per-player death → respawn).

- **Player bullet vs other player** — a new pass in `_collisions` (after the
  bullet-vs-enemy pass, before bullet-vs-asteroid, mirroring the host's
  enemies-then-rocks order). Same shield-oval / swept-hull test as the enemy
  path; on hit `self.bullets.remove(b)` + `_handle_ship_hit(other, hit_pt)`.
- **Missile vs other player** — same, in the missile-vs-enemy pass region.
  `m.dmg` damage points via `register_hit` (a player's `register_hit` takes
  one shield point per call, so loop `m.dmg` times, exactly like the enemy
  missile path).
- **Beam vs other player** — `_resolve_beam` extends its enemy loop to also
  test the other player's ship (distance to `beam.end` < `collision_radius +
  4`, same as the enemy test). `vis_end` lands on their shield oval via
  `shield_impact_point`; the damage loop calls `register_hit(impact)`
  `beam.damage` times. The beam entry + T_BEAM event carry the target (see
  2.6).

**Result:** bullets, missiles, and beams all deal damage to the other player,
with shield flash + burst on the host. Per-player death + respawn already
exist (`_handle_ship_hit` sets `ship.dead`; `respawn_player` handles R).

### 2.5 Client-side lock-on (Phase 3)

The ghost must see the remote player's ship as a target, or its predicted
lock/charge/launch desyncs from the host.

- **Remote-player proxy** — a new `_GhostPlayerProxy` (mirrors
  `_GhostEnemyProxy`) built from the buffer's `pos['ships'][remote_index]`
  entry. Exposes `.pos`, `.vel`, `.id` (a stable key, e.g. `"p<index>"`),
  `.team` (the remote index), and a `.ship` stand-in (shield oval + power
  fields, from the local `self.players[remote]` hull — the same stand-in the
  remote render uses). Built in a new `_ghost_player_proxy(pos)` alongside
  `_ghost_enemy_proxies`.
- **Ghost pickers** — `PredictedShip._pick_laser_target` /
  `_predict_missile_target` take the remote-player proxy in addition to the
  enemy proxies and apply the same team predicate (the ghost's own ship is
  `TEAM_<local_index>`; the proxy is `TEAM_<remote_index>`; enemies neutral).
  Fed in from `advance` / `reconcile_rewind` / `step` (the same call sites
  that pass `enemies`).
- **Ghost missile homing** — `GhostMissile.update` re-resolves its target by
  id each step (`step_local_missiles` builds `by_id = {e.id: e}`). The
  remote-player proxy's id (`"p<index>"`) is a different key space from the
  enemy numeric ids, so the `by_id` map must include the player proxy too.
  The missile's stored target id comes from the launch (see 2.6 for the
  encoding). A missile locked on the remote player homes on the proxy; if the
  proxy leaves the buffer (shouldn't happen — ships don't turn over) it
  coasts, same as a dead enemy.

**Result:** the client's predicted lock/charge/launch matches the host
(lockstep preserved). The ghost's missile homes on the other player.

### 2.6 Wire additions (Phase 3 — both low-risk, no snapshot change)

Two small, additive wire changes. The **snapshot is untouched** (still the
pruned 11-tuple; the ship snapshot stays 25 fields).

1. **T_BEAM gains a `target` field.** The beam is already an *event* message
   (not snapshot state), so extending it is cheap. `target` is:
   - `None` — the beam hit a rock (the existing rock path), or
   - an enemy ship id (int) — the beam hit an AI enemy, or
   - `"p0"` / `"p1"` (a string, the player index) — the beam hit a player.
   The host's `_emit_beam_event` sets it from the resolved target; the client
   reads it to know "this beam hit *me*" (Phase 4) and to anchor the beam's
   endpoint to the right ship's interpolated pose.

2. **Missile snapshot target encoding.** `Missile.snapshot` currently stores
   `self.target.ship.id` (an enemy ship id, or None). Extend it to also
   encode a player target: store `"p<index>"` (a string) when the target is a
   player ship, the enemy ship id (int) when it's an enemy, else None.
   `Game.apply_snapshot` re-wires `m.target` accordingly: int → the enemy via
   the existing `id_map`; `"p<i>"` → a small **homing adapter** wrapping
   `self.players[i]` that exposes `.hp` (1 if not dead else 0), `.pos`, and
   `.lead_point` (delegating to `Ship.lead_point`) — so `Missile.update`'s
   existing `self.target.hp > 0` + `self.target.lead_point(...)` work
   unchanged for a player target. The adapter is host-side only (the client's
   ghost uses `_GhostPlayerProxy` instead).

**Why not serialize `team`:** the client derives it from indices (2.1). The
only wire changes are the two additive event/field extensions above, both
backward-tolerant (a missing `target` field reads as None → the rock path).

### 2.7 Client-side hit FX (Phase 4)

The existing `_local_incoming_hit_fx` detects a hit on the local ship by a
**jump in the ghost's authoritative `shield_dump`** (a hit spikes it by a full
`power_hit`; the ghost re-anchors at each reconcile). That signal is
**weapon-agnostic** — it fires for *any* hit that raises the shield dump,
including the other player's bullets and missiles. So:

- **Bullets + missiles (the other player's)** — the existing shield_dump-jump
  detection already covers them. Extend the impact-point search to also
  consider **incoming player bullets** (buffer `pos['bullets']` with
  `kind == 'player'` and `owner == remote_index`), not just enemy bullets, so
  the spark lands on the correct impact point. (Missiles have no buffer
  "incoming" entry to project from — the shield_dump jump + the oval-front
  fallback is sufficient, same as today's enemy-missile case.)
- **Beams** — a T_BEAM event whose `target` is the local player
  (`"p<local_index>"`) spawns a shield_burst at the beam's endpoint (the
  `end` the host computed on the local ship's shield oval) + the throttled
  `shield_hit` SFX, in the same frame the beam is drawn. This is the one new
  detection (beams are hitscan — there is no in-flight projectile to watch).

**Result:** shield flash + particle burst + SFX when the other player's
weapons impact the local ship, on the client — for all three weapon types.

## 3. Work breakdown

### Session 1 — Team foundation + unified targeting (host + ghost lock-on)

1. **config.py**: `TEAM_NEUTRAL = -1` (and `TEAM_0 = 0`, `TEAM_1 = 1` for
   clarity).
2. **ship.py**: `Ship.team` field (default `TEAM_NEUTRAL`); new
   `Ship.lead_point(shooter_pos, bullet_speed)` (constant-velocity intercept,
   mirrors `AIEnemy.lead_point` but reads `self.pos`/`self.vel`, no accel).
3. **game.py**: `can_target(attacker, target)` module helper;
   `Game._targetable_candidates(ship)`; set `ship.team` in `reset()` /
   `set_player_ship()` (players by index) and on enemy spawn (neutral).
   Rewrite `_pick_laser_target` / `_pick_missile_target` to loop
   `_targetable_candidates` instead of `self.enemies`.
4. **netcode.py**: `_GhostPlayerProxy` (pos/vel/id/team/ship stand-in);
   `PredictedShip._pick_laser_target` / `_pick_missile_target` accept the
   remote-player proxy + apply the team predicate; thread the proxy through
   `advance` / `reconcile_rewind` / `step`.
5. **game.py (client)**: `_ghost_player_proxy(pos)` (from
   `pos['ships'][remote_index]`); feed it into the ghost's advance +
   reconcile alongside `_ghost_enemy_proxies`.

**Verification:** 2P — both ships in range → each picks the other as laser +
missile target (host); the ghost picks the remote player (client).
Single-player bit-identical (the other player does not exist → candidates ==
enemies). Required suite green (test_determinism, test_snapshot,
test_14_2/14_3 lockstep, test_10_3b, test_net_process, test_render_model).

### Session 2 — Host-side damage (the other player takes hits)

1. **game.py `_collisions`**: player-bullet-vs-other-player pass +
   missile-vs-other-player pass (same shield-oval / swept-hull criteria as
   the enemy path; `_handle_ship_hit` on hit; `m.dmg` loop for missiles).
2. **game.py `_resolve_beam`**: extend the enemy loop to the other player's
   ship (distance test, `vis_end` on their shield oval, `beam.damage`
   `register_hit` calls). Beam entry + `_emit_beam_event` carry the target
   (int enemy id / `"p<i>"` / None).
3. **bullets.py**: `Missile.snapshot` target encoding (int enemy id /
   `"p<i>"` / None); `Game.apply_snapshot` re-wires `m.target` (enemy via
   `id_map`, player via the homing adapter exposing `.hp`/`.pos`/
   `.lead_point`).

**Verification:** 2P — bullet / missile / beam each reduce the remote ship's
shield then hull; a kill → per-player death + respawn (the other player keeps
playing). Single-player regression (no other player → no new hits). Required
suite green.

### Session 3 — Client-side lock-on + wire (ghost sees the remote player)

1. **net.py / sim_thread.py**: T_BEAM message gains `target` (None / int /
   `"p<i>"`); the host's `_emit_beam_event` + the sim thread's T_BEAM send
   carry it. The client's T_BEAM receive (`_step_remote_beams` / the
   `remote_beams` entry) stores it.
2. **netcode.py**: `GhostMissile` homing re-resolves the remote-player proxy
   by its `"p<i>"` id (`step_local_missiles` `by_id` includes the player
   proxy); a missile locked on the remote player homes on it.
3. **game.py (client)**: confirm the ghost's predicted lock/charge/launch
   matches the host given the remote-player proxy (the Session-1 pickers now
   have a real target to home/charge against end-to-end).

**Verification:** 2P — the client's ghost locks + charges + launches on the
remote player on the same ticks the host does (lockstep); a ghost missile
homes on the remote player; the reconcile snap stays ~0. Required suite green
(test_14_2/14_3, test_10_3b, test_10_4/10_4d beams).

### Session 4 — Client-side hit FX + tests + docs

1. **game.py `_local_incoming_hit_fx`**: extend the impact-point search to
   incoming **player** bullets (`kind == 'player'`, `owner == remote_index`);
   the shield_dump-jump detection already covers the other player's bullets +
   missiles.
2. **game.py (client)**: a T_BEAM whose `target` is the local player spawns a
   shield_burst at the beam endpoint + throttled `shield_hit` SFX (in the
   `remote_beams` draw/step path).
3. **New test** `test_19_2p_combat.py` (standalone `main()`, the test_15_4 /
   test_10_7 scratch-client scaffolding):
   - **TEAM**: `can_target` (off-team alive → True; same team → False; dead
     → False; enemy neutral → True). `Ship.lead_point` (a converging intercept
     point; None when the target outruns the bullet).
   - **HOST TARGETING**: 2P, both in range → each picks the other (laser +
     missile); single-player → enemies only (bit-identical).
   - **HOST DAMAGE**: bullet / missile / beam each reduce the remote ship's
     shield then hull; a kill → `dead` + the other player alive.
   - **GHOST LOCK-ON**: the ghost picks the remote player as laser/missile
     target; a ghost missile homes on the `_GhostPlayerProxy`.
   - **CLIENT HIT FX**: the other player's bullet / missile / beam each
     trigger the local shield flash + burst on the client.
4. **Docs**: ARCHITECTURE.md — a "2P player combat (plan_19)" section: the
   team model + `can_target` predicate, the unified targetable surface, the
   two additive wire changes (T_BEAM `target`, missile target encoding), and
   the client hit-FX generalization. PERFORMANCE.md — no snapshot bandwidth
   change (the two wire additions are additive event fields; T_BEAM is a
   rare event, the missile target id is one small field).

**Verification:** `test_19_2p_combat.py` all gates pass; required suite green
(the full 10.x/14.x/15.x/17.x/18.x client-path set + test_determinism +
test_snapshot + test_net_process + test_render_model). Manual 2P playtest:
two players lock on, shoot, and see shield flash + burst on each other's
ships; a kill respawns the dead player while the other keeps playing.

## 4. What is NOT changing (scope boundary)

- **The snapshot** — untouched (still the pruned 11-tuple; the ship snapshot
  stays 25 fields). `team` is derived locally, not serialized.
- **The authoritative sim's structure, the ghost's PHYSICS, the reconcile,
  the interpolation buffer** — untouched. The ghost still does no
  player-vs-player collision (a predicted clip-through is corrected by the
  next reconcile, same as the deferred ghost rock collision from plan_17 §4).
- **Single-player** — bit-identical (the other player does not exist, so
  `_targetable_candidates` == `self.enemies` and no new damage path fires).
- **AI enemy behavior** — unchanged (they still target the nearest player;
  they are `TEAM_NEUTRAL` and remain targetable by both players).
- **`tracked` targeting-assist count + sensor contacts** — stay
  **enemies-only** for now. The remote player's live power signature is not
  available to the client's ghost (the host never saw the client's V/G/T), so
  including the remote player in `tracked` / contacts would desync the
  ghost. Counting the remote player in `tracked` (host-side only) and in
  sensor contacts is a small follow-up if desired — flagged, not in scope.
- **Friendly fire** — OFF by default (the `can_target` team predicate). A
  future "all-attack" mode is a one-line change to the predicate.

## 5. Risks / edge cases

- **Lockstep (the load-bearing invariant):** the fire decision must stay a
  pure function of input-local state (plan_14). Targeting reads only
  `t.pos` (synced/interpolated) + the ship's own weapon state, so adding the
  remote player as a candidate does not break it — but the ghost MUST see the
  same candidate the host does (the remote-player proxy), or the predicted
  lock/charge desyncs. Session 1's verification pins this.
- **Missile homing on a player:** `Missile.update` gates on `self.target.hp >
  0` and calls `self.target.lead_point`. A player `Ship` has neither `.hp`
  nor (before this plan) `.lead_point` — the homing adapter (2.6) + the new
  `Ship.lead_point` (2.2) close both gaps without changing `Missile.update`.
- **Beam `target` field backward-tolerance:** a missing `target` reads as
  None → the rock path (today's behavior). LAN, both peers run the same
  codebase — no version negotiation needed.
- **Dead remote player:** a frozen corpse is not a target (`can_target`
  checks `.dead`) and takes no hits (the existing "a dead ship is frozen"
  rule). The remote-player proxy is still built (ships don't turn over), but
  the pickers skip it while `dead`.
- **Shield_dump-jump false positives:** the existing 0.5×`power_hit` threshold
  is well below a real hit and well above clock-drift noise (plan_18 / the
  current `_local_incoming_hit_fx`). Generalizing it to player bullets adds
  no new false-positive source (same signal, same threshold).
- **Beam hit FX timing:** a T_BEAM is a 0.15 s flash drawn immediately on
  receipt (one-way LAN latency ~10-20 ms is imperceptible). The shield_burst
  fires in the same frame the beam is drawn — on time with the flash.
- **SFX volume:** sustained mutual fire plays `shield_hit` per hit — the host
  does exactly the same (throttled). If loud in the playtest, the existing
  `_sfx_throttled` interval is the one-line knob.