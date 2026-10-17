# Plan 14 — Input-Local Fire Decision (the Bullet-Cap Foundation Fix)

**Status:** Phase 0 (measure) DONE — drift confirmed, mechanism pinned down
(see §6). Phase 1 (host fire path) DONE — host fire decision is now
input-local, cull-oldest safety net in place, single-player sustained fire
= 60/s with pool at the natural 90 (see §6). Phase 2 (ghost fire path)
DONE — the ghost's local-draw gate is gone, the ghost is in bullet_seq /
cooldown lockstep with the host across 30 reconciles, and no ghost-only id
is ever drawn (see §6). Phase 3 (missiles) DONE — the ghost's missile
local-draw gate is gone, the ghost is in missile_seq / lock lockstep with
the host across 30 reconciles, and no ghost-only missile id is ever drawn
(see §6). Phase 4 (balance) DONE — the gun is now a max-auto-burst
weapon: 6 shots at 20/s, then a 0.4 s rest (~8.6/s sustained, the old
effective 8/s feel with a snappy burst), input-local on both peers so the
lockstep holds (see §6). Phase 5 (regression + live playtest) is next.
**Scope:** the player bullet + missile fire path (host + ghost). The "early
enemy-bullet hit" presentation bug is **out of scope** here (it is a separate
time-domain offset) — but this plan de-risks it by making the bullet system a
stable, provably-in-lockstep foundation first.

---

## 1. The problem

### 1.1 The cap is a pool size being used as a fire-rate control

The three relevant numbers (`config.py`):

| Constant | Value | Meaning |
|---|---|---|
| `FIRE_COOLDOWN` | `0.01` | min interval between shots (shorter than a tick) |
| `BULLET_LIFE` | `1.5` | seconds a bullet lives |
| `MAX_BULLETS` | `12` | the "cap" |

> **Rate correction (measured, Phase 0):** `FIRE_COOLDOWN` (0.01) is shorter
> than `TICK` (0.0167), so the gun fires **once per tick = 60/s**, not the
> nominal 100/s. The capture confirms ~53–60/s. Natural live count =
> **60 × 1.5 = 90 per player, 180 world (2P)** — not 150/300.

Without the cap, holding fire produces a steady **90 live bullets per player
(180 world)**. The cap is **12 — ~7% of the natural count**. So the cap is not a
loose safety bound; it is **permanently hard-bound** the instant the trigger
is held. That is the "overflow": the natural bullet population overflows the
pool ~12×, so the **pool — not the input — is the dominant constraint**.

What that produces, tick by tick, holding fire:

- **t = 0 → 0.2 s:** the gun fires at its native 60/s, dumping all 12 (a burst).
- **t = 0.2 → 1.5 s:** the pool is full → **no fire** (a 1.3 s dead zone).
- **t = 1.5 → 1.7 s:** the 12 burst bullets expire one/tick, the pool refills, 12 more fire.
- **repeat every 1.5 s.**

So the cap silently converts a smooth 60/s laser into a **12-shot burst every
1.5 s (8% duty cycle)**. That is almost certainly not the intended feel — it is
a side effect of using a *pool size* as a *fire-rate* control.

### 1.2 The architectural root cause (the double-bullet)

The host's fire decision (`game.py` `Game._step`, ~1306):

```python
if p_inp.fire and len(self.bullets) >= MAX_BULLETS:
    p_inp = replace(p_inp, fire=False)   # world cap: no room
...
shots, beams, missiles = p.update(dt, p_inp)
```

`len(self.bullets)` is **world state** — it depends on how many bullets *both*
players have alive, where they hit, where they expired. When it binds, the host
sets `fire=False` *before* `p.update()`, so it **does not fire, does not reset
the gun cooldown, and does not bump `bullet_seq`**.

The ghost's fire decision (`netcode.py` `PredictedShip.step`, ~1686):

```python
shots, beams, missiles = s.update(dt, inp)     # fires on cooldown+power, NO cap
for shot in shots:
    if len(self.local_bullets) < MAX_BULLETS:  # local DRAW count, not world
        ... s.bullet_seq += 1 ...
        self.local_bullets.append(b)
```

Two independent mismatches:

1. **Different input.** The host gates on the **world** count (both players);
   the ghost gates on its **local draw** count (`local_bullets` — the bullets it
   is *currently drawing*, which are handed off ~`INTERP_DELAY` after firing, so
   that count sits at 0–2). The ghost's cap almost never binds while the host's
   binds hard.
2. **Different side effects.** The ghost calls `s.update()` (which resets the gun
   cooldown) *before* the cap check; the cap only decides whether the bullet is
   *added to `local_bullets`*. So when the ghost's cap binds, the ghost's
   **cooldown still resets** and (if a shot slipped through) its **`bullet_seq`
   still advances** — but the host, whose cap bound, did *neither*.

Net effect: the ghost and host drift out of phase on both cooldown and
`bullet_seq`. The ghost fires a shot the host never fired → its id
`(local_index, seq)` matches no buffer bullet → **the id-dedup fails** → the
ghost's predicted bullet *and* the host's buffered bullet are both drawn. That
is the "one stream spawning out of empty space." The seq resync on reconcile
only fixes the *counter*, not the already-in-flight ghost-only bullet, so the
double-draw persists until that bullet ages out (~1.5 s).

> **Measured (Phase 0):** the divergence is real and bounded — the seq gap
> sits at exactly +1 (the ghost fires one extra bullet per reconcile cycle,
> always its "13th"), and 554 such host-never-fired bullets were captured in a
> 63 s session. See §6 for the full measurement.

**The deeper principle:** *the fire decision must be a pure function of state
the ghost can reproduce (its own input + weapon state). A world bullet count is
not reproducible, so it cannot be part of the fire decision.*

### 1.3 The missile cap has the identical bug

`MAX_MISSILES = 3`, `MISSILE_COOLDOWN = 1.2`, `MISSILE_LIFE = 3.0` → natural
count = 3.0 / 1.2 = **2.5**, so the cap of 3 *barely* binds (less visible than
bullets). But it is the same class of flaw: world-dependent fire decision
(`len(self.missiles) >= MAX_MISSILES` on the host vs.
`len(self.local_missiles) < MAX_MISSILES` on the ghost) → the same latent
`missile_seq` divergence. Fix it in the same pass.

---

## 2. Goal and principles

**Separate the two things the cap currently conflates:**

| Concern | What it should depend on | Reproducible by the ghost? |
|---|---|---|
| **Fire decision** (may I fire *now*?) | the player's *own* cooldown / recent-fire history + power — **input-local** | **Yes** — the ghost *is* the one firing |
| **Bullet population** (how many live bullets exist) | world state — **performance/presentation only** | **No — and it does not need to be** |

Principles:

1. **The fire decision is input-local.** It reads only the player's own weapon
   state (cooldown, charge, power allocation) and input. The ghost reproduces it
   bit-for-bit, so `bullet_seq` / `missile_seq` and cooldowns stay in lockstep
   with the host → ids always match → dedup always works → the double-bullet
   disappears *at the root*.
2. **The pool is a performance safety net, not a fire gate.** When the live
   count exceeds a hard ceiling, cull the *oldest* projectile. This is
   world-dependent, but it never changes *whether* a shot fires, so the ghost
   does not need to reproduce it (its projectiles are presentation-only; worst
   case it keeps drawing a culled bullet for the last fraction of its life — a
   3 px dot, imperceptible, and it never collides on the client).
3. **The fire rate becomes an explicit, tunable, balance-facing knob** instead
   of an emergent artifact of `12 / 1.5`.

---

## 3. Design

### 3.1 The fire-rate knob (input-local)

`Ship._fire` (`ship.py` ~678) is **already input-local** — it fires when
`inp.fire and w.cooldown <= 0 and w.allocation > 0`, with no pool term. So the
fire *mechanism* needs no change; we only remove the world-dependent gate that
sits *around* it in `Game._step`.

Two ways to express the sustained rate (both input-local):

- **Option A — steady laser (recommended to start):** `FIRE_COOLDOWN` *is* the
  rate knob. Rate = `1 / FIRE_COOLDOWN`. Simplest, already exists, already
  reproducible. Choose a value that gives a reasonable live-bullet count (see
  §7).
- **Option B — burst-then-throttle (later, if the design wants it):** add a
  sliding-window limit ("N shots per T seconds") on top of `FIRE_COOLDOWN`.
  More state (a per-weapon fire-history ring), more to keep in lockstep. Do
  **not** build this now — only if a burst feel is explicitly wanted.

### 3.2 The pool ceiling (performance, cull-oldest)

After appending a new projectile, if the live count exceeds the ceiling, cull
the oldest:

```python
while len(self.bullets) > BULLET_POOL_MAX:
    self.bullets.pop(0)
```

The list is in fire order (new bullets append to the back; the life-cull
`[b for b in self.bullets if b.life > 0]` removes the oldest first and preserves
order), so `pop(0)` is the oldest. Set the ceiling **above the natural count**
so it **never binds in normal play** — it is a safety net against a bug that
spawns unbounded projectiles, not a gameplay mechanic.

- Bullet natural count (per player) = `BULLET_LIFE / FIRE_COOLDOWN`.
- World natural count = 2× that (two players share the host's `self.bullets`).
- `BULLET_POOL_MAX` = ~2× the world natural count (depends on the chosen
  `FIRE_COOLDOWN` — see §7).

The ghost mirrors the same cull on its presentation `local_bullets` /
`local_missiles` (cheap, keeps them bounded; it never conflicts with the
handoff because the handoff removes bullets well before the ceiling is near).

### 3.3 What the fire decision reads (the lockstep contract)

For the ghost to reproduce the host, the fire decision must read **only**:

- `inp.fire` / `inp.missile_fire` (the input),
- the weapon's `cooldown` / `charge` / `lock_progress` (weapon state, synced via
  the ship snapshot),
- the weapon's `allocation` (power, derived from synced power state).

All of these are already synced / reproducible. **No world count.** That is the
invariant to preserve: *any future term added to the fire path must be
input-local or the lockstep (and the dedup) breaks again.*

---

## 4. Exact changes (file by file)

### 4.1 `config.py`

```python
# BEFORE
BULLET_SPEED = 860.0
BULLET_LIFE = 1.5
MAX_BULLETS = 12
FIRE_COOLDOWN = 0.01
...
MISSILE_LIFE = 3.0
MISSILE_COOLDOWN = 1.2
MAX_MISSILES = 3

# AFTER
BULLET_SPEED = 860.0
BULLET_LIFE = 1.5
FIRE_COOLDOWN = <chosen rate knob>     # input-local; rate = 1/FIRE_COOLDOWN
BULLET_POOL_MAX = <safety net>         # cull-oldest ceiling, above the natural
                                       # world count (see §7). NOT a fire gate.
...
MISSILE_LIFE = 3.0
MISSILE_COOLDOWN = 1.2                 # input-local missile rate knob
MISSILE_POOL_MAX = 6                   # cull-oldest ceiling, above natural (2.5)
```

(`MAX_BULLETS` / `MAX_MISSILES` are removed or repurposed — nothing should read
them as a fire gate anymore. Grep for all uses before deleting.)

### 4.2 `game.py` — host fire path (`Game._step`, ~1306–1343)

**Remove** the world-dependent fire gates:

```python
# DELETE these two lines:
if p_inp.fire and len(self.bullets) >= MAX_BULLETS:
    p_inp = replace(p_inp, fire=False)
if p_inp.missile_fire and len(self.missiles) >= MAX_MISSILES:
    p_inp = replace(p_inp, missile_fire=False)
```

**Add** the cull-oldest safety net after the append loops:

```python
for shot in shots:
    bid = (i, p.bullet_seq); p.bullet_seq += 1
    self.bullets.append(Bullet(shot.pos, shot.vel, owner=shot.owner, bid=bid))
# performance safety net (never binds in normal play)
while len(self.bullets) > BULLET_POOL_MAX:
    self.bullets.pop(0)
```

Same for the missile append loop (`while len(self.missiles) > MISSILE_POOL_MAX:
self.missiles.pop(0)`).

> Note: the input-echo block (`self._remote_input_echo.append((self.sim_time -
> dt, p_inp))`) currently echoes the **post-cap** `p_inp`. With the cap gone,
> `p_inp` is the raw input — which is now exactly what the host applies, so the
> echo is still correct (and the ghost's replay no longer needs the cap to
> match). Leave the echo as-is; just confirm it now echoes the raw input.

### 4.3 `netcode.py` — ghost fire path (`PredictedShip.step`, ~1686–1707)

**Remove** the local-draw gate so the ghost adds every shot its (now
input-local) fire decision produces:

```python
# BEFORE
for shot in shots:
    if len(self.local_bullets) < MAX_BULLETS:
        seq_before = s.bullet_seq
        bid = (self.local_index, seq_before)
        s.bullet_seq += 1
        if any(tuple(b.id) == bid for b in self.local_bullets):
            continue
        b = Bullet(shot.pos, shot.vel, owner=shot.owner, bid=bid)
        b.pending = False
        self.local_bullets.append(b)

# AFTER
for shot in shots:
    seq_before = s.bullet_seq
    bid = (self.local_index, seq_before)
    s.bullet_seq += 1
    if any(tuple(b.id) == bid for b in self.local_bullets):
        continue          # re-fire guard (reconcile replay) — KEEP
    b = Bullet(shot.pos, shot.vel, owner=shot.owner, bid=bid)
    b.pending = False
    self.local_bullets.append(b)
# performance safety net (presentation-only; mirrors the host's cull)
while len(self.local_bullets) > BULLET_POOL_MAX:
    self.local_bullets.pop(0)
```

Same for the missile loop (`local_missiles`, `MISSILE_POOL_MAX`, keep the
existing re-fire guard + `_mtel` telemetry).

**Keep** the re-fire guard (`if any(...): continue`) — it is what makes the
reconcile replay idempotent, and it now works *because* the seqs are in
lockstep.

### 4.4 `ship.py`, `bullets.py`

No change. `_fire` is already input-local; the snapshot/`apply_snapshot`
round-trip already carries `bullet_seq` / `missile_seq` (ship snapshot fields
21 and 24) and the projectile ids.

---

## 5. What is NOT changing (scope boundary)

- The interpolation buffer, `RenderPoint`, the ghost clock, and
  `reconcile_rewind` — untouched.
- The bullet/missile **snapshot format** — untouched (still carries the id).
- The **dedup** in `predicted_view` (skip the buffer's copy of a ghost-drawn
  projectile) — untouched; it now *always* matches because the ids are in
  lockstep.
- The **handoff** (ghost → buffer, pull-forward / offset-decay) — untouched.
- **Enemy bullets** — untouched (not player-fired; no cap problem).
- The "early enemy-bullet hit" presentation bug — **out of scope** (separate
  time-domain offset; revisit after this lands).
- **Ghost bullet spawn/angle misalignment at rest — DEFERRED (user report,
  2026-10-17, observed in a live 2P test).** Symptom: after moving the ship
  (rotating or flying) then coming to COMPLETE rest (the B button), firing —
  the bullets do not align with the front of the ship; they are off by a
  small angle, or appear to spawn a tiny distance from the "correct" visual
  spawn point. **Hypothesis (UNVERIFIED — confirm the exact settling when we
  dig in):** a render-pose-vs-physics-pose offset. A bullet spawns from the
  ghost's PHYSICS pose (`shot.pos` / `shot.vel` captured at the fire tick),
  but the ship is DRAWN at the RENDER pose (the smoothed/interpolated
  `_prev→curr` lerp + `v·_acc`). Beams hide this because they re-anchor to
  the render pose every frame; bullets can't, because they have already
  flown off. It is most visible after a hard stop because that is when the
  render pose/angle settles to a value slightly off from the physics
  pose/angle (during motion the continuous lerp keeps them close, or the
  motion masks it). **NOT a Phase 2 regression** — Phase 2 only changed how
  MANY bullets the ghost holds (removed the local-draw cap, added the
  cull-oldest net); it did not touch WHERE a bullet spawns or WHICH WAY it
  flies (still `shot.pos` / `shot.vel` from the fire tick's physics pose,
  byte-for-byte as before). Same class as the deferred "early enemy-bullet
  hit" presentation bug (render vs. sim / time-domain). **Action:** bundle
  with the early-enemy-bullet-hit into ONE focused presentation pass after
  the lockstep foundation (Phases 3–5) lands; do not interleave with the
  missile work.

---

## 6. Phases (small, sequential, testable)

**Phase 0 — Measure (no behavior change). DONE.** Added `bullet_telemetry.py`
(client-side, F3-gated, writes `bullet_debug.csv`) logging the bullet
lifecycle (FIRE / RECONCILE / HANDBACK / DEDUP) + `tools/analyze_bullet_drift.py`.
The RECONCILE row carries both the ghost's predicted `bullet_seq` + gun
cooldown and the host's authoritative `bullet_seq` (snapshot field 24) +
at-snapshot cooldown (field 18) — so the client file alone is sufficient (no
host files needed).

**Measured result (63 s 2P capture, sustained fire): DRIFT CONFIRMED.**
- **SEQ GAP bounded at exactly +1** (2021 reconciles at 0, 1785 at +1, 3 at
  −1, never above +1). Each reconcile resyncs the ghost's `bullet_seq` to the
  host's, then the ghost fires **one more** before the next reconcile.
- **The extra bullet is always the ghost's "13th"** — `1:12` when the host's
  seq is 12. 554 true-extra bullets captured (a lower bound; DEDUP uses
  change-detection). This is the double-draw.
- **COOLDOWN DELTA: 47% of reconciles |delta| > 5 ms.** The negative host
  cooldowns (down to −6.98 s) are *real*: in `Ship._fire`, `w.cooldown -= dt`
  runs unconditionally and only resets to `fire_cooldown` when the gun fires,
  so a host gun that is *ready but capped* runs its cooldown negative
  ("ready but suppressed ~7 s"); the ghost, having fired, sits at +0.01.
- **Mechanism refinement (vs. §1.2's assumption):** the ghost's local-draw
  count is *not* tiny — it climbs to ~11 (mean 4.9; at 60 shots/s with a
  0.1 s handoff delay the ghost holds ~10 of its own bullets). So the ghost's
  cap *does* bind — but the **host's WORLD cap binds first** (it counts both
  players' bullets), so the host stops firing while the ghost's *local* count
  is still < 12 → the ghost fires one more → the +1 gap. Conclusion unchanged:
  the world-vs-local cap mismatch is the driver.

**Phase 1 — Host fire path. DONE.** Removed the two world gates from
`Game._step`; added the cull-oldest safety net (`BULLET_POOL_MAX = 400`,
`MISSILE_POOL_MAX = 12` — temporary, finalized in Phase 4). `MAX_BULLETS` /
`MAX_MISSILES` stay in config (the ghost still reads them until Phase 2).
The input-echo block now echoes the raw input (which is exactly what the
host applies, so it is still correct).

- **New test `test_14_1_sustained_fire`** (single-player, isolated world):
  holding fire for 240 ticks fires EXACTLY 1 shot/tick (no burst, no dead
  zone) = 60/s; the live pool climbs to the natural 90 (not capped at 12);
  the pool never exceeds `BULLET_POOL_MAX`. PASS.
- **`test_10_10_input_echo`** section 2 rewritten: WORLD-CAP → RAW-ECHO
  (a full world no longer strips fire; the echo carries fire=True and the
  host fires). PASS.
- **`test_snapshot` latent bug fixed:** its local oracle compared
  `particles`, but `Game.snapshot()` deliberately omits them
  (presentation-only — the client regenerates its own). It passed before
  only because the world cap made collisions rare enough that Game A often
  had no live particles at tick 600; sustained fire made that deterministic.
  The test now clears both games' particles before comparing (synced state
  only). PASS.
- **Determinism:** `test_determinism` PASS (the fire decision + cull are
  pure functions of state — the host stays deterministic).
- **Full required suite PASS:** `test_determinism`, `test_snapshot`,
  `test_interpolation`, `test_10_7_bullet_dedup`,
  `test_10_3b_ghost_missiles`, `test_net_process`. (Pre-existing failures
  unrelated to this phase: `test_10_3a_ghost_beams`,
  `test_10_11c_phase_lock`, `test_network`, `test_fog_pixel` — identical on
  the clean tree.)
- **Expected temporary asymmetry (per the brief):** the ghost is UNCHANGED —
  it still gates on its local-draw count, so in 2P it will now fire *fewer*
  than the host in some cases. That is what Phase 2 fixes.

**Phase 2 — Ghost fire path. DONE.** Removed the ghost's local-draw gate
(`len(self.local_bullets) < MAX_BULLETS` in `PredictedShip.step`) and added
the matching cull-oldest safety net (`while len(self.local_bullets) >
BULLET_POOL_MAX: pop(0)`), mirroring the host. The re-fire guard
(`if any(tuple(b.id) == bid ...): continue`) is KEPT — it makes the
reconcile replay idempotent, and it now works *because* the seqs are in
lockstep. `MAX_BULLETS` is no longer read by the ghost (only the host's
missile path still reads `MAX_MISSILES` until Phase 3). The `_btel` FIRE
row's `f_cap_before` column is kept for CSV compatibility (it now just
tracks the presentation list's depth — the cap it measured is gone).

- **New test `test_14_2_ghost_lockstep`** (2P host + client ghost, same
  input stream, fixed 3-tick one-way latency, isolated world): drives 300
  ticks of sustained fire with a snapshot every 10 ticks and proves,
  measured PRE-reconcile (the prediction vs the host's actual state at the
  same instant — a post-reconcile comparison would be masked by the rewind
  replay):
  1. **SEQ LOCKSTEP** — the ghost's predicted `bullet_seq` == the host's at
     all 30 reconciles (gap 0 every time; Phase 0 was +1 at 1785/2021).
  2. **COOLDOWN LOCKSTEP** — max |ghost − host| predicted gun cooldown =
     0.0 s over 30 reconciles (Phase 0: 47% with |delta| > 5 ms).
  3. **NO GHOST-ONLY ID** — every in-flight ghost bullet id was fired by
     the host (host fired 300 total; Phase 0 captured 554 host-never-fired).
  4. **GHOST FIRES THE SAME SHOTS** — 300 shots over 300 ticks (exactly
     1/tick, == the host's), list grew past the old cap 12, stayed ≤
     `BULLET_POOL_MAX`. (The list settles ~70, below the host's natural 90,
     because the rewind replay re-steps the presentation bullets — a
     harmless presentation artifact, not a fire-path bug; a no-reconcile
     run reaches exactly 90.)
- **`test_interpolation` (i) CAP check updated:** the ghost's local-bullet
  list is no longer capped at `MAX_BULLETS`, so the check now asserts
  `max_list <= BULLET_POOL_MAX` (measured 60). The host-side `bullets —`
  buffer check (pop-in / persistent / drop-out) went from the known
  pre-existing VACUOUS FAIL to a real PASS (20 pop-in, 721 persistent, 1
  drop-out) — a Phase-1 side effect (the host now fires continuously, so
  the interpolation window exercises all three cases), not a Phase-2 change.
- **Full gate PASS:** `test_14_2_ghost_lockstep`, `test_10_7_bullet_dedup`,
  `test_determinism`, `test_snapshot`, `test_interpolation`,
  `test_10_3b_ghost_missiles`, `test_net_process`, `test_10_10_input_echo`.
  Full suite green except the 4 known pre-existing failures (identical on
  the clean tree): `test_10_3a_ghost_beams`, `test_10_9_ghost_pose`,
  `test_10_11c_phase_lock`, `test_network` (the last is a harness
  `AttributeError: 'ship5.__main__' has no attribute 'Host'`, not a
  fire/lockstep assertion).
- **Lockstep is now proven on BOTH peers.** The fire decision is
  input-local on the host (Phase 1) and the ghost (Phase 2), so the ids,
  `bullet_seq`, and gun cooldown stay in lockstep → the id-dedup always
  matches → the double-bullet is gone at the root. The 2P asymmetry
  expected after Phase 1 (ghost firing fewer than the host) is resolved.

**Phase 3 — Missiles. DONE.** Same two edits as Phase 2, mirrored for the
missile path in `PredictedShip.step`: removed the ghost's local-draw gate
(`len(self.local_missiles) < MAX_MISSILES`) and added the matching
cull-oldest safety net (`while len(self.local_missiles) > MISSILE_POOL_MAX:
pop(0)`), mirroring the host (Phase 1) and the bullet net (Phase 2). The
re-fire guard (`if any(tuple(g.id) == mid ...): continue`) is KEPT — it
makes the reconcile replay idempotent, and it now works *because* the seqs
are in lockstep. The `_mtel` FIRE telemetry (f_guard=0 created / f_guard=1
suppressed) is kept. `MAX_MISSILES` is no longer read by the ghost — and
after this edit **nothing reads it at all** (the host's missile gate was
removed in Phase 1); the constant stays in config for Phase 4 (balance).
`netcode.py` now imports `MISSILE_POOL_MAX` (dropped `MAX_MISSILES`).

- **New test `test_14_3_missile_lockstep`** (2P host + client ghost, same
  input stream — player 1 holds the missile key — fixed 3-tick one-way
  latency, isolated world with ONE stationary TestTarget 300 px ahead of
  the nose so the missile lock actually fires): drives 300 ticks with a
  snapshot every 10 ticks and proves, measured PRE-reconcile (the
  prediction vs the host's actual state at the same instant — a
  post-reconcile comparison would be masked by the rewind replay):
  1. **SEQ LOCKSTEP** — the ghost's predicted `missile_seq` == the host's
     at all 30 reconciles (gap 0 every time).
  2. **LOCK LOCKSTEP** — max |ghost − host| predicted `lock_progress` =
     0.0 over 30 reconciles (the lock is the missile's rate limiter — a
     drift here desyncs the NEXT launch even if the current seqs agree).
  3. **NO GHOST-ONLY ID** — every in-flight ghost missile id was launched
     by the host (3 in flight; host launched 6 total).
  4. **GHOST LAUNCHES THE SAME MISSILES** — 6 launches == the host's 6
     over 300 ticks (the 0.8 s lock-rebuild cadence); the presentation
     list stayed ≤ `MISSILE_POOL_MAX` (max 3; natural ~3.75).
  **Test harness note:** unlike `test_14_2` (the gun fires regardless of
  target), the missile path needs the reconcile REWIND REPLAY to see the
  target — the replay builds its enemy proxies from `render_point.now()`
  + the buffer window, so the test delivers a snapshot to the buffer each
  tick and advances the render point + latency each tick (mirroring the
  game loop). Without that the replay runs targetless and resets the
  ghost's lock (a false lockstep break).
- **Cadence correction (vs §1.3 / the Phase-3 brief's "2.5 natural
  count"):** the missile cadence is the **lock-rebuild** (0.8 s —
  `lock_progress` resets to 0 on launch and rebuilds over
  `missile_lock_time`), NOT `MISSILE_COOLDOWN` (1.2 s) — grep confirms
  `MISSILE_COOLDOWN` is read by nothing (only config.py references it).
  Natural live count = `MISSILE_LIFE / 0.8` = **3.75/player** (measured:
  6 launches in 5 s). Phase 4 should size `MISSILE_POOL_MAX` off the
  0.8 s cadence, not 1.2 s.
- **Full gate PASS:** `test_14_3_missile_lockstep`,
  `test_10_3b_ghost_missiles` (its RECONCILE-DEDUP check depends on the
  re-fire guard — still green), `test_14_2_ghost_lockstep` (bullets still
  in lockstep — no regression), `test_determinism`, `test_snapshot`,
  `test_interpolation`, `test_10_7_bullet_dedup`, `test_net_process`.
  Full suite green except the 4 known pre-existing failures (identical on
  the clean tree): `test_10_3a_ghost_beams`, `test_10_9_ghost_pose`,
  `test_10_11c_phase_lock`, `test_network`.
- **Missiles are now input-local on BOTH peers**, like bullets: the fire
  decision reads only the player's own input + weapon state (the
  lock-rebuild is the rate limiter), so `missile_seq` and lock stay in
  lockstep → the id-dedup always matches → the double-missile is gone at
  the root.

**Phase 4 — Balance. DONE.** User decision: no continuous auto-fire —
the gun is a **max-auto-burst** weapon: `GUN_BURST_SIZE = 6` shots at the
within-burst rate, then a `GUN_BURST_REST = 0.4` s gap. `FIRE_COOLDOWN =
0.049` (20/s within-burst — see the §7 rate-table correction: 0.05 would
quantize to 15/s), `BULLET_POOL_MAX = 80` (safety net, ~3× the natural
26 world). Sustained rate = 6 / (6×0.049 + 0.4) ≈ **8.6/s** — the old
effective 8/s feel, but a snappy burst instead of a dead zone. Natural
live bullets track the SUSTAINED rate: 8.6/s × 1.5 s ≈ 13/player (26
world). Bullet damage stays 1.0 (judge in the Phase-5 playtest).

Mechanism: `Weapon.burst_count` (new field, synced as weapon-snapshot
field 4) + ~5 lines in `Ship._fire` — after the Nth shot the burst
resets and the cooldown gets the extra rest. The decision is
**input-local** (a pure function of input + the ship's own
`burst_count`), so the host and the client's ghost (which runs the same
`Ship._fire`) fire the same bursts — the Phase 1–3 lockstep contract is
preserved with **no netcode change**. Releasing fire does NOT reset the
burst (no tap-tap-tap to dodge the gap); a brownout pauses it (the count
only advances when a shot actually fires). Enemies are unaffected (their
own `fire_timer` path).

Tests: `test_14_1` now proves the burst pattern (bursts of exactly 6,
max 1 shot/tick, total == an exact float-cooldown reference simulation);
`test_14_2` check 4 is now "ghost total fired == host total fired" (the
real fire-gate guard; the retired "grew past MAX_BULLETS 12" check no
longer discriminates at the low rate); `test_interpolation`'s bullet
window scan extended 20 → 120 ticks (a drop-out needs a full bullet
lifetime to be observable at the lower rate). Full gate green:
test_14_1/2/3, test_10_3b, test_render_model, test_determinism,
test_snapshot, test_interpolation, test_10_7, test_net_process.

**Phase 5 — Regression.** Full suite: `test_interpolation`,
`test_determinism`, `test_10_*`, `test_snapshot`, `test_net_process`. Live
2P playtest: the double-bullet is gone at all fire rates; the gun feels
like a snappy 6-shot burst with a rest (no continuous auto-fire); the
bullet damage (still 1.0) is judged here — if too hot, add a
`BULLET_DAMAGE` constant and re-tune; dead→respawn and reconcile are
still clean (`snap_px` / `render_snap_px` small). Re-run
`tools/analyze_bullet_drift.py` on a fresh capture (expect seq gap 0,
true-extra 0).

---

## 7. Balance decisions (resolved in Phase 4)

1. **The fire rate.** RESOLVED: the user does not want continuous auto-fire —
   the gun is a **max-auto-burst** weapon (6 shots, 0.4 s rest). The
   within-burst rate is set by `FIRE_COOLDOWN`.

   **Rate-table correction (measured, not derived):** the gun's cooldown is a
   FLOAT (`cd -= TICK; if cd <= 0: fire; cd = FIRE_COOLDOWN`), so the tick
   grid QUANTIZES the rate — the original table below was wrong. The
   achievable rates (simulated with the exact code pattern):

   | `FIRE_COOLDOWN` | original table | **actual** | live/player (×1.5 s) |
   |---|---|---|---|
   | 0.01 | 60/s | 60/s ✓ | 90 |
   | 0.02 | 50/s | **30/s** | 45 |
   | 0.05 | 20/s | **15/s** (slips to the 4-tick band) | 22.5 |
   | 0.049 | — | **20/s** (3-tick band) | 30 |
   | 0.033 | — | 30/s | 45 |
   | 0.10 | 10/s | **8.6/s** | 12.9 |

   0.05 lands just past the 3-tick boundary (0.05 > 3×0.01667) and drops to
   15/s; **0.049** is the value that actually gives 20/s (any value in
   [0.045, 0.05) works; 0.049 sits mid-band, robust to float wobble).
2. **`BULLET_POOL_MAX` = 80.** RESOLVED. ~3× the natural world count (26 at
   the 8.6/s sustained rate). Purely a safety net.
3. **Steady laser (Option A) vs. burst window (Option B).** RESOLVED: the
   burst feel is now explicit (the max-auto-burst gate), so Option B's
   motivation is moot; the laser is unchanged.
4. **Bullet damage.** RESOLVED: stays 1.0 (hardcoded in
   `Ship.register_hit` — there is no `BULLET_DAMAGE` constant). At 8.6/s
   sustained the DPS is ~2.5× the old effective 8/s × 1.0 — judge in the
   Phase-5 playtest; if too hot, add a `BULLET_DAMAGE` constant then.

---

## 8. Risks and mitigations

| Risk | Why | Mitigation |
|---|---|---|
| **Performance** — more live bullets → more bullet-vs-asteroid/ship checks | was 180 bullets × ~20 rocks = 3600 checks/tick at the 0.01 rate | RESOLVED (Phase 4): the burst gate drops the natural world count to ~26 (8.6/s sustained × 1.5 s × 2P) — ~676 checks/tick, back below the pre-bug level. No spatial work needed. |
| **Balance** — the sustained rate changes from an emergent 8/s to a chosen value | the cap was silently setting the rate | RESOLVED (Phase 4): explicit max-auto-burst (6 shots, 0.4 s rest) ≈ 8.6/s sustained — the old feel, deliberately. Damage stays 1.0; judged in the Phase-5 playtest. |
| **Ghost draws a host-culled bullet** | the cull is world-dependent; the ghost doesn't reproduce it | Imperceptible (a 3 px dot for < 1 tick of its remaining life; presentation-only, never collides on the client). The ceiling is set above the natural count so this essentially never happens |
| **A future world term re-breaks lockstep** | the whole fix rests on the fire decision being input-local | Document the invariant (§3.3) at the fire path; add a Phase-2 test that fails if ghost/host seqs ever diverge |
| **Determinism** | the host fire path changes | The fire decision + cull are pure functions of state → the host stays deterministic; the Phase-1 determinism-oracle run proves it |

---

## 9. Verification checklist

- [x] Phase 0 telemetry confirms the seq/cooldown divergence + its correlation
      with the double-draw (before) — DONE: seq gap bounded at +1, 554
      host-never-fired bullets, 47% cooldown delta. (Re-run after Phase 1–2 to
      confirm lockstep: gap stays 0, no true-extra bullets.)
- [x] Single-player: the fire pattern is as designed; pool ≤
      `BULLET_POOL_MAX`. (Phase 1: `test_14_1_sustained_fire` — 60/s steady,
      pool 90. Phase 4: re-proven as the BURST pattern — bursts of exactly
      `GUN_BURST_SIZE` 6, max 1 shot/tick, total == an exact
      float-cooldown reference sim, pool ~13, never > 80.)
- [x] 2P (bullets, Phase 4 burst): the ghost fires the SAME bursts as the
      host — ghost total fired == host total fired, at most 1 shot/tick,
      seq/cooldown lockstep still gap 0 (the burst gate is input-local, so
      the Phase-2 lockstep survives the rate change).
      (`test_14_2_ghost_lockstep` — 44 == 44 shots, gap 0 at all 30
      reconciles, cooldown delta 0.0 s, 0 ghost-only ids.)
- [x] 2P (bullets): ghost `bullet_seq` + gun cooldown == host's across N
      reconciles; **no ghost-only bullet id ever drawn** (no double-draw).
      (Phase 2: `test_14_2_ghost_lockstep` — gap 0 at all 30 reconciles,
      cooldown delta 0.0 s, 0 ghost-only ids.)
- [x] 2P (missiles): ghost `missile_seq` == host's across N reconciles; no
      ghost-only missile id ever drawn. (Phase 3: `test_14_3_missile_lockstep`
      — gap 0 at all 30 reconciles, lock delta 0.0, 0 ghost-only ids, 6
      launches == the host's 6.)
- [x] `test_10_7_bullet_dedup`, `test_10_3b_ghost_missiles`,
      `test_interpolation`, `test_determinism`, `test_snapshot`,
      `test_net_process` all pass. (Phase 1: all green; `test_snapshot`
      needed the particles-oracle fix — see §6 Phase 1.)
- [ ] Tick profile at the chosen rate is within budget.
- [ ] Live 2P playtest: the "stream from empty space" is gone; the gun feels
      like a steady weapon; dead→respawn and reconcile are still clean
      (`snap_px` / `render_snap_px` small).