# Belt Fighter — Performance Review

Measured 2026-10-20 in a headless container (SDL dummy driver), warmed 2P
world (~100 rocks, 3 enemies, sustained fire). Reproduce with:

```
cd /mnt
python -m ship5.profile_tick        # sim tick cost
python -m ship5.profile_render      # host draw() + component breakdown
python -m ship5.tools._perf_review  # wire + client predicted_view cProfile
python -m ship5.tools._fog_breakdown# fog cost decomposition
```

## Headline

**The sim and the netcode are NOT the bottleneck. The render's fog is ~70%
of draw cost and is mostly irreducible with CPU software blending.**

| Subsystem | Measured | Verdict |
|---|---|---|
| Sim tick (`Game._step`, 2P sustained fire, ~100 rocks) | **2.05 ms** mean / 2.39 max vs 16.67 ms budget | ✅ 8× headroom |
| Wire (host: snapshot + prune + `json.dumps`) | 0.30 ms, 7.4 KB/snap, 432 KB/s @ 60 Hz | ✅ runs in the net process, off the render thread |
| Client (`json.loads` + `push_snapshot` incl. ghost reconcile) | 0.07 + 0.13 ms | ✅ negligible |
| `render_model()` | 0.23 ms, zero GC-tracked growth | ✅ fine |
| **Host `draw()`** | **4.25 ms/frame** headless (≈235 FPS ceiling here) | ⚠️ the place to look |
| — of which **`draw_fog`** | **3.05 ms (~70%)** | 🔴 the bottleneck |
| — world entities (102 rocks, 3 enemies, 90 stars) | 1.51 ms | ⚠️ secondary |
| — HUD / local ship / `screen.fill` | < 1.2 ms | ✅ fine |

Note: the headless container has a *fast* CPU. The Mac host (the slower
machine, and the one that does the software blending) is the one that
actually feels the fog cost — expect its draw() to be a meaningful multiple
of the 4.25 ms above.

## The fog breakdown (the 3.05 ms)

- **Final full-res SRCALPHA `screen.blit`: 1.16–1.33 ms** — the single
  biggest item in the whole frame.
- `fog_surf.fill`: 0.59 ms
- **Front-lobe `rotate` (0.19) + `scale` (0.45) = 0.62 ms every frame**
- All point-light SUB blits: ~0.1 ms total (the texture caching from the
  8.x work already did its job).

Structural fact: **pygame `Surface` blits are CPU software blending even on
a 4090** — the GPU is not involved in the fog at all. The irreducible floor
with this approach is ~2.5 ms (fill + final blit + lobe transform).

## Recommendations (priority order)

### A. GPU fog — the only real ceiling-breaker
Move the fog to one GPU pass (pygame 2.1+ `Renderer` or a small fragment
shader computing `darkness = base + Σ lights` per pixel). Sub-ms on the
4090, and light count scales for free (matters in combat). Biggest win on
the **Mac host**, whose slower CPU does the software blending.

Bigger project: it changes pixel output (gate with `test_fog_pixel.py`) and
needs the software path kept as a fallback for the Mac.

### B. Viewport culling — small, safe, future-proofing
Nothing was culled before this change: only ~21% of asteroids are on-screen
in a 2-screen band (3% over a 6-screen band). Skipping off-screen
asteroids/stars/enemies/projectiles saves ~0.3 ms/frame now (measured:
4.08 → 3.77 ms draw) and keeps draw cost flat as the belt gets denser.
**Implemented** (see below).

### C. GC tuning in the render loop
`gc.freeze()` (or disable) after warmup: measured draw max 5.63→5.11 ms,
p99 4.44→4.26 ms. Modest but free. Not yet applied.

## Tested and rejected (do not re-test)

- **Half-res fog + upscale**: 3.00 vs 3.03 ms — no win (the final blit is
  full-res either way).
- **16-bit SRCALPHA surface**: final blit blew up to 23.6 ms — catastrophic.
- **Gray-multiply fog** (opaque gray + `BLEND_RGB_ADD` lights +
  `BLEND_RGB_MULT` final): pixel-correct to 1/255 but *slower* (3.14 vs
  2.36 ms) — the ADD pipeline costs more than SUB.
- **Pre-rotated front-lobe ring**: 302 MB at full res / 38 MB at half-res
  for only 0.16 ms saved.
- **Star sprite-sheet blit**: 0.59 ms vs 0.03 ms for 90 per-star circles —
  per-star `draw.circle` is faster.

## Viewport culling (implemented)

**Where:** `game.py`
- `_view_bounds(cam)` — module helper returning the world-space screen
  rect inflated by `CULL_MARGIN` (config, 80 px). A pure function of the
  camera, so it is deterministic and safe for the pixel-parity gate.
- `Game.draw()` (host / single-player): culls asteroids, enemies, bullets,
  missiles, enemy bullets, and particles against the bounds.
- `Game.predicted_view()` (client): culls asteroids, enemies, and the
  buffer-projectile draw.

**Client vs host (the 2P asymmetry the design must respect):**
- The **host** draws every entity from the authoritative `RenderModel`
  (`model["asteroids"]`, `model["enemies"]`, …) at the sim's current time.
- The **client** draws remote entities from the **interpolation buffer at
  the render point** (`pos['asteroids']`, `pos['enemies']`, `pos['bullets']`
  — `rp = newest_snapshot − delay`, a point in the *past*), and its own ship
  from the prediction ghost. Culling uses the *interpolated* positions
  (what is actually drawn), so a rock that is off-screen at the render point
  is skipped even if it is on-screen at the newest snapshot.
- Culling is **presentation-only**: it skips the *draw* of an entity. It
  never touches the dedup / handoff / hit-FX logic, which still runs over
  the full buffer (a ghost bullet that is off-screen is still deduped and
  handed off correctly — only its sprite is skipped). The local ship, fog,
  HUD, beams, and scan pulse are never culled (the local ship is always
  on-screen by camera construction; the fog is full-screen by design).

**Margin:** `CULL_MARGIN = 80` px > largest rock radius (50) + vertex jitter
(~12) + edge stroke, so a rock is never culled while any part of it is
visible. The same margin is safe for enemies (~20 px hulls) and projectiles.

**Measured win:** with culling on vs. off (2P, ~85 rocks, sustained fire,
headless): draw mean **4.08 → 3.77 ms** (−0.31 ms, ~8% of draw), p99
4.72 → 4.52 ms. The win scales with rock count — at ~100 rocks only ~21%
are on-screen, so a denser belt saves proportionally more.

**Test:** `test_viewport_culling.py` —
- PURE-BOUNDS: `_view_bounds` is the camera rect inflated by `CULL_MARGIN`;
  `_in_view` is the inclusive point-in-rect.
- HOST-ON / HOST-OFF / HOST-MARGIN: a rock at screen center is drawn
  (pixel diff > 0), a rock far off-screen is culled (pixel diff == 0), and
  a rock off-screen but within `CULL_MARGIN` is still drawn (the margin is
  respected — a rock is never culled while any part of it is visible).
- CLIENT-ON / CLIENT-OFF / CLIENT-MARGIN: the same three via the client's
  `predicted_view` (the interpolation-buffer path).
- STRUCTURAL: both `draw()` and `predicted_view()` compute
  `bounds = _view_bounds(self.cam)` and guard the loops with `_in_view`.

The pixel-diff method renders a baseline frame (no rock) and a frame with a
single rock at a controlled world position (camera locked at the ship, zero
velocity, lead 0), then counts pixels that changed. A drawn rock changes its
pixels; a culled rock changes none (the sprite is skipped, so the frame is
identical to the baseline).

**Regression gate:** `test_render_model.py` (pixel parity + determinism)
still passes — the cull is a pure function of the camera, so the
never-stepped shell and the live game render identically.