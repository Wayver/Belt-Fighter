# Plan 16 — GPU Fog (the ceiling-breaker)

**Status:** BRIEF (feasibility verified, not started).
**Scope:** move the fog from CPU software blending to one GPU pass
(`pygame._sdl2.video.Renderer` + `Texture` targets). Presentation-only: the
sim, the wire, the netcode, and the world rendering (asteroids, ships,
beams, HUD) are all untouched. The GPU does the **fog + the final
composite over the world**; the world itself stays on the CPU
`pygame.Surface` path.

**Session fit:** M0 (probe + fallback scaffold) is one focused edit cluster
inside a single 64k session. M1 (the GPU pass) is a second session. M2/M3
are separate decisions.

---

## 1. Why (the measured motivation)

From the performance review (PERFORMANCE.md): host `draw()` is 4.25 ms
headless, of which `draw_fog` is **3.05 ms (~70%)**. The fog breakdown:

- Final full-res SRCALPHA `screen.blit`: **1.16–1.33 ms** (the single
  biggest item in the frame)
- `fog_surf.fill`: 0.59 ms
- Front-lobe `rotate` + `scale`: 0.62 ms every frame
- Point-light SUB blits: ~0.1 ms (already cached)

Structural fact: **pygame `Surface` blits are CPU software blending even on
a 4090** — the GPU is not involved in the fog at all. The irreducible
software floor is ~2.5 ms. The Mac host (the slower CPU, the one that
actually feels it) is a meaningful multiple of the headless numbers.

A GPU pass replaces `fill + lobe transform + full-res SRCALPHA blit` with
`clear + a few ADD blits + one MOD composite + present` — sub-ms on the
4090, and **light count scales for free** (matters during combat, when the
point-light count peaks).

**Fog-only, not everything — by necessity, not just convenience:**
`pygame.draw` (polygon/circle/line) and font rendering only work on
`Surface`s. The renderer API has `draw_point/line/rect` but **no polygon**
and **no text**. Our world is mostly polygons (rocks, hulls) + text (HUD).
Rewriting all of that to GPU primitives is a massive, high-risk effort for
little gain — and the fog is the bottleneck, so fog-only captures the win.

---

## 2. Feasibility (verified by probing the real API, pygame 2.6.1)

### 2.1 The critical constraint (tested, not assumed)

> **`Renderer.from_window(window)` FAILS in the headless/dummy
> environment.** The window creates fine, but renderer creation raises
> `error('')` (empty `SDL_GetError`); `SDL_RENDERER=software` and `W.show()`
> do not help. The dummy video driver cannot back a renderer.

Consequences:

1. The GPU path **cannot run in the CI/test container at all**.
2. The **software fallback is MANDATORY** — it is what keeps every existing
   test passing headless. It is not an optional "nice to have."
3. The GPU path can only be validated on a **real display** (the 4090 box /
   the Mac). Its pixel gate therefore needs a `--display` mode and is
   skipped in CI.

### 2.2 API surface (all confirmed present)

| Capability | Status |
|---|---|
| `Renderer.from_window(window)` | ✅ (1 arg only, no driver param; fails headless) |
| `renderer.target` (→ a `Texture`, or `None` = window) | ✅ |
| `renderer.blit(source, dest, area, special_flags)` | ✅ — **`special_flags` "have no effect"**; blend mode is per-TEXTURE, not per-blit |
| `renderer.clear` / `present` | ✅ |
| `renderer.to_surface()` | ✅ — **"very slow"; one-shot parity readback only, never per-frame** |
| `Texture(renderer, size, depth, static/streaming/target=True)` | ✅ — `target=True` = VRAM render target |
| `Texture.from_surface(renderer, surface)` | ✅ — upload a `Surface` |
| `texture.update(surface)` | ✅ — **"fairly slow"; for static textures** |
| `texture.blend_mode` (settable property) | ✅ — the per-overlay blend mechanism |

Blend constants are **not exported by name** — use the SDL2 ints:
`NONE=0, BLEND=1, ADD=2, MOD=4, MUL=8`.

---

## 3. Architecture

```
CPU (unchanged):  world -> screen Surface (fill, stars, rocks, ships,
                   beams, HUD, fog lights... exactly as today)
        |
        |  Texture.from_surface(renderer, screen)   [per-frame upload]
        v
GPU:  world_tex (NONE)  ─┐
                          ├─> composite on window target ─> present()
fog_texture (MOD)      ─┘
        ^
        |  built each frame on the fog_texture target:
        |    clear(base_brightness)
        |    blit(ship_light_circle, ADD)
        |    blit(front_lobe, ADD)
        |    blit(point_light_i, ADD) ...
```

**Two compositing stages; the blend modes fall out naturally:**

**Stage 1 — build the light map** (target = `fog_texture`):
- `clear` to the base brightness (≈ `255 − FOG_ALPHA` ≈ 180, the ambient
  floor — the exact value is tuned in M1 to match the software fog).
- Blit the ship light circle + front lobe + each point light with
  **`ADD`** (each is a separate radial-gradient texture,
  `blend_mode=ADD`). This is the "soft radial gradient for the ship's light
  circle."

**Stage 2 — composite over the world** (target = window):
- Blit the uploaded world texture with **`NONE`** (opaque, just place it).
- Blit the `fog_texture` with **`MOD`** — `world × light_map / 255`. Where
  the map is bright (lights), the world shows; where it is the base floor,
  the world is dimmed. **This is the `SDL_BLENDMODE_MOD` on the
  `fog_texture`.**

**Why the composite must be on the GPU:** the 1.16–1.33 ms full-res blit is
the fog's biggest cost. If we composited on the CPU (blit the fog texture
back onto the screen `Surface`), we would *not* save it. The ceiling-break
only comes from compositing on the GPU (upload world → blit fog over it →
`present()`), which replaces the full-res CPU blit with a GPU upload + GPU
composite.

**Per-overlay blend dispatch (criterion 3):** the same per-texture pattern
extends to any overlay that moves to the GPU composite — weapon beams get
their own texture with `blend_mode=ADD` (glow), sensor tags with
`blend_mode=NONE` (crisp). Note beams/tags are currently drawn in the
**world pass** (CPU), so pulling them into the GPU composite is a scope
expansion (M2), not part of the fog win.

---

## 4. The `GameRenderer` abstraction (criterion 4)

A single seam with two backends, selected by a capability probe + env
override:

```
GameRenderer (abstract)
  ├── SoftwareFog   # today's pygame.Surface path, byte-identical
  └── GpuFog        # pygame._sdl2.video path
```

- **Probe:** try `Renderer.from_window(window)`; on failure (headless / no
  GPU) select `SoftwareFog`.
- **Override:** `BELT_FOG=auto|software|gpu` env var forces the choice
  (default `auto`). `gpu` on a machine without a renderer is a hard error
  (a misconfiguration, not a silent fallback).
- **Contract:** both backends expose the same `draw_fog(screen, ship, cam,
  lights)` entry point the call sites already use. `SoftwareFog` is the
  existing `fog.py` code, untouched.

---

## 5. Milestones

### M0 — Capability probe + fallback scaffold (no pixel change, CI-verifiable)

- `GameRenderer` abstraction + `SoftwareFog` (wraps today's `fog.py`,
  byte-identical) + `GpuFog` stub (raises "not implemented" until M1).
- The probe + `BELT_FOG` override.
- Wire the call sites through the abstraction (they already call
  `draw_fog`; this is a routing change, not a behavior change).

**Gate:** in headless, `auto` → `SoftwareFog`, so **every existing test
passes unchanged** (run the full suite). This milestone is safe and fully
testable in CI.

### M1 — GPU fog light-map + composite (display-only validation)

- Implement Stage 1 (light map: `clear` + `ADD` gradients) + Stage 2
  (composite: world `NONE` + fog `MOD` + `present`).
- Tune the base-brightness value, the `ADD` gradients, and the `MOD`
  composite to match the software fog's output.
- **Gate:** extend `test_fog_pixel.py` to compare software vs GPU fog
  (max/mean per-pixel delta, the same metric the 8.x fog work used). This
  **can only run on a real display** — it gets a `--display` mode and is
  skipped in CI. Expect the same sub-perceptual tolerance as the 8.x work
  (max ~5/255, mean ~0.02).

### M2 — Per-overlay blend dispatch (beams `ADD`, tags `NONE`) — *optional*

Only if we decide beams/tags should glow/crisp on the GPU. Each becomes its
own texture with its own `blend_mode` in the composite. **Scope expansion —
a separate decision, not part of the fog win.**

### M3 — Measure + cutover

- Profile GPU vs software fog on the **4090 box and the Mac** (different
  renderer drivers: OpenGL vs Metal). Confirm the win (expect ~2.5–3 ms off
  draw).
- Measure the **world-upload cost** (`from_surface` per frame) — the one
  cost we cannot predict from here.
- Then make `auto` default to GPU on capable displays.

---

## 6. Risks (ranked)

1. **Blend math is untestable in CI.** The exact `ADD`/`MOD`/base-brightness
   values that reproduce the software fog can only be validated on a real
   display. This is the main reason M1 is gated on a display-only test +
   manual validation, and why the software fallback must stay the CI
   default.
2. **World-upload cost is unmeasured.** `from_surface` each frame is a
   CPU→GPU upload. We *expect* it is cheaper than the full-res CPU blit it
   replaces, but we must measure it (M3) before declaring the win.
3. **Cross-driver variance.** OpenGL (4090) and Metal (Mac) both go through
   SDL2's abstraction, so the blend math *should* be identical, but validate
   on both.
4. **`to_surface`/`update` are slow** — keep them out of the per-frame path
   (readback only for the one-shot parity test).

---

## 7. Out of scope (explicitly)

- The world (asteroids, ships, beams, HUD) — stays on the CPU `Surface`
  path (no polygon/text primitives on the renderer).
- The sim, the wire, the netcode, the interpolation buffer — untouched
  (presentation-only).
- Region-based fog updates (the "only update the region around the ship"
  idea) — rejected: the lights move with the ship, so the dirty region is
  the whole screen; and the final full-res composite is the cost, which a
  region update does not avoid. The GPU pass is the correct fix for the
  composite cost.