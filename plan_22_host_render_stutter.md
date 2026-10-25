# Plan 22 — Host / SP local-ship stutter: measure, then fix

## Symptom
The host player (2P) and the single-player player see an occasional (SP) or
somewhat periodic (2P host) snap in their OWN ship on screen. The remote
(client) ship is already smooth (plans 10–21). This is the local render path.

## Architecture (confirmed in code — do NOT re-derive, just trust this)
- SP and 2P-host are the SAME architecture: sim runs on a `SimThread`
  (real-time 60 Hz clock, dynamic sleep to tick boundaries), the render thread
  is a "local client" of that sim. (`__main__.py` SP loop ~L1574, host loop
  `run_host` ~L229.)
- The sim thread publishes ONE model per wake (latest-wins), stamped with
  `sim_time` (T) and `step_alpha` = the sim's own `acc/STEP` at publish
  (`sim_thread.py` ~L240–300; `g.acc = acc` is written before `render_model()`).
- The render thread computes its OWN alpha via `_HostRenderClock`
  (`__main__.py` L74–131):
    - NEW model  → hard re-anchor: `t = T + step_alpha*STEP` (dt NOT added)
    - SAME model  → `t += dt` (advance at 1x real time)
    - `alpha = clamp((t - T)/STEP, 0, 1)`
- The rendered pose = `_ship_pose(pack, alpha)` = `lerp(prev_pos, pos, alpha)`,
  where prev/curr span the window `[T-STEP, T]` (`game.py` L564, L2613, L2645).
  Both loops overwrite the model's `step_alpha` with the render-clock alpha
  before `draw()` (host L712, SP L1631).

## The theory under test (from the prior session)
> The render clock's alpha is driven by the render frame rate vs the sim tick
> rate, which are not locked. The model is published per-wake; the render clock
> re-anchors alpha on each new model. When render frame and sim tick are not
> 1:1, alpha sawtooths between `step_alpha` and 1.0 → the rendered position
> oscillates (periodic stutter, worse at speed).

## Refined hypothesis space (what the measurement must distinguish)
The hard re-anchor is the mechanism that converts cadence mismatch into a
position jump. Three concrete causes all flow through it:
- **C1 — re-anchor pull-back:** t (advanced by dt) runs ahead of the new
  stamp (which is ~publish-latency old); re-anchor snaps t backward. Small wobble.
- **C2 — re-read sawtooth:** the same model is re-read for 2+ frames (sim tick
  slightly behind the render frame); alpha ramps 0→1 then snaps back to ~0 on
  the next re-anchor. This is the literal "sawtooth between step_alpha and 1.0."
- **C3 — batched-wake T-jump:** the OS wakes the sim ~1 tick late ~19% of the
  time (per `sim_thread.py` comments), so a wake steps 2 ticks and the per-wake
  model's T jumps 2·STEP; the re-anchor snaps t forward ~1·STEP. Most likely
  source of the *periodic* 2P-host snap.

All three point at the same fix class (smooth the render clock's time estimate
instead of hard re-anchoring), which de-risks the plan — but the measurement
decides which dominates and sets the fix's parameters.

## ✅ S3 VERDICT (SP capture, 2532 frames / 41.5 s, `render_clock.csv`)
**Confirmed + refined.** The stutter is a **fall-short** (not jump-forward):
- **58 slow frames @ 1.40/s**, **0 fast frames**. Shortfall **median 6.85 px,
  p90 8.32, max 8.82**, and it **scales with speed** (3.4 px <300 px/s →
  5.6 px ≥300 px/s) — matches the "worse at speed" report.
- **36/58 slow frames are immediately after a same-model re-read.**
- `alpha` == `model_step_alpha` (the sim's `acc/STEP`) on **nearly every
  frame** → the render clock re-anchors on every new model; its free-run
  branch is dead. It is effectively just using the sim's accumulator.
- The sim's `acc` at publish **oscillates** (0.086 → 1.000) because the render
  thread and sim thread are both ~60 Hz but **not phase-locked** (a beat
  frequency → "somewhat periodic"). When the rendered pose
  `lerp(prev, curr, alpha)` sees `alpha` drop (e.g. 1.0 → 0.18), it falls
  short by `(Δalpha)·v·STEP` ≈ 6.85 px at 520 px/s. **That is the snap.**

**Refined mechanism (supersedes C1/C2/C3 as the primary cause):** the render
clock re-anchors to the sim's **`acc`** (`t = T + acc·STEP`) on every new
model, coupling the rendered pose to the sim's publish *phase*. The fix is to
**stop re-anchoring to `acc`**: free-run `t` at 1× real time and apply only a
**bounded correction toward the sim's time `T`** (not `T + acc·STEP`). That
decouples the rendered pose from the sim's publish phase → smooth. (C2
re-reads still occur ~0.96/s but are no longer the stutter source once the
clock free-runs.)

## Why current CSVs can't answer it
`ghost_clock.csv` / `net_debug.csv` / `client_snap_debug.csv` all describe the
CLIENT's ghost + HostTimeEstimator. None capture the host/SP render clock's
`alpha`, the model's `step_alpha`, or the rendered local-ship pose. The only
host-side log is the F3 `t,raw_dt,draw_dt,sim_time,alpha` line (host loop only,
not SP) — it has `alpha` but not the pose, the model stamp, or model identity,
so it can't separate C1/C2/C3.

---

## Session plan (each session is context-bounded: read a few small ranges,
## write ONE focused file, run ONE check)

### S1 — Instrumentation (additive, behind F3 `debug_net`)  ✅ DONE
Goal: a per-frame "render clock" CSV for BOTH SP and host that lets the
analyzer separate C1/C2/C3.

**Implemented** in `__main__.py`:
- `_render_clock_log(game, t, raw_dt, dt, model, alpha)` helper (after
  `_HostRenderClock`). Writes to `render_clock.csv` (workspace root), opened
  lazily on the first row, gated by `game.debug_net` (F3).
- Called from BOTH loops right after `render_clock.advance(...)`: host loop
  (`run_host`, ~L700) and SP loop (~L1700). Setup (`_dbg_rc_log_path`,
  `_dbg_rc_log_f`, `_rc_model_id`, `_rc_model_gen`) added to both; file closed
  in both `finally` blocks.
- **Exact columns** (S2 must parse these):
  `t, raw_dt, dt, model_gen, model_sim_time, model_step_alpha, alpha,
   ship_x, ship_y, ship_speed`
  - `model_gen` = monotonic id that bumps on each NEW model object (robust to
    `id()` reuse) — the analyzer uses it to find same-model re-read runs (C2).
  - `model_sim_time` = T = `model["sim_time"]`; `model_step_alpha` = the sim's
    acc/STEP at publish; `alpha` = the render clock's alpha this frame.
  - `ship_x/y` = the RENDERED local-ship pose `lerp(prev_pos, pos, alpha)`
    (the thing that snaps); `ship_speed` = `|pack["vel"]|`.
  - Local ship = `model["players"][game.local_index]` (index 0 for SP + host).
- Verified: `py_compile` clean; synthetic helper test (new/same/None model)
  writes correct rows + `model_gen` 1,1,2; `test_sim_thread`,
  `test_render_model`, `test_determinism` all PASS.
- **Done when:** running SP and 2P-host with F3 on writes a non-empty CSV with
  all columns; a 10-line `head` looks sane (alpha in [0,1], T increasing).
- Add a small shared helper in `__main__.py`, e.g.
  `_render_clock_log(f, t, raw_dt, dt, model, alpha, rpos, ship_vel)`, and call
  it from BOTH loops right after `alpha = render_clock.advance(...)`.
- Columns (minimal; the rest is derived in the analyzer):
  `t, raw_dt, dt, model_id, model_sim_time, model_step_alpha, alpha,
   ship_x, ship_y, ship_speed`
  - `model_id` = `id(model)` (detects new-vs-same + re-read runs)
  - `model_sim_time` = `model["sim_time"]` (T)
  - `model_step_alpha` = `model["step_alpha"]` (the sim's acc/STEP at publish)
  - `ship_x/y` = the rendered pose (reuse `_ship_pose(local_pack, alpha)`)
  - `ship_speed` = `|pack["vel"]|` (normalizes wobble by speed)
- Reuse the existing F3 `debug_net` gate + `game._dbg_sim_log_*` pattern; give
  SP the same log it currently lacks. Do NOT change any sim/render behavior.
- **Done when:** running SP and 2P-host with F3 on writes a non-empty CSV with
  all columns; a 10-line `head` looks sane (alpha in [0,1], T increasing).

### S2 — Analyzer (validate on synthetic data FIRST)
Goal: `tools/analyze_render_clock.py` that reads the CSV and emits a verdict.
- Per frame compute:
  - `render_sim_time = T - STEP + alpha*STEP` (the sim time actually rendered)
  - `dst = render_sim_time - prev_render_sim_time` (per-frame advance; smooth ≈ dt)
  - `T_jump = T - prev_T` on new-model frames (C3 if > 1.5·STEP)
  - `same_model_run` = length of the current same-model-id run (C2 if ≥ 2)
- Flag **snap frames**: `|dst - dt| > THRESH` (start THRESH ≈ 0.5·STEP) or
  `dst < 0` (backward) or `dst > 1.5·dt` (forward lurch).
- Report: snap rate (%), snap magnitude distribution (px, using ship_speed·|dst|),
  and the **breakdown by cause** (C1 re-anchor-pull / C2 re-read / C3 T-jump) by
  checking which signature each snap frame carries.
- **Validate on a synthetic CSV** (generated in-test: a clean 1:1 case → 0 snaps;
  a C2 case → sawtooth; a C3 case → periodic forward lurch) BEFORE touching a
  real capture, so the tool is trusted.
- **Done when:** synthetic cases produce the expected verdicts; script runs clean
  on an empty/short CSV without crashing.

### S3 — Capture + verdict  ✅ SP DONE (2P-host pending)
Goal: real numbers, SP and 2P-host.
- **SP: DONE** (see "S3 VERDICT" above). 58 slow frames @ 1.40/s, median
  shortfall 6.85 px, scales with speed, 36/58 right after a re-read.
- **2P-host: TODO** — run the 2P host with F3 on, fly at speed ~30–60 s, save
  `render_clock.csv`, run the same analysis. Confirm the same mechanism (and
  whether the 2P-host snap is more periodic, as reported).
- **Done when:** both SP and 2P-host show the same fall-short signature.

### S4 — Fix  ✅ IMPLEMENTED + SIMULATED (real-capture verification pending)
The fix is in `_HostRenderClock` ONLY (NOT the sim, NOT the wire). The problem
was that `advance()` re-anchored `t = T + step_alpha·STEP` on every new model,
coupling the rendered pose to the sim's publish *phase* (its `acc`).

**Implemented design** (`__main__.py` `_HostRenderClock`):
- Every frame: `t += dt` (1× real time — the host's sim rate).
- On a NEW model: gentle **EMA correction toward `T`** (offset 0, NOT
  `T + acc·STEP`): `t += A·(T - t)` with `A = 0.2`. Keeps `t` honest (no
  drift after a stall, no permanent clamp) WITHOUT inheriting the sim's `acc`
  oscillation.
- `alpha = clamp((t - T)/STEP, 0, 1)` as before.
- **Offset is 0 (track `T`, not `T + 0.5·STEP`)** — the key finding. With
  steady-state alpha ~0, a same-model re-read frame's free-run only pushes
  alpha to ~0.96 (just under the clamp), so the pose advance is preserved.
  With offset 0.5·STEP (steady-state alpha ~0.5), a re-read frame pushes alpha
  to ~1.46 (clamped to 1.0) and the pose advances only 0.5·STEP → a fall-short
  stutter. Measured on the SP capture: **offset 0 → 0 slow frames; offset
  0.5·STEP → 38 slow frames.** The rendered pose is `T - STEP` (full-step
  latency — the SAME latency the old design had), just smooth now.

**Simulated against the real SP capture (strict >20% thresholds):**
- OLD: 38 slow, 1 fast, jitter std 0.98 px, max dev 8.32 px.
- NEW (implemented): **0 slow, 0 fast, jitter std 0.32 px, max dev 1.59 px.**

**Tests:** updated `check_render_clock()` in `test_sim_thread.py` for the new
behavior (steady-state alpha ~0, NOT the sim's acc; hiccup clamps to 1;
recovery eases to ~0). `test_sim_thread`, `test_render_model`,
`test_determinism` all PASS.

**Remaining (S4 close-out):**
- **Real-capture verification:** run SP + 2P-host with F3 on, fly at speed,
  confirm the slow-frame rate is ≈ 0 in the fresh `render_clock.csv` (the
  simulation predicts it; the live capture confirms it end-to-end).
- **2P-host capture** (completes S3): confirm the same mechanism + fix.
- **Done when:** a fresh SP + 2P-host capture shows slow-frame rate ≈ 0 at
  speed, no new drift/pin, and the three test suites still pass.

## Context-safety rules (for every session)
- Never read all of `game.py` (4700 lines) or `netcode.py` (2776). Read only the
  small ranges listed above.
- One focused file change per session; run one check; stop.
- Keep the instrumentation additive and behind F3 so normal runs are untouched.

## Open questions to resolve in S3 (not blockers)
- Is the 2P-host snap truly periodic at ~1/5 frame (→ C3, the 19% batch rate)
  or irregular (→ C2 phase drift)?
- Does SP's "occasional" snap correlate with GC pauses / stalls (raw_dt spikes)
  rather than steady-state cadence?
- How much does snap px scale with ship_speed (confirms the "worse at speed"
  report and sets the THRESH for the analyzer)?