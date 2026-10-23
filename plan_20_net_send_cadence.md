# Session prompt: smooth the host's net-process send cadence (fix the residual 2P stutter)

Paste everything below the line into a fresh session. It is self-contained.

---

## Task
Belt Fighter 2P, host-authoritative. The host runs three processes: **render** (pygame draw), **sim** (60 Hz authoritative), **net** (socket + snapshot send). The net process drains the sim's net Queue and sends snapshots to the client.

A recent fix (commit `8c32708`) replaced the snapshot rate limiter with **LATEST-WINS** (each loop iteration sends only the newest pending snap). That fixed a multi-second staleness bug (the game is playable again), but a **residual stutter remains**. This session fixes the send cadence.

## Hard data (from the live CSVs — do not re-derive, trust these)
- **Sim publish: clean 60.0 Hz** (0% of gaps <2 ms — there is NO burst at the source).
- **Net send: 53.9 Hz**, ragged 17–26 ms gaps (mean 18.56 ms).
- **Client receive: 53.9 Hz**, 315 gaps >30 ms (p90 = 32 ms, p99 = 33 ms).
- `serialize_snapshot` measures **0.032 ms** (negligible) — the slow loop is NOT CPU work.
- The net loop is `while True: <poll/drain/send> ; time.sleep(0.001)`. The 1 ms sleep is stretching to ~17 ms, so the loop only iterates ~54×/sec.

**Working hypothesis:** the OS is descheduling the net process (the host runs render + sim + net concurrently; the render process is CPU-heavy). The 1 ms sleep is not being honored. **This is a hypothesis, not a confirmed root cause — measure first (step 1) before picking a fix.**

## Constraint (important — read before designing)
- The host currently runs **Ubuntu (Linux) on a Mac**. Do NOT use macOS-specific APIs.
- The solution must work **cross-platform: Linux, macOS, and Windows**.
- Prefer **stdlib-only** solutions (`time.monotonic` + busy-wait) over platform-specific APIs.
- If a platform-specific API is genuinely needed (e.g. Windows `timeBeginPeriod` for timer resolution, `os.sched_setaffinity` / `os.nice` on Linux, `SetPriorityClass` on Windows), wrap it in `try/except` with a stdlib fallback so the game still runs if the API is unavailable.
- Note: Windows' default timer resolution is ~15.6 ms, so `time.sleep(0.001)` actually sleeps ~15 ms on Windows. A deadline + busy-wait hybrid (below) sidesteps that without needing `timeBeginPeriod`.

## Goal
Make the net process send snapshots at a **steady 60 Hz (16.7 ms cadence)** regardless of the loop's actual iteration rate or sleep precision. The client should receive ~one snap per frame, eliminating the ragged 17–26 ms gaps and the visible stutter.

## Micro-plan (small, testable steps — do them in order)
1. **Measure first (do not guess).** Instrument the net loop in `net_process.py` to log one row per iteration: `(sleep_requested, sleep_actual, work_time, total_period)`. Run a live 2P test, capture the CSV, and determine whether the 18.56 ms period is (a) **sleep overshoot** (the 1 ms sleep returns after ~17 ms → descheduling), (b) **work time** (the loop body is slow → unlikely, serialize is 0.03 ms), or (c) **timer imprecision** (the sleep returns early/late by a few ms). This measurement decides the fix in step 2.
2. **Pick the fix based on the measurement:**
   - **If sleep overshoot / descheduling** (most likely): the process is being starved. Two levers, both cross-platform-wrappable:
     - Raise the net process's scheduling priority so the OS wakes it on time. Linux/macOS: `os.nice(-N)` (needs permission) or `os.sched_setscheduler`; Windows: `ctypes` `SetPriorityClass`. Wrap in `try/except`, fall back to normal priority.
     - Reduce the contention: check whether the host's **render process is uncapped** (a busy draw loop burning a core) and cap it to ~60 FPS if so. This is a side fix, not the primary one.
   - **If timer imprecision** (sleep returns early/late by a few ms): use a **deadline + busy-wait hybrid**. Compute `next_send = last_send + TICK`; `time.sleep(max(0.0, next_send - now - 0.002))` (sleep until 2 ms before the deadline); then `while time.monotonic() < next_send: pass` (busy-wait the last 2 ms). This gives ~1 ms precision with stdlib only, on all three OSes. The busy-wait burns ~12% of one core (2 ms per 16.7 ms) — acceptable.
   - **If both** (overshoot AND imprecision): apply both levers.
3. **Verify.** Run a live 2P test, capture `net_snap_debug.csv`, and confirm the send cadence is a steady ~16.7 ms (p90 < 20 ms, **no gaps >30 ms**). Confirm the client's stutter is gone in a playtest.
4. **Keep LATEST-WINS** (commit `8c32708`) — it is correct and orthogonal. LATEST-WINS decides *which* snap to send (the newest, to avoid staleness); the cadence fix decides *when* to send it (a steady 16.7 ms). Do not remove or restructure LATEST-WINS.

## Files
- `net_process.py` — the net loop (`while True: ... time.sleep(WAIT)` inside `net_process_main`). The fix goes here. `WAIT = 0.001` is the current sleep.
- `test_net_process.py` — the test harness. `test_net_process_latest_wins` shows the pattern for a fake client that records arrival times; add a cadence test that asserts the send gaps are ~16.7 ms (p90 < 20 ms).
- `ARCHITECTURE.md` — update the snapshot bullet (the "LATEST-WINS" paragraph) if the send mechanism changes.

## Do NOT
- Do NOT re-add a rate limiter (it backfired — see commit `8c32708`'s message: the FIFO grew unbounded and the client rendered seconds-old state).
- Do NOT use macOS-specific APIs (the host is Ubuntu).
- Do NOT try to make the sim publish smoother (it is already clean at 60 Hz — 0% gaps <2 ms).
- Do NOT remove LATEST-WINS (it prevents staleness if the loop is ever slow).
- Do NOT brute-force a platform-specific timer API without a stdlib fallback (the game must still run on all three OSes).

## Definition of done
- `net_snap_debug.csv` from a live 2P test shows a steady ~16.7 ms send cadence (p90 < 20 ms, zero gaps >30 ms).
- The client's residual stutter is gone in a playtest.
- The fix is cross-platform (Linux/macOS/Windows) with stdlib-only or wrapped platform APIs.
- `test_net_process.py` has a cadence regression test; the full net/sim process suite is green.
- `ARCHITECTURE.md` reflects the new send mechanism.