"""Phase 0 (plan_14) bullet fire-drift analyzer.

Reads `bullet_debug.csv` (written by bullet_telemetry, F3-gated, client-only)
and answers the Phase 0 question: does the ghost's fire decision diverge from
the host's when the host's WORLD bullet cap binds?

The signals:

  1. SEQ GAP (primary): per RECONCILE row, gap = r_gseq_before - r_snap_bseq
     (the ghost's PREDICTED bullet_seq minus the host's authoritative
     bullet_seq at the snapshot). A monotonic counter, so a growing /
     non-zero gap = the ghost fired shots the host did not (its local-draw
     cap didn't bind while the host's world cap did). With the cap fixed,
     the gap should stay ~0.

  2. COOLDOWN DELTA (secondary): per RECONCILE row, cd_delta = r_gcd -
     r_host_cd (the ghost's predicted gun cooldown minus the host's
     at-snapshot cooldown, both at the snapshot's time). A persistent
     non-zero delta = the two guns are out of phase (the ghost reset its
     cooldown on a shot the host didn't fire).

  3. DOUBLE-DRAW (the visible symptom): per DEDUP row, a ghost bullet id
     that is in NEITHER the buffer's drawn NOR skipped set = the ghost drew
     a bullet the host never fired (its id drifted from the host's, so the
     buffer has no matching copy to dedup against). Count these.

  4. CAP BINDING (context): FIRE rows' f_cap_before (the ghost's local-draw
     count before the append) — if it stays tiny (0-2) while the host's
     world count is near MAX_BULLETS, the ghost cap is not binding (the
     divergence enabler).

Run: python tools/analyze_bullet_drift.py   (from the workspace root)
"""
import csv
import os
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(os.path.dirname(_HERE), "bullet_debug.csv")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(label, xs):
    if not xs:
        print("  %-16s n=0" % label)
        return
    print("  %-16s n=%-5d mean=%8.3f  p50=%8.3f  p95=%8.3f  max=%8.3f"
          % (label, len(xs), st.mean(xs), pct(xs, 50), pct(xs, 95),
             max(xs)))


def parse_ids(s):
    """'1:0|1:1' -> {(1,0),(1,1)} (empty string -> empty set)."""
    if not s:
        return set()
    out = set()
    for part in s.split("|"):
        if ":" in part:
            p, q = part.split(":")
            out.add((int(p), int(q)))
    return out


def main():
    if not os.path.exists(_PATH):
        print("No bullet_debug.csv — run a 2P session with F3 (debug_net) "
              "on, holding fire, then re-run this.")
        return
    rows = load(_PATH)
    rec = [r for r in rows if r["event"] == "RECONCILE"]
    ded = [r for r in rows if r["event"] == "DEDUP"]
    fir = [r for r in rows if r["event"] == "FIRE"]
    if not rec:
        print("bullet_debug.csv has no RECONCILE rows — the client never "
              "reconciled (no 2P session, or F3 was off).")
        return

    t0 = float(rec[0]["t"])
    t1 = float(rec[-1]["t"])
    print("== SESSION: bullet_debug %.1f s (%d reconciles, %d dedup, "
          "%d fire), %.3f -> %.3f" % (t1 - t0, len(rec), len(ded),
                                      len(fir), t0, t1))

    # ---- 1. SEQ GAP (primary drift signal) ----
    gaps = []
    for r in rec:
        g = int(r["r_gseq_before"]) - int(r["r_snap_bseq"])
        gaps.append((float(r["t"]), g))
    gvals = [g for _, g in gaps]
    print("\n== 1. SEQ GAP (ghost predicted bullet_seq - host authoritative)")
    dist("gap", gvals)
    nonzero = [g for g in gvals if g != 0]
    print("  non-zero gaps: %d / %d (%.1f%%)"
          % (len(nonzero), len(gvals),
             100.0 * len(nonzero) / len(gvals) if gvals else 0.0))
    if nonzero:
        worst = sorted(gaps, key=lambda x: abs(x[1]), reverse=True)[:8]
        print("  worst |gap| (t, gap):")
        for t, g in worst:
            print("    t=%.3f  gap=%+d" % (t, g))
    else:
        print("  (no drift — the ghost's bullet_seq tracks the host's)")

    # ---- 2. COOLDOWN DELTA (secondary) ----
    cds = []
    for r in rec:
        if r["r_gcd"] and r["r_host_cd"]:
            cds.append((float(r["t"]),
                        float(r["r_gcd"]) - float(r["r_host_cd"])))
    cdvals = [d for _, d in cds]
    print("\n== 2. COOLDOWN DELTA (ghost predicted - host at-snapshot, s)")
    dist("cd_delta", cdvals)
    if cdvals:
        big = [d for d in cdvals if abs(d) > 0.005]
        print("  |delta| > 5 ms: %d / %d (%.1f%%)"
              % (len(big), len(cdvals),
                 100.0 * len(big) / len(cdvals) if cdvals else 0.0))

    # ---- 3. DOUBLE-DRAW (the visible symptom) ----
    dd_total = 0
    dd_frames = 0
    dd_examples = []
    for r in ded:
        ghost = parse_ids(r["d_ghost_bids"])
        drawn = parse_ids(r["d_buf_bdrawn"])
        skipped = parse_ids(r["d_buf_bskipped"])
        # A ghost bullet with no buffer counterpart (not drawn, not
        # skipped) = the ghost drew a shot the host never fired.
        orphan = ghost - (drawn | skipped)
        if orphan:
            dd_total += len(orphan)
            dd_frames += 1
            if len(dd_examples) < 8:
                dd_examples.append((float(r["t"]),
                                    sorted(orphan), sorted(ghost)))
    print("\n== 3. DOUBLE-DRAW (ghost bullets with no buffer counterpart)")
    print("  orphan bullets across DEDUP rows: %d (in %d frames)"
          % (dd_total, dd_frames))
    if dd_examples:
        print("  examples (t, orphan_ids, all_ghost_ids):")
        for t, orphan, ghost in dd_examples:
            print("    t=%.3f  orphan=%s  ghost=%s"
                  % (t,
                     ",".join("%d:%d" % i for i in orphan),
                     ",".join("%d:%d" % i for i in ghost)))
    else:
        print("  (no double-draw — every ghost bullet has a buffer copy)")

    # ---- 4. CAP BINDING (context) ----
    caps = [int(r["f_cap_before"]) for r in fir if r["f_cap_before"] != ""]
    print("\n== 4. GHOST CAP (local-draw count before append, FIRE rows)")
    dist("cap_before", [float(c) for c in caps])
    if caps:
        print("  (the ghost cap binds at >= MAX_BULLETS; if this stays "
              "tiny while the host's world count is near the cap, the "
              "ghost cap is not binding — the divergence enabler)")

    # ---- verdict ----
    print("\n== VERDICT")
    drift = bool(nonzero) or (cdvals and any(abs(d) > 0.005 for d in cdvals))
    if drift and dd_total:
        print("  DRIFT CONFIRMED: the ghost's fire decision diverges from")
        print("  the host's (seq gap / cooldown delta) AND a double-draw is")
        print("  visible. The cap is the driver. Proceed to Phase 1.")
    elif drift:
        print("  DRIFT PRESENT (seq/cooldown) but no double-draw captured")
        print("  this session — the divergence is real; the double-draw may")
        print("  need sustained fire to surface. Proceed to Phase 1.")
    elif dd_total:
        print("  DOUBLE-DRAW visible but no seq/cooldown drift captured —")
        print("  unexpected; inspect the RECONCILE rows manually.")
    else:
        print("  NO DRIFT, NO DOUBLE-DRAW — the cap is not the driver in")
        print("  this session (or fire was not sustained). Re-capture with")
        print("  continuous fire to confirm.")


if __name__ == "__main__":
    main()