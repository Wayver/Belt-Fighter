"""Session 9.x M1: the RenderModel builder is a faithful, plain-data,
non-mutating, deterministic capture of everything draw() reads.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_render_model

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init(), so no window
opens.

This is the M1 gate. render_model() is ADDITIVE (draw() is untouched), so
the test proves four things:

  1. SHAPE     — the model carries every field draw() reads (see the
                 FIELD_MAP below for the explicit draw()-read -> model-field
                 mapping the plan's "carries every field draw() reads"
                 claim rests on).
  2. PARITY    — every field the model carries EQUALS the live sim state at
                 the same instant. Checked at EVERY tick of a scripted run,
                 so whenever beams/particles/contacts/enemies are non-empty
                 the parity is asserted on real data, not just empty lists.
  3. PLAIN     — the model is plain data only (tuples/lists/dicts of
                 numbers/strings/None): no pygame objects, no live entity
                 references. This is what makes the M3 atomic reference
                 swap race-free.
  4. PURE      — building the model does NOT mutate the sim (snapshot()
                 before == after; the live lists are not rebound) and is
                 DETERMINISTIC (two same-seed games produce identical
                 models at every tick).

M3 (plan_12) adds the NEVER-STEPPED SHELL audit (test_m3_shell_audit):
  5. SHELL     — a shell Game (real surfaces, both players' hulls, NEVER
                 stepped) fed the SAME model renders PIXEL-IDENTICAL to
                 the live stepped game (draw() reads only surfaces +
                 model + render-side presentation state), shell.draw(dt,
                 None) works (the pre-first-model frame), and the model
                 carries the plain "sfx" list (D.1).

The script_input pattern is copied verbatim from test_snapshot.py so the
fire paths (bullets, lasers, missiles, stop) are exercised identically.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

import pygame

from .config import WIDTH, HEIGHT, LOOK_AHEAD
from .fog import make_light_texture
from .game import Game, STEP
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .hulls import PLAYER_HULLS, default_loadout
from .ship import Ship
from .intent import ShipInput

TICKS = 600   # 10 simulated seconds at 60 Hz (same as test_snapshot)
SEED = 1234


class Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """A canned input pattern that exercises every fire path:
    turn cycles, thrust bursts, bullets, lasers, missiles, stop.
    Copied verbatim from test_snapshot.py."""
    return Keys(
        {pygame.K_q: 1 if (t // 30) % 3 == 0 else 0,
           pygame.K_e: 1 if (t // 30) % 3 == 2 else 0,
           pygame.K_w: 1 if (t // 60) % 2 == 0 else 0,
           pygame.K_a: 1 if (t // 15) % 2 == 0 else 0,
           pygame.K_d: 1 if (t // 15) % 2 == 1 else 0,
           pygame.K_SPACE: 1 if (t // 30) % 2 == 0 else 0,
           pygame.K_r: 1 if (t % 45) < 5 else 0,
           pygame.K_2: 1 if (t % 60) < 3 else 0,
           pygame.K_b: 1 if (t % 90) < 5 else 0})


def make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
              seed, players=1):
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                seed=seed, players=players)


# --- the explicit "every field draw() reads" mapping (the SHAPE claim) ---
# Each entry: (model path, what draw() reads it for). If draw() later reads a
# new live field, it MUST be added here + to render_model() + to the parity
# check, or the M2 grep gate (no live self.* reads in the render path) will
# catch the regression.
FIELD_MAP = {
    "sim_time":        "snapshot stamp / render interpolation clock",
    "step_alpha":      "the acc/STEP alpha draw() feeds p.sync_render",
    "camera":          "cam.update(dt, ship) target (pos/vel/dampening)",
    "stars":           "the parallax background loop",
    "asteroids":       "for a in self.asteroids: a.draw  (pos/angle/verts)",
    "enemies":         "for e in self.enemies: e.draw + _draw_lead + "
                       "lead_point (tag/pos/angle/vel/acc_smooth/"
                       "collision_radius/local_poly/shield_dump/"
                       "shield_clock — 10.2b)",
    "bullets":         "for b in self.bullets  (pos/vel)",
    "enemy_bullets":   "for b in self.enemy_bullets  (pos/vel)",
    "missiles":        "for m in self.missiles  (pos/vel/boost/life)",
    "beams":           "for local,target,d,vis_end,age,ttl,owner in self.beams "
                       "(target_id replaces the live enemy ref; owner is the "
                       "firing player's index — 10.4d)",
    "particles":       "for p in self.particles: p.draw  "
                       "(pos/vel/color/life/max_life)",
    "players":         "per-ship presentation: flame_mags/arcs/"
                       "shield_impacts/scan_pulse/contacts + prev/curr pose "
                       "+ HUD power/shield/vel + fog lights",
    "game_over":       "the game-over branch + draw_game_over",
    "protect_timer":   "the spawn-protection shield ring",
    "snapshot":        "the synced state (== snapshot(); the strongest check)",
}


def assert_plain(obj, path="root"):
    """Recursively assert the model is plain data only (no pygame objects,
    no live entity references). This is what makes the M3 atomic reference
    swap race-free."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, tuple):
        for i, v in enumerate(obj):
            assert_plain(v, "%s[%d]" % (path, i))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_plain(v, "%s[%d]" % (path, i))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, (str, int, float, bool, tuple)), \
                "%s: non-plain key %r" % (path, k)
            assert_plain(v, "%s.%s" % (path, k))
    else:
        raise AssertionError("%s: non-plain object %r" % (path, type(obj)))


def check_parity(m, g):
    """Assert every field the model carries EQUALS the live sim state.
    Returns the number of live entities the check covered (for the report)."""
    ship = g.ship
    # --- scalar / flag fields ---
    assert m["sim_time"] == g.sim_time, "sim_time"
    assert m["step_alpha"] == g.acc / STEP, "step_alpha"
    assert m["game_over"] == g.game_over, "game_over"
    assert m["protect_timer"] == g.protect_timer, "protect_timer"
    # --- the synced state (the strongest check) ---
    assert m["snapshot"] == g.snapshot(), "snapshot"
    # --- camera ---
    cpos, cvel, cdamp = m["camera"]
    assert cpos == (ship.pos.x, ship.pos.y), "camera.pos"
    assert cvel == (ship.vel.x, ship.vel.y), "camera.vel"
    assert cdamp == ship.dampening, "camera.dampening"
    # --- stars ---
    assert m["stars"] == list(g.stars), "stars"
    # --- asteroids ---
    assert len(m["asteroids"]) == len(g.asteroids), "asteroids len"
    for (mpos, mang, mverts), a in zip(m["asteroids"], g.asteroids):
        assert mpos == (a.pos.x, a.pos.y), "asteroid pos"
        assert mang == a.angle, "asteroid angle"
        assert mverts == tuple((v.x, v.y) for v in a.verts), "asteroid verts"
    # --- enemies ---
    assert len(m["enemies"]) == len(g.enemies), "enemies len"
    for (mtag, mid, mpos, mang, mvel, macc, mcr, mpoly,
             mdump, mclock, mflame), e in zip(m["enemies"], g.enemies):
        assert mtag == g._enemy_tag(e), "enemy tag"
        assert mid == e.ship.id, "enemy ship_id"
        assert mpos == (e.pos.x, e.pos.y), "enemy pos"
        assert mang == e.ship.angle, "enemy angle"
        assert mvel == (e.ship.vel.x, e.ship.vel.y), "enemy vel"
        assert macc == (e._acc_smooth.x, e._acc_smooth.y), "enemy acc_smooth"
        assert mcr == e.collision_radius, "enemy collision_radius"
        assert mpoly == tuple(e.ship.collision.local_poly), "enemy local_poly"
        # 10.2b: the model carries the enemy's shield-impact state so the
        # host's rendered enemies flash (restores the M2a deliberate loss).
        assert mdump == e.ship.shield_dump, "enemy shield_dump"
        assert mclock == e.ship.shield_clock, "enemy shield_clock"
        # 10.6: the model carries the enemy's flame_mags so the host's
        # rendered enemies show their exhaust (restores the M2a deliberate
        # loss of enemy flames, parity with the client's buffer path).
        assert mflame == dict(e.ship.flame_mags), "enemy flame_mags"
    # --- projectiles ---
    assert len(m["bullets"]) == len(g.bullets), "bullets len"
    for (mpos, mvel), b in zip(m["bullets"], g.bullets):
        assert mpos == (b.pos.x, b.pos.y), "bullet pos"
        assert mvel == (b.vel.x, b.vel.y), "bullet vel"
    assert len(m["enemy_bullets"]) == len(g.enemy_bullets), "enemy_bullets len"
    for (mpos, mvel), b in zip(m["enemy_bullets"], g.enemy_bullets):
        assert mpos == (b.pos.x, b.pos.y), "enemy_bullet pos"
        assert mvel == (b.vel.x, b.vel.y), "enemy_bullet vel"
    assert len(m["missiles"]) == len(g.missiles), "missiles len"
    for (mpos, mvel, mboost, mlife), ms in zip(m["missiles"], g.missiles):
        assert mpos == (ms.pos.x, ms.pos.y), "missile pos"
        assert mvel == (ms.vel.x, ms.vel.y), "missile vel"
        assert mboost == ms.boost, "missile boost"
        assert mlife == ms.life, "missile life"
    # --- beams (target_id must resolve back to the live enemy) ---
    assert len(m["beams"]) == len(g.beams), "beams len"
    for (mls, mtid, md, mvend, mage, mttl, mowner), beam in \
            zip(m["beams"], g.beams):
        assert mls == beam[0], "beam local_start"
        want_tid = beam[1].ship.id if beam[1] is not None else None
        assert mtid == want_tid, "beam target_id"
        assert md == (beam[2].x, beam[2].y), "beam d"
        assert mvend == (beam[3].x, beam[3].y), "beam vis_end"
        assert mage == beam[4], "beam age"
        assert mttl == beam[5], "beam ttl"
        assert mowner == beam[6], "beam owner (10.4d)"
    # --- particles ---
    assert len(m["particles"]) == len(g.particles), "particles len"
    for (mpos, mvel, mcolor, mlife, mmax), p in \
            zip(m["particles"], g.particles):
        assert mpos == (p.pos.x, p.pos.y), "particle pos"
        assert mvel == (p.vel.x, p.vel.y), "particle vel"
        assert mcolor == tuple(p.color), "particle color"
        assert mlife == p.life, "particle life"
        assert mmax == p.max_life, "particle max_life"
    # --- per-ship presentation (the rich local model) ---
    assert len(m["players"]) == len(g.players), "players len"
    for mp, p in zip(m["players"], g.players):
        assert mp["id"] == p.id, "ship id"
        assert mp["pos"] == (p.pos.x, p.pos.y), "ship pos"
        assert mp["vel"] == (p.vel.x, p.vel.y), "ship vel"
        assert mp["angle"] == p.angle, "ship angle"
        assert mp["prev_pos"] == (p.prev_pos.x, p.prev_pos.y), "ship prev_pos"
        assert mp["prev_angle"] == p.prev_angle, "ship prev_angle"
        assert mp["dampening"] == p.dampening, "ship dampening"
        assert mp["flame_mags"] == dict(p.flame_mags), "ship flame_mags"
        assert mp["arcs"] == [([tuple(pt) for pt in pts], age, ttl)
                              for pts, age, ttl in p.arcs], "ship arcs"
        assert mp["shield_impacts"] == [(th, age, ttl)
                                        for th, age, ttl in
                                        p.shield_impacts], "ship shield_impacts"
        assert mp["scan_pulse"] == p.scan_pulse, "ship scan_pulse"
        assert mp["contacts"] == [((pos.x, pos.y), dist, strength, confirmed)
                                  for pos, dist, strength, confirmed
                                  in p.contacts], "ship contacts"
        assert mp["targeting_on"] == p.targeting_on, "ship targeting_on"
        assert mp["tracked"] == p.tracked, "ship tracked"
        assert mp["sensor_on"] == p.sensor_on, "ship sensor_on"
        assert mp["scan_cd"] == p.scan_cd, "ship scan_cd"
        assert mp["scan_reveal"] == p.scan_reveal, "ship scan_reveal"
        assert mp["shield_charge"] == p.shield_charge, "ship shield_charge"
        assert mp["shield_dump"] == p.shield_dump, "ship shield_dump"
        assert mp["shield_clock"] == p.shield_clock, "ship shield_clock"
        assert mp["brownout"] == p.brownout, "ship brownout"
        assert mp["power_used"] == p.power_used, "ship power_used"
        assert mp["power_supply"] == p.power_supply, "ship power_supply"
        assert mp["compute_used"] == p.compute_used, "ship compute_used"
        assert mp["compute_supply"] == p.compute_supply, "ship compute_supply"
        assert mp["weapons"] == [(w.cooldown, w.charge, w.lock_progress,
                                 w.burst_count)
                                for w in p.weapons], "ship weapons"
    # --- report the coverage so the PASS line is meaningful ---
    return (len(g.asteroids), len(g.enemies), len(g.bullets),
            len(g.enemy_bullets), len(g.missiles), len(g.beams),
            len(g.particles))


# ---------------------------------------------------------------------------
# M3 (plan_12): the NEVER-STEPPED SHELL audit.
#
# The render process draws a SHELL Game: built with the real surfaces +
# both players' hulls, then NEVER stepped (the sim runs in a separate
# process; the shell's draw(dt, model) runs entirely from the piped
# model). This section proves the shell is a valid draw target:
#
#   S1. PIXEL PARITY — a game stepped in-process and a never-stepped
#       shell fed the SAME model render IDENTICAL pixels (draw() reads
#       only surfaces + model + render-side presentation state — no
#       accidental live read).
#   S2. MODEL=NONE SMOKE — shell.draw(dt, None) falls back to
#       self.render_model() (the shell's own construction state) without
#       crashing: the run_host loop draws before the first piped model
#       arrives.
#   S3. SFX FIELD — the model carries the "sfx" list (plan_12 D.1),
#       plain data, drained per publish.
#
# The pixel comparison must be EXACT, so every input to draw() is made
# deterministic:
#   * pygame.time.get_ticks is FROZEN — the draw path has wall-clock
#     animations (the targeting reticle's 100 ms blink, the shield
#     swirl/phase in ship.py). Both draws run inside the freeze.
#   * the GLOBAL random module is RE-SEEDED before each draw — the draw
#     path itself samples it (the flame's per-frame length jitter, the
#     shield's flicker in ship.py). Seeding both draws identically
#     isolates the question under test (does draw() read LIVE sim state?)
#     from draw-side nondeterminism.
#   * The camera is SYNCED (not eased) for both games: cam.pos = the
#     model's camera target, cam.lead = LOOK_AHEAD — the same starting
#     state for both, so the one update(dt, ...) call lands them in the
#     same place.
#   * The shell's _standins/_remote_enemies are pre-built: draw() builds
#     them lazily from players[i].hull/components — the SAME hulls the
#     live game's standins use (the shell is built with the identical
#     hulls/loadouts).
#   * The per-player shield Surfaces (Game.shields) are pre-drawn in
#     __init__ from each player's hull — identical for both games.
# ---------------------------------------------------------------------------

TICKS_S = 300              # 5 simulated seconds of scripted input
FROZEN_TICKS = 12345       # the frozen wall clock (inside the 100 ms blink
                           # period: (12345 // 100) % 2 == 1)


def _sync_cam(g, model):
    """Put the camera at the exact state draw()'s one update(dt) call
    produces from a cold start at the model's camera target (so the live
    game and the shell, both starting cold, end up pixel-identical)."""
    cpos, cvel, cdamp = model["camera"]
    g.cam.pos = pygame.Vector2(cpos)
    g.cam.lead = LOOK_AHEAD * (0.35 if cdamp else 1.0)


def _grab_screen(screen):
    """A pixel-exact grab of the screen (read the pixels, not the
    Surface object — the grab must survive the next draw)."""
    return pygame.image.tostring(screen, "RGBA")


def test_m3_shell_audit(screen, font, big_font, light_tex, fog_surf,
                        light_surf):
    """The never-stepped shell audit (S1 + S2 + S3). Returns True when all
    checks pass."""
    ok = True
    # Two DIFFERENT hulls (host + client), like the real 2P game: the
    # hulls matter for the ship-body/shield rendering (different radii,
    # colors, panels) and for the standin build.
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]

    # --- the live game: stepped in-process (the "reference" draw) ---
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    live = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=default_loadout(host_hull),
                seed=SEED, players=2)
    live.set_player_ship(1, Ship(hull=client_hull,
                                 loadout=default_loadout(client_hull)))

    # --- the shell: built identically, NEVER stepped ---
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    shell = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                 hull=host_hull, loadout=default_loadout(host_hull),
                 seed=SEED, players=2)
    shell.set_player_ship(1, Ship(hull=client_hull,
                                  loadout=default_loadout(client_hull)))
    # Pre-build the presentation standins (draw() would build them lazily
    # from the same hulls — pre-building makes the construction explicit
    # and deterministic).
    shell._get_standin(0)
    shell._get_standin(1)
    shell._get_remote_enemies()

    # --- S3: the model carries the "sfx" list (plain, drained) ---
    # The list may hold the constructor's reset event (("thruster", False)
    # — the engine loop is off at construction) plus whatever the scripted
    # run fired; the invariants are: present, a list, plain data, and
    # DRAINED per publish (a second render_model() with no step in
    # between is empty).
    m0 = live.render_model()
    s3_shape = ("sfx" in m0 and isinstance(m0["sfx"], list))
    assert_plain(m0)
    drained = ("sfx" in live.render_model()
               and live.render_model()["sfx"] == [])
    if not s3_shape or not drained:
        ok = False
        print("FAIL: S3 — 'sfx' list missing/not plain/not drained: "
              "present=%s drained=%s first=%r"
              % (s3_shape, drained, m0.get("sfx")))
    else:
        print("PASS: S3 — model carries a plain 'sfx' list, drained per "
              "publish (first publish: %r)" % (m0["sfx"],))

    # --- S1: pixel parity, EVERY tick of a scripted run ---
    orig_ticks = pygame.time.get_ticks
    pygame.time.get_ticks = lambda: FROZEN_TICKS
    worst = 0
    try:
        for t in range(TICKS_S):
            live._step(STEP, ShipInput.from_keys(script_input(t)))
            model = live.render_model()
            # Same camera start for both (see _sync_cam).
            _sync_cam(live, model)
            _sync_cam(shell, model)
            random.seed(999)
            live.draw(STEP, model)
            px_live = _grab_screen(screen)
            random.seed(999)
            shell.draw(STEP, model)
            px_shell = _grab_screen(screen)
            if px_live != px_shell:
                # Count the differing pixels for the report (a few
                # antialiasing-adjacent pixels would still be a FAIL —
                # the claim is exact parity, not approximate).
                n_diff = sum(1 for a, b in zip(px_live, px_shell)
                             if a != b) // 4
                worst = max(worst, n_diff)
                ok = False
                print("FAIL: S1 — shell draw differs from live draw at "
                      "tick %d (%d px differ)" % (t, n_diff))
                break
    finally:
        pygame.time.get_ticks = orig_ticks
    if ok:
        print("PASS: S1 — never-stepped shell renders PIXEL-IDENTICAL to "
              "the live game at every tick (%d ticks; scripted input, "
              "2P with %s + %s hulls)"
              % (TICKS_S, host_hull.id, client_hull.id))

    # --- S2: shell.draw(dt, None) — the pre-first-model frame ---
    # The run_host loop draws with model=None until the first piped model
    # arrives; draw() then falls back to self.render_model() (the shell's
    # own construction state). It must not crash (no live sim state is
    # being stepped, so the fallback model is the static construction
    # state — which is exactly what the pre-game frame should show).
    try:
        _sync_cam(shell, shell.render_model())
        shell.draw(STEP, None)
        print("PASS: S2 — shell.draw(dt, None) renders the shell's own "
              "construction state (the pre-first-model frame)")
    except Exception as e:
        ok = False
        print("FAIL: S2 — shell.draw(dt, None) raised %r" % (e,))

    return ok


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ok = True

    # --- Game A: run N ticks; at EVERY tick build the model and check
    #     SHAPE + PARITY + PLAIN. This is the core M1 assertion: the model
    #     matches live state at every instant, including the moments when
    #     beams/particles/contacts/enemies are non-empty. ---
    a = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED)
    max_cov = (0, 0, 0, 0, 0, 0, 0)
    for t in range(TICKS):
        a._step(STEP, ShipInput.from_keys(script_input(t)))
        m = a.render_model()
        # SHAPE: every field draw() reads is present.
        for key in FIELD_MAP:
            assert key in m, "model missing field %r (%s)" % (
                key, FIELD_MAP[key])
        # PLAIN: no pygame objects / live references anywhere.
        assert_plain(m)
        # PARITY: every field equals the live state.
        cov = check_parity(m, a)
        max_cov = tuple(max(x, y) for x, y in zip(max_cov, cov))
    print("PASS: SHAPE+PARITY+PLAIN at every tick (%d ticks; max live "
          "coverage: %d rocks, %d enemies, %d bullets, %d ebullets, "
          "%d missiles, %d beams, %d particles)"
          % (TICKS, *max_cov))

    # --- PURE #1: building the model does NOT mutate the sim. ---
    snap_before = a.snapshot()
    id_before = (id(a.bullets), id(a.enemy_bullets), id(a.missiles),
                 id(a.particles), id(a.asteroids), id(a.enemies), id(a.beams))
    a.render_model()
    snap_after = a.snapshot()
    id_after = (id(a.bullets), id(a.enemy_bullets), id(a.missiles),
                id(a.particles), id(a.asteroids), id(a.enemies), id(a.beams))
    if snap_before == snap_after and id_before == id_after:
        print("PASS: PURE — render_model() does not mutate the sim "
              "(snapshot + live-list identity unchanged)")
    else:
        ok = False
        print("FAIL: PURE — render_model() mutated the sim "
              "(snapshot_changed=%s, list_ids_changed=%s)"
              % (snap_before != snap_after, id_before != id_after))

    # --- PURE #2: the model is DETERMINISTIC — two same-seed games produce
    #     identical models at every tick (critical for M3, where the sim
    #     thread builds the model).
    #
    # NOTE: the games must NOT be stepped interleaved (step c, then step d):
    # Asteroid._next_id / AIEnemy._next_id are CLASS-level (shared) counters,
    # so stepping c first advances the shared counter and d's spawned
    # entities get different ids. Instead: run c fully, recording its model
    # at every tick, then run d (fresh, counter reset to 1) and compare
    # d's tick-t model against c's recorded tick-t model. Both start at
    # counter=1 with identical inputs, so the id streams (and thus the
    # models) match tick-for-tick. ---
    c = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED)
    models_c = []
    for t in range(TICKS):
        c._step(STEP, ShipInput.from_keys(script_input(t)))
        models_c.append(c.render_model())
    d = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED)
    det_ok = True
    for t in range(TICKS):
        d._step(STEP, ShipInput.from_keys(script_input(t)))
        if d.render_model() != models_c[t]:
            det_ok = False
            print("FAIL: DETERMINISM — models diverged at tick %d" % t)
            break
    if det_ok:
        print("PASS: DETERMINISM — two same-seed games produce identical "
              "render models at every tick (%d ticks)" % TICKS)
    else:
        ok = False

    # --- 2P smoke: the host is a 2-ship sim; render_model must pack BOTH
    #     players and stay plain (the per-ship path for player 1). ---
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    two = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
               seed=SEED, players=2)
    two.set_player_ship(1, Ship())
    for t in range(120):
        two._step(STEP, ShipInput.from_keys(script_input(t)))
    m2 = two.render_model()
    if len(m2["players"]) == 2:
        assert_plain(m2)
        check_parity(m2, two)
        print("PASS: 2P — render_model packs both players, plain + parity")
    else:
        ok = False
        print("FAIL: 2P — expected 2 players, got %d" % len(m2["players"]))

    # --- M3: the never-stepped shell audit (plan_12) ---
    print("\nM3: never-stepped shell Game audit (plan_12)")
    ok &= test_m3_shell_audit(screen, font, big_font, light_tex, fog_surf,
                              light_surf)

    pygame.quit()
    print("\nRENDER MODEL (M1+M3):", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()