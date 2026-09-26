"""Session 9.x M2b: the LOCAL ship + camera + sensor contacts + HUD + laser
beams + targeting reticle + fog all render from the plain-data RenderModel,
with NO live sim-state reads in the render path.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_m2b_local

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

M2a moved the WORLD entities onto the RenderModel. M2b moves the rest of
draw() — the local ship (hull + flames + shield + arcs + scan pulse), the
camera, the laser beams, the targeting reticle, the fog lights, the sensor
contacts, and the HUD — so the render path reads NO live sim state. That is
what makes the M3 atomic reference swap safe: the render thread will hold a
published model and call draw(dt, model), never self.*.

Why this test is VALUE-level, not a full-frame pixel compare:
  * draw() IS the model path now — there is no separate "live" full-frame
    path left to diff against (the old live code was replaced in M2b).
  * The local ship's flames and shield flash use per-frame random.random()
    and pygame.time.get_ticks(), so two full frames are never bit-identical
    even when the render is correct.
So the gate proves the parts that MUST be exact (the interpolated pose, the
beam/reticle/contacts geometry, the fog light list) are identical to the
live math, and it proves the render path reads no live state. The one thing
left to a human eye is the beam's *look* (fade timing / interpolation),
which is intentionally out of scope for a pixel gate.

The checks:
  1. SMOKE        — draw() runs end-to-end (model=None and explicit model).
  2. NO-LIVE-READS— the grep gate: every self.<attr> in draw()'s source is a
                    render resource / config, never a live sim-state field.
                    This is the M2b headline assertion (the M3 safety net).
  3. POSE         — the model's interpolated local-ship pose (_ship_pose)
                    equals the live ship.sync_render pose, at EVERY tick.
  4. FOG-LIGHTS   — _build_lights_model(model, standin) equals the live
                    Game._build_lights(ship) light list, at EVERY tick
                    (bullets/missiles/shield-impacts; reticle when on).
  5. LOCAL-SHIP   — the local ship's hull/panels/shield render
                    pixel-identically via the synced stand-in (checked on a
                    fresh, unstepped ship with no flames/flash/arcs, so the
                    render is deterministic).
  6. BEAM         — the laser beam's geometry is identical: the stand-in's
                    shield oval == the live target's oval, and the model's
                    end-point math (_shield_impact_point_pose) matches the
                    live Ship.shield_impact_point; the start matches the
                    live pose. (The default hull has no laser, so this is a
                    focused geometry check, not a scripted discharge.)
  7. RETICLE      — with targeting on, the model lead point + alignment
                    match the live AIEnemy.lead_point / Game._lead_aligned,
                    and the fog reticle lights match.
  8. CONTACTS     — with the sensor on + a scan fired, the sensor-contact
                    layer (blips / edge arrows / distance text) renders
                    pixel-identically via the model.

Pixel comparison uses pygame.image.tostring (Surface == is identity-based,
not pixel-based).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import ast
import inspect
import textwrap

import pygame

from .config import (WIDTH, HEIGHT, BG, BULLET_SPEED,
                    TARGETING_USE_ACCEL, TARGETING_RANGE)
from .fog import make_light_texture
from .game import (Game, STEP,
                   _ship_pose, _sync_local_ship, _draw_local_ship,
                   _shield_impact_point_pose, _lead_point, _lead_aligned,
                   _build_lights_model, _draw_sensor_contacts_model)
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .intent import ShipInput

TICKS = 600   # 10 simulated seconds at 60 Hz (same as M1 / M2a)
SEED = 1234
EPS = 1e-6


class Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """Canned input that exercises the fire paths (bullets, stop, thrust,
    turn). Copied from test_m2a_world.py. Deliberately has NO T/V/G, so the
    full run keeps targeting/sensor OFF — those are covered by the focused
    checks below (the default hull has no laser, so beams are focused too)."""
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


def _surface():
    s = pygame.Surface((WIDTH, HEIGHT))
    s.fill(BG)
    return s


def _pixels(surf):
    return pygame.image.tostring(surf, "RGB")


def _v2_close(a, b, eps=EPS):
    return abs(a.x - b.x) <= eps and abs(a.y - b.y) <= eps


# --- the NO-LIVE-READS grep gate ------------------------------------------
#
# The M2b goal: the render path (draw) reads NO live sim state, so the M3
# atomic reference swap is safe. draw() may only touch render RESOURCES and
# config (screen, camera, fonts, fog/light surfaces, the per-player shield
# rings, the local index, the model builder, and the stand-in lookups) —
# never a live sim field (asteroids/enemies/bullets/.../ship/players/
# game_over/protect_timer/acc/sim_time/stars). This gate asserts exactly
# that by parsing draw()'s source with `ast` and collecting every REAL
# self.<attr> attribute access (comments and the docstring are ignored —
# a naive regex over the source would false-positive on docstring prose
# like "mirroring the live `enumerate(self.players)`").
ALLOWED_SELF_IN_DRAW = {
    # render resources / config (not live sim state)
    "screen", "cam", "light_tex", "fog_surf", "light_surf",
    "font", "big_font", "shields", "local_index",
    # the sanctioned model builder (the local-build seam; M3 moves it to the
    # sim thread — the render thread always passes a model, so this is not
    # hit on the networked path) and the stand-in lookups (presentation).
    "render_model", "_get_remote_enemies", "_get_standin",
}


def check_no_live_reads():
    """Return (ok, found_set, bad_set). Asserts every REAL self.<attr>
    attribute access in draw()'s code is in ALLOWED_SELF_IN_DRAW (i.e. no
    live sim-state read). Uses ast so comments/docstrings are ignored."""
    src = textwrap.dedent(inspect.getsource(Game.draw))
    found = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            found.add(node.attr)
    bad = found - ALLOWED_SELF_IN_DRAW
    return (not bad, found, bad)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ok = True

    # --- 1. SMOKE: draw() runs end-to-end (model=None builds internally; an
    #     explicit model is also accepted). The "does it run" gate. ---
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED)
    g._step(STEP, ShipInput.from_keys(script_input(0)))
    g.draw(0.016)                          # model=None (builds internally)
    g.draw(0.016, model=g.render_model())  # explicit model
    print("PASS: SMOKE — draw() runs with model=None and an explicit model")

    # --- 2. NO-LIVE-READS: the grep gate (the M2b headline assertion). ---
    nl_ok, found, bad = check_no_live_reads()
    if nl_ok:
        print("PASS: NO-LIVE-READS — draw() touches only render resources "
              "(self.%s); no live sim-state reads"
              % ", ".join(sorted(found)))
    else:
        ok = False
        print("FAIL: NO-LIVE-READS — draw() reads live sim state: %s"
              % ", ".join("self." + b for b in sorted(bad)))

    # --- 3+4. FULL RUN: at EVERY tick, the model's interpolated local-ship
    #     POSE equals the live sync_render pose, and the model-built FOG
    #     LIGHT list equals the live _build_lights list. Coverage is
    #     reported so the PASS line is meaningful. ---
    max_pose_drift = 0.0
    max_lights = 0
    pose_ok = True
    lights_ok = True
    for t in range(TICKS):
        g._step(STEP, ShipInput.from_keys(script_input(t)))
        m = g.render_model()
        local_pack = m["players"][g.local_index]
        alpha = m["step_alpha"]
        # --- POSE: model _ship_pose == live ship.sync_render ---
        ship = g.ship
        ship.sync_render(alpha)
        mpos, mang = _ship_pose(local_pack, alpha)
        drift = max(abs(mpos.x - ship.rpos.x), abs(mpos.y - ship.rpos.y),
                    abs(mang - ship.rangle))
        max_pose_drift = max(max_pose_drift, drift)
        if drift > EPS:
            pose_ok = False
        # --- FOG LIGHTS: model list == live list (pos/radius/intensity) ---
        standin = g._get_standin(g.local_index)
        standin.pos = pygame.Vector2(local_pack["pos"])
        standin.vel = pygame.Vector2(local_pack["vel"])
        standin.angle = local_pack["angle"]
        _sync_local_ship(standin, local_pack)
        model_lights = _build_lights_model(m, standin)
        live_lights = g._build_lights(ship)
        max_lights = max(max_lights, len(model_lights))
        if len(model_lights) != len(live_lights):
            lights_ok = False
        else:
            for ml, ll in zip(model_lights, live_lights):
                if (not _v2_close(ml.pos, ll.pos)
                        or ml.radius != ll.radius
                        or ml.intensity != ll.intensity):
                    lights_ok = False
                    break
    if pose_ok:
        print("PASS: POSE — model local-ship pose == live sync_render at "
              "every tick (%d ticks; max drift %.2e)"
              % (TICKS, max_pose_drift))
    else:
        ok = False
        print("FAIL: POSE — model local-ship pose diverged from live "
              "(max drift %.3e)" % max_pose_drift)
    if lights_ok:
        print("PASS: FOG-LIGHTS — model fog light list == live _build_lights "
              "at every tick (%d ticks; max %d lights)"
              % (TICKS, max_lights))
    else:
        ok = False
        print("FAIL: FOG-LIGHTS — model fog light list differs from live "
              "_build_lights")

    # --- 5. LOCAL-SHIP hull pixel parity: the synced stand-in renders the
    #     local ship pixel-identically to the live ship.draw. Checked on a
    #     FRESH (unstepped) ship: no thruster flames / shield flash / arcs /
    #     scan pulse, so the render is deterministic and the stand-in (which
    #     carries no enemy presentation state of its own) draws EXACTLY the
    #     live hull + panels + shield ring. ---
    fresh = make_game(screen, font, big_font, light_tex, fog_surf,
                      light_surf, SEED)
    fm = fresh.render_model()
    fpack = fm["players"][fresh.local_index]
    fstandin = fresh._get_standin(fresh.local_index)
    fcam = fresh.cam
    s_live = _surface()
    s_model = _surface()
    fship = fresh.ship
    fship.draw(s_live, fcam, fship.pos, fship.angle)
    _draw_local_ship(s_model, fcam, fstandin, fpack, fm["step_alpha"])
    if _pixels(s_live) == _pixels(s_model):
        print("PASS: LOCAL-SHIP — the local ship renders pixel-identically "
              "via the synced stand-in (hull + panels + shield ring)")
    else:
        ok = False
        print("FAIL: LOCAL-SHIP — model-driven local ship differs from the "
              "live ship.draw")

    # --- 6. BEAM geometry: the laser beam's start + end are computed by
    #     identical math in the live and model paths. The default hull has
    #     no laser (so the full run never discharges one), so this is a
    #     focused geometry check on a real enemy: (a) the stand-in's shield
    #     oval == the live target's oval (the model's only source of the
    #     oval), and (b) the model end-point math matches the live
    #     Ship.shield_impact_point, and (c) the start matches the live pose.
    #     Together these prove the beam lands exactly where the live code
    #     would — the "laser looks the same" guarantee. ---
    beam_ok = True
    oval_ok = True
    end_ok = True
    start_ok = True
    for e in fresh.enemies:
        proxy = fresh._get_remote_enemies().get(fresh._enemy_tag(e))
        if proxy is None:
            continue
        if proxy.ship.shield_oval != e.ship.shield_oval:
            oval_ok = False
    # end-point math on the first stand-in-backed enemy
    for e in fresh.enemies:
        proxy = fresh._get_remote_enemies().get(fresh._enemy_tag(e))
        if proxy is None:
            continue
        d = (fship.pos - e.pos)
        if d.length_squared() < 1e-6:
            d = pygame.Vector2(1, 0)
        d.normalize_ip()
        world_pt = e.pos - d * e.collision_radius
        live_end = e.ship.shield_impact_point(world_pt)
        model_end = _shield_impact_point_pose(e.pos, e.ship.angle,
                                              proxy.ship.shield_oval,
                                              world_pt)
        if not _v2_close(live_end, model_end):
            end_ok = False
        # start: model pose == live pose (same rpos/rangle)
        fship.sync_render(fm["step_alpha"])
        mpos, mang = _ship_pose(fpack, fm["step_alpha"])
        if not (_v2_close(mpos, fship.rpos) and abs(mang - fship.rangle) <= EPS):
            start_ok = False
        break
    beam_ok = oval_ok and end_ok and start_ok
    if beam_ok:
        print("PASS: BEAM — beam geometry is identical (stand-in shield oval "
              "== live oval; end-point math == Ship.shield_impact_point; "
              "start == live pose)")
    else:
        ok = False
        print("FAIL: BEAM — beam geometry differs (oval=%s end=%s start=%s)"
              % (oval_ok, end_ok, start_ok))

    # --- 7. RETICLE: with targeting on, the model lead point + alignment
    #     match the live AIEnemy.lead_point / Game._lead_aligned, and the
    #     fog reticle lights match. A focused check (the full run keeps
    #     targeting off). ---
    reticle_ok = True
    lead_ok = True
    aligned_ok = True
    ret_light_ok = True
    for e in fresh.enemies:
        fship.targeting_on = True
        fship.tracked = sum(1 for x in fresh.enemies
                            if x.pos.distance_to(fship.pos) <= TARGETING_RANGE)
        live_p = e.lead_point(fship.pos, BULLET_SPEED, TARGETING_USE_ACCEL)
        m = fresh.render_model()
        mp = _lead_point(e.pos, e.ship.vel, e._acc_smooth, fship.pos,
                         BULLET_SPEED, TARGETING_USE_ACCEL)
        if (live_p is None) != (mp is None):
            lead_ok = False
        elif live_p is not None and not _v2_close(live_p, mp):
            lead_ok = False
        if live_p is not None and (live_p - fship.pos).length() <= TARGETING_RANGE:
            if fship._lead_aligned(e, live_p, fship) != _lead_aligned(
                    e.pos, live_p, fship.pos, fship.angle):
                aligned_ok = False
    # fog reticle lights: model vs live with targeting on
    m = fresh.render_model()
    fpack = m["players"][fresh.local_index]
    fstandin = fresh._get_standin(fresh.local_index)
    fstandin.pos = pygame.Vector2(fpack["pos"])
    fstandin.vel = pygame.Vector2(fpack["vel"])
    fstandin.angle = fpack["angle"]
    _sync_local_ship(fstandin, fpack)
    ml = _build_lights_model(m, fstandin)
    ll = fresh._build_lights(fship)
    if len(ml) != len(ll):
        ret_light_ok = False
    else:
        for a, b in zip(ml, ll):
            if not _v2_close(a.pos, b.pos) or a.radius != b.radius \
                    or a.intensity != b.intensity:
                ret_light_ok = False
                break
    reticle_ok = lead_ok and aligned_ok and ret_light_ok
    if reticle_ok:
        print("PASS: RETICLE — model lead point + alignment + fog reticle "
              "lights match the live targeting math")
    else:
        ok = False
        print("FAIL: RETICLE — model reticle differs (lead=%s aligned=%s "
              "lights=%s)" % (lead_ok, aligned_ok, ret_light_ok))

    # --- 8. CONTACTS: with the sensor on + a scan fired, the sensor-contact
    #     layer (blips / edge arrows / distance text) renders
    #     pixel-identically via the model. A focused check (the full run
    #     keeps the sensor off). The contacts render is deterministic (no
    #     random / get_ticks), so a pixel compare is valid here. ---
    fresh.ship.sensor_on = True
    fresh.ship.fire_scan()          # reveal: confirmed contacts in range
    fresh._step(STEP, ShipInput())  # _update_contacts populates contacts
    cm = fresh.render_model()
    cpack = cm["players"][fresh.local_index]
    if fresh.ship.contacts:
        # live contacts layer onto a fresh surface
        saved_screen = fresh.screen
        s_live = _surface()
        fresh.screen = s_live
        fresh._draw_sensor_contacts()
        live_px = _pixels(s_live)
        # model contacts layer onto another fresh surface
        s_model = _surface()
        _draw_sensor_contacts_model(s_model, fresh.cam, fresh.font,
                                    pygame.Vector2(cpack["pos"]),
                                    cpack["contacts"])
        model_px = _pixels(s_model)
        fresh.screen = saved_screen
        if live_px == model_px:
            print("PASS: CONTACTS — sensor-contact layer renders "
                  "pixel-identically via the model (%d contacts)"
                  % len(fresh.ship.contacts))
        else:
            ok = False
            print("FAIL: CONTACTS — model sensor-contact layer differs from "
                  "the live path (%d contacts)" % len(fresh.ship.contacts))
    else:
        ok = False
        print("FAIL: CONTACTS — no contacts produced (check is vacuous)")

    pygame.quit()
    print("\nM2b LOCAL RENDER:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()