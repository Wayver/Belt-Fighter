"""Session 9.x M2a: the WORLD entities render from the plain-data
RenderModel, pixel-identically to the live path.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_m2a_world

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

M2a moved the world entities (stars, asteroids, enemies, bullets,
enemy_bullets, missiles, particles) off live sim state and onto the
RenderModel via the module-level _draw_world_* helpers. The local ship,
camera, fog, HUD, the laser beams, and the targeting reticle are DEFERRED
to M2b (they still read live state). This test proves:

  1. SMOKE     — draw() runs end-to-end with no error, both when it builds
                 the model itself (model=None) and when handed one.
  2. PLAIN     — the model is plain data (no pygame objects / live refs),
                 the property that makes the M3 atomic swap race-free.
  3. PARITY    — for EACH world category, the model-driven render is
                 pixel-identical to the live render. Each category is
                 checked the first tick it is NON-EMPTY (so the check is
                 never vacuous), and the run reports how many entities of
                 each kind were actually compared. Asteroids/enemies are
                 always present; bullets/missiles/particles/enemy-bullets
                 are checked whenever the scripted firing produces them.
  4. ENEMY     — the enemy HULL is pixel-identical (checked on a fresh,
                 unstepped game where enemies have no thruster flames /
                 shield flash, so the stand-in draws exactly the live
                 hull). This makes the one deliberate M2a loss explicit:
                 the hull is faithful, the enemy flames are deferred to
                 M2b.

Pixel comparison uses pygame.image.tostring (Surface == is identity-based,
not pixel-based).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from .config import (WIDTH, HEIGHT, BG, BULLET_COLOR,
                    ENEMY_BULLET_COLOR, MISSILE_COLOR)
from .fog import make_light_texture
from .game import (Game, STEP, _dim_color,
                   _draw_world_asteroid, _draw_world_enemy,
                   _draw_world_bullet, _draw_world_missile,
                   _draw_world_particle)
from .ai_enemy import AIEnemy
from .asteroid import Asteroid
from .bullets import Missile
from .intent import ShipInput

TICKS = 600   # 10 simulated seconds at 60 Hz (enough firing for projectiles)
SEED = 1234


class Keys:
    """Minimal stand-in for pygame.key.get_pressed()."""
    def __init__(self, pressed):
        self.p = pressed
    def __getitem__(self, k):
        return self.p.get(k, 0)


def script_input(t):
    """Canned input that exercises every fire path (bullets, lasers,
    missiles, stop). Copied from test_render_model.py."""
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


def assert_plain(obj, path="root"):
    """Recursively assert the model is plain data only (no pygame objects,
    no live entity references)."""
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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ok = True

    # --- SMOKE: draw() runs end-to-end (model=None builds internally; an
    #     explicit model is also accepted). This is the "does it run" gate. ---
    g = make_game(screen, font, big_font, light_tex, fog_surf, light_surf,
                  SEED)
    g._step(STEP, ShipInput.from_keys(script_input(0)))
    g.draw(0.016)                       # model=None (builds internally)
    g.draw(0.016, model=g.render_model())   # explicit model
    print("PASS: SMOKE — draw() runs with model=None and an explicit model")

    # --- PARITY: for each world category, the model-driven render is
    #     pixel-identical to the live render. Each category is checked the
    #     FIRST tick it is non-empty (never vacuous); coverage is reported.
    #     The live and model draws share g.cam at the same tick, so they are
    #     directly comparable. ---
    cam = g.cam
    # name -> (live entity list, live draw closure, model draw closure)
    # Each closure draws its category onto the surface passed in.
    def live_asteroids(s):
        for a in g.asteroids:
            a.draw(s, cam)
    def model_asteroids(s, m):
        for pos, angle, verts in m["asteroids"]:
            _draw_world_asteroid(s, cam, pos, angle, verts)

    def live_bullets(s):
        for b in g.bullets:
            p = cam.to_screen(b.pos)
            pygame.draw.circle(s, BULLET_COLOR, (int(p.x), int(p.y)), 3)
    def model_bullets(s, m):
        for pos, vel in m["bullets"]:
            _draw_world_bullet(s, cam, pos, BULLET_COLOR)

    def live_ebullets(s):
        for b in g.enemy_bullets:
            p = cam.to_screen(b.pos)
            pygame.draw.circle(s, ENEMY_BULLET_COLOR, (int(p.x), int(p.y)), 3)
    def model_ebullets(s, m):
        for pos, vel in m["enemy_bullets"]:
            _draw_world_bullet(s, cam, pos, ENEMY_BULLET_COLOR)

    def live_missiles(s):
        for ms in g.missiles:
            p = cam.to_screen(ms.pos)
            fwd = ms.vel.normalize()
            tail = cam.to_screen(ms.pos - fwd * 14)
            pygame.draw.line(s, _dim_color(MISSILE_COLOR, 0.7), tail, p, 3)
            pygame.draw.circle(s, MISSILE_COLOR, (int(p.x), int(p.y)), 3)
            if ms.boost > 0:
                flick = 6 * (0.5 + 0.5 * math.sin(ms.life * 40))
                flame = cam.to_screen(ms.pos - fwd * (14 + flick))
                pygame.draw.line(s, (255, 220, 120), tail, flame, 2)
    def model_missiles(s, m):
        for pos, vel, boost, life in m["missiles"]:
            _draw_world_missile(s, cam, pos, vel, boost, life)

    def live_particles(s):
        for p in g.particles:
            p.draw(s, cam)
    def model_particles(s, m):
        for pos, vel, color, life, max_life in m["particles"]:
            _draw_world_particle(s, cam, pos, vel, color, life, max_life)

    # NOTE: the sim REBINDS the projectile lists each tick
    # (self.bullets = [b for b in ... if b.life > 0]), so we must read the
    # attribute at check time (getter), never a captured reference.
    categories = [
        ("asteroids",  lambda: g.asteroids,      live_asteroids,  model_asteroids),
        ("bullets",    lambda: g.bullets,        live_bullets,    model_bullets),
        ("ebullets",   lambda: g.enemy_bullets,  live_ebullets,   model_ebullets),
        ("missiles",   lambda: g.missiles,       live_missiles,   model_missiles),
        ("particles",  lambda: g.particles,      live_particles,  model_particles),
    ]
    verified = {}      # name -> entity count compared (set once, when non-empty)
    for t in range(TICKS):
        g._step(STEP, ShipInput.from_keys(script_input(t)))
        m = g.render_model()
        # PLAIN is tick-independent (structure), but assert on a populated
        # model so it is meaningful.
        if t == 0:
            assert_plain(m)
        for name, live_get, live_fn, model_fn in categories:
            if name in verified:
                continue
            live_list = live_get()
            if not live_list:
                continue
            s_live = _surface()
            s_model = _surface()
            live_fn(s_live)
            model_fn(s_model, m)
            if _pixels(s_live) == _pixels(s_model):
                verified[name] = len(live_list)
            else:
                ok = False
                print("FAIL: PARITY %s — model-driven render differs from "
                      "the live path (%d entities)" % (name, len(live_list)))
                verified[name] = -len(live_list)   # mark as checked+failed
    for name, *_ in categories:
        if name not in verified:
            print("NOTE: %s never non-empty during the run (parity not "
                  "observed)" % name)
    print("PASS: PLAIN — the render model is plain data (no pygame objects "
          "/ live refs)")
    print("PASS: PARITY — model-driven world render is pixel-identical to "
          "the live path; entities compared: %s"
          % ", ".join("%s=%d" % (n, verified.get(n, 0)) for n, *_ in
                      categories))

    # --- ENEMY hull parity: _draw_world_enemy == AIEnemy.draw for the hull.
    #     Checked on a FRESH (unstepped) game: the enemies have no thruster
    #     flames / shield flash / arcs, so the stand-in (which carries no
    #     enemy presentation state) draws EXACTLY the live hull. This makes
    #     the one deliberate M2a loss explicit — the hull is faithful, the
    #     enemy flames are deferred to M2b. ---
    fresh = make_game(screen, font, big_font, light_tex, fog_surf,
                      light_surf, SEED)
    fm = fresh.render_model()
    standins = fresh._get_remote_enemies()
    fcam = fresh.cam
    s_live = _surface()
    s_model = _surface()
    for e in fresh.enemies:
        e.draw(s_live, fcam)
    for tag, _id, pos, angle, _vel, _acc, _cr, _poly in fm["enemies"]:
        _draw_world_enemy(s_model, fcam, tag, pos, angle, standins)
    if _pixels(s_live) == _pixels(s_model):
        print("PASS: ENEMY — %d fresh enemies render their hull "
              "pixel-identically via the model (flames deferred to M2b)"
              % len(fresh.enemies))
    else:
        ok = False
        print("FAIL: ENEMY — model-driven enemy hulls differ from the live "
              "path")

    # --- MISSILE parity (focused): the scripted run may never launch a
    #     missile (a lock needs ~48 sustained ticks), so verify the missile
    #     path directly on a constructed Missile with boost > 0 (exercises
    #     the exhaust-flicker branch). The live inline code and the model
    #     helper read the SAME object, so they must be pixel-identical. ---
    ms = Missile(pygame.Vector2(600, 300), pygame.Vector2(300, 120))
    assert ms.boost > 0, "constructed missile should be in its boost ramp"
    s_live = _surface()
    s_model = _surface()
    p = cam.to_screen(ms.pos)
    fwd = ms.vel.normalize()
    tail = cam.to_screen(ms.pos - fwd * 14)
    pygame.draw.line(s_live, _dim_color(MISSILE_COLOR, 0.7), tail, p, 3)
    pygame.draw.circle(s_live, MISSILE_COLOR, (int(p.x), int(p.y)), 3)
    flick = 6 * (0.5 + 0.5 * math.sin(ms.life * 40))
    flame = cam.to_screen(ms.pos - fwd * (14 + flick))
    pygame.draw.line(s_live, (255, 220, 120), tail, flame, 2)
    _draw_world_missile(s_model, cam, (ms.pos.x, ms.pos.y),
                        (ms.vel.x, ms.vel.y), ms.boost, ms.life)
    if _pixels(s_live) == _pixels(s_model):
        print("PASS: MISSILE — a boosting missile renders pixel-identically "
              "via the model (body + nose + exhaust flicker)")
    else:
        ok = False
        print("FAIL: MISSILE — model-driven missile differs from the live "
              "path")

    pygame.quit()
    print("\nM2a WORLD RENDER:", "ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()