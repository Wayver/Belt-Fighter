"""Session 10.2b: the AI enemy's shield-impact flash.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_2b_enemy_shield

Headless: sets SDL_VIDEODRIVER=dummy before pygame.init().

10.2b makes the AI ENEMY's shield-impact flash render on BOTH peers, the
same blue->white glow a ship's shield produces when struck. Before 10.2b
NEITHER peer showed it: the client's remote enemies were drawn from the
interpolation buffer (which carried no shield state), and the host's
rendered enemies were drawn from the RenderModel (whose enemy tuple
deliberately dropped shield state — the M2a "deliberate loss"). Only the
old live AIEnemy.draw flashed, and that is no longer on the render path
after M2b.

The fix (no wire change — the enemy's Ship snapshot already carries
shield_dump (e_s[0][6]) + shield_clock (e_s[0][7])):
  * netcode.interp_positions carries shield_dump + shield_clock through
    the buffer's enemy entry, LERP'd across the window like the pose. The
    entry is now (tag, x, y, angle, vx, vy, id, shield_dump, shield_clock).
  * game.render_model carries them in the model's enemy tuple (now a
    10-tuple), and _draw_world_enemy feeds them to the stand-in before
    drawing (host path).
  * game._draw_remote_enemy_hull feeds them to the stand-in before
    drawing (client path, both remote_view + predicted_view).

This test proves it:
  1. BUFFER-CARRIES — the buffer's enemy entry is a 9-tuple with
                     shield_dump at [7] and shield_clock at [8].
  2. LERP           — interp_positions lerps the enemy's shield state
                     between two snapshots (midpoint), endpoint-exact at
                     alpha 0/1.
  3. FEEDS-STANDIN  — after predicted_view, the enemy stand-in's
                     shield_dump/shield_clock match the buffer's entry.
  4. PIXEL-FLASH    — the enemy's impact flash renders on the CLIENT: a
                     fresh hit (dump=power_hit) produces bluish-white
                     flash pixels around the enemy; no hit (dump=0) does
                     not.
  5. DECAY          — a decaying dump (a partial hit) flashes dimmer than
                     a fresh hit (fewer flash pixels).
  6. HOST-MODEL     — the host's render path shows the same flash: the
                     model's enemy tuple carries the shield state, and
                     _draw_world_enemy renders the flash (parity with the
                     client — the 10.x goal is "client looks like host").
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, _draw_world_enemy
from .netcode import (interp_positions, HostTimeEstimator, LatencyTracker,
                      RenderPoint, PredictedShip)
from .ship import Ship
from .ai_enemy import AIEnemy
from .hulls import PLAYER_HULLS

SEED = 1234
# The target enemy sits this far to the right of the local ship (the ghost,
# the fog's light source) so its flash survives the fog. 150 px is close
# enough that a fresh hit (env=1.0) is well above the flashy-pixel
# threshold — a distant enemy's flash IS dimmer (correct fog behavior), but
# the test needs the flash to survive the fog to measure it. The enemy's
# shield uses power_hit=25.0 (same as the player), so the 10.2 values
# (fresh=25, partial=15) apply directly.
ENEMY_OFFSET = 150.0
ENEMY_IDX = 0


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def make_client(screen, font, big_font, light_tex, fog_surf, light_surf):
    """A client Game (local_index=1) with the host's hull on player 0 and
    the client's hull on player 1 — the same layout run_client builds."""
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             hull=host_hull, loadout=None, seed=SEED, players=2,
             local_index=1)
    g.set_player_ship(1, Ship(hull=client_hull, loadout=None))
    g.ghost = PredictedShip(hull=client_hull, loadout=None, local_index=1)
    g.host_time = HostTimeEstimator()
    g.latency = LatencyTracker()
    g.render_point = RenderPoint(g.latency)
    return g


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless).

    Resets AIEnemy._next_id so every scratch game's enemies get the SAME
    ship ids (1, 2, 3, ...) — the buffer matches enemies BY ID, so two
    snapshots fed to interp_positions must share ids or the prev enemy
    won't match (and the lerp would fall back to curr)."""
    AIEnemy._next_id = 1
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=None, seed=SEED, players=2)


def make_snap(enemy_dump, enemy_clock):
    """A valid 2-player Game snapshot with the TARGET enemy (ENEMY_IDX) at
    a fixed pose (ENEMY_OFFSET right of center) and a controllable
    shield_dump / shield_clock. All other enemies keep their (zero) shield
    state, so only the target flashes."""
    tmp = _scratch_game()
    e = tmp.enemies[ENEMY_IDX]
    e.ship.pos = pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)
    e.ship.angle = 0.0
    e.ship.shield_dump = enemy_dump
    e.ship.shield_clock = enemy_clock
    return tmp.snapshot()


def push_stream(g, enemy_dump, enemy_clock):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the target enemy's
    shield state held at (enemy_dump, enemy_clock)."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(enemy_dump, enemy_clock), now=t)


def render_frame(g):
    """Advance the render point one frame and render via predicted_view."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def enemy_world():
    return pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)


def flashy_pixels(screen, cam, world, half=40):
    """Count bluish-white 'flash' pixels in a box around `world`. The
    shield-impact flash (Ship._draw_shield) lerps toward
    SHIELD_COLOR_BRIGHT (120, 200, 255) — a saturated blue far brighter /
    bluer than the dim hull or the fog, so 'b > 150 and b > r + 20 and
    g > 120' isolates it."""
    s = cam.to_screen(world)
    cx, cy = int(s.x), int(s.y)
    box = screen.subsurface(cx - half, cy - half, 2 * half, 2 * half)
    data = pygame.image.tostring(box, "RGB")
    n = 0
    for i in range(0, len(data), 3):
        r, gg, b = data[i], data[i + 1], data[i + 2]
        if b > 150 and b > r + 20 and gg > 120:
            n += 1
    return n


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g, = (make_client(screen, font, big_font, light_tex, fog_surf,
                      light_surf),)

    # --- 1. BUFFER-CARRIES: the enemy entry is a 10-tuple with the shield
    #     state at [7] (dump) and [8] (clock). 10.3c grew it to a 10-tuple
    #     (+ power_used at [9]) — the shape assert tracks that. ---
    prev = make_snap(0.0, 0.0)
    curr = make_snap(25.0, 0.05)
    P = interp_positions(prev, curr, 1.0, dt=0.1)
    entry = P['enemies'][ENEMY_IDX]
    assert isinstance(entry, tuple) and len(entry) == 10, \
        "enemy entry must be a 10-tuple, got %r" % (entry,)
    # At alpha=1 the entry sits exactly on curr (endpoint-exact): the
    # target enemy's dump/clock are curr's (25.0, 0.05).
    assert entry[7] == 25.0 and entry[8] == 0.05, \
        "shield_dump/clock must be at [7]/[8], endpoint-exact: %r" % (entry,)
    print("PASS: BUFFER-CARRIES — enemy entry is "
          "(tag,x,y,angle,vx,vy,id,dump,clock,power_used)")

    # --- 2. LERP: interp_positions lerps the enemy's shield state between
    #     two snapshots (midpoint), endpoint-exact at alpha 0/1. ---
    # prev dump=0, curr dump=25 -> midpoint alpha=0.5 gives 12.5.
    Pm = interp_positions(prev, curr, 0.5, dt=0.1)
    assert abs(Pm['enemies'][ENEMY_IDX][7] - 12.5) < 1e-9, \
        "midpoint shield_dump must lerp to 12.5, got %r" \
        % (Pm['enemies'][ENEMY_IDX][7],)
    # Endpoint-exact: alpha=0 -> prev's dump (0.0), alpha=1 -> curr's (25.0).
    assert interp_positions(prev, curr, 0.0, dt=0.1)['enemies'][ENEMY_IDX][7] == 0.0
    assert interp_positions(prev, curr, 1.0, dt=0.1)['enemies'][ENEMY_IDX][7] == 25.0
    print("PASS: LERP — enemy shield_dump lerps across the window "
          "(midpoint 12.5)")

    # --- 3. FEEDS-STANDIN: _draw_remote_enemy_hull feeds the given shield
    #     state to the stand-in before drawing. (The stand-in is SHARED
    #     per tag — all 3 enemies here are 'ai' — so reading it after a
    #     full predicted_view frame is ambiguous (it holds the LAST
    #     enemy of that tag drawn). The feed is atomic per enemy: each is
    #     fed its own state then drawn immediately, so the RENDER is
    #     correct. To verify the feed unambiguously, call the helper
    #     directly with known values and check the stand-in reflects them.)
    standin = g._get_remote_enemies()['ai']
    g._draw_remote_enemy_hull(g.screen, 'ai', 0.0, 0.0, 0.0, 25.0, 0.05)
    assert abs(standin.ship.shield_dump - 25.0) < 1e-9, \
        "_draw_remote_enemy_hull must feed shield_dump to the stand-in: " \
        "standin=%r" % (standin.ship.shield_dump,)
    assert abs(standin.ship.shield_clock - 0.05) < 1e-9, \
        "_draw_remote_enemy_hull must feed shield_clock to the stand-in: " \
        "standin=%r" % (standin.ship.shield_clock,)
    print("PASS: FEEDS-STANDIN — _draw_remote_enemy_hull feeds the "
          "stand-in's shield state before drawing")

    # --- 4. PIXEL-FLASH: a fresh hit (dump=power_hit=25) renders the
    #     bluish-white flash around the enemy on the CLIENT; no hit
    #     (dump=0) does not. ---
    push_stream(g, 0.0, 0.0)
    render_frame(g)
    base = flashy_pixels(g.screen, g.cam, enemy_world())
    push_stream(g, 25.0, 0.05)
    render_frame(g)
    hit = flashy_pixels(g.screen, g.cam, enemy_world())
    assert hit > base + 50, \
        "a fresh enemy shield hit must render flash pixels: hit=%d base=%d" \
        % (hit, base)
    print("PASS: PIXEL-FLASH — the enemy's impact flash renders on the "
          "client (%d vs %d baseline pixels)" % (hit, base))

    # --- 5. DECAY: a partial hit (dump=15, a decaying flash) flashes
    #     dimmer than a fresh hit (dump=25) — fewer flash pixels. The
    #     flash has a per-frame random flicker jitter (Ship._draw_shield),
    #     so the margins are generous (the 10.2 measured gap is large). ---
    push_stream(g, 15.0, 0.2)
    render_frame(g)
    partial = flashy_pixels(g.screen, g.cam, enemy_world())
    assert partial < hit - 20, \
        "a decaying enemy dump must flash dimmer than a fresh hit: " \
        "partial=%d fresh=%d" % (partial, hit)
    assert partial > base + 20, \
        "a partial enemy hit must still flash (above the no-hit baseline): " \
        "partial=%d base=%d" % (partial, base)
    print("PASS: DECAY — a decaying enemy dump flashes dimmer "
          "(%d vs %d fresh)" % (partial, hit))

    # --- 6. HOST-MODEL: the host's render path shows the same flash. The
    #     model's enemy tuple carries the shield state, and
    #     _draw_world_enemy renders the flash (parity with the client —
    #     the 10.x goal is "client looks like host"). ---
    host = _scratch_game()
    he = host.enemies[ENEMY_IDX]
    he.ship.pos = pygame.Vector2(WIDTH / 2 + ENEMY_OFFSET, HEIGHT / 2)
    he.ship.angle = 0.0
    he.ship.shield_dump = 25.0
    he.ship.shield_clock = 0.05
    m = host.render_model()
    met = m["enemies"][ENEMY_IDX]
    assert len(met) == 10, "model enemy tuple must be a 10-tuple, got %r" \
        % (met,)
    assert met[8] == 25.0 and met[9] == 0.05, \
        "model enemy tuple must carry shield_dump/clock at [8]/[9]: %r" \
        % (met,)
    host.cam.pos = enemy_world().copy()   # center the cam on the enemy
    standins = host._get_remote_enemies()
    s_hit = pygame.Surface((WIDTH, HEIGHT))
    _draw_world_enemy(s_hit, host.cam, met[0], met[2], met[3], standins,
                      met[8], met[9])
    host_hit = flashy_pixels(s_hit, host.cam, enemy_world())
    # No-hit baseline on the host path.
    he.ship.shield_dump = 0.0
    he.ship.shield_clock = 0.0
    m0 = host.render_model()
    met0 = m0["enemies"][ENEMY_IDX]
    s_base = pygame.Surface((WIDTH, HEIGHT))
    _draw_world_enemy(s_base, host.cam, met0[0], met0[2], met0[3], standins,
                      met0[8], met0[9])
    host_base = flashy_pixels(s_base, host.cam, enemy_world())
    assert host_hit > host_base + 50, \
        "the host's rendered enemy must flash on a fresh hit: " \
        "hit=%d base=%d" % (host_hit, host_base)
    print("PASS: HOST-MODEL — the host's rendered enemy flashes too "
          "(%d vs %d baseline pixels; parity with the client)"
          % (host_hit, host_base))

    print("ALL PASS: 10.2b (AI enemy's shield-impact flash, both peers)")


if __name__ == "__main__":
    main()