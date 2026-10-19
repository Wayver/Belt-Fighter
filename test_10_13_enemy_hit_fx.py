"""Session 10.13: the client-side bullet-impact spark (option A) fires on
AI ENEMIES and respects the SHIELD.

Root cause this test guards (found via hit_fx_debug.csv): the first pass of
the option-A trigger (the pass-through fix) built its target list from
`pos['ships']` ONLY — the other PLAYER — and never tested `pos['enemies']`
(AI enemies). In normal 2P play the player shoots AI enemies, so the target
list was effectively empty and no spark ever fired (27 enemy_hit frames,
0 HIT rows). The fix extends the swept segment-circle test to the buffer's
AI enemies.

The SHIELD rule (this session): the host sparks at the SHIELD BOUNDARY when
the target's shield is up (conveying "the shield absorbed it") and at the
HULL when it's down. The client mirrors that:
  * shield UP   -> test the shield oval (circle approx, radius max(a,b)) and
                   spark at the shield boundary with `shield_burst` + SFX.
  * shield DOWN -> test the hull radius, small `burst` at the hull, no SFX.
The buffer carries the enemy's `shield_charge` (snapshot index 5, a state
latch) so the client knows which case it is.

This test drives `predicted_view` headless with a ghost bullet whose
per-frame segment crosses an AI enemy's drawn hull and proves:
  1. SHIELD-UP   — shield_charge>0: a `shield_burst` (16 particles) spawns,
                   the bullet is culled, and the CSV tags it 'enemy:ai:shield'.
  2. SHIELD-DOWN — shield_charge=0: a small `burst` (14 particles) spawns,
                   the bullet is culled, and the CSV tags it 'enemy:ai'.
  3. MISS        — a bullet 200 px away spawns no spark and is not culled.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_13_enemy_hit_fx
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import csv

import pygame

from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game
from .netcode import (HostTimeEstimator, LatencyTracker, RenderPoint,
                      PredictedShip)
from .ship import Ship
from .bullets import Bullet
from .hulls import PLAYER_HULLS
from . import hit_fx_telemetry

SEED = 1234
# The AI enemy's drawn position (frozen across the buffer window). 300 px
# right of the local ship (center) — far enough to be a distinct target.
ENEMY_POS = (WIDTH / 2 + 300.0, HEIGHT / 2)
# Particle counts (particles.py): shield_burst n=16; burst(radius=6) is
# not `big` -> n=14. These distinguish the shield path from the hull path.
SHIELD_BURST_N = 16
HULL_BURST_N = 14


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
    return g, host_hull, client_hull


def make_snap(enemy_pos, shield_charge, t):
    """A valid 2-player Game snapshot with a SINGLE AI enemy at
    `enemy_pos` (frozen — the same pose in every snapshot, so the buffer
    interpolates it to exactly `enemy_pos`) and a controllable
    `shield_charge` (the buffer carries it as the 12th field; the trigger
    reads it to pick the shield vs hull path). The local ship (index 1)
    sits at center. `t` is unused (the caller stamps it on push)."""
    tmp = _scratch_game()
    tmp.players[0].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    tmp.players[1].pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    # Replace the reset()'s 3 random enemies with ONE at a known pose.
    # A FIXED ship id (999) so all snapshots carry the SAME enemy — the
    # buffer matches enemies by id, so a fresh id per snapshot would make
    # it see 3 different enemies instead of one persistent one.
    tmp.enemies.clear()
    from .ai_enemy import AIEnemy
    e = AIEnemy(pygame.Vector2(enemy_pos), rng=tmp.rng)
    e.ship.id = 999
    e.ship.shield_charge = shield_charge
    tmp.enemies.append(e)
    return tmp.snapshot()


def _scratch_game():
    """A minimal 2-player Game for snapshot-shape purposes (headless)."""
    screen = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    host_hull, client_hull = PLAYER_HULLS[0], PLAYER_HULLS[1]
    return Game(screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=host_hull, loadout=None, seed=SEED, players=2)


def push_stream(g, enemy_pos, shield_charge):
    """Reset the client's buffer + ghost + render point, then push a 3-
    snapshot stream at 10 Hz (t = 0, 0.1, 0.2) with the enemy frozen at
    `enemy_pos` and its shield_charge held at `shield_charge`."""
    g.snap_buf = g.snap_buf.__class__()
    g.ghost = PredictedShip(hull=g.players[1].hull,
                            loadout=g.players[1].components,
                            local_index=1)
    g.render_point = RenderPoint(g.latency)
    for k in range(3):
        t = k * 0.1
        g.push_snapshot(t, make_snap(enemy_pos, shield_charge, t), now=t)


def seed_bullet(g, prev, pos, bid=(1, 0)):
    """Give the ghost a single local bullet with a known per-frame segment
    (prev -> pos). The trigger tests this segment against the enemy hull.

    vel is (0, 0) ON PURPOSE: predicted_view advances the ghost BEFORE the
    trigger, and `advance` -> `step_local_bullets` overwrites prev_pos/pos
    (pos += vel*TICK). With vel=0 the bullet stays exactly where I put it
    regardless of how many ticks the ghost steps this frame, so the segment
    the trigger sees is deterministic: it degenerates to the point `pos`.
    The hit case therefore places `pos` at the enemy CENTER (distance 0 <
    both the shield radius and the hull radius); the miss case places it
    200 px away (distance 200 > both)."""
    b = Bullet(pygame.Vector2(pos), pygame.Vector2(0, 0), owner=1, bid=bid)
    b.prev_pos = pygame.Vector2(prev)
    b.pos = pygame.Vector2(pos)
    g.ghost.local_bullets = [b]
    return b


def render_frame(g):
    """Advance the render point one frame and render via predicted_view
    (no input keys — the ghost must not fire its own gun)."""
    g.latency.tick(0.016)
    g.render_point.advance(0.016, g.snap_buf.newest_time())
    g.predicted_view(0.016, pygame.key.get_pressed())


def read_hits():
    """The HIT rows from the hit_fx CSV (empty list if the file is absent)."""
    try:
        rows = list(csv.DictReader(open(hit_fx_telemetry._path)))
    except FileNotFoundError:
        return []
    return [r for r in rows if r["event"] == "HIT"]


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()
    g, host_hull, client_hull = make_client(screen, font, big_font,
                                            light_tex, fog_surf, light_surf)
    ex, ey = ENEMY_POS

    # --- 1. SHIELD-UP: shield_charge>0 -> a `shield_burst` (16 particles)
    #     spawns at the shield boundary, the bullet is culled, and the CSV
    #     tags it 'enemy:ai:shield'. ---
    push_stream(g, ENEMY_POS, shield_charge=3.0)
    seed_bullet(g, (ex, ey), (ex, ey), bid=(1, 0))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == SHIELD_BURST_N, \
        "a shield-up hit must spawn a shield_burst (%d particles): " \
        "got %d" % (SHIELD_BURST_N, len(g._hit_fx))
    assert (1, 0) in g._hit_fx_cull_bids, \
        "the hitting bullet must be marked for cull: %r" \
        % (g._hit_fx_cull_bids,)
    remaining = [tuple(b.id) for b in g.ghost.local_bullets
                 if b.id is not None]
    assert (1, 0) not in remaining, \
        "the hitting bullet must be removed from local_bullets: %r" \
        % (remaining,)
    print("PASS: SHIELD-UP — a shield-up hit spawns a shield_burst "
          "(%d particles) and is culled" % len(g._hit_fx))

    # --- 1b. LOCAL-GLOW: a shield-up hit ALSO flashes the stand-in's
    #     shield (the immediate glow, same frame as the particle burst —
    #     the buffer's shield_dump is the host's DELAYED replay). The
    #     trigger sets _enemy_glow['ai'] to the shield's power_hit;
    #     _step_hit_fx decays it at SHIELD_DUMP_DECAY until it clears. ---
    from .config import SHIELD_DUMP_DECAY
    # The glow is keyed by ENEMY ID (not tag) — the test's enemy has id
    # 999 (see make_snap). Two enemies of the same tag must not share a
    # glow, so the key is the per-enemy id.
    assert 999 in g._enemy_glow, \
        "a shield-up hit must set the local glow (keyed by enemy id): " \
        "%r" % (g._enemy_glow,)
    _glow_dump, _glow_clock = g._enemy_glow[999]
    _ph = g._get_remote_enemies()['ai'].ship.shield_comp.power_hit
    assert abs(_glow_dump - _ph) < 1e-9, \
        "the local glow must start at the shield's power_hit (%.1f): " \
        "got %.1f" % (_ph, _glow_dump)
    # Decay: step the glow enough frames that it reaches zero and clears.
    _frames = int(_ph / (SHIELD_DUMP_DECAY * 0.016)) + 2
    for _ in range(_frames):
        g._step_hit_fx(0.016)
    assert 999 not in g._enemy_glow, \
        "the local glow must decay to zero and clear: %r" \
        % (g._enemy_glow,)
    print("PASS: LOCAL-GLOW — a shield-up hit flashes the stand-in's "
          "shield (power_hit %.0f) and it decays away" % _ph)

    # --- 2. SHIELD-DOWN: shield_charge=0 -> a small `burst` (14 particles)
    #     spawns at the hull, the bullet is culled, and the CSV tags it
    #     'enemy:ai' (no ':shield'). ---
    push_stream(g, ENEMY_POS, shield_charge=0.0)
    seed_bullet(g, (ex, ey), (ex, ey), bid=(1, 1))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == HULL_BURST_N, \
        "a shield-down hit must spawn a small burst (%d particles): " \
        "got %d" % (HULL_BURST_N, len(g._hit_fx))
    assert (1, 1) in g._hit_fx_cull_bids, \
        "the hitting bullet must be marked for cull: %r" \
        % (g._hit_fx_cull_bids,)
    remaining = [tuple(b.id) for b in g.ghost.local_bullets
                 if b.id is not None]
    assert (1, 1) not in remaining, \
        "the hitting bullet must be removed from local_bullets: %r" \
        % (remaining,)
    print("PASS: SHIELD-DOWN — a shield-down hit spawns a small burst "
          "(%d particles) and is culled" % len(g._hit_fx))

    # --- 3. MISS: a bullet 200 px from the enemy spawns no spark and is
    #     not culled (the test is not a no-op). ---
    push_stream(g, ENEMY_POS, shield_charge=3.0)
    seed_bullet(g, (ex, ey - 200.0), (ex, ey - 200.0), bid=(1, 2))
    g._hit_fx = []
    g._hit_fx_cull_bids.clear()
    render_frame(g)
    assert len(g._hit_fx) == 0, \
        "a bullet 200 px from the enemy must NOT spawn a spark: %r" \
        % (g._hit_fx,)
    assert (1, 2) not in g._hit_fx_cull_bids, \
        "a missing bullet must NOT be culled: %r" \
        % (g._hit_fx_cull_bids,)
    remaining = [tuple(b.id) for b in g.ghost.local_bullets
                 if b.id is not None]
    assert (1, 2) in remaining, \
        "a missing bullet must stay in local_bullets: %r" % (remaining,)
    print("PASS: MISS — a bullet 200 px from the enemy spawns no spark "
          "and is not culled")

    # --- 4. TELEMETRY: the hit_fx CSV records HIT rows tagged
    #     'enemy:ai:shield' (gate 1) and 'enemy:ai' (gate 2). ---
    hit_fx_telemetry.set_enabled(True)
    try:
        push_stream(g, ENEMY_POS, shield_charge=3.0)
        seed_bullet(g, (ex, ey), (ex, ey), bid=(1, 3))
        g._hit_fx = []
        g._hit_fx_cull_bids.clear()
        render_frame(g)
        push_stream(g, ENEMY_POS, shield_charge=0.0)
        seed_bullet(g, (ex, ey), (ex, ey), bid=(1, 4))
        g._hit_fx = []
        g._hit_fx_cull_bids.clear()
        render_frame(g)
        hits = read_hits()
        assert any(r["tgt"] == "enemy:ai:shield" and r["bid"] == "1:3"
                   for r in hits), \
            "the CSV must record a 'enemy:ai:shield' HIT for the " \
            "shield-up bullet: %r" % (hits,)
        assert any(r["tgt"] == "enemy:ai" and r["bid"] == "1:4"
                   for r in hits), \
            "the CSV must record an 'enemy:ai' HIT for the shield-down " \
            "bullet: %r" % (hits,)
    finally:
        hit_fx_telemetry.set_enabled(False)
        hit_fx_telemetry.close()
    print("PASS: TELEMETRY — the CSV records 'enemy:ai:shield' and "
          "'enemy:ai' HIT rows")

    print("ALL PASS: 10.13 (client-side bullet-impact spark fires on "
          "AI enemies, respecting the shield)")


if __name__ == "__main__":
    main()