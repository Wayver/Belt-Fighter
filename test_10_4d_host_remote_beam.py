"""Session 10.4d: the HOST anchors the REMOTE player's beam to the REMOTE
ship, not the local (host) ship.

BUG (confirmed in code): the host's authoritative sim runs BOTH players
(Session 6.2a), so when the CLIENT fires, the host's sim creates that beam
in ``self.beams``. But ``draw()`` anchored EVERY beam in the model to the
LOCAL (host) ship's pose — so the client's beam appeared to emanate from
the host's ship ("looks like the host player just fired").

FIX: each beam now carries its owner's index (the firing player's index,
set in ``_resolve_beam``/``_beam_hit_asteroid`` and serialized by
``render_model``), and ``draw()`` anchors each beam to THAT ship's
interpolated pose (the same pose ``_draw_local_ship`` draws that ship at).
Single-player (1-ship list) is bit-identical: every beam's owner is
0 == local_index.

The beam's END is shared between the buggy and correct anchoring (it
resolves the target by ship_id), so the discriminator is purely the MUZZLE
pose. We prove the fix two ways:
  1. MODEL-OWNER — a beam the REMOTE player (index 1) fired carries
     owner=1 in the render model (not 0).
  2. ANCHOR-POSE — a spy on ``_draw_world_beam`` proves ``draw()`` passes
     the REMOTE ship's interpolated pose (not the local/host ship's pose)
     as the beam's anchor. If the bug regressed (every beam anchored to
     the local ship), the spy would capture the local pose and the
     assertion would fail.

Run from the repo root (the directory that CONTAINS ship5/):

    python -m ship5.test_10_4d_host_remote_beam
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

import pygame

from . import game as game_mod
from .config import WIDTH, HEIGHT
from .fog import make_light_texture
from .game import Game, STEP, _draw_world_beam
from .ship import Ship
from .intent import ShipInput
from .hulls import SILAS_HULL, default_loadout
from .ai_enemy import AIEnemy
from .asteroid import Asteroid

SEED = 1234
EPS = 1e-3

# The remote (client) ship sits mid-screen, facing +x. The local (host)
# ship stays at its spawn (top-left). They are far apart in Y, so the two
# candidate anchor poses are unambiguously different.
REMOTE_POS = pygame.Vector2(960.0, 640.0)
REMOTE_ANGLE = 0.0
ENEMY_OFFSET = 100.0   # enemy 100 px in front of the remote ship


def make_resources():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    return screen, font, big_font, light_tex, fog_surf, light_surf


def make_host(screen, font, big_font, light_tex, fog_surf, light_surf):
    """A 2P HOST Game (local_index=0). Player 0 (the host's own ship) is
    the default scout hull; player 1 (the remote/client ship) is a SILAS
    (its LASER_360 eyes give it a laser)."""
    AIEnemy._next_id = 1
    Asteroid._next_id = 1
    g = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
             seed=SEED, players=2)
    g.set_player_ship(1, Ship(hull=SILAS_HULL,
                              loadout=default_loadout(SILAS_HULL)))
    return g


def _charge_all(s):
    for w in s.weapons:
        w.charge = 1.0


def main():
    screen, font, big_font, light_tex, fog_surf, light_surf = \
        make_resources()

    # --- 1. MODEL-OWNER: a beam the REMOTE player (index 1) fired carries
    #     owner=1 in the render model (not 0). The host's sim runs both
    #     players, so the remote's beam lands in self.beams; the fix tags
    #     it with the firing player's index. ---
    g = make_host(screen, font, big_font, light_tex, fog_surf, light_surf)
    g.protect_timer = 0.0
    # Remote (client) ship mid-screen, facing +x.
    r = g.players[1]
    r.pos = REMOTE_POS.copy()
    r.vel = pygame.Vector2(0.0, 0.0)
    r.angle = REMOTE_ANGLE
    # Enemy 100 px in front of the remote ship (in the LASER_360 wedge).
    e = AIEnemy(REMOTE_POS + pygame.Vector2(ENEMY_OFFSET, 0.0),
                rng=g.rng)
    e.ship.vel = pygame.Vector2(0.0, 0.0)
    e.ship.shield_charge = 10.0   # absorb the 4-damage beam; target survives
    g.enemies = [e]
    g.asteroids = []
    _charge_all(r)
    # The remote player (index 1) is stepped with self.remote_input (NOT
    # the local `inp`), so the laser_fire must go there. The local (host)
    # player (index 0) stays idle.
    g.remote_input = ShipInput(laser_fire=True)
    for _ in range(10):
        g._step(STEP, ShipInput())   # local (host) player stays idle
        if g.beams:
            break
        _charge_all(r)               # recharge if the discharge reset it
    assert g.beams, "the remote player's laser must fire a beam into the " \
        "host's self.beams (charge 1.0, laser_fire, enemy in range): " \
        "beams=%r" % (g.beams,)
    assert g.beams[0][6] == 1, \
        "the remote player's beam must carry owner=1 (the firing player's " \
        "index), got %r" % (g.beams[0][6],)
    # The enemy's shield absorbs the beam, so the target survives and the
    # beam's target_id resolves to it (the live-target anchor path).
    assert e.hp > 0, "the enemy must survive the beam (shield absorbs): " \
        "hp=%r" % (e.hp,)
    m = g.render_model()
    assert m["beams"], "the model must carry the beam(s): %r" \
        % (m["beams"],)
    for mb in m["beams"]:
        assert len(mb) == 7, "the model beam must be a 7-tuple (with " \
            "owner): %r" % (mb,)
        assert mb[6] == 1, "every remote beam must carry owner=1: %r" \
            % (mb,)
    print("PASS: MODEL-OWNER — the remote player's beam carries owner=1 "
          "in the render model (7-tuple, owner at [6])")

    # --- 2. ANCHOR-POSE: draw() anchors the remote beam to the REMOTE
    #     ship's interpolated pose, NOT the local (host) ship's pose. The
    #     beam's END is shared between the two anchors (it resolves the
    #     target by ship_id), so the muzzle pose is the discriminator. A
    #     spy on _draw_world_beam captures the (rpos, rangle) draw()
    #     passes; it must equal the remote ship's pose and differ from the
    #     local ship's pose. If the bug regressed (every beam anchored to
    #     the local ship), the spy would capture the local pose and this
    #     would fail. ---
    g2 = make_host(screen, font, big_font, light_tex, fog_surf, light_surf)
    g2.protect_timer = 0.0
    # Local (host) ship far from the remote ship (both default to center),
    # so the two candidate anchor poses are unambiguously different.
    local = g2.players[0]
    local.pos = pygame.Vector2(200.0, 200.0)
    local.vel = pygame.Vector2(0.0, 0.0)
    local.angle = 0.0
    r2 = g2.players[1]
    r2.pos = REMOTE_POS.copy()
    r2.vel = pygame.Vector2(0.0, 0.0)
    r2.angle = REMOTE_ANGLE
    e2 = AIEnemy(REMOTE_POS + pygame.Vector2(ENEMY_OFFSET, 0.0),
                 rng=g2.rng)
    e2.ship.vel = pygame.Vector2(0.0, 0.0)
    e2.ship.shield_charge = 10.0  # absorb the beam; target survives
    g2.enemies = [e2]
    g2.asteroids = []
    _charge_all(r2)
    g2.remote_input = ShipInput(laser_fire=True)   # remote fires; local idle
    for _ in range(10):
        g2._step(STEP, ShipInput())
        if g2.beams:
            break
        _charge_all(r2)
    assert g2.beams, "the remote player's beam must fire: beams=%r" \
        % (g2.beams,)
    # The local (host) ship is far from the remote ship (top-left vs
    # mid-screen), so the two candidate anchor poses are unambiguous.
    assert local.pos.distance_to(r2.pos) > 300, \
        "the local and remote ships must be far apart for the anchor " \
        "check to be unambiguous: local=%r remote=%r" \
        % (local.pos, r2.pos)
    # Center the camera on the local ship (the draw() camera target) so
    # the beam is on-screen; the anchor pose is independent of the camera.
    g2.cam.pos = local.pos.copy()
    g2.cam.lead = 0.0
    # Spy on _draw_world_beam to capture the anchor pose draw() passes.
    # draw() looks up the module-global _draw_world_beam at call time, so
    # we patch the GAME MODULE's attribute (not this test's local name).
    captured = []
    real = game_mod._draw_world_beam

    def spy(screen_, cam_, rpos, rangle, beam, enemies_by_id, standins):
        captured.append((pygame.Vector2(rpos), rangle))
        return real(screen_, cam_, rpos, rangle, beam, enemies_by_id,
                    standins)

    game_mod._draw_world_beam = spy
    try:
        g2.draw(0.016, model=g2.render_model())
    finally:
        game_mod._draw_world_beam = real
    assert captured, "draw() must anchor at least one beam: captured=%r" \
        % (captured,)
    # The anchor must be the REMOTE ship's interpolated pose for EVERY
    # beam. With step_alpha=0 (acc=0) the interpolated pose == the current
    # pose.
    want_pos = pygame.Vector2(r2.pos.x, r2.pos.y)
    want_ang = r2.angle
    for got_pos, got_ang in captured:
        assert got_pos.distance_to(want_pos) < EPS, \
            "draw() must anchor the remote beam to the REMOTE ship's " \
            "pose: got=%r want=%r" % (got_pos, want_pos)
        da = (got_ang - want_ang + math.pi) % (2 * math.pi) - math.pi
        assert abs(da) < EPS, \
            "draw() must anchor the remote beam to the REMOTE ship's " \
            "angle: got=%r want=%r" % (got_ang, want_ang)
        # And it must NOT be the local (host) ship's pose (the bug).
        assert got_pos.distance_to(local.pos) > 300, \
            "draw() must NOT anchor the remote beam to the LOCAL (host) " \
            "ship's pose (the bug): got=%r local=%r" \
            % (got_pos, local.pos)
    print("PASS: ANCHOR-POSE — draw() anchors all %d remote beam(s) to "
          "the REMOTE ship's pose (%r), not the local (host) ship's pose "
          "(%r)" % (len(captured),
                    tuple(round(v, 1) for v in want_pos),
                    tuple(round(v, 1) for v in local.pos)))

    print("ALL PASS: 10.4d (host anchors the remote beam to the remote "
          "ship)")


if __name__ == "__main__":
    main()