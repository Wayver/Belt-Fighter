"""Game state: entities, collisions, and per-frame update/draw.

Networked-ready:
- update() runs the whole sim on a fixed timestep (STEP) via an
  accumulator, so every entity steps on the same deterministic tick.
- Player intent (movement + fire) arrives as a ShipInput, not raw
  key state — the same type a remote player's input will be.
"""
import math
import random
from dataclasses import replace

import pygame

from .config import (WIDTH, HEIGHT, SPAWN_PROTECT, MAX_BULLETS,
                    BULLET_SPEED,
                    ROCK_SPLIT, ROCK_SIZES, BG, STAR_COLOR,
                    BULLET_COLOR, ENEMY_BULLET_COLOR,
                    TARGETING_ASSIST, TARGETING_COLOR, TARGETING_HORIZON,
                    TARGETING_STEPS, TARGETING_RANGE,
                    TARGETING_USE_ACCEL, TARGETING_MAX_LEAD,
                    TARGETING_COLOR_GREEN, TARGETING_ALIGN_TOL, LASER_COLOR,
                    BEAM_IMPACT_SPREAD, SENSOR_COLOR, SENSOR_SCAN_COLOR,
                    SENSOR_SIGNATURE_THRESHOLD, SENSOR_SIG_FULL, SENSOR_ARROW_MARGIN,
                    SCAN_DUMP_DECAY, DEBUG_COLLISION,
                    MISSILE_COLOR, MAX_MISSILES,
                    SFX_LASER_MIN_INTERVAL, SFX_ENEMY_LASER_MIN_INTERVAL,
                    SFX_SHIELD_HIT_MIN_INTERVAL)

from .bullets import Bullet, EnemyBullet, Missile, MissileShot

from .ship import Ship, wrapped_delta, _dim_color
from .intent import ShipInput
from .asteroid import Asteroid
from .bullets import Bullet, EnemyBullet
from .particles import burst, shield_burst
from .spawning import spawn_enemy, make_stars, update_field, TestTarget
from .ai_enemy import AIEnemy, MoteEnemy

from .fog import draw_fog, LightSource

from .hud import draw_hud, draw_game_over
from .camera import Camera
from .netcode import SnapshotBuffer, PredictedShip
from .config import (INTERP_DELAY, SNAPSHOT_INTERVAL, ROCK_FILL, ROCK_EDGE,
                    ENEMY_FILL, ENEMY_EDGE)

STEP = 1 / 60   # fixed simulation timestep


class Game:
    def __init__(self, screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=None, loadout=None, test_mode=False, seed=None, sound=None,
                local_index=0, players=1):
        self.screen = screen
        self.font = font
        self.big_font = big_font
        self.light_tex = light_tex
        self.fog_surf = fog_surf
        self.light_surf = light_surf
        # One rng for the whole sim: same seed -> same run. None -> random
        # each launch (the pre-seed behavior).
        self.rng = random.Random(seed)
        self.stars = make_stars(rng=self.rng)

        # Player ships (Session 6.1): a list, not a single attribute.
        # Single-player is a 1-ship list; 2P is 2 ships in the same world.
        # `self.ship` (the property below) is players[0] — the backward-
        # compat alias the sim, the HUD, and the tests still read.
        # players: 1 = single-player (the local hull/loadout); 2 = 2P host
        # (Session 6.5) — player 0 is the host's ship, player 1 is a
        # PLACEHOLDER (the client's hull/loadout is unknown until the join
        # handshake). The host replaces the placeholder after the handshake
        # via `set_player_ship` (the client's shield ring is rebuilt then).
        if players == 1:
            self.players = [Ship(hull=hull, loadout=loadout)]
        else:
            self.players = [Ship(hull=hull, loadout=loadout), Ship()]
        # local_index: which player this Game controls locally (0 = host's
        # own ship, 1 = the client's). The prediction ghost tracks
        # players[local_index] (Session 6.1 layout; the host applies the
        # remote player's input in 6.2).
        self.local_index = local_index
        # Per-player shield ring (Session 6.2a): one pre-drawn Surface per
        # player, sized by that player's hull radius (hulls may differ in 2P).
        self.shields = [self._make_shield(p) for p in self.players]

        self.cam = Camera(self.ship.pos)
        self.bullets = []
        self.missiles = []
        self.beams   = []
        self.enemy_bullets = []
        self.particles = []
        self.asteroids = []
        self.enemies = []
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.test_mode = test_mode
        self.sound = sound      # SoundBank or None (silent, e.g. headless)
        # --- networking: remote-interpolation render state (Session 5b.3) ---
        # sim_time: seconds of sim this Game has advanced. The authoritative
        # peer stamps each snapshot with it; the remote peer renders
        # sim_time - INTERP_DELAY (see remote_view).
        self.sim_time = 0.0
        self._snap_tick = 0        # sim ticks since the last snapshot stamp
        self.snap_buf = SnapshotBuffer()
        self.ghost = PredictedShip()   # local-ship prediction ghost (5b.4b)
        # Remote player's latest input (Session 6.2a): the host stores the
        # client's ShipInput here (set_remote_input) and _step applies it to
        # player 1. Empty default = no thrust/fire; single-player never sets
        # it (only player 0 exists), so the sim is untouched.
        self.remote_input = ShipInput()
        self.reset()

    @property
    def ship(self):
        """Backward-compat alias: the local player's ship (players[0]).

        Session 6.1 generalized the sim to a players list (2P: two ships in
        the same world). Everything that reads self.ship — the sim, the HUD,
        the tests — keeps working; 2P code uses self.players directly.
        Read-only: assign to self.players, not self.ship."""
        return self.players[0]

    def _make_shield(self, p):
        """Pre-drawn shield ring for one player ship, sized by that hull's
        collision radius (hulls may differ in 2P)."""
        r = p.collision_radius
        sh = pygame.Surface((int(r * 2 + 10), int(r * 2 + 10)),
                            pygame.SRCALPHA)
        pygame.draw.circle(sh, (120, 200, 255, 100),
                           (sh.get_width() // 2, sh.get_height() // 2),
                           r + 5, 2)
        return sh

    def set_player_ship(self, i, ship):
        """Replace player slot `i` with a built Ship (Session 6.5).

        The host calls this once, after the join handshake, to swap the
        player-1 placeholder for the client's real hull/loadout. The
        placeholder has never been stepped (the handshake happens before
        the first tick), so replacing it in place is safe; the shield ring
        is rebuilt for the new hull's radius. The caller must build the
        Ship itself (it owns the hull/loadout from the wire).
        """
        self.players[i] = ship
        self.shields[i] = self._make_shield(ship)

    def _sfx(self, name):
        """Play a sound if a SoundBank is attached. Pure side effect:
        the sim never reads sound state, so determinism is untouched."""
        if self.sound:
            self.sound.play(name)

    def _sfx_throttled(self, name, min_interval):
        """Throttled variant for rapid-fire weapons (see SoundBank)."""
        if self.sound:
            self.sound.play_throttled(name, min_interval)

    def _sfx_thruster(self, on):
        """Engine loop on/off (see SoundBank.set_thruster)."""
        if self.sound:
            self.sound.set_thruster(on)

    def reset(self):
        # Reset ALL player ships (Session 6.2a): each to center, idle.
        # (Both players spawn at center in v1 — no ship-vs-ship collision,
        # so they pass through each other.)
        for p in self.players:
            p.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
            p.vel = pygame.Vector2(0, 0)
            p.angle = -math.pi / 2
            p.prev_pos = p.pos.copy()
            p.prev_angle = p.angle
            p.reset_shield()
            p.targeting_on = False
            p.tracked = 0
            p.reset_missiles()
            p.reset_lasers()
            p.reset_sensors()
        self.bullets.clear()
        self.enemy_bullets.clear()
        self.beams.clear()
        self.missiles.clear()
        self.particles.clear()
        self.asteroids.clear()
        self.enemies.clear()
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.cam.pos = self.ship.pos.copy()
        self._sfx_thruster(False)   # never carry the engine loop across a reset
        if self.test_mode:
            self._setup_test_scene()
            return
        update_field(self.asteroids, [p.pos for p in self.players], 0,
                     rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)

    # --- networking: whole-sim snapshot (see Ship.snapshot for the
    # ship-level classification) ---
    #
    # SYNCED (serialized, must be identical on both peers):
    #   players              index 0: tuple of one Ship.snapshot() per
    #                        player (21-field tuple each, see ship.py).
    #                        2P: both ships in the same world (6.1).
    #   enemies              per enemy: variant tag + e.snapshot()
    #                        tag = 'mote' (MoteEnemy) | 'test' (TestTarget)
    #                        | 'ai' (AIEnemy). Order matters: MoteEnemy and
    #                        TestTarget are both AIEnemy subclasses, so check
    #                        them first.
    #   bullets, enemy_bullets, missiles
    #                        per projectile: pos/vel/owner/life (+ boost for
    #                        missiles). Missile target is stored as the
    #                        enemy's ship id; apply_snapshot re-wires it.
    #   asteroids            per rock: id/pos/vel/size/angle/spin/verts
    #   rng state            the sim's single rng — the strongest check;
    #                        without it the peers diverge on the next roll
    #   game_over, protect_timer
    #   AIEnemy._next_id     class-level id counter (not instance state —
    #                        a naive snapshot misses it)
    #   Asteroid._next_id    same for rocks (Session 5b.1)
    #
    # NOT SYNCED (presentation / config / local timing — never serialized):
    #   stars                cosmetic background; regenerated locally
    #   shield               pre-drawn pygame.Surface ring (derived from
    #                        ship radius)
    #   cam                  Camera; re-derived from ship.pos on apply
    #   beams, particles     presentation (0.15 s beam ttl, cosmetic bursts)
    #   acc                  fixed-timestep accumulator — local timing, not
    #                        sim state
    #   screen, font, big_font, light_tex, fog_surf, light_surf
    #                        render resources
    #   test_mode, sound     mode flag + side-effect layer
    #   (score/wave/wave_timer no longer exist — removed in the wave/score
    #   cleanup; do not re-add them here.)

    def snapshot(self):
        """Capture the whole sim as ONE plain-data tuple: no pygame
        objects, no references — safe to pickle / send over the wire."""
        return (
            tuple(p.snapshot() for p in self.players),
            tuple((self._enemy_tag(e), e.snapshot()) for e in self.enemies),
            tuple(b.snapshot() for b in self.bullets),
            tuple(b.snapshot() for b in self.enemy_bullets),
            tuple(m.snapshot() for m in self.missiles),
            tuple(a.snapshot() for a in self.asteroids),
            self.rng.getstate(),
            self.game_over,
            self.protect_timer,
            AIEnemy._next_id,
            Asteroid._next_id,
        )

    @staticmethod
    def _enemy_tag(e):
        # Order matters: MoteEnemy and TestTarget are both AIEnemy
        # subclasses, so they must be checked before the base class.
        if isinstance(e, MoteEnemy):
            return 'mote'
        if isinstance(e, TestTarget):
            return 'test'
        return 'ai'

    def apply_snapshot(self, s):
        """Restore a snapshot() tuple into this Game. Entities are
        constructed via the same paths the sim uses (so invariants hold),
        then their synced fields are overwritten."""
        (players_s, enemies_s, bullets_s, enemy_bullets_s, missiles_s,
         asteroids_s, rng_state, game_over, protect_timer, next_id,
         rock_next_id) = s

        # Player ships (Session 6.1): index 0 is a tuple of one
        # Ship.snapshot() per player; apply each to the matching slot.
        for p, p_s in zip(self.players, players_s):
            p.apply_snapshot(p_s)

        # Enemies: build with a THROWAWAY rng so construction (which rolls
        # ship.angle and consumes _next_id) neither burns the real rng
        # stream nor matters — apply_snapshot overwrites the synced fields.
        throwaway = random.Random(0)
        origin = pygame.Vector2(0, 0)
        self.enemies.clear()
        for tag, e_s in enemies_s:
            if tag == 'mote':
                e = MoteEnemy(origin, rng=throwaway)
            elif tag == 'test':
                e = TestTarget(origin, rng=throwaway)
            else:
                e = AIEnemy(origin, rng=throwaway)
            e.apply_snapshot(e_s)
            self.enemies.append(e)

        # Bullets: construct minimally, then overwrite.
        self.bullets.clear()
        for b_s in bullets_s:
            px, py, vx, vy, owner, _life = b_s
            b = Bullet(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                       owner=owner)
            b.apply_snapshot(b_s)
            self.bullets.append(b)

        self.enemy_bullets.clear()
        for b_s in enemy_bullets_s:
            px, py, vx, vy, owner, _life = b_s
            b = EnemyBullet(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                            owner=owner)
            b.apply_snapshot(b_s)
            self.enemy_bullets.append(b)

        # Missiles: same pattern; target is re-wired below, once every
        # enemy exists.
        self.missiles.clear()
        for m_s in missiles_s:
            px, py, vx, vy, owner, _life, _boost, _tid = m_s
            m = Missile(pygame.Vector2(px, py), pygame.Vector2(vx, vy),
                        owner=owner)
            m.apply_snapshot(m_s)
            self.missiles.append(m)
        id_map = {e.ship.id: e for e in self.enemies}
        for m in self.missiles:
            m.target = id_map.get(m._target_id)

        # Asteroids: construct minimally (vel=None rolls a throwaway vel
        # from the GLOBAL random module — the sim's rng is untouched, and
        # apply_snapshot overwrites it anyway); apply_snapshot rebuilds
        # radius/collision_radius from ROCK_SIZES[size] and restores the
        # rock's id.
        self.asteroids.clear()
        for a_s in asteroids_s:
            _aid, px, py, _vx, _vy, size, _angle, _spin, _verts = a_s
            a = Asteroid(pygame.Vector2(px, py), size)
            a.apply_snapshot(a_s)
            self.asteroids.append(a)

        self.rng.setstate(rng_state)
        self.game_over = game_over
        self.protect_timer = protect_timer
        # AFTER constructing enemies/asteroids: their construction already
        # consumed ids from the class counters, so restore the snapshotted
        # values now (same pattern for both counters).
        AIEnemy._next_id = next_id
        Asteroid._next_id = rock_next_id
        # Camera is not synced — re-derive it from the restored ship.
        self.cam.pos = self.ship.pos.copy()

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_t and not self.game_over:
                    self.ship.targeting_on = not self.ship.targeting_on
                elif event.key == pygame.K_r and self.game_over:
                    self.reset()
                elif event.key == pygame.K_v and not self.game_over:
                    self.ship.sensor_on = not self.ship.sensor_on
                elif event.key == pygame.K_g and not self.game_over:
                    self.ship.fire_scan()
                elif event.key == pygame.K_f and (self.test_mode or self.game_over):
                    self.reset()
        return True

    def update(self, dt, keys):
        # Sample input once per frame; apply it to each fixed step.
        inp = ShipInput.from_keys(keys)
        self.acc += min(dt, 0.25)   # clamp: no spiral of death after a hitch
        while self.acc >= STEP:
            self._step(STEP, inp)
            self.acc -= STEP
    def _step(self, dt, inp):
        self.sim_time += dt   # sim clock: every fixed step advances it
        if not self.game_over:
            # Movement + fire for EVERY player (Session 6.2a). Player 0 uses
            # the local keys' input; the remote player (index 1) uses the
            # host's latest received input (self.remote_input). Single-player
            # is a 1-ship list, so this loop runs once with the local input —
            # bit-identical to the old single-ship path.
            for i, p in enumerate(self.players):
                p_inp = inp if i == 0 else self.remote_input
                # Per-ship targeting (Session 6.2b): each player gets its own
                # tracked count / laser target / missile target / sensor
                # contacts, all set BEFORE p.update() so this tick's
                # _allocate()/_update_lasers()/_update_missiles() see them.
                # Player 0 is self.ship, so single-player is bit-identical.
                p.tracked = sum(
                    1 for e in self.enemies
                    if e.pos.distance_to(p.pos) <= TARGETING_RANGE)
                p.laser_target = self._pick_laser_target(p)
                p.missile_target = self._pick_missile_target(p)
                self._update_contacts(p)
                if p_inp.fire and len(self.bullets) >= MAX_BULLETS:
                    p_inp = replace(p_inp, fire=False)   # world cap: no room
                if p_inp.missile_fire and len(self.missiles) >= MAX_MISSILES:
                    p_inp = replace(p_inp, missile_fire=False)
                shots, beams, missiles = p.update(dt, p_inp)

                for shot in shots:
                    self.bullets.append(Bullet(shot.pos, shot.vel,
                                               owner=shot.owner))
                # S3: fire sounds. The gun fires ~100 shots/s (FIRE_COOLDOWN
                # 0.01), so the laser blip is throttled to a human rate.
                if shots:
                    self._sfx_throttled("laser", SFX_LASER_MIN_INTERVAL)

                for m in missiles:
                    self.missiles.append(Missile(m.pos, m.vel, owner=m.owner,
                                                 target=m.target))
                if missiles:
                    self._sfx("missile_launch")

                for beam in beams:
                    self._resolve_beam(beam)
                if beams:
                    # Hitscan laser: one discharge per charge cycle (~1/s),
                    # so no throttle needed — unlike the rapid-fire gun blip.
                    self._sfx("beam")
            
            self.protect_timer -= dt
            if not self.test_mode:
                update_field(self.asteroids, [p.pos for p in self.players],
                             dt, rng=self.rng)

        # Engine loop: on while the ship is alive and any fitted thruster
        # has resolved force (demand * allocation > 0). Death or a power
        # brownout cuts the sound; set_thruster(False) is a cheap no-op
        # when the loop is already off.
        self._sfx_thruster(not self.game_over
                           and any(t.force > 0.0
                                   for p in self.players
                                   for t in p.thrusters))

        enemy_fired = False
        for e in self.enemies:
            # Each enemy targets the NEAREST player (Session 6.2a). With one
            # player this is that player — bit-identical to the old path.
            shots = e.update(dt, self._nearest_player(e.pos), self.asteroids)
            for shot in shots:
                self.enemy_bullets.append(EnemyBullet(shot.pos, shot.vel, owner=shot.owner))
            if shots:
                enemy_fired = True
        if enemy_fired:
            self._sfx_throttled("enemy_laser", SFX_ENEMY_LASER_MIN_INTERVAL)

        for b in self.bullets:
            b.update(dt)
        for b in self.enemy_bullets:
            b.update(dt)
        for m in self.missiles:
            m.update(dt)
        for a in self.asteroids:
            a.update(dt)
        for p in self.particles:
            p.update(dt)

        # cull expired projectiles and particles
        self.bullets = [b for b in self.bullets if b.life > 0]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.life > 0]
        self.missiles = [m for m in self.missiles if m.life > 0]
        self.particles = [p for p in self.particles if p.life > 0]

        # collisions after movement, so this tick's motion counts
        self._collisions()

        # age the beam visuals
        for b in self.beams:
            b[4] += dt
        self.beams = [b for b in self.beams if b[4] < b[5]]

    def set_remote_input(self, inp):
        """Store the remote player's latest ShipInput (Session 6.2a). The
        host calls this when an 'input' message arrives; _step applies it to
        player 1 (index 1) each tick. Single-player never calls it."""
        self.remote_input = inp

    def _nearest_player(self, pos):
        """The player ship nearest to `pos` (Session 6.2a). Enemies target
        the nearest player; with one player this is that player."""
        return min(self.players,
                   key=lambda p: p.pos.distance_squared_to(pos))

    
    def _build_lights(self, ship):
        """Whitelist of things that shine through the fog of war.
        Returns a list of LightSource in world coords, built fresh each frame.
        (Session 6.2b: per-ship — called with the local ship.)"""
        lights = []


        # --- Targeting reticle: a small light at each predicted lead point.
        # Mirrors the guard in _draw_lead so the light and the reticle
        # appear/disappear together.
        if TARGETING_ASSIST and ship.targeting_on:
            for e in self.enemies:
                p = e.lead_point(ship.pos, BULLET_SPEED, TARGETING_USE_ACCEL)
                if p is not None and (p - ship.pos).length() <= TARGETING_RANGE:
                    lights.append(LightSource(p, 50, 0.6))

        # --- Weapon fire: bullets glow as they fly through the dark.
        for b in self.bullets:
            lights.append(LightSource(b.pos, 30, 0.5))
        for b in self.enemy_bullets:
            lights.append(LightSource(b.pos, 24, 0.4))

        for m in self.missiles:
            lights.append(LightSource(m.pos, 30, 0.5))

        # --- Shield impacts: a fading flash at the hit point on the oval.
        # shield_impacts stores [theta, age, ttl] in local hull space, so
        # convert to world the same way _draw_shield_impacts does.
        if ship.shield_impacts:
            a, b, cx, cy = ship.shield_oval
            fwd, right = ship.axes()
            for theta, age, ttl in ship.shield_impacts:
                fade = 1.0 - age / ttl
                world = (ship.pos + fwd * (cx + a * math.cos(theta))
                                 + right * (cy + b * math.sin(theta)))
                lights.append(LightSource(world, 40, 0.5 * fade))

        return lights


    def _update_contacts(self, ship):
        """Build the ship's contact list from the fitted sensor.

        Passive: enemies in sensor_range whose active power draw
        (power_used - idle) crosses the signature threshold. Active:
        while a ping's reveal lasts, everything in scan_range is
        confirmed. A confirmed contact replaces a passive one.
        (Session 6.2b: per-ship, called once per player.)
        """
        c = ship.sensor_comp
        ship.contacts = []
        if c is None:
            return
        found = {}
        if ship.sensor_on and c.sensor_range > 0:
            for e in self.enemies:
                d = e.pos.distance_to(ship.pos)
                if d <= c.sensor_range:
                    sig = e.ship.power_used - e.ship.power_idle_total
                    if sig >= SENSOR_SIGNATURE_THRESHOLD:
                        found[e] = [e.pos.copy(), d,
                                    min(1.0, sig / SENSOR_SIG_FULL), False]
        if ship.scan_reveal > 0 and c.scan_range > 0:
            for e in self.enemies:
                d = e.pos.distance_to(ship.pos)
                if d <= c.scan_range:
                    found[e] = [e.pos.copy(), d, 1.0, True]
        ship.contacts = list(found.values())

    def _draw_sensor_contacts(self):
        """Contacts above the fog: on-screen blips, off-screen edge
        arrows. Confirmed (scanned) contacts are brighter + show distance."""
        screen = self.screen
        ship = self.ship
        if not ship.contacts:
            return
        sp = self.cam.to_screen(ship.pos)
        for pos, dist, strength, confirmed in ship.contacts:
            s = self.cam.to_screen(pos)
            base = SENSOR_SCAN_COLOR if confirmed else SENSOR_COLOR
            color = _dim_color(base, 0.35 + 0.65 * strength)
            if -20 <= s.x <= WIDTH + 20 and -20 <= s.y <= HEIGHT + 20:
                pygame.draw.circle(screen, color, (int(s.x), int(s.y)), 5, 2)
                continue
            d = pygame.Vector2(s.x - sp.x, s.y - sp.y)
            if d.length_squared() < 1:
                continue
            d.normalize_ip()
            m = SENSOR_ARROW_MARGIN
            ts = []
            if d.x > 0:
                ts.append((WIDTH - m - sp.x) / d.x)
            elif d.x < 0:
                ts.append((m - sp.x) / d.x)
            if d.y > 0:
                ts.append((HEIGHT - m - sp.y) / d.y)
            elif d.y < 0:
                ts.append((m - sp.y) / d.y)
            if not ts:
                continue
            p = sp + d * min(ts)
            ang = math.atan2(d.y, d.x)
            for off in (0.5, -0.5):
                pygame.draw.line(screen, color, (p.x, p.y),
                                  (p.x - math.cos(ang + off) * 9,
                                   p.y - math.sin(ang + off) * 9), 2)
            if confirmed:
                txt = self.font.render(f"{dist:.0f}", True, color)
                screen.blit(txt, (p.x - txt.get_width() / 2, p.y + 10))

    def _pick_laser_target(self, ship):
        """Nearest enemy within the max laser range; None if no laser fitted.
        (Session 6.2b: per-ship, called once per player.)"""
        max_range = max((w.comp.laser_range for w in ship.weapons
                         if w.comp.laser_range > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(ship.pos)
            if d <= best_d:
                best, best_d = e, d
        return best


    def _pick_missile_target(self, ship):
        """Nearest enemy within the max missile lock range; None if no
        missile fitted. (Session 6.2b: per-ship, called once per player.)"""
        max_range = max((w.comp.missile_lock_range for w in ship.weapons
                         if w.comp.missile_speed > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(ship.pos)
            if d <= best_d:
                best, best_d = e, d
        return best


    def _resolve_beam(self, beam):
        """Hitscan: hit the first enemy near the beam's end point.

        The beam *line* is drawn to the point on the target's shield/hull
        facing the muzzle, so beams from different muzzles land at different
        spots. Each damage point additionally jitters its shield flash.
        """
        rock, hit_pt = self._beam_blocked(beam)
        if rock is not None:
            self._beam_hit_asteroid(rock, hit_pt, beam)
            return

        for i, e in enumerate(self.enemies):
            if e.pos.distance_to(beam.end) < e.collision_radius + 4:
                approach = e.pos - beam.start
                if approach.length_squared() < 1e-6:
                    approach = pygame.Vector2(1, 0)
                base_ang = math.atan2(approach.y, approach.x)
                # Where the beam line visually lands: on the shield oval (or
                # hull) facing the muzzle.
                d = pygame.Vector2(math.cos(base_ang), math.sin(base_ang))
                vis_end = e.ship.shield_impact_point(e.pos - d * e.collision_radius)
                for _ in range(beam.damage):
                    ang = base_ang + self.rng.uniform(-BEAM_IMPACT_SPREAD,
                                                BEAM_IMPACT_SPREAD)
                    d2 = pygame.Vector2(math.cos(ang), math.sin(ang))
                    impact = e.pos - d2 * e.collision_radius
                    if not e.register_hit(impact):
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                        break
                    shield_burst(self.particles,
                             e.ship.shield_impact_point(impact),
                              rng=self.rng)
                self.beams.append([beam.local_start, e, d,  vis_end, 0.0, 0.15])
                return


    def _beam_blocked(self, beam):
        """Nearest asteroid intersecting the beam segment, or (None, None)."""
        seg = beam.end - beam.start
        seg_len2 = seg.length_squared()
        if seg_len2 < 1e-6:
            return None, None
        best_t, best_a, best_pt = 1.0, None, None
        for a in self.asteroids:
            t = (a.pos - beam.start).dot(seg) / seg_len2
            t = max(0.0, min(1.0, t))          # clamp: only between muzzle and target
            closest = beam.start + seg * t
            if closest.distance_to(a.pos) < a.collision_radius and t < best_t:
                best_t, best_a, best_pt = t, a, closest
        return best_a, best_pt

    def _beam_hit_asteroid(self, a, hit_pt, beam):
        # Same kill/split as a bullet hitting a rock.
        burst(self.particles, hit_pt, a.radius, rng=self.rng)
        self._sfx("small_explosion")
        child_size = ROCK_SPLIT[a.size]
        if child_size:
            for _ in range(2):
                ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                ka = self.rng.uniform(0, 2 * math.pi)
                kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                self.asteroids.append(Asteroid(a.pos, child_size,
                                           vel=a.vel * 0.5 + kick,
                                           rng=self.rng))
        self.asteroids.remove(a)
        # Beam visual: fixed endpoint on the rock's surface. target=None makes
        # draw() use vis_end (the "target already gone" path).
        d = hit_pt - a.pos
        if d.length_squared() < 1e-6:
            d = beam.end - beam.start
        d = d.normalize()
        vis_end = a.pos + d * a.collision_radius
        self.beams.append([beam.local_start, None, d, vis_end, 0.0, 0.15])

    def _draw_targeting(self, screen, e):
        pts = e.predict_path(TARGETING_HORIZON, TARGETING_STEPS)
        n = len(pts)
        for i, p in enumerate(pts):
            s = self.cam.to_screen(p)          # world -> screen [3]
            t = i / (n - 1)                    # 0 at enemy, 1 at far end
            r = max(1, int(3 * (1.0 - t)))     # dots shrink toward the far end
            c = tuple(int(ch * (1.0 - 0.6 * t)) for ch in TARGETING_COLOR)
            pygame.draw.circle(screen, c, (int(s.x), int(s.y)), r)

    def _draw_lead(self, screen, e):
        p = e.lead_point(self.ship.pos, BULLET_SPEED, TARGETING_USE_ACCEL)
        if p is None or (p - self.ship.pos).length() > TARGETING_RANGE:
            return
        if self._lead_aligned(e, p):
            # flash green: alternate between green and the base light blue
            c = (TARGETING_COLOR_GREEN if (pygame.time.get_ticks() // 100) % 2
                 else TARGETING_COLOR)
        else:
            c = TARGETING_COLOR
        s = self.cam.to_screen(p)
        x, y = int(s.x), int(s.y)
        R, gap = 8, 3
        pygame.draw.line(screen, c, (x, y - R), (x, y - gap), 2)
        pygame.draw.line(screen, c, (x, y + R), (x, y + gap), 2)
        pygame.draw.line(screen, c, (x - R, y), (x - gap, y), 2)
        pygame.draw.line(screen, c, (x + R, y), (x + gap, y), 2)

    def _lead_aligned(self, e, p):
        """True when the player's nose points between the reticle and the
        enemy: the facing direction lies in the angular span (plus tolerance)
        between the direction to the reticle and the direction to the enemy."""
        ship = self.ship
        to_ret = p - ship.pos
        to_en = e.pos - ship.pos
        if to_ret.length() < 1 or to_en.length() < 1:
            return False
        ang_r = math.atan2(to_ret.y, to_ret.x)
        ang_e = math.atan2(to_en.y, to_en.x)
        d_f = wrapped_delta(ang_r, ship.angle, 2 * math.pi)   # reticle -> facing
        d_e = wrapped_delta(ang_r, ang_e, 2 * math.pi)        # reticle -> enemy
        if d_f * d_e < 0:
            return False   # facing on the far side of the reticle
        return abs(d_f) <= abs(d_e) + TARGETING_ALIGN_TOL


    def _handle_ship_hit(self, ship, source_pos):
        """Handle a hit on a player ship (Session 6.2a: takes the ship).
        Returns True if the ship survives."""
        if ship.register_hit():
            impact = ship.shield_impact_point(source_pos)
            shield_burst(self.particles, impact, rng=self.rng)
            # Throttled: 3 enemies can hit ~20/s; the ping is 0.3 s long.
            self._sfx_throttled("shield_hit", SFX_SHIELD_HIT_MIN_INTERVAL)
            return True
        self.game_over = True
        burst(self.particles, ship.pos, 30, big=True, rng=self.rng)
        self._sfx("explosion")
        self._sfx("game_over")
        return False

    def _collisions(self):
        # player bullet vs enemy (swept segment vs hull polygon)
        for b in self.bullets[:]:
            for i, e in enumerate(self.enemies):
                hit, hit_pt = e.ship.collision_segment(b.prev_pos, b.pos)
                if hit:
                    self.bullets.remove(b)
                    impact = pygame.Vector2(hit_pt)
                    if e.register_hit(impact):
                        burst(self.particles, impact, 6, rng=self.rng)
                    else:
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                    break

        # bullet vs asteroid
        for b in self.bullets[:]:
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    self.bullets.remove(b)
                    break

        # missile vs enemy (swept segment vs hull polygon)
        for m in self.missiles[:]:
            for i, e in enumerate(self.enemies):
                hit, hit_pt = e.ship.collision_segment(m.prev_pos, m.pos)
                if hit:
                    self.missiles.remove(m)
                    impact = pygame.Vector2(hit_pt)
                    alive = True
                    for _ in range(m.dmg):
                        alive = e.register_hit(impact)
                    if alive:
                        burst(self.particles, impact, 6, rng=self.rng)
                    else:
                        burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                        self._sfx("explosion")
                        self.enemies.pop(i)
                        if self.test_mode:
                            self._setup_test_scene()   # re-arm: fresh rock + target
                        else:
                            spawn_enemy(self.enemies, self.ship, rng=self.rng)
                    break

        # missile vs asteroid (detonate, same as a bullet)
        for m in self.missiles[:]:
            for i, a in enumerate(self.asteroids):
                if m.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    self.missiles.remove(m)
                    break

        # enemy bullet vs asteroid
        for b in self.enemy_bullets:
            if b.life <= 0:
                continue
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius, rng=self.rng)
                    self._sfx("small_explosion")
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = self.rng.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = self.rng.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick,
                                                           rng=self.rng))
                    self.asteroids.pop(i)
                    b.life = 0.0          # was: self.enemy_bullets.remove(b)
                    break

        # enemy bullet vs EACH player ship (Session 6.2a: shield surface if
        # up, else swept hull). A bullet is consumed by the first ship it
        # hits; one dead ship ends the game (game_over).
        if self.protect_timer <= 0:
            for b in self.enemy_bullets:
                if b.life <= 0:
                    continue
                for p in self.players:
                    if p.shield_on:
                        if p.shield_contains(b.pos):
                            b.life = 0.0
                            if not self._handle_ship_hit(p, b.pos):
                                break
                            break   # bullet consumed by this ship
                    else:
                        hit, hit_pt = p.collision_segment(b.prev_pos, b.pos)
                        if hit:
                            b.life = 0.0
                            if not self._handle_ship_hit(p,
                                                         pygame.Vector2(hit_pt)):
                                break
                            break   # bullet consumed by this ship
                if self.game_over:
                    break


        # EACH player ship vs enemy (ram) — SAT on both hull polygons
        if self.protect_timer <= 0 and not self.game_over:
            for p in self.players:
                for e in self.enemies:
                    if p.collision_overlaps_ship(e.ship):
                        if not self._handle_ship_hit(p, e.pos):
                            break
                if self.game_over:
                    break

        # EACH player ship vs asteroid — swept hull polygon vs circle
        if self.protect_timer <= 0 and not self.game_over:
            for p in self.players:
                for a in self.asteroids:
                    if p.collision_swept_overlaps_circle(a.pos, a.collision_radius):
                        if not self._handle_ship_hit(p, a.pos):
                            break
                if self.game_over:
                    break
        
        # enemy vs asteroid (rocks are hazards for everyone)
        for i, e in enumerate(self.enemies[:]):
            for a in self.asteroids:
                if e.pos.distance_to(a.pos) < a.collision_radius + e.collision_radius:
                    burst(self.particles, e.pos, 20, big=True, rng=self.rng)
                    self._sfx("explosion")
                    self.enemies.pop(i)
                    spawn_enemy(self.enemies, self.ship, rng=self.rng)   # instant respawn
                    break

    def draw(self, dt):
        screen = self.screen
        self.cam.update(dt, self.ship)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)
        for a in self.asteroids:
            a.draw(screen, self.cam)
        for e in self.enemies:
            e.draw(screen, self.cam)
            if TARGETING_ASSIST and self.ship.targeting_on:
                self._draw_lead(screen, e)
        for b in self.bullets:
            s = self.cam.to_screen(b.pos)
            pygame.draw.circle(screen, BULLET_COLOR, (int(s.x), int(s.y)), 3)
        

        for m in self.missiles:

            s = self.cam.to_screen(m.pos)
            fwd = m.vel.normalize()
            tail = self.cam.to_screen(m.pos - fwd * 14)
            # body: longer, thicker than a bullet
            pygame.draw.line(screen, _dim_color(MISSILE_COLOR, 0.7), tail, s, 3)
            # nose: bright tip
            pygame.draw.circle(screen, MISSILE_COLOR, (int(s.x), int(s.y)), 3)
            # exhaust: only during the boost ramp, flickering length
            if m.boost > 0:
                flick = 6 * (0.5 + 0.5 * math.sin(m.life * 40))
                flame = self.cam.to_screen(m.pos - fwd * (14 + flick))
                pygame.draw.line(screen, (255, 220, 120), tail, flame, 2)


        for local, target, d, vis_end, age, ttl in self.beams:
            fade = 1.0 - age / ttl
            c = tuple(int(ch * fade) for ch in LASER_COLOR)
            fwd = pygame.Vector2(math.cos(self.ship.rangle),
                                 math.sin(self.ship.rangle))
            right = pygame.Vector2(-fwd.y, fwd.x)
            start = self.ship.rpos + fwd * local[0] + right * local[1]
            if target in self.enemies:
                end = target.ship.shield_impact_point(
                    target.pos - d * target.collision_radius)
            else:
                end = vis_end
            pygame.draw.line(screen, c, self.cam.to_screen(start),
                             self.cam.to_screen(end), 2)


        for b in self.enemy_bullets:
            s = self.cam.to_screen(b.pos)
            pygame.draw.circle(screen, ENEMY_BULLET_COLOR, (int(s.x), int(s.y)), 3)
        for p in self.particles:
            p.draw(screen, self.cam)
        if not self.game_over:
            # Draw EVERY player ship (Session 6.2a), each with its own
            # shield ring (self.shields[i], sized by that hull's radius).
            for i, p in enumerate(self.players):
                p.sync_render(self.acc / STEP)
                p.draw(screen, self.cam, p.rpos, p.rangle)
                if self.protect_timer > 0:
                    sh = self.shields[i]
                    ssx, ssy = self.cam.to_screen(p.rpos)
                    screen.blit(sh, (ssx - sh.get_width() // 2,
                                     ssy - sh.get_height() // 2))
        draw_fog(screen, self.ship, self.cam, self.light_tex, self.fog_surf, self.light_surf, self._build_lights(self.ship))
        # Scan pulse above the fog: a bright ring sweeping through the dark
        if not self.game_over:
            self.ship._draw_scan_pulse(screen, self.cam, self.ship.rpos)
        self._draw_sensor_contacts()
        draw_hud(screen, self.font, self.enemies, self.ship)
        if self.game_over:
            draw_game_over(screen, self.big_font, self.font)

        if DEBUG_COLLISION:
            self.ship.draw_collision(screen, self.cam)
            for e in self.enemies:
                e.ship.draw_collision(screen, self.cam)

    # --- networking: remote-interpolation render (Session 5b.3) ---
    #
    # The remote peer does NOT run the sim for remote entities. It pushes
    # each received snapshot (stamped with the authoritative peer's sim_time)
    # into self.snap_buf, then renders INTERP_DELAY seconds in the PAST:
    # positions_at(sim_time - INTERP_DELAY) interpolates between the two
    # snapshots that bracket that time. The local player's own ship, bullets,
    # beams, particles, and fog are drawn by the normal draw() path — this
    # hook only draws the REMOTE entities (ship, enemies, asteroids) at their
    # interpolated positions, so the two paths never fight over a pixel.

    def remote_view(self, dt):
        """Draw the remote entities at their interpolated positions.

        Called by the remote peer's render loop INSTEAD of (or before) the
        local draw() for the shared entities. It reads only self.snap_buf and
        self.sim_time — it never mutates sim state, so it is safe to call
        every frame and it leaves the local path untouched.

        Returns the interpolated positions dict (or None when the buffer has
        not yet filled a window) so a caller can, e.g., skip the frame or
        draw a "waiting for snapshots" state.
        """
        screen = self.screen
        self.cam.update(dt, self.ship)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)

        pos = self.snap_buf.positions_at(self.sim_time - INTERP_DELAY)
        if pos is None:
            # Not enough snapshots yet to interpolate: draw nothing for the
            # remote entities (the caller may show a waiting state).
            return None

        # Asteroids: fill + edge polygon at the interpolated center. The
        # shape (verts) and spin come from the latest snapshot's data; only
        # the center is interpolated, which is what the buffer provides.
        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        for (x, y) in pos['enemies']:
            self._draw_remote_enemy(screen, x, y)
        # All player ships (Session 6.1): the buffer's 'ships' list, by index.
        # Session 6.8: each is drawn as its REAL hull at the interpolated
        # (pos, angle) — the buffer lerps the angle with lerp_angle —
        # instead of a dot.
        for i, (x, y, ang) in enumerate(pos['ships']):
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        draw_hud(screen, self.font, self.enemies, self.ship)
        return pos

    def push_snapshot(self, sim_time, snap):
        """The single seam where a received authoritative snapshot enters
        the remote peer: record it in the interpolation buffer, then feed
        the local ship's prediction ghost — seed on the first snapshot,
        reconcile (full-snap) on every one after. snap[0] is the tuple of
        per-player ship snapshots (Session 6.1); the ghost is the LOCAL
        player's ship, so it takes snap[0][self.local_index]. The network
        layer (and the 5b.4c test) call this."""
        self.snap_buf.push(sim_time, snap)
        local_s = snap[0][self.local_index]
        self.ghost.seed(local_s) if not self.ghost.seeded \
            else self.ghost.reconcile(local_s)

    def predicted_view(self, dt, keys):
        """Draw the frame with the LOCAL ship taken from the prediction
        ghost instead of the sim (client-side prediction, Session 5b.4b).

        Mirrors remote_view, except the local ship is the ghost: each frame
        the ghost is stepped with the local player's input at the fixed
        timestep, and the ghost's ship is drawn at its own (predicted)
        position. Remote entities (enemies, asteroids) still come from the
        interpolation buffer at sim_time - INTERP_DELAY, exactly as
        remote_view does. The buffer's 'ships' entry is drawn for every
        player EXCEPT the local one (Session 6.6: the remote ship in 2P;
        Session 6.8: as its real hull at the interpolated (pos, angle));
        the local ship's buffer entry is NOT drawn — it IS the local ship,
        now predicted.

        The HUD reads the ghost ship (the local player's power/shield/vel)
        and the buffer's enemy count — self.ship (players[0]) and
        self.enemies are the host's ship / the client's stale initial spawn
        on a client, so they would be wrong here.

        The ghost is PRESENTATION-ONLY: it is never fed back into the sim
        (update() still runs on self.ship). Returns the interpolated
        positions dict (or None while waiting for the first snapshot).
        """
        if not self.ghost.seeded:
            # No authoritative snapshot has arrived yet: nothing to predict
            # from. The caller may show a waiting state.
            return None

        inp = ShipInput.from_keys(keys)
        self.ghost.step(STEP, inp)

        screen = self.screen
        self.cam.update(dt, self.ghost.ship)
        screen.fill(BG)
        for x, y, r in self.stars:
            sx = (x - self.cam.pos.x * 0.2) % WIDTH
            sy = (y - self.cam.pos.y * 0.2) % HEIGHT
            pygame.draw.circle(screen, STAR_COLOR, (sx, sy), r)

        pos = self.snap_buf.positions_at(self.sim_time - INTERP_DELAY)
        if pos is None:
            # Not enough snapshots yet to interpolate the remote entities.
            return None

        for (x, y) in pos['asteroids']:
            self._draw_remote_rock(screen, x, y)
        for (x, y) in pos['enemies']:
            self._draw_remote_enemy(screen, x, y)
        # Remote ships (Session 6.6, hulls drawn in 6.8): every player
        # EXCEPT the local one, from the buffer by index (Session 6.1:
        # ships matched by index). The local ship is the ghost, drawn
        # below — so skip it here. Each is drawn as its REAL hull at the
        # interpolated (pos, angle) (the buffer lerps the angle with
        # lerp_angle, Session 6.8) — the remote peer knows the remote
        # hull from the join/welcome handshake, so self.players[i] IS
        # the remote ship's hull/loadout.
        for i, (x, y, ang) in enumerate(pos['ships']):
            if i == self.local_index:
                continue
            self.players[i].draw(screen, self.cam,
                                 pygame.Vector2(x, y), ang)

        # The LOCAL ship, from the ghost (its own flame_mags render).
        self.ghost.ship.draw(screen, self.cam,
                             self.ghost.ship.pos, self.ghost.ship.angle)

        # HUD: the LOCAL (ghost) ship's power/shield/velocity, and the real
        # (interpolated) enemy count from the buffer. self.ship is players[0]
        # (the HOST's ship on a client) and self.enemies is the client's stale
        # initial spawn (the client never runs the sim), so both would be
        # wrong here.
        draw_hud(screen, self.font, pos['enemies'], self.ghost.ship)
        return pos

    def _draw_remote_rock(self, screen, x, y):
        """Draw a remote asteroid at an interpolated world position.

        The buffer only interpolates the CENTER (x, y); the rock's shape and
        orientation are presentation detail. We draw a simple filled circle
        sized by the latest snapshot's rock — good enough for the remote
        view, and it keeps this hook free of per-rock state it does not own.
        """
        s = self.cam.to_screen(pygame.Vector2(x, y))
        # Latest rock snapshot for the size (first rock; the remote view is
        # a coarse stand-in, not a pixel-faithful replica).
        r = 20
        pygame.draw.circle(screen, ROCK_FILL, (int(s.x), int(s.y)), r)
        pygame.draw.circle(screen, ROCK_EDGE, (int(s.x), int(s.y)), r, 2)

    def _draw_remote_enemy(self, screen, x, y):
        """Draw a remote enemy at an interpolated world position (coarse)."""
        s = self.cam.to_screen(pygame.Vector2(x, y))
        pygame.draw.circle(screen, ENEMY_FILL, (int(s.x), int(s.y)), 10)
        pygame.draw.circle(screen, ENEMY_EDGE, (int(s.x), int(s.y)), 10, 2)

# --- test range: static scene for laser occlusion testing ---
    TEST_ROCK_POS   = (960, 340)   # 300 px ahead, dead center (large)
    TEST_TARGET_POS = (960, 40)    # 600 px ahead, behind the rock

    def _setup_test_scene(self):
        self.asteroids.clear()
        self.enemies.clear()

        # Dead-center large rock (the primary occluder)
        rock = Asteroid(pygame.Vector2(self.TEST_ROCK_POS), 'large')
        rock.vel = pygame.Vector2(0, 0)
        rock.spin = 0.0
        self.asteroids.append(rock)

        # Medium rock, slightly off-axis and further out.
        # Tests nearest-t: if both are on the beam line, the closer
        # (large) rock should win.
        rock2 = Asteroid(pygame.Vector2(990, 250), 'medium')
        rock2.vel = pygame.Vector2(0, 0)
        rock2.spin = 0.0
        self.asteroids.append(rock2)

        # Small rock, off-axis and closer in. Should NOT be hit by a
        # center-line beam (verifies the segment test doesn't false-positive).
        rock3 = Asteroid(pygame.Vector2(920, 480), 'small')
        rock3.vel = pygame.Vector2(0, 0)
        rock3.spin = 0.0
        self.asteroids.append(rock3)

        # Medium rock, well off-axis to the right. Should never be hit
        # unless you deliberately turn the ship.
        rock4 = Asteroid(pygame.Vector2(1060, 380), 'medium')
        rock4.vel = pygame.Vector2(0, 0)
        rock4.spin = 0.0
        self.asteroids.append(rock4)

        self.enemies.append(TestTarget(pygame.Vector2(self.TEST_TARGET_POS)))
