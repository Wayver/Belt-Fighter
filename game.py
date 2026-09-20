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

STEP = 1 / 60   # fixed simulation timestep


class Game:
    def __init__(self, screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=None, loadout=None, test_mode=False, seed=None, sound=None):
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

        self.ship = Ship(hull=hull, loadout=loadout)
        r = self.ship.collision_radius
        self.shield = pygame.Surface((int(r * 2 + 10), int(r * 2 + 10)),
                                     pygame.SRCALPHA)
        pygame.draw.circle(self.shield, (120, 200, 255, 100),
                           (self.shield.get_width() // 2, self.shield.get_height() // 2),
                           r + 5, 2)

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
        self.reset()

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
        self.ship.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.ship.vel = pygame.Vector2(0, 0)
        self.ship.angle = -math.pi / 2
        self.ship.prev_pos = self.ship.pos.copy()
        self.ship.prev_angle = self.ship.angle
        self.ship.reset_shield()
        self.ship.targeting_on = False
        self.ship.tracked = 0
        self.bullets.clear()
        self.enemy_bullets.clear()
        self.beams.clear()
        self.missiles.clear()
        self.ship.reset_missiles()
        self.ship.reset_lasers()
        self.ship.reset_sensors()
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
        update_field(self.asteroids, [self.ship.pos], 0, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)
        spawn_enemy(self.enemies, self.ship, rng=self.rng)

    # --- networking: whole-sim snapshot (see Ship.snapshot for the
    # ship-level classification) ---
    #
    # SYNCED (serialized, must be identical on both peers):
    #   ship                 player Ship (21-field tuple, see ship.py)
    #   enemies              per enemy: variant tag + e.snapshot()
    #                        tag = 'mote' (MoteEnemy) | 'test' (TestTarget)
    #                        | 'ai' (AIEnemy). Order matters: MoteEnemy and
    #                        TestTarget are both AIEnemy subclasses, so check
    #                        them first.
    #   bullets, enemy_bullets, missiles
    #                        per projectile: pos/vel/owner/life (+ boost for
    #                        missiles). Missile target is stored as the
    #                        enemy's ship id; apply_snapshot re-wires it.
    #   asteroids            per rock: pos/vel/size/angle/spin/verts
    #   rng state            the sim's single rng — the strongest check;
    #                        without it the peers diverge on the next roll
    #   game_over, protect_timer
    #   AIEnemy._next_id     class-level id counter (not instance state —
    #                        a naive snapshot misses it)
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
            self.ship.snapshot(),
            tuple((self._enemy_tag(e), e.snapshot()) for e in self.enemies),
            tuple(b.snapshot() for b in self.bullets),
            tuple(b.snapshot() for b in self.enemy_bullets),
            tuple(m.snapshot() for m in self.missiles),
            tuple(a.snapshot() for a in self.asteroids),
            self.rng.getstate(),
            self.game_over,
            self.protect_timer,
            AIEnemy._next_id,
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
        (ship_s, enemies_s, bullets_s, enemy_bullets_s, missiles_s,
         asteroids_s, rng_state, game_over, protect_timer, next_id) = s

        self.ship.apply_snapshot(ship_s)

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

        # Asteroids: construct minimally; apply_snapshot rebuilds
        # radius/collision_radius from ROCK_SIZES[size].
        self.asteroids.clear()
        for a_s in asteroids_s:
            px, py, _vx, _vy, size, _angle, _spin, _verts = a_s
            a = Asteroid(pygame.Vector2(px, py), size)
            a.apply_snapshot(a_s)
            self.asteroids.append(a)

        self.rng.setstate(rng_state)
        self.game_over = game_over
        self.protect_timer = protect_timer
        # AFTER constructing enemies: their construction already consumed
        # ids from the class counter, so restore the snapshotted value now.
        AIEnemy._next_id = next_id
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
        if not self.game_over:
            # Targeting sensor: count enemies in range before the ship
            # steps, so _allocate() sees this tick's tracked count.
            self.ship.tracked = sum(
                1 for e in self.enemies
                if e.pos.distance_to(self.ship.pos) <= TARGETING_RANGE)
            


            # Laser target: nearest enemy in range, set before the ship
            # steps so _update_lasers() sees it this tick.
            self.ship.laser_target = self._pick_laser_target()
            
            # Missile target: same pattern, for the lock state machine.
            self.ship.missile_target = self._pick_missile_target()

            # Sensor contacts: passive + active reveals, same pattern.
            self._update_contacts()


            if inp.fire and len(self.bullets) >= MAX_BULLETS:
                inp = replace(inp, fire=False)   # world cap: no room, no shot
            #shots, beams, missiles = self.ship.update(dt, inp)
            
            if inp.missile_fire and len(self.missiles) >= MAX_MISSILES:
                inp = replace(inp, missile_fire=False)
            shots, beams, missiles = self.ship.update(dt, inp)

            for shot in shots:
                self.bullets.append(Bullet(shot.pos, shot.vel, owner=shot.owner))

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
                update_field(self.asteroids, [self.ship.pos], dt,
                             rng=self.rng)

        # Engine loop: on while the ship is alive and any fitted thruster
        # has resolved force (demand * allocation > 0). Death or a power
        # brownout cuts the sound; set_thruster(False) is a cheap no-op
        # when the loop is already off.
        self._sfx_thruster(not self.game_over
                           and any(t.force > 0.0 for t in self.ship.thrusters))

        enemy_fired = False
        for e in self.enemies:
            shots = e.update(dt, self.ship, self.asteroids)
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

    
    def _build_lights(self):
        """Whitelist of things that shine through the fog of war.
        Returns a list of LightSource in world coords, built fresh each frame."""
        lights = []
        ship = self.ship


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


    def _update_contacts(self):
        """Build the ship's contact list from the fitted sensor.

        Passive: enemies in sensor_range whose active power draw
        (power_used - idle) crosses the signature threshold. Active:
        while a ping's reveal lasts, everything in scan_range is
        confirmed. A confirmed contact replaces a passive one.
        """
        ship = self.ship
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

    def _pick_laser_target(self):
        """Nearest enemy within the max laser range; None if no laser fitted."""
        max_range = max((w.comp.laser_range for w in self.ship.weapons
                         if w.comp.laser_range > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(self.ship.pos)
            if d <= best_d:
                best, best_d = e, d
        return best


    def _pick_missile_target(self):
        """Nearest enemy within the max missile lock range; None if no
        missile fitted."""
        max_range = max((w.comp.missile_lock_range for w in self.ship.weapons
                         if w.comp.missile_speed > 0), default=0.0)
        if max_range <= 0:
            return None
        best, best_d = None, max_range
        for e in self.enemies:
            d = e.pos.distance_to(self.ship.pos)
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


    def _handle_ship_hit(self, source_pos):
        """Handle a hit on the ship. Returns True if the ship survives."""
        if self.ship.register_hit():
            impact = self.ship.shield_impact_point(source_pos)
            shield_burst(self.particles, impact, rng=self.rng)
            # Throttled: 3 enemies can hit ~20/s; the ping is 0.3 s long.
            self._sfx_throttled("shield_hit", SFX_SHIELD_HIT_MIN_INTERVAL)
            return True
        self.game_over = True
        burst(self.particles, self.ship.pos, 30, big=True, rng=self.rng)
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

        # enemy bullet vs ship (shield surface if up, else swept hull)
        if self.protect_timer <= 0:
            for b in self.enemy_bullets:
                if b.life <= 0:
                    continue
                if self.ship.shield_on:
                    if self.ship.shield_contains(b.pos):
                        b.life = 0.0      # was: self.enemy_bullets.remove(b)
                        if not self._handle_ship_hit(b.pos):
                            break
                else:
                    hit, hit_pt = self.ship.collision_segment(b.prev_pos, b.pos)
                    if hit:
                        b.life = 0.0      # was: self.enemy_bullets.remove(b)
                        if not self._handle_ship_hit(pygame.Vector2(hit_pt)):
                            break


        # ship vs enemy (ram) — SAT on both hull polygons
        if self.protect_timer <= 0 and not self.game_over:
            for e in self.enemies:
                if self.ship.collision_overlaps_ship(e.ship):
                    if not self._handle_ship_hit(e.pos):
                        break

        # ship vs asteroid — swept hull polygon vs circle (speed-independent)
        if self.protect_timer <= 0 and not self.game_over:
            for a in self.asteroids:
                if self.ship.collision_swept_overlaps_circle(a.pos, a.collision_radius):
                    if not self._handle_ship_hit(a.pos):
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
            self.ship.sync_render(self.acc / STEP)
            self.ship.draw(screen, self.cam, self.ship.rpos, self.ship.rangle)
            if self.protect_timer > 0:
                ssx, ssy = self.cam.to_screen(self.ship.rpos)
                screen.blit(self.shield, (ssx - self.shield.get_width() // 2,
                                      ssy - self.shield.get_height() // 2))
        draw_fog(screen, self.ship, self.cam, self.light_tex, self.fog_surf, self.light_surf, self._build_lights())
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
