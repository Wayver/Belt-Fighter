"""Game state: entities, collisions, waves, and per-frame update/draw.

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
                    BULLET_SPEED, ENEMY_SCORE,
                    ROCK_SPLIT, ROCK_SIZES, BG, STAR_COLOR,
                    BULLET_COLOR, ENEMY_BULLET_COLOR, WAVE_INTERVAL,
                    TARGETING_ASSIST, TARGETING_COLOR, TARGETING_HORIZON,
                    TARGETING_STEPS, TARGETING_RANGE,
                    TARGETING_USE_ACCEL, TARGETING_MAX_LEAD,
                    TARGETING_COLOR_GREEN, TARGETING_ALIGN_TOL, LASER_COLOR,
                    BEAM_IMPACT_SPREAD)
from .ship import Ship, wrapped_delta
from .intent import ShipInput
from .asteroid import Asteroid
from .bullets import Bullet, EnemyBullet
from .particles import burst, shield_burst
from .spawning import spawn_enemy, make_stars, update_field
from .fog import draw_fog
from .hud import draw_hud, draw_game_over
from .camera import Camera

STEP = 1 / 60   # fixed simulation timestep


class Game:
    def __init__(self, screen, font, big_font, light_tex, fog_surf, light_surf,
                hull=None, loadout=None):
        self.screen = screen
        self.font = font
        self.big_font = big_font
        self.light_tex = light_tex
        self.fog_surf = fog_surf
        self.light_surf = light_surf
        self.stars = make_stars()

        self.ship = Ship(hull=hull, loadout=loadout)
        r = self.ship.collision_radius
        self.shield = pygame.Surface((int(r * 2 + 10), int(r * 2 + 10)),
                                     pygame.SRCALPHA)
        pygame.draw.circle(self.shield, (120, 200, 255, 100),
                           (self.shield.get_width() // 2, self.shield.get_height() // 2),
                           r + 5, 2)

        self.cam = Camera(self.ship.pos)
        self.bullets = []
        self.beams   = []
        self.enemy_bullets = []
        self.particles = []
        self.asteroids = []
        self.enemies = []
        self.score = 0
        self.wave = 1
        self.wave_timer = 0.0
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0

        self.reset()

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
        self.ship.reset_lasers()
        self.particles.clear()
        self.asteroids.clear()
        self.enemies.clear()
        self.score = 0
        self.wave = 1
        self.wave_timer = 0.0
        self.game_over = False
        self.protect_timer = SPAWN_PROTECT
        self.acc = 0.0
        self.cam.pos = self.ship.pos.copy()
        update_field(self.asteroids, [self.ship.pos], self.wave, 0)
        spawn_enemy(self.enemies, self.ship)
        spawn_enemy(self.enemies, self.ship)
        spawn_enemy(self.enemies, self.ship)

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
            if inp.fire and len(self.bullets) >= MAX_BULLETS:
                inp = replace(inp, fire=False)   # world cap: no room, no shot
            shots, beams = self.ship.update(dt, inp)
            
            for shot in shots:
                self.bullets.append(Bullet(shot.pos, shot.vel, owner=shot.owner))
            
            for beam in beams:
                self._resolve_beam(beam)
            
            self.protect_timer -= dt
            self.wave_timer += dt
            if self.wave_timer >= WAVE_INTERVAL:
                self.wave_timer = 0.0
                self.wave += 1
            update_field(self.asteroids, [self.ship.pos], self.wave, dt)

        for e in self.enemies:
            shots, _ = e.update(dt, self.ship, self.asteroids)
            for shot in shots:
                self.enemy_bullets.append(EnemyBullet(shot.pos, shot.vel, owner=shot.owner))

        for b in self.bullets:
            b.update(dt)
        for b in self.enemy_bullets:
            b.update(dt)
        for a in self.asteroids:
            a.update(dt)
        for p in self.particles:
            p.update(dt)

        # cull expired projectiles and particles
        self.bullets = [b for b in self.bullets if b.life > 0]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.life > 0]
        self.particles = [p for p in self.particles if p.life > 0]

        # collisions after movement, so this tick's motion counts
        self._collisions()

        # age the beam visuals
        for b in self.beams:
            b[4] += dt
        self.beams = [b for b in self.beams if b[4] < b[5]]

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

    def _resolve_beam(self, beam):
        """Hitscan: hit the first enemy near the beam's end point.

        The beam *line* is drawn to the point on the target's shield/hull
        facing the muzzle, so beams from different muzzles land at different
        spots. Each damage point additionally jitters its shield flash.
        """
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
                    ang = base_ang + random.uniform(-BEAM_IMPACT_SPREAD,
                                                BEAM_IMPACT_SPREAD)
                    d2 = pygame.Vector2(math.cos(ang), math.sin(ang))
                    impact = e.pos - d2 * e.collision_radius
                    if not e.register_hit(impact):
                        self.score += ENEMY_SCORE
                        burst(self.particles, e.pos, 20, big=True)
                        self.enemies.pop(i)
                        spawn_enemy(self.enemies, self.ship)
                        break
                    shield_burst(self.particles,
                             e.ship.shield_impact_point(impact))
                self.beams.append([beam.local_start, e, d,  vis_end, 0.0, 0.15])
                return


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
            shield_burst(self.particles, impact)
            return True
        self.game_over = True
        burst(self.particles, self.ship.pos, 30, big=True)
        return False

    def _collisions(self):
        # player bullet vs enemy
        for b in self.bullets[:]:
            for i, e in enumerate(self.enemies):
                if b.pos.distance_to(e.pos) < e.collision_radius + 4:
                    self.bullets.remove(b)
                    if e.register_hit(b.pos):
                        burst(self.particles, b.pos, 6)
                    else:
                        self.score += ENEMY_SCORE
                        burst(self.particles, e.pos, 20, big=True)
                        self.enemies.pop(i)
                        spawn_enemy(self.enemies, self.ship)
                    break

        # bullet vs asteroid
        for b in self.bullets[:]:
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    self.score += a.score
                    burst(self.particles, a.pos, a.radius)
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = random.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = random.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick))
                    self.asteroids.pop(i)
                    self.bullets.remove(b)
                    break

        # enemy bullet vs asteroid
        for b in self.enemy_bullets[:]:
            for i, a in enumerate(self.asteroids):
                if b.pos.distance_to(a.pos) < a.collision_radius:
                    burst(self.particles, a.pos, a.radius)
                    child_size = ROCK_SPLIT[a.size]
                    if child_size:
                        for _ in range(2):
                            ks = random.uniform(*ROCK_SIZES[child_size]['speed'])
                            ka = random.uniform(0, 2 * math.pi)
                            kick = pygame.Vector2(math.cos(ka) * ks, math.sin(ka) * ks)
                            self.asteroids.append(Asteroid(a.pos, child_size,
                                                           vel=a.vel * 0.5 + kick))
                    self.asteroids.pop(i)
                    self.enemy_bullets.remove(b)
                    break

        # enemy bullet vs ship (shield surface if up, else hull)
        if self.protect_timer <= 0:
            for b in self.enemy_bullets[:]:
                if self.ship.shield_on:
                    if self.ship.shield_contains(b.pos):
                        self.enemy_bullets.remove(b)
                        if not self._handle_ship_hit(b.pos):
                            break
                elif b.pos.distance_to(self.ship.pos) < self.ship.collision_radius + 4:
                    self.enemy_bullets.remove(b)
                    if not self._handle_ship_hit(b.pos):
                        break

        # ship vs enemy (ram)
        if self.protect_timer <= 0 and not self.game_over:
            for e in self.enemies:
                if self.ship.pos.distance_to(e.pos) < e.collision_radius + self.ship.collision_radius:
                    if not self._handle_ship_hit(e.pos):
                        break

        # ship vs asteroid
        if self.protect_timer <= 0 and not self.game_over:
            for a in self.asteroids:
                if self.ship.pos.distance_to(a.pos) < a.collision_radius + self.ship.collision_radius:
                    if not self._handle_ship_hit(a.pos):
                        break

        # enemy vs asteroid (rocks are hazards for everyone)
        for i, e in enumerate(self.enemies[:]):
            for a in self.asteroids:
                if e.pos.distance_to(a.pos) < a.collision_radius + e.collision_radius:
                    burst(self.particles, e.pos, 20, big=True)
                    self.score += ENEMY_SCORE
                    self.enemies.pop(i)
                    spawn_enemy(self.enemies, self.ship)   # instant respawn
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
        draw_fog(screen, self.ship, self.cam, self.light_tex, self.fog_surf, self.light_surf)
        draw_hud(screen, self.font, self.score, self.wave, self.enemies, self.ship)
        if self.game_over:
            draw_game_over(screen, self.big_font, self.font, self.score)
