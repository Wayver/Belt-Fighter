"""Quantify the pixel difference between the OLD fog (light_surf round-trip
+ full-screen BLEND_RGBA_SUB) and the NEW fog (direct per-light SUB onto
fog_surf + cached/half-res textures). Headless.

Run: python -m ship5.test_fog_pixel
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import pygame

from .config import (WIDTH, HEIGHT, FOG_ALPHA, FOG_BASE_RADIUS,
                     FOG_FRONT_RADIUS, FOG_SIDE_RADIUS, FOG_FRONT_OFFSET)
from .fog import make_light_texture, LightSource, draw_fog as new_draw_fog


# --- the OLD fog, copied verbatim from the pre-optimization fog.py --------
def old_blit_light(light, tex, pos, angle, radius_x, radius_y):
    scaled = pygame.transform.scale(tex, (max(2, int(radius_x * 2)),
                                          max(2, int(radius_y * 2))))
    rotated = pygame.transform.rotate(scaled, -math.degrees(angle))
    light.blit(rotated, rotated.get_rect(center=(int(pos.x), int(pos.y))))


def old_blit_light_circle(light, tex, pos, radius, intensity=1.0):
    size = max(2, int(radius * 2))
    scaled = pygame.transform.scale(tex, (size, size))
    if intensity < 1.0:
        scaled.fill((255, 255, 255, int(255 * intensity)),
                    special_flags=pygame.BLEND_RGBA_MULT)
    light.blit(scaled, scaled.get_rect(center=(int(pos.x), int(pos.y))))


def old_draw_fog(screen, ship, cam, light_tex, fog_surf, light_surf,
                 lights=()):
    light_surf.fill((0, 0, 0, 0))
    fwd, _ = ship.axes()
    sp = cam.to_screen(ship.pos)
    old_blit_light(light_surf, light_tex, sp, ship.angle,
                   FOG_BASE_RADIUS, FOG_BASE_RADIUS)
    old_blit_light(light_surf, light_tex, sp + fwd * FOG_FRONT_OFFSET,
                   ship.angle, FOG_FRONT_RADIUS, FOG_SIDE_RADIUS)
    for ls in lights:
        old_blit_light_circle(light_surf, light_tex, cam.to_screen(ls.pos),
                              ls.radius, ls.intensity)
    fog_surf.fill((0, 0, 0, FOG_ALPHA))
    fog_surf.blit(light_surf, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    screen.blit(fog_surf, (0, 0))


class FakeShip:
    def __init__(self, pos, angle):
        self.pos = pos
        self.angle = angle
    def axes(self):
        return (pygame.Vector2(math.cos(self.angle), math.sin(self.angle)),
                pygame.Vector2(-math.sin(self.angle), math.cos(self.angle)))


class FakeCam:
    def __init__(self, off):
        self.off = off
    def to_screen(self, p):
        return pygame.Vector2(p.x - self.off.x, p.y - self.off.y)


def render_fog(draw_fn, ship, cam, tex, fog_surf, light_surf, lights,
               scene_color):
    screen = pygame.Surface((WIDTH, HEIGHT))
    screen.fill(scene_color)
    draw_fn(screen, ship, cam, tex, fog_surf, light_surf, lights)
    return pygame.image.tostring(screen, "RGB")


def compare(label, old, new):
    n = len(old) // 3
    maxd = 0
    sumd = 0
    ndiff = 0
    for i in range(0, len(old), 3):
        d = max(abs(old[i] - new[i]), abs(old[i+1] - new[i+1]),
                abs(old[i+2] - new[i+2]))
        if d > 0:
            ndiff += 1
            sumd += d
            if d > maxd:
                maxd = d
    pct = 100.0 * ndiff / n
    mean = sumd / n
    print("  %-28s maxΔ=%3d  meanΔ=%.4f  %%pixels-differ=%.3f%%"
          % (label, maxd, mean, pct))
    return maxd


def main():
    pygame.init()
    tex = make_light_texture()
    cam = FakeCam(pygame.Vector2(WIDTH // 2 - 100, HEIGHT // 2 - 50))

    cases = []
    # 1. ship only (base + front lobe), no point lights.
    cases.append(("ship only", FakeShip(pygame.Vector2(WIDTH // 2 + 100,
                                                       HEIGHT // 2 + 50),
                                        0.7), []))
    # 2. ship + a handful of point lights (bullets/missiles/reticle).
    lights = [LightSource(pygame.Vector2(WIDTH // 2 + 200, HEIGHT // 2),
                          30, 0.5),
              LightSource(pygame.Vector2(WIDTH // 2 + 260, HEIGHT // 2 + 40),
                          30, 0.5),
              LightSource(pygame.Vector2(WIDTH // 2 + 150, HEIGHT // 2 - 60),
                          50, 0.6),
              LightSource(pygame.Vector2(WIDTH // 2 + 300, HEIGHT // 2 + 90),
                          24, 0.4)]
    cases.append(("ship + 4 point lights",
                  FakeShip(pygame.Vector2(WIDTH // 2 + 100,
                                          HEIGHT // 2 + 50), 0.7), lights))
    # 3. many overlapping faint point lights (worst case for the SUB
    #    ordering difference).
    many = [LightSource(pygame.Vector2(WIDTH // 2 + 120 + (i % 6) * 30,
                                       HEIGHT // 2 + (i // 6) * 30),
                        40, 0.5) for i in range(24)]
    cases.append(("ship + 24 overlapping lights",
                  FakeShip(pygame.Vector2(WIDTH // 2 + 100,
                                          HEIGHT // 2 + 50), 0.7), many))
    # 4. a different ship angle (lobe rotated the other way).
    cases.append(("ship angle -1.2",
                  FakeShip(pygame.Vector2(WIDTH // 2 + 100,
                                          HEIGHT // 2 + 50), -1.2), lights))

    print("=== OLD vs NEW fog: per-pixel difference ===")
    worst = 0
    for label, ship, lights in cases:
        fog_old = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        light_old = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fog_new = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        light_new = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        old = render_fog(old_draw_fog, ship, cam, tex, fog_old, light_old,
                         lights, (90, 100, 120))
        new = render_fog(new_draw_fog, ship, cam, tex, fog_new, light_new,
                         lights, (90, 100, 120))
        worst = max(worst, compare(label, old, new))
    print()
    print("  worst-case maxΔ over all cases: %d / 255" % worst)
    print("  (Δ is the per-channel difference in the final composited scene)")


if __name__ == "__main__":
    main()