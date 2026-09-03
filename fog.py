"""Fog of war: a light bubble around the ship, with a forward lobe."""
import math

import pygame

from .config import (WIDTH, HEIGHT, FOG_ALPHA, FOG_BASE_RADIUS,
                     FOG_FRONT_RADIUS, FOG_SIDE_RADIUS, FOG_FRONT_OFFSET)


def make_light_texture(size=256):
    """Radial gradient: opaque at center, fading to transparent at the edge."""
    tex = pygame.Surface((size, size), pygame.SRCALPHA)
    half = size // 2
    for r in range(half, 0, -1):
        a = int(255 * (1.0 - r / half) ** 1.6)
        pygame.draw.circle(tex, (0, 0, 0, a), (half, half), r)
    return tex


def blit_light(light, tex, pos, angle, radius_x, radius_y):
    """Blit the light texture as an ellipse; radius_x is along the heading."""
    scaled = pygame.transform.scale(tex, (max(2, int(radius_x * 2)),
                                          max(2, int(radius_y * 2))))
    rotated = pygame.transform.rotate(scaled, -math.degrees(angle))
    light.blit(rotated, rotated.get_rect(center=(int(pos.x), int(pos.y))))

def draw_fog(screen, ship, cam, light_tex, fog_surf, light_surf):
    light_surf.fill((0, 0, 0, 0))
    fwd, _ = ship.axes()
    sp = cam.to_screen(ship.pos)
    # small all-around bubble at the ship
    blit_light(light_surf, light_tex, sp, ship.angle,
               FOG_BASE_RADIUS, FOG_BASE_RADIUS)
    # elongated lobe biased to the front
    blit_light(light_surf, light_tex, sp + fwd * FOG_FRONT_OFFSET,
               ship.angle, FOG_FRONT_RADIUS, FOG_SIDE_RADIUS)
    fog_surf.fill((0, 0, 0, FOG_ALPHA))
    fog_surf.blit(light_surf, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    screen.blit(fog_surf, (0, 0))
