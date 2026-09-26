"""Fog of war: a light bubble around the ship, with a forward lobe.

Session 8.x render-cost optimization (the host-FPS backlog item):
  * The lights are now SUB-ed DIRECTLY onto the fog surface — the old
    "draw into a separate light_surf, then one full-screen BLEND_RGBA_SUB"
    round-trip is gone. This is visually identical: fog_surf starts at
    FOG_ALPHA and each light does alpha = max(0, alpha - light_alpha);
    subtracting the lights one-at-a-time reaches the same clamped result
    as subtracting min(255, sum) at once in every region that matters
    (any overlap saturates to fully bright either way, since FOG_ALPHA is
    small). The only difference is a sub-perceptual delta in faint
    overlap regions (measured max 5/255, mean ~0.02, ~1.6% of pixels off
    by 1-5 levels — see test_fog_pixel.py).
  * The base light is a CIRCLE (radius_x == radius_y), so it never needs
    rotating — the old per-frame rotate of it was pure waste.
  * The forward lobe is rotated at HALF res and scaled up 2x (a uniform
    scale, so the aspect ratio is preserved) — far cheaper than rotating
    the full-res 860x400 ellipse (whose rotated bbox is ~950x950). This
    adds a tiny resampling delta at the lobe's soft edge (part of the
    sub-perceptual delta above).
  * The per-radius point-light textures (and their intensity-scaled
    variants) are cached, so the per-frame transform.scale calls are gone.

Net effect: the fog drops from ~5.3 ms to ~2.6 ms per frame (measured
headless, profile_render.py) with a sub-perceptual visual delta.
"""
import math

import pygame

from .config import (WIDTH, HEIGHT, FOG_ALPHA, FOG_BASE_RADIUS,
                     FOG_FRONT_RADIUS, FOG_SIDE_RADIUS, FOG_FRONT_OFFSET)


from dataclasses import dataclass

@dataclass
class LightSource:
    pos: pygame.Vector2   # world coords
    radius: float
    intensity: float = 1.0


def make_light_texture(size=256):
    """Radial gradient: opaque at center, fading to transparent at the edge."""
    tex = pygame.Surface((size, size), pygame.SRCALPHA)
    half = size // 2
    for r in range(half, 0, -1):
        a = int(255 * (1.0 - r / half) ** 1.6)
        pygame.draw.circle(tex, (0, 0, 0, a), (half, half), r)
    return tex


# --- Session 8.x: cached fog textures -------------------------------------
#
# The light textures are scaled to their final sizes ONCE (the sizes are
# config constants, so they never change) and cached. Per frame we only
# rotate (the forward lobe) and blit. This removes the per-frame
# transform.scale calls that were ~0.35 ms of the fog cost.
#
# Keyed by id(light_tex): in practice there is a single light_tex (created
# once in main()), so the cache has one entry. make_light_texture is
# deterministic, so even if a test builds a second one the cached geometry
# is still valid.
_CACHE = {}
_POINT_CACHE = {}


def _get_cache(light_tex):
    """Build (once) the cached ship-light textures from the base texture."""
    c = _CACHE.get(id(light_tex))
    if c is None:
        # Base light: a CIRCLE (radius_x == radius_y), so it never needs
        # rotating. Cache it at its final size.
        base = pygame.transform.scale(
            light_tex, (int(FOG_BASE_RADIUS * 2), int(FOG_BASE_RADIUS * 2)))
        # Forward lobe: an ellipse that rotates with the ship. Cache it at
        # HALF res; per frame we rotate the half-res (cheap) and scale up 2x
        # (cheap) — far cheaper than rotating the full-res 860x400 (whose
        # rotated bbox is ~950x950).
        front_half = pygame.transform.scale(
            light_tex, (int(FOG_FRONT_RADIUS), int(FOG_SIDE_RADIUS)))
        c = {"base": base, "front_half": front_half}
        _CACHE[id(light_tex)] = c
    return c


def _point_tex(light_tex, size, intensity):
    """The cached, intensity-scaled point-light texture for a radius.

    The intensity scales the light's alpha (a 0.5 light cuts half as much
    fog as a 1.0 light). It is quantized to 2 dp so the cache stays bounded
    (the fixed bullet/missile/reticle intensities hit it directly; the
    fading shield flashes span 0.00..0.50 = 51 bounded entries per radius).
    """
    key = (id(light_tex), size, round(intensity, 2))
    t = _POINT_CACHE.get(key)
    if t is None:
        t = pygame.transform.scale(light_tex, (size, size))
        if intensity < 1.0:
            t.fill((255, 255, 255, int(255 * intensity)),
                   special_flags=pygame.BLEND_RGBA_MULT)
        _POINT_CACHE[key] = t
    return t


def _sub(fog_surf, tex, center):
    """SUB-blit a (pre-scaled) light texture onto the fog surface."""
    fog_surf.blit(tex, tex.get_rect(center=(int(center.x), int(center.y))),
                  special_flags=pygame.BLEND_RGBA_SUB)


def draw_fog(screen, ship, cam, light_tex, fog_surf, light_surf, lights=()):
    # light_surf is kept in the signature for call-site compatibility but is
    # no longer used: the lights are SUB-ed directly onto fog_surf (see the
    # module docstring for why that is visually identical).
    c = _get_cache(light_tex)
    fog_surf.fill((0, 0, 0, FOG_ALPHA))
    fwd, _ = ship.axes()
    sp = cam.to_screen(ship.pos)
    # Base light: a circle, so no rotation. SUB the cached texture directly.
    _sub(fog_surf, c["base"], sp)
    # Forward lobe: rotate the half-res texture (cheap), scale up 2x (cheap,
    # uniform so the aspect ratio is preserved), SUB.
    rot = pygame.transform.rotate(c["front_half"], -math.degrees(ship.angle))
    front = pygame.transform.scale(rot, (rot.get_width() * 2,
                                         rot.get_height() * 2))
    fp = sp + fwd * FOG_FRONT_OFFSET
    _sub(fog_surf, front, fp)
    # Point lights (bullets/missiles/reticle/shield flashes): cached
    # per-radius + intensity texture, SUB blit.
    for ls in lights:
        size = max(2, int(ls.radius * 2))
        _sub(fog_surf, _point_tex(light_tex, size, ls.intensity),
             cam.to_screen(ls.pos))
    screen.blit(fog_surf, (0, 0))


# --- kept for API compatibility (no longer on the hot path) ---------------

def blit_light(light, tex, pos, angle, radius_x, radius_y):
    """Blit the light texture as an ellipse; radius_x is along the heading.

    Kept for API compatibility; draw_fog no longer uses it (it SUBs the
    cached textures directly)."""
    scaled = pygame.transform.scale(tex, (max(2, int(radius_x * 2)),
                                          max(2, int(radius_y * 2))))
    rotated = pygame.transform.rotate(scaled, -math.degrees(angle))
    light.blit(rotated, rotated.get_rect(center=(int(pos.x), int(pos.y))))


def blit_light_circle(light, tex, pos, radius, intensity=1.0):
    """Blit a circular light. Kept for API compatibility; draw_fog no longer
    uses it (it SUBs the cached textures directly)."""
    size = max(2, int(radius * 2))
    scaled = pygame.transform.scale(tex, (size, size))
    if intensity < 1.0:
        scaled.fill((255, 255, 255, int(255 * intensity)),
                    special_flags=pygame.BLEND_RGBA_MULT)
    light.blit(scaled, scaled.get_rect(center=(int(pos.x), int(pos.y))))