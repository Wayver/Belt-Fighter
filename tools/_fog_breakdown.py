"""Break down draw_fog's cost: fill vs SUB blits vs final blit vs transforms.
Throwaway.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math
import time

import pygame

from ship5.config import (WIDTH, HEIGHT, FOG_ALPHA, FOG_BASE_RADIUS,
                          FOG_FRONT_RADIUS, FOG_SIDE_RADIUS,
                          FOG_FRONT_OFFSET)
from ship5.fog import make_light_texture, _get_cache, _point_tex, LightSource


class FakeShip:
    def __init__(self, pos, angle):
        self.pos = pos
        self.angle = angle
    def axes(self):
        return (pygame.Vector2(math.cos(self.angle), math.sin(self.angle)),
                pygame.Vector2(-math.sin(self.angle), math.cos(self.angle)))


class FakeCam:
    def __init__(self, pos):
        self.pos = pos
    def to_screen(self, wp):
        return wp - self.pos + pygame.Vector2(WIDTH / 2, HEIGHT / 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    light_tex = make_light_texture()
    ship = FakeShip(pygame.Vector2(WIDTH / 2, HEIGHT / 2), 0.7)
    cam = FakeCam(pygame.Vector2(WIDTH / 2, HEIGHT / 2))
    lights = [LightSource(pygame.Vector2(200 + i * 90, 150 + (i % 3) * 120),
                          30 + (i % 4) * 20, 0.5) for i in range(10)]
    fog = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    c = _get_cache(light_tex)
    fwd, _ = ship.axes()
    sp = cam.to_screen(ship.pos)
    fp = sp + fwd * FOG_FRONT_OFFSET
    rot = pygame.transform.rotate(c["front_half"], -math.degrees(ship.angle))
    front = pygame.transform.scale(rot, (rot.get_width() * 2, rot.get_height() * 2))
    ptexs = []
    for ls in lights:
        size = max(2, int(ls.radius * 2))
        ptexs.append((_point_tex(light_tex, size, ls.intensity),
                      cam.to_screen(ls.pos)))

    def bench(label, fn, frames=200):
        fn()
        t0 = time.perf_counter()
        for _ in range(frames):
            fn()
        t1 = time.perf_counter()
        print("  %-34s %.2f ms" % (label, (t1 - t0) / frames * 1000))

    print("=== draw_fog breakdown (10 point lights) ===")
    bench("fog_surf.fill", lambda: fog.fill((0, 0, 0, FOG_ALPHA)))
    bench("base SUB blit (320x320)",
          lambda: fog.blit(c["base"], c["base"].get_rect(center=(int(sp.x), int(sp.y))),
                           special_flags=pygame.BLEND_RGBA_SUB))
    bench("front SUB blit (860x400)",
          lambda: fog.blit(front, front.get_rect(center=(int(fp.x), int(fp.y))),
                           special_flags=pygame.BLEND_RGBA_SUB))
    def sub_all():
        for t, p in ptexs:
            fog.blit(t, t.get_rect(center=(int(p.x), int(p.y))),
                     special_flags=pygame.BLEND_RGBA_SUB)
    bench("10 point SUB blits", sub_all)
    bench("final screen.blit (full res)", lambda: screen.blit(fog, (0, 0)))
    bench("rotate+scale front lobe",
          lambda: pygame.transform.scale(
              pygame.transform.rotate(c["front_half"], -math.degrees(ship.angle)),
              (rot.get_width() * 2, rot.get_height() * 2)))

    # What if the fog surface were 32-bit opaque (no alpha channel)?
    fog32 = pygame.Surface((WIDTH, HEIGHT))
    bench("opaque fog_surf.fill (32-bit)", lambda: fog32.fill((0, 0, 0)))
    bench("opaque screen.blit (32-bit)", lambda: screen.blit(fog32, (0, 0)))

    # And a 16-bit SRCALPHA surface?
    fog16 = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA, 16)
    bench("16-bit SRCALPHA fog_surf.fill", lambda: fog16.fill((0, 0, 0, FOG_ALPHA)))
    bench("16-bit SRCALPHA screen.blit", lambda: screen.blit(fog16, (0, 0)))


if __name__ == "__main__":
    main()