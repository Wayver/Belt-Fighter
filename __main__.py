"""Entry point: run with  python -m ship5"""
import pygame

from .config import WIDTH, HEIGHT, FPS
from .fog import make_light_texture
from .game import Game


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids — 4-thruster ship")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)

    light_tex = make_light_texture()
    fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    game = Game(screen, font, big_font, light_tex, fog_surf, light_surf)

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        running = game.handle_events()
        keys = pygame.key.get_pressed()
        game.update(dt, keys)
        game.draw(dt)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
