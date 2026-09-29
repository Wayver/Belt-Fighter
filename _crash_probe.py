"""Headless crash probe: run the single-player path (SimThread + render)
for a few seconds and print any traceback. Bypasses the menu."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import traceback
import pygame

from ship5.config import WIDTH, HEIGHT, FPS
from ship5.fog import make_light_texture
from ship5.game import Game
from ship5.hulls import PLAYER_HULLS, default_loadout
from ship5.sim_thread import SimThread
from ship5.sound import SoundBank
from ship5.intent import ShipInput

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas,menlo,monospace", 18)
big_font = pygame.font.SysFont("consolas,menlo,monospace", 40)
light_tex = make_light_texture()
fog_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
sfx = SoundBank()
try:
    sfx.init()
except Exception:
    pass

hull = PLAYER_HULLS[0]
game = Game(screen, font, big_font, light_tex, fog_surf, light_surf,
            hull=hull, loadout=default_loadout(hull),
            test_mode=True, seed=1234, sound=sfx)
sim_thread = SimThread(game)
sim_thread.start()

# Drive: hold W (thrust) + toggle V (sensor) + G (scan) periodically so the
# sensor-contact path runs.
import time
t0 = time.time()
frame = 0
try:
    while time.time() - t0 < 6.0:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        # Simulate key events for V/G/T/R occasionally.
        if frame % 120 == 0:
            for k in (pygame.K_v, pygame.K_g, pygame.K_t):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                                     key=k))
        if not game.handle_events(sim_thread.command):
            break
        keys = pygame.key.get_pressed()
        # Force W held by posting a key state? get_pressed reads SDL state;
        # with dummy driver it's all False. Instead publish a direct input.
        inp = ShipInput(thrust_fwd=1.0, fire=True, laser_fire=True,
                        missile_fire=(frame % 300 < 1))
        sim_thread.publish_input(inp)
        model = sim_thread.latest_model
        if model:
            game.draw(dt, model)
        pygame.display.flip()
        frame += 1
    print("RAN %d frames without crash" % frame)
except Exception:
    traceback.print_exc()
finally:
    sim_thread.stop()
    pygame.quit()