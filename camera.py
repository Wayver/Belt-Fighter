import math
import pygame
from .config import (WIDTH, HEIGHT, LOOK_AHEAD, CAMERA_SMOOTHING,
                     CAMERA_LEAD_SMOOTHING)


class Camera:
    def __init__(self, start_pos):
        self.pos = pygame.Vector2(start_pos)
        self.lead = LOOK_AHEAD   # current lead (seconds of velocity), eased

    def update(self, dt, ship):
        # Dampener shift: while B is held, ease the lead down so the
        # target point itself moves smoothly and the camera glides back
        # toward center before the ship has fully stopped.
        target_lead = LOOK_AHEAD * (0.35 if ship.dampening else 1.0)
        t = 1.0 - math.exp(-CAMERA_LEAD_SMOOTHING * dt)
        self.lead += (target_lead - self.lead) * t
        target = ship.pos + ship.vel * self.lead
        t = 1.0 - math.exp(-CAMERA_SMOOTHING * dt)
        self.pos += (target - self.pos) * t

    def to_screen(self, world_pos):
        return world_pos - self.pos + pygame.Vector2(WIDTH / 2, HEIGHT / 2)
