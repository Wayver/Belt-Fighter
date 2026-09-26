import math
import pygame
from .config import (WIDTH, HEIGHT, LOOK_AHEAD, CAMERA_SMOOTHING,
                     CAMERA_LEAD_SMOOTHING)


class Camera:
    def __init__(self, start_pos):
        self.pos = pygame.Vector2(start_pos)
        self.lead = LOOK_AHEAD   # current lead (seconds of velocity), eased

    def update(self, dt, pos, vel, dampening):
        """Ease the camera toward the ship's look-ahead point.

        Takes the target as PLAIN values (pos, vel, dampening) rather than
        a live ship, so the render path can feed it from the RenderModel
        (Session 9.x M2b) without touching live sim state. `pos`/`vel` are
        (x, y) tuples or Vector2s; `dampening` is the ship's stop flag."""
        pos = pygame.Vector2(pos)
        vel = pygame.Vector2(vel)
        # Dampener shift: while B is held, ease the lead down so the
        # target point itself moves smoothly and the camera glides back
        # toward center before the ship has fully stopped.
        target_lead = LOOK_AHEAD * (0.35 if dampening else 1.0)
        t = 1.0 - math.exp(-CAMERA_LEAD_SMOOTHING * dt)
        self.lead += (target_lead - self.lead) * t
        target = pos + vel * self.lead
        t = 1.0 - math.exp(-CAMERA_SMOOTHING * dt)
        self.pos += (target - self.pos) * t

    def to_screen(self, world_pos):
        return world_pos - self.pos + pygame.Vector2(WIDTH / 2, HEIGHT / 2)
