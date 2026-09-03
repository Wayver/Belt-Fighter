"""Player input, sampled once per frame at the edge.

This is the future network packet: everything the sim needs to know
about a player's actions, and nothing pygame-specific. `ShipInput` is
flat and serializable so it can be sent over the wire and replayed
deterministically.

Thrust is recorded per key, not netted: holding W and S together is a
real (wasteful) action — both thrusters fire, the forces cancel in the
physics, and the power draw reflects it. Netting here would hide the
inefficiency from the sim.
"""
from dataclasses import dataclass

import pygame


@dataclass
class ShipInput:
    turn: float = 0.0            # -1 (Q), 0, +1 (E)
    thrust_fwd: float = 0.0      # W, 0 or 1
    thrust_rev: float = 0.0      # S, 0 or 1
    thrust_left: float = 0.0     # A, 0 or 1
    thrust_right: float = 0.0    # D, 0 or 1
    stop: bool = False           # B: auto-stop guidance
    fire: bool = False           # SPACE

    @classmethod
    def from_keys(cls, keys):
        turn = 0.0
        if keys[pygame.K_q]:
            turn -= 1.0
        if keys[pygame.K_e]:
            turn += 1.0
        return cls(turn=turn,
                   thrust_fwd=1.0 if keys[pygame.K_w] else 0.0,
                   thrust_rev=1.0 if keys[pygame.K_s] else 0.0,
                   thrust_left=1.0 if keys[pygame.K_a] else 0.0,
                   thrust_right=1.0 if keys[pygame.K_d] else 0.0,
                   stop=keys[pygame.K_b],
                   fire=keys[pygame.K_SPACE])
