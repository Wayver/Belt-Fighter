"""Hull and component type definitions — pure data, no rendering.

A HullType is a polygon plus a set of named mount points (slots). A
ComponentType declares which slot types it fits and its stats. A Ship is
built from a HullType plus a loadout of components.

Orientation semantics (important):
- Slot.orientation is the direction of the FORCE on the ship (the way the
  part pushes it), in hull-local coords. The exhaust is the opposite.
- Slot.flame_dir is the visual exhaust direction, defaulting to
  -orientation. It exists so a thruster can push one way but show its
  flame another (e.g. RCS pods: the force is across the ship, but the
  flame is drawn outward so it stays visible past the hull edge).

Thrust values are expressed off config.SHIP_ACCEL so config.py remains
the tuning surface.
"""
from dataclasses import dataclass

from .config import (SHIP_ACCEL, FIRE_COOLDOWN, BULLET_SPEED,
                     ENEMY_FIRE_COOLDOWN, ENEMY_BULLET_SPEED)

@dataclass(frozen=True)
class Slot:
    """A mount point on a hull.

    name:        stable id, used to key flames and the loadout
    slot_type:   what kind of component fits ('thruster', 'weapon', ...)
    position:    (lx, ly) in hull-local coords (+x = nose, +y = starboard)
    orientation: (dx, dy) local direction of the FORCE on the ship
    flame_dir:   (dx, dy) visual exhaust direction; None = -orientation
    flame_key:   which flame_mags bucket this slot renders under ('' = none)
    flame_scale / flame_width: visual tuning for the flame
    """
    name: str
    slot_type: str
    position: tuple
    orientation: tuple = (1, 0)
    flame_dir: tuple = None
    flame_key: str = ''
    flame_scale: float = 0.6
    flame_width: int = 3

    def __post_init__(self):
        if self.flame_dir is None:
            object.__setattr__(self, 'flame_dir',
                               (-self.orientation[0], -self.orientation[1]))


@dataclass(frozen=True)
class HullType:
    """Static geometry + slots for one hull design."""
    id: str
    polygon: tuple                 # hull outline in local coords
    slots: tuple                   # tuple of Slot
    base_mass: float = 1.0
    collision_radius: float = 12.0
    nose: tuple = (18, 0)          # fallback muzzle / ram anchor
    cockpit: tuple = (8, 0)        # small cockpit dot


@dataclass(frozen=True)
class ComponentType:
    """A mountable part: fit + stats.

    `thrust` is the force this part applies at full demand. Power/compute:
    `power_idle` is drawn whenever fitted, `power_active` while active
    (scaled by demand), `compute_demand` while active. Generators provide
    `power_supply` / `compute_supply`. `priority` (lower = first) decides
    who keeps power when supply is short.
    """
    id: str
    name: str
    slot_types: tuple              # slot types this part fits
    mass: float = 0.0
    thrust: float = 0.0
    power_idle: float = 0.0
    power_active: float = 0.0
    compute_demand: float = 0.0
    power_supply: float = 0.0
    compute_supply: float = 0.0
    priority: int = 0
    power_hit: float = 0.0
    shield_max_charge: float = 0.0
    shield_recharge_rate: float = 0.0
    fire_cooldown: float = 0.0   # seconds between shots
    bullet_speed: float = 0.0    # px/s, expressed off config below


# --- The current ship, as data ---
# orientation = force direction on the ship (exhaust is the opposite).

FORWARD_S = Slot('forward_s', 'thruster', (-13, 8), (1, 0), flame_key='forward')
FORWARD_P = Slot('forward_p', 'thruster', (-13, -8), (1, 0), flame_key='forward')
REVERSE   = Slot('reverse', 'thruster', (16, 0), (-1, 0), flame_key='reverse')
# RCS: force is across the ship (port/starboard), but the flame is drawn
# outward (flame_dir) so it stays visible instead of ending under the hull.
RCS_LEFT  = Slot('to_left', 'thruster', (-10, -9.5), (0, -1),
                 flame_dir=(0, -1), flame_key='to_left',
                 flame_scale=0.5, flame_width=2)
RCS_RIGHT = Slot('to_right', 'thruster', (-10, 9.5), (0, 1),
                 flame_dir=(0, 1), flame_key='to_right',
                 flame_scale=0.5, flame_width=2)
GUN       = Slot('gun', 'weapon', (18, 0), (1, 0))
# Generators: plain mount points, no orientation/flame semantics.
REACTOR   = Slot('reactor', 'reactor', (-4, 0))
COMPUTER  = Slot('computer', 'computer', (2, 0))
SHIELD    = Slot('shield', 'shield', (0, 0))

DEFAULT_HULL = HullType(
    id='scout',
    polygon=(
        (18, 0),     # nose tip
        (14, 3.5),
        (8, 6.5),
        (0, 8),
        (-8, 8),
        (-12, 11),   # starboard pod, outer rear (sticks out)
        (-12, 5),    # starboard pod, inner rear
        (-9, 3),     # notch between engines
        (-9, -3),
        (-12, -5),   # port pod, inner rear
        (-12, -11),  # port pod, outer rear (sticks out)
        (-8, -8),
        (0, -8),
        (8, -6.5),
        (14, -3.5),
    ),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN,
           REACTOR, COMPUTER, SHIELD),
    base_mass=1.0,
    collision_radius=12.0,
    nose=(18, 0),
    cockpit=(8, 0),
)

# Component types for the default loadout.
# Two main engines sum to SHIP_ACCEL straight ahead; nose + RCS match the
# old single-vector magnitudes.
#
## Stock power/compute budget (deliberately generous — feel must not change):
#   power idle:   2*1 + 1 + 2*1 + 2 + 5 + 2 = 14
#   power steady: + 10 (shield) = 24 at rest
#   power max:    + 2*20 + 15 + 2*5 + 5 = 94  (of 100)  -> never brownouts
#   hit dump:     + 25 (transient, decays 50/s) -> brief brownout at full thrust
#
# Priority: RCS (1) is fine control near rocks, mains (2) propulsion,
# reverse (3) sheds first.
MAIN_ENGINE   = ComponentType('main_engine', 'Main Engine', ('thruster',),
                              mass=2.0, thrust=SHIP_ACCEL / 2,
                              power_idle=1.0, power_active=20.0, priority=2)
NOSE_THRUSTER = ComponentType('nose_thruster', 'Nose Thruster', ('thruster',),
                              mass=1.0, thrust=SHIP_ACCEL,
                              power_idle=1.0, power_active=15.0, priority=3)
RCS           = ComponentType('rcs', 'RCS', ('thruster',),
                              mass=0.5, thrust=SHIP_ACCEL,
                              power_idle=1.0, power_active=5.0,
                              compute_demand=5.0, priority=1)
GUN_TYPE      = ComponentType('gun', 'Pulse Gun', ('weapon',), mass=1.0,
                              power_idle=2.0, power_active=5.0,
                              fire_cooldown=FIRE_COOLDOWN,
                              bullet_speed=BULLET_SPEED, priority=2)
REACTOR_TYPE  = ComponentType('reactor', 'Reactor', ('reactor',),
                              mass=3.0, power_supply=100.0)
COMPUTER_TYPE = ComponentType('computer', 'Computer', ('computer',),
                              mass=2.0, compute_supply=50.0)
SHIELD_TYPE = ComponentType('shield', 'Shield', ('shield',),
                            mass=2.0,
                            power_idle=2.0, power_active=10.0,
                            power_hit=25.0,
                            shield_max_charge=3.0,
                            shield_recharge_rate=0.5)

E_SHIELD_TYPE = ComponentType('shield', 'Shield', ('shield',),
                             mass=2.0,
                             power_idle=2.0, power_active=10.0,
                             power_hit=45.0,
                             shield_max_charge=2.0,
                             shield_recharge_rate=0.5)


def default_loadout():
    """slot_name -> ComponentType, reproducing the current ship's fittings."""
    return {
        'forward_s': MAIN_ENGINE,
        'forward_p': MAIN_ENGINE,
        'reverse': NOSE_THRUSTER,
        'to_left': RCS,
        'to_right': RCS,
        'gun': GUN_TYPE,
        'reactor': REACTOR_TYPE,
        'computer': COMPUTER_TYPE,
        'shield'  : SHIELD_TYPE,
    }


# --- Enemy hull: slender dart body + forward wing gun pods ---
# Standard thruster slot names (forward_s/forward_p) so Ship._set_demands
# works unchanged. The wings carry the guns (gun_s/gun_p), not RCS.

E_FORWARD_S = Slot('forward_s', 'thruster', (-14, 2.5), (1, 0), flame_key='forward')
E_FORWARD_P = Slot('forward_p', 'thruster', (-14, -2.5), (1, 0), flame_key='forward')
# braking/strafing: same slot names the player uses, so Ship._set_demands
# and auto-stop work unchanged
E_REVERSE   = Slot('reverse', 'thruster', (24, 0), (-1, 0), flame_key='reverse')
E_RCS_L     = Slot('to_left', 'thruster', (-7, -8), (0, -1),
                   flame_dir=(0, -1), flame_key='to_left',
                   flame_scale=0.5, flame_width=2)
E_RCS_R     = Slot('to_right', 'thruster', (-7, 8), (0, 1),
                   flame_dir=(0, 1), flame_key='to_right',
                   flame_scale=0.5, flame_width=2)
E_GUN_S     = Slot('gun_s', 'weapon', (21, 11), (1, 0))
E_GUN_P     = Slot('gun_p', 'weapon', (21, -11), (1, 0))
E_REACTOR   = Slot('reactor', 'reactor', (-4, 0))
E_REACTOR_2 = Slot('reactor2', 'reactor', (-8, 0))
E_COMPUTER  = Slot('computer', 'computer', (2, 0))
E_SHIELD    = Slot('shield', 'shield', (0, 0))


ENEMY_HULL = HullType(
    id='interceptor',
    polygon=(
        (28, 0),     # nose tip (extended)
        (12, 3),     # fuselage shoulder (slender body)
        (10, 6),     # wing root leading edge (notch before the pod)
        (22, 9),     # starboard gun pod leading edge (extended)
        (22, 13),    # starboard gun pod front outer (extended)
        (4, 13),     # starboard wingtip (pod outer rear)
        (-0, 9),     # starboard wing trailing edge
        (-7, 7),    # starboard wingtip rear
        (-10, 3),    # fuselage rear corner (starboard)
        (-10, -3),   # fuselage rear corner (port)
        (-7, -7),   # port wingtip rear
        (-0, -9),    # port wing trailing edge
        (4, -13),    # port wingtip (pod outer rear)
        (22, -13),   # port gun pod front outer (extended)
        (22, -9),    # port gun pod leading edge (extended)
        (10, -6),    # port wing root leading edge
        (12, -3),    # fuselage shoulder (port)
    ),
    slots=(E_FORWARD_S, E_FORWARD_P, E_GUN_S, E_GUN_P, E_REACTOR, E_COMPUTER, E_SHIELD),
    base_mass=3.0,
    collision_radius=16.0,
    nose=(28, 0),
    cockpit=(10, 0),
)

# Enemy guns use the enemy's own config values (not the player's) — see notes.
ENEMY_GUN = ComponentType('enemy_gun', 'Enemy Gun', ('weapon',), mass=1.0,
                          power_idle=2.0, power_active=5.0,
                          fire_cooldown=ENEMY_FIRE_COOLDOWN,
                          bullet_speed=ENEMY_BULLET_SPEED, priority=2)


def enemy_loadout():
    return {
        'forward_s': MAIN_ENGINE,
        'forward_p': MAIN_ENGINE,
        'reverse': NOSE_THRUSTER,
        'to_left': RCS,
        'to_right': RCS,
        'gun_s': ENEMY_GUN,
        'gun_p': ENEMY_GUN,
        'reactor': REACTOR_TYPE,
        'reactor2': REACTOR_TYPE,
        'computer': COMPUTER_TYPE,
        'shield'  : E_SHIELD_TYPE,
    }
